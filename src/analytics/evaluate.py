"""Layer 3 evaluator: splits, labels, matching, metrics, tuning (plan §1, §4).

The only Layer 3 module allowed to read labels (``fault_events.parquet``,
``meter_frame_true.parquet``, the P2 scenario labels) and to use the event
record's ``event_id`` (to select a split half and to match alerts to faults).

Definitions (fixed before any run; run sheet §3):

- **Burst window** of a labelled event: from 0.5 s before the fault (the
  record's pre-trigger) to 1.0 s after clearing; a quiet window spans 1.5 s.
- **Matching:** an engine event (or a flat-baseline alert) matches the
  labelled event whose burst window contains its onset.  A test fault is
  detected if at least one alert matches it.  An alert matching no fault
  (including one in a quiet window) is a false positive.  Extra alerts on an
  already-detected fault (a split) are counted and reported, not as false
  positives.
- **Precision / recall / F1:** pooled, P = alerts matching a fault / all
  alerts, R = detected faults / faults.  Per fault type, recall uses the true
  type; precision attributes each true-positive alert to the true type of its
  fault and each false positive to the engine's predicted type.
- **Type and zone** are scored on detected faults, using the earliest
  matching alert.  Bus-level accuracy is scored where the network allows a
  bus answer (not at 671, 680 or 692, which share zone E, F8).  Error
  direction: the predicted node is an ancestor of the true bus (too far
  upstream), a descendant of it (too far downstream), or neither (wrong branch).
- **Wilson 95 % interval** for every proportion (plan §4, z = 1.96).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from src.data_source import faults as F
from src.data_source.config import PROJECT_ROOT, YEAR_CACHE_DIR

from . import fault_engine as E
from . import zones as Z

FAULT_EVENTS = PROJECT_ROOT / "results" / "faults_2024" / "fault_events.parquet"
SPLIT_YAML = PROJECT_ROOT / "config" / "fault_split.yaml"
Z95 = 1.96
NOT_BUS_RESOLVABLE = ("671", "680", "692")          # zone E (F8)

# Step D grids, objectives and tie-breaks (run sheet §2, fixed 27 Oct before any run)
GRID_RHO = tuple(np.round(np.arange(1.05, 3.0001, 0.05), 2))
GRID_WINDOW = tuple(range(1, 11))
GRID_TOL = tuple(np.round(np.arange(0.05, 0.5001, 0.05), 2))
GRID_DELTA = tuple(np.round(np.arange(0.25, 3.0001, 0.25), 2))


# --------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> dict:
    """Wilson score interval (plan §4)."""
    if n == 0:
        return {"k": int(k), "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h)}


# --------------------------------------------------------------------------
# Labels and splits
# --------------------------------------------------------------------------

def load_split(path: str | Path = SPLIT_YAML) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: [int(x) for x in d[k]] for k in ("faults_tune", "faults_test", "quiet_tune", "quiet_test")}


def _ist_ms(ts) -> int:
    return int(pd.Timestamp(ts).tz_localize("Asia/Kolkata").value // 1_000_000)


def load_labels(path: str | Path = FAULT_EVENTS) -> pd.DataFrame:
    """Labelled events with their burst windows (epoch ms) and true zones."""
    ev = pd.read_parquet(path).set_index("event_id").sort_index()
    t_ev = ev.step + pd.to_timedelta(ev.offset_s, unit="s")
    start = t_ev - pd.Timedelta(seconds=F.PRE_S)
    n = []
    for r in ev.itertuples():
        if r.kind == "fault":
            n.append(sum(F.burst_lengths(float(r.t_clear_s))))
        else:
            n.append(F.N_PRE + F.N_POST)
    out = pd.DataFrame(index=ev.index)
    out["kind"] = ev.kind
    out["ftype"] = ev.ftype
    out["phases"] = ev.phases
    out["bus"] = ev.bus
    out["rf_class"] = np.where(ev.rf_model == "arc", "arc",
                               ev.rf_ohm.map(lambda r: "" if pd.isna(r) else f"{r:g}"))
    out["t_event_ms"] = [_ist_ms(t) for t in t_ev]
    out["win_start_ms"] = [_ist_ms(t) for t in start]
    out["win_end_ms"] = out.win_start_ms + np.asarray(n) * F.SAMPLE_MS
    out["t_clear_s"] = ev.t_clear_s
    out["element"] = ev.element
    out["zone"] = [Z.zone_of_bus(b) if k == "fault" else None for b, k in zip(ev.bus, ev.kind)]
    return out


def read_docs(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_docs(docs: Iterable[dict], event_ids: Iterable[int]) -> list[dict]:
    """The burst documents of a split half (by the record's event_id)."""
    ids = set(int(x) for x in event_ids)
    return [d for d in docs if int(d["event_id"]) in ids]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def match_onsets(onsets_ms: Iterable[int], labels: pd.DataFrame) -> list[Optional[int]]:
    """Event id whose burst window contains each onset (None if none)."""
    lab = labels.sort_values("win_start_ms")
    s, e, ids = lab.win_start_ms.to_numpy(), lab.win_end_ms.to_numpy(), lab.index.to_numpy()
    out = []
    for t in onsets_ms:
        k = int(np.searchsorted(s, t, side="right")) - 1
        out.append(int(ids[k]) if k >= 0 and t <= e[k] else None)
    return out


def events_table(events: list, labels: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([e.record() for e in events])
    if df.empty:
        df = pd.DataFrame(columns=["onset_ms"])
    df["matched"] = match_onsets(df.onset_ms, labels)
    return df


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _rel(node: Optional[str], true_bus: str) -> str:
    if node is None:
        return "no_dt_detection"
    if node == true_bus:
        return "same_node"
    if node in Z.path_to_root(true_bus):
        return "upstream"
    if true_bus in Z.path_to_root(node):
        return "downstream"
    return "wrong_branch"


def detection_metrics(ev: pd.DataFrame, labels: pd.DataFrame, fault_ids, quiet_ids,
                      pred_type_col: Optional[str] = "ftype") -> dict:
    lab_f = labels.loc[list(fault_ids)]
    fault_set = set(fault_ids)
    m = ev.matched
    is_tp = m.isin(fault_set)
    n_alerts = int(len(ev))
    tp_alerts = int(is_tp.sum())
    fp = int((~is_tp).sum())
    detected = set(m[is_tp].astype(int))
    rec = wilson(len(detected), len(fault_set))
    prec = wilson(tp_alerts, n_alerts)
    out = {"alerts": n_alerts, "tp_alerts": tp_alerts, "false_positives": fp,
           "precision": prec, "recall": rec, "f1": _f1(prec["p"], rec["p"]),
           "quiet_windows_with_alarm": wilson(len(set(m[m.isin(set(quiet_ids))].astype(int))), len(quiet_ids)),
           "faults_with_more_than_one_alert": int((m[is_tp].value_counts() > 1).sum())}
    per_type = {}
    for t in E.TYPES:
        ids = set(lab_f.index[lab_f.ftype == t])
        tp_t = int(m[is_tp].isin(ids).sum())
        fp_t = int(((~is_tp) & (ev[pred_type_col] == t)).sum()) if pred_type_col else 0
        r = wilson(len(detected & ids), len(ids))
        p = wilson(tp_t, tp_t + fp_t)
        per_type[t] = {"precision": p, "recall": r, "f1": _f1(p["p"], r["p"])}
    out["per_type"] = per_type
    for col, name in (("rf_class", "per_rf_class"), ("bus", "per_bus")):
        out[name] = {str(v): wilson(len(detected & set(g.index)), len(g))
                     for v, g in lab_f.groupby(col)}
    out["detected_ids"] = sorted(detected)
    return out


def _f1(p, r):
    return None if not p or not r else 2 * p * r / (p + r)


def first_alert(ev: pd.DataFrame, fault_ids) -> pd.DataFrame:
    """Earliest matching alert per detected fault."""
    tp = ev[ev.matched.isin(set(fault_ids))].sort_values("onset_ms")
    return tp.groupby("matched").head(1).set_index("matched")


def type_metrics(first: pd.DataFrame, labels: pd.DataFrame) -> dict:
    lab = labels.loc[first.index]
    true, pred = lab.ftype, first.ftype.fillna("undetermined")
    ok = (true == pred)
    two = {"LL": "2PH", "LLG": "2PH"}
    ok2 = true.replace(two) == pred.replace(two)
    conf = pd.crosstab(true, pred).to_dict()
    sub = lab.ftype.isin(["LL", "LLG"])
    return {"accuracy_4class": wilson(int(ok.sum()), len(ok)),
            "accuracy_merged_two_phase": wilson(int(ok2.sum()), len(ok2)),
            "confusion_true_by_pred": {str(p): {str(t): int(n) for t, n in col.items()} for p, col in conf.items()},
            "ll_llg_confusion": pd.crosstab(true[sub], pred[sub]).to_dict(),
            "per_type_accuracy": {t: wilson(int(ok[true == t].sum()), int((true == t).sum())) for t in E.TYPES},
            "per_rf_class_accuracy": {str(c): wilson(int(ok[lab.rf_class == c].sum()), int((lab.rf_class == c).sum()))
                                      for c in sorted(lab.rf_class.unique())}}


def zone_metrics(first: pd.DataFrame, labels: pd.DataFrame) -> dict:
    lab = labels.loc[first.index]
    ok = first.zone == lab.zone
    base_ok = first.baseline_zone == lab.zone
    rel = pd.Series([_rel(n, b) for n, b in zip(first.zone_node, lab.bus)], index=first.index)
    wrong = ~ok
    bus_ok_mask = ~lab.bus.isin(NOT_BUS_RESOLVABLE)
    bus_ok = first.zone_node[bus_ok_mask] == lab.bus[bus_ok_mask]
    out = {"accuracy": wilson(int(ok.sum()), len(ok)),
           "baseline_deepest_dt_accuracy": wilson(int(base_ok.sum()), len(base_ok)),
           "per_type": {t: wilson(int(ok[lab.ftype == t].sum()), int((lab.ftype == t).sum())) for t in E.TYPES},
           "per_rf_class": {str(c): wilson(int(ok[lab.rf_class == c].sum()), int((lab.rf_class == c).sum()))
                            for c in sorted(lab.rf_class.unique())},
           "per_true_zone": {str(z): wilson(int(ok[lab.zone == z].sum()), int((lab.zone == z).sum()))
                             for z in sorted(lab.zone.unique())},
           "baseline_per_type": {t: wilson(int(base_ok[lab.ftype == t].sum()), int((lab.ftype == t).sum()))
                                 for t in E.TYPES},
           "confusion_true_by_pred": {str(p): {str(t): int(n) for t, n in col.items()}
                                      for p, col in pd.crosstab(lab.zone, first.zone.fillna("none")).to_dict().items()},
           "bus_level_accuracy": wilson(int(bus_ok.sum()), int(bus_ok_mask.sum())),
           "error_direction": {k: int(v) for k, v in rel[wrong].value_counts().items()},
           "baseline_error_direction": {k: int(v) for k, v in pd.Series(
               [_rel(b, t) for b, t in zip(first.baseline_bus, lab.bus)], index=first.index)[~base_ok]
               .value_counts().items()}}
    return out


def timing_metrics(first: pd.DataFrame, labels: pd.DataFrame) -> dict:
    lab = labels.loc[first.index]
    delay = (first.onset_ms - lab.t_event_ms).astype(float)
    obs = first.clearing_ms.astype(float) / 1e3
    diff = first.t_expected_clear_s.astype(float) - obs
    clr_err = obs - lab.t_clear_s

    def q(x):
        x = x[np.isfinite(x)]
        return None if not len(x) else {"n": int(len(x)), "min": float(x.min()), "p50": float(x.median()),
                                        "p95": float(np.percentile(x, 95)), "max": float(x.max())}
    return {"detection_delay_ms": q(delay),
            "detection_delay_ms_per_type": {t: q(delay[lab.ftype == t]) for t in E.TYPES},
            "observed_clearing_minus_label_s": q(clr_err),
            "idmt_expected_minus_observed_s": q(diff),
            "idmt_expected_minus_observed_s_by_element": {
                el: q(diff[lab.element == el]) for el in sorted(lab.element.unique())},
            "idmt_phase_element_inf_count": int(np.isinf(first.t_expected_clear_s.astype(float)).sum())}


def no_dip_metrics(first: pd.DataFrame, ev_an: dict, labels: pd.DataFrame, fault_ids) -> dict:
    """F2: test faults with no dip at the feeder head (head Step 1 = no_dip)."""
    ids = [e for e in fault_ids if ev_an.get(e, {}).get("head_kind") == "no_dip"
           and not ev_an[e]["head_swell"]]
    det = [e for e in ids if e in first.index]
    by_dt_step1 = [e for e in det if ev_an[e]["dt_step1"]]
    rise_only = [e for e in det if not ev_an[e]["dt_step1"] and ev_an[e]["rise"]]
    return {"no_dip_faults": len(ids), "detected": wilson(len(det), len(ids)),
            "found_by_step1_at_a_dt_meter": len(by_dt_step1), "found_only_by_current_rise": len(rise_only),
            "head_current_rise_detected": int(sum(ev_an[e]["head_rise"] for e in ids))}


def burst_facts(an: list, labels: pd.DataFrame) -> dict:
    """Per labelled event: head Step 1 kind, DT Step 1 detections, rises (from analyses)."""
    ids = match_onsets([a.t0_ms + F.N_PRE * F.SAMPLE_MS for a in an], labels)
    out: dict = {}
    for a, e in zip(an, ids):
        if e is None:
            continue
        d = out.setdefault(e, {"head_kind": None, "head_swell": False, "dt_step1": False, "rise": False,
                               "head_rise": False})
        if a.role == "feeder_head":
            d["head_kind"] = a.kind
            d["head_swell"] = "swell" in a.by
            d["head_rise"] = "current_rise" in a.by
        elif "is17036" in a.by:
            d["dt_step1"] = True
        if "current_rise" in a.by:
            d["rise"] = True
    return out


def flat_metrics(alerts_ms: list[int], labels: pd.DataFrame, fault_ids, quiet_ids) -> dict:
    m = pd.Series(match_onsets(alerts_ms, labels), dtype="object")
    fset, qset = set(fault_ids), set(quiet_ids)
    is_tp = m.isin(fset)
    detected = set(m[is_tp].astype(int))
    prec = wilson(int(is_tp.sum()), len(m))
    rec = wilson(len(detected), len(fset))
    lab_f = labels.loc[list(fault_ids)]
    return {"alerts": int(len(m)), "tp_alerts": int(is_tp.sum()), "false_positives": int((~is_tp).sum()),
            "precision": prec, "recall": rec, "f1": _f1(prec["p"], rec["p"]),
            "quiet_windows_with_alarm": wilson(len(set(m[m.isin(qset)].astype(int))), len(qset)),
            "per_type_recall": {t: wilson(len(detected & set(lab_f.index[lab_f.ftype == t])),
                                          int((lab_f.ftype == t).sum())) for t in E.TYPES},
            "per_rf_class_recall": {str(c): wilson(len(detected & set(g.index)), len(g))
                                    for c, g in lab_f.groupby("rf_class")}}


# --------------------------------------------------------------------------
# Unbalance truth (feeder head, 2024) and P2 scoring
# --------------------------------------------------------------------------

def head_weekly_truth(cache: Path = YEAR_CACHE_DIR) -> pd.DataFrame:
    """Feeder-head V_N / V_P per step from the full phasors of meter_frame_true, weekly."""
    from src.data_source.unbalance_scenarios import vu_true
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "phase", "voltage_v", "voltage_angle_deg"],
                        filters=[("load_id", "==", "feeder_head_650")])
    v = vu_true(f)
    v["week"] = v.timestamp.dt.normalize() - pd.to_timedelta(v.timestamp.dt.weekday, unit="D")
    g = v.groupby("week").vu_true_pct
    out = g.agg(samples="size", vu_max_pct="max").reset_index()
    out["share_above_limit"] = g.apply(lambda x: float((x > E.VU_LIMIT_PCT).mean())).to_numpy()
    out["exceeds"] = out.share_above_limit > E.WEEK_EXCEED_SHARE
    return out


def p2_scores(engine_weekly: pd.DataFrame, labels_weekly: pd.DataFrame) -> dict:
    """Alarm precision / recall per meter (scenario-level weekly flags)."""
    m = engine_weekly.merge(labels_weekly, on=["scenario_id", "load_id"], validate="one_to_one")
    out = {}
    for lid, g in m.groupby("load_id"):
        tp = int((g.alarm & g.exceeds_limit).sum())
        fp = int((g.alarm & ~g.exceeds_limit).sum())
        fn = int((~g.alarm & g.exceeds_limit).sum())
        out[lid] = {"positives": int(g.exceeds_limit.sum()), "alarms": int(g.alarm.sum()),
                    "precision": wilson(tp, tp + fp), "recall": wilson(tp, tp + fn)}
    return out


# --------------------------------------------------------------------------
# Step D: tuning on the tuning half only
# --------------------------------------------------------------------------

def best_plateau_middle(grid, scores) -> tuple[float, list]:
    """Middle of the best plateau: the longest run of consecutive grid values
    with the maximum score (ties: the lowest run); its middle value, the lower
    middle when the run has an even length."""
    scores = np.asarray(scores, float)
    best = np.nanmax(scores)
    runs, cur = [], []
    for k, s in enumerate(scores):
        if s == best:
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    run = max(runs, key=len)                        # max() keeps the first (lowest) on ties
    return grid[run[(len(run) - 1) // 2]], [grid[k] for k in run]


def tune(docs_tune: list[dict], labels: pd.DataFrame, split: dict) -> dict:
    """Run sheet Step D, in order: rho*, coincidence window, phase-balance
    tolerance, delta (decisions of 27 Oct for the rho* objective and merging)."""
    relay = E.load_relay()
    bursts = [E.burst_from_doc(d) for d in docs_tune]
    ft, qt = split["faults_tune"], split["quiet_tune"]
    ft_set, qt_set = set(ft), set(qt)
    # rho* --------------------------------------------------------------------
    an0 = [E.analyse(b, GRID_RHO[0]) for b in bursts]
    facts = burst_facts(an0, labels)
    # decision 27 Oct: the tuning faults with no IS 17036 detection (dip, interruption, swell) at the head
    target = sorted(e for e in ft if facts[e]["head_kind"] == "no_dip" and not facts[e]["head_swell"])
    rho_curve = []
    an_by_rho = {}
    for r in GRID_RHO:
        an = [E.analyse(b, r) for b in bursts]
        an_by_rho[r] = an
        ids = match_onsets([a.onset_ms if a.detected else -1 for a in an], labels)
        rise_found = {e for a, e in zip(an, ids) if a.detected and "current_rise" in a.by and e in set(target)}
        quiet_det = sum(1 for a, e in zip(an, ids) if a.detected and (e in qt_set or e is None))
        rho_curve.append({"rho": float(r), "recall": len(rise_found) / len(target) if target else None,
                          "found": len(rise_found), "target": len(target), "quiet_detections": quiet_det})
    feas = [c for c in rho_curve if c["quiet_detections"] == 0]
    if not feas:
        return {"stop": "no rho* gives zero events on the tuning quiet windows", "rho_curve": rho_curve}
    best_rec = max(c["found"] for c in feas)
    rho_star = max(c["rho"] for c in feas if c["found"] == best_rec)
    an = an_by_rho[rho_star]
    # coincidence window ----------------------------------------------------------
    win_curve = []
    for w in GRID_WINDOW:
        evs = E.merge(an, w)
        mt = match_onsets([e.onset_ms for e in evs], labels)
        per_fault = pd.Series([x for x in mt if x in ft_set]).value_counts()
        # an event merging two different faults: its members' bursts match two fault windows
        two = 0
        for e in evs:
            ids = {x for x in match_onsets([a.t0_ms + F.N_PRE * F.SAMPLE_MS for a in e.members], labels)
                   if x in ft_set}
            two += len(ids) > 1
        win_curve.append({"window_samples": w, "faults_split": int((per_fault > 1).sum()),
                          "events_merging_two_faults": two, "events": len(evs)})
    key = min((c["faults_split"], c["events_merging_two_faults"]) for c in win_curve)
    window = min(c["window_samples"] for c in win_curve
                 if (c["faults_split"], c["events_merging_two_faults"]) == key)
    # phase-balance tolerance and delta on detected tuning faults --------------
    tol_curve, delta_curve = [], []
    evs = E.merge(an, window)
    for tol in GRID_TOL:
        p = E.Params(rho_star, window, float(tol), GRID_DELTA[0])
        first = first_alert(events_table([E.finalise(_copy(e), p, relay) for e in evs], labels), ft)
        acc = type_metrics(first, labels)["accuracy_4class"]
        tol_curve.append({"balance_tol": float(tol), "accuracy": acc["p"], "correct": acc["k"], "n": acc["n"]})
    tol, tol_plateau = best_plateau_middle(GRID_TOL, [c["correct"] for c in tol_curve])
    for d in GRID_DELTA:
        p = E.Params(rho_star, window, float(tol), float(d))
        first = first_alert(events_table([E.finalise(_copy(e), p, relay) for e in evs], labels), ft)
        acc = zone_metrics(first, labels)["accuracy"]
        delta_curve.append({"delta_pp": float(d), "accuracy": acc["p"], "correct": acc["k"], "n": acc["n"]})
    delta, delta_plateau = best_plateau_middle(GRID_DELTA, [c["correct"] for c in delta_curve])
    strict = E.merge(an, window, mode="strict")
    mt = pd.Series([x for x in match_onsets([e.onset_ms for e in strict], labels) if x in ft_set]).value_counts()
    return {"frozen": {"rho_star": float(rho_star), "coincidence_samples": int(window),
                       "balance_tol": float(tol), "delta_pp": float(delta)},
            "rho_target_faults": target, "rho_curve": rho_curve, "window_curve": win_curve,
            "tol_curve": tol_curve, "tol_plateau": [float(x) for x in tol_plateau],
            "delta_curve": delta_curve, "delta_plateau": [float(x) for x in delta_plateau],
            "strict_onset_faults_split_tuning": int((mt > 1).sum())}


def _copy(e):
    import copy
    return copy.copy(e)



# ==========================================================================
# Theft engine (plan §2; Stage 2 run sheet §3-4)
# ==========================================================================
# Truth read here only: households.parquet (thief, attack, onset), the P3 label
# file (theft_kwh, offline_kwh), household_daily's true / reported kWh,
# meter_frame_true and state.parquet (true losses).

NTL_TUNE = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"))      # run sheet §0
NTL_TEST = (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"))
NTL_YEAR = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
NTL_WINDOWS = {"jul_dec_test": NTL_TEST, "jan_jun_reference": NTL_TUNE}
ENERGY_THEFT = ("h1", "h2", "h3", "h4")                                   # N4
TARIFF_SHIFT = ("h5", "h6")


def load_household_labels(cache: Path = YEAR_CACHE_DIR) -> pd.DataFrame:
    """household -> dt, thief, attack, onset (Timestamp or NaT)."""
    h = pd.read_parquet(Path(cache) / "households.parquet",
                        columns=["household", "dt", "thief", "attack", "onset"])
    h["onset"] = pd.to_datetime(h.onset.replace("", None))
    return h.set_index("household")


def theft_truth(cache: Path = YEAR_CACHE_DIR) -> dict:
    """Daily energy-theft truth (h1-h4; P3: true - would-have-reported on
    attack-active days, offline days included) per feeder-day and DT-day, the
    offline meters per DT-day and the would-have-reported offline energy."""
    hh = load_household_labels(cache)
    lab = pd.read_parquet(Path(cache) / "household_daily_labels.parquet",
                          columns=["household", "date", "offline", "theft_kwh", "offline_kwh"])
    lab["dt"] = lab.household.map(hh.dt)
    lab["energy_theft_kwh"] = np.where(lab.household.map(hh.attack).isin(ENERGY_THEFT), lab.theft_kwh, 0.0)
    lab["date"] = pd.to_datetime(lab.date)
    g = lab.groupby("date")
    gd = lab.groupby(["date", "dt"])
    return {
        "feeder_theft": g.energy_theft_kwh.sum(),
        "feeder_theft_all_attacks": g.theft_kwh.sum(),
        "feeder_offline": g.offline_kwh.sum(),
        "dt_theft": gd.energy_theft_kwh.sum().unstack(),
        "dt_offline_meters": gd.offline.sum().unstack(),
    }


def bias_mae(est: pd.Series, truth: pd.Series) -> dict:
    """Bias and MAE (kWh) of an estimate against the truth, aligned on the index."""
    e, t = est.align(truth, join="inner")
    err = e - t
    return {"n": int(len(err)), "bias_kwh": float(err.mean()), "mae_kwh": float(err.abs().mean()),
            "corr": float(np.corrcoef(e, t)[0, 1]) if len(e) > 1 and t.std() > 0 else None,
            "mean_truth_kwh": float(t.mean()), "mean_estimate_kwh": float(e.mean())}


def _in(idx, w) -> np.ndarray:
    d = pd.DatetimeIndex(idx)
    return np.asarray((d >= w[0]) & (d <= w[1]))


def _dt_long(df: pd.DataFrame, name: str) -> pd.Series:
    s = df.stack()
    s.index.names = ["date", "dt"]
    return s.rename(name)


def score_feeder(estimates: dict[str, pd.Series], truth: dict) -> dict:
    """Feeder-day bias / MAE per estimate (netted, un-netted C), per window."""
    t = truth["feeder_theft"]
    return {w: {name: bias_mae(s[_in(s.index, r)], t) for name, s in estimates.items()}
            for w, r in NTL_WINDOWS.items()}


def score_dt(estimates: dict[str, pd.DataFrame], truth: dict, windows=None) -> dict:
    """DT-day bias / MAE per estimate, per window, on all DT-days and on the
    fully online DT-days (every household meter on the DT online)."""
    t = _dt_long(truth["dt_theft"], "theft")
    online = _dt_long(truth["dt_offline_meters"], "off") == 0
    out = {}
    for w, r in (windows or NTL_WINDOWS).items():
        out[w] = {}
        for name, df in estimates.items():
            e = _dt_long(df, name)
            e = e[_in(e.index.get_level_values("date"), r)]
            fo = online.reindex(e.index).fillna(False).to_numpy(bool)
            out[w][name] = {"all_dt_days": bias_mae(e, t), "fully_online_dt_days": bias_mae(e[fo], t)}
    return out


def ntl_energy_identity(window=NTL_TEST, cache: Path = YEAR_CACHE_DIR) -> dict:
    """Feeder energy identity on a window (as tests/test_energy_identity.py on
    the year): head metered - billed = MV and DT losses + theft + offline +
    meter error.  Raises if it does not close within the cache's own
    tolerance (power-flow balance error x steps + float32 rounding)."""
    cache = Path(cache)
    end = window[1] + pd.Timedelta(days=1)
    f = [("load_id", "==", "feeder_head_650"), ("timestamp", ">=", window[0]), ("timestamp", "<", end)]
    cols = ["timestamp", "load_id", "active_power_kw"]
    head_true = pd.read_parquet(cache / "meter_frame_true.parquet", columns=cols,
                                filters=f).active_power_kw.sum() * 0.25
    head_met = pd.read_parquet(cache / "meter_frame_metered.parquet", columns=cols,
                               filters=f).active_power_kw.sum() * 0.25
    st = pd.read_parquet(cache / "state.parquet", columns=["timestamp", "line_losses_mw", "dt_losses_mw"],
                         filters=f[1:])
    df = [("date", ">=", window[0]), ("date", "<=", window[1])]
    hd = pd.read_parquet(cache / "household_daily.parquet",
                         columns=["date", "true_kwh", "reported_kwh", "metered_kwh"], filters=df)
    lab = pd.read_parquet(cache / "household_daily_labels.parquet", columns=["date", "theft_kwh", "offline_kwh"],
                          filters=df)
    acc = json.loads((cache / "acceptance.json").read_text(encoding="utf-8"))
    true = hd.true_kwh.to_numpy(float)
    billed = float(np.nansum(hd.metered_kwh.to_numpy(float)))
    reported = float(np.nansum(hd.reported_kwh.to_numpy(float)))
    losses = float((st.line_losses_mw + st.dt_losses_mw).sum() * 1e3 * 0.25)
    theft, offline = float(lab.theft_kwh.sum()), float(lab.offline_kwh.sum())
    meter_error = float((head_met - head_true) - (billed - reported))
    resid = float((head_met - billed) - (losses + theft + offline + meter_error))
    tol = float(acc["feeder_balance_max_abs_mw"] * 1e3 * 0.25 * len(st) + 2.0 ** -23 * np.abs(true).sum())
    rec = {"window": [str(window[0].date()), str(window[1].date())], "steps": int(len(st)),
           "household_days": int(len(hd)), "head_metered_kwh": float(head_met), "billed_kwh": billed,
           "losses_kwh": losses, "theft_kwh": theft, "offline_kwh": offline, "meter_error_kwh": meter_error,
           "residual_kwh": resid, "tolerance_kwh": tol, "closes": bool(abs(resid) <= tol)}
    if not rec["closes"]:
        raise AssertionError(f"feeder energy identity does not close: {rec}")
    return rec


def true_losses_daily(cache: Path = YEAR_CACHE_DIR) -> pd.DataFrame:
    """True MV line and DT losses per day (kWh), from the power-flow state."""
    st = pd.read_parquet(Path(cache) / "state.parquet",
                         columns=["timestamp", "line_losses_mw", "dt_losses_mw"]).set_index("timestamp")
    d = st[["line_losses_mw", "dt_losses_mw"]] * 1e3 * 0.25
    d = d.groupby(pd.DatetimeIndex(d.index).normalize()).sum()
    d.columns = ["line_losses_kwh", "dt_losses_kwh"]
    return d


def ltech_vs_true(ltech_day: pd.DataFrame, cache: Path = YEAR_CACHE_DIR) -> dict:
    """L^_tech against the true losses (year and per window), MWh."""
    t = true_losses_daily(cache)
    out = {}
    for w, r in {"year": NTL_YEAR, **NTL_WINDOWS}.items():
        e, tt = ltech_day[_in(ltech_day.index, r)], t[_in(t.index, r)]
        out[w] = {"ltech_mwh": float(e.ltech_kw.sum() / 1e3),
                  "ltech_lines_mwh": float(e.line_losses_kw.sum() / 1e3),
                  "ltech_dts_mwh": float(e.dt_losses_kw.sum() / 1e3),
                  "true_mwh": float((tt.line_losses_kwh + tt.dt_losses_kwh).sum() / 1e3),
                  "true_lines_mwh": float(tt.line_losses_kwh.sum() / 1e3),
                  "true_dts_mwh": float(tt.dt_losses_kwh.sum() / 1e3)}
    return out


def offline_vs_true(eoff_day: pd.Series, truth: dict) -> dict:
    """E^_offline against the would-have-reported offline energy, MWh, and per day."""
    out = {}
    for w, r in {"year": NTL_YEAR, **NTL_WINDOWS}.items():
        e = eoff_day[_in(eoff_day.index, r)]
        out[w] = {"estimate_mwh": float(e.sum() / 1e3),
                  "true_mwh": float(truth["feeder_offline"][_in(truth["feeder_offline"].index, r)].sum() / 1e3),
                  "per_day": bias_mae(e, truth["feeder_offline"])}
    return out


def worked_example(est: dict, truth: dict, cache: Path = YEAR_CACHE_DIR) -> dict:
    """The plan's 2024 feeder-balance table (§2), engine-estimated and true-loss
    versions, MWh and share of the head energy."""
    cache = Path(cache)
    head_true = pd.read_parquet(cache / "meter_frame_true.parquet", columns=["load_id", "active_power_kw"],
                                filters=[("load_id", "==", "feeder_head_650")]).active_power_kw.sum() * 0.25
    t = true_losses_daily(cache)
    billed = float(est["billed"].sum())
    theft_e = float(truth["feeder_theft"].sum())

    def table(head, lines, dts, offline):
        c = head - billed
        net = c - lines - dts - offline
        mwh = {"head_energy": head, "household_metered": billed, "C_un_netted": c,
               "technical_losses_lines": lines, "technical_losses_dts": dts, "offline": offline,
               "netted_estimate": net, "true_energy_theft_h1_h4": theft_e,
               "netted_minus_true_theft": net - theft_e}
        return {**{k: v / 1e3 for k, v in mwh.items()},
                "share_of_head": {k: mwh[k] / head for k in ("C_un_netted", "technical_losses_lines",
                                                            "technical_losses_dts", "offline",
                                                            "netted_estimate", "true_energy_theft_h1_h4")},
                "un_netted_over_true_theft": c / theft_e}

    ld = est["ltech_day"]
    return {
        "units": "MWh (except share_of_head and the ratio)",
        "engine_estimated": table(float(est["head_day"].sum()), float(ld.line_losses_kw.sum()),
                                  float(ld.dt_losses_kw.sum()), float(est["eoff_day"].sum())),
        "true_losses_and_offline": table(float(head_true), float(t.line_losses_kwh.sum()),
                                         float(t.dt_losses_kwh.sum()), float(truth["feeder_offline"].sum())),
        "true_theft_all_attacks_mwh": float(truth["feeder_theft_all_attacks"].sum() / 1e3),
        "note": "engine_estimated: metered head energy, L^_tech and E^_offline; true_losses_and_offline: "
                "the plan's table (true head, true losses, would-have-reported offline energy)"}


# --------------------------------------------------------------------------
# Consumer detector scoring (run sheet §4)
# --------------------------------------------------------------------------

def fold_targets(labels: pd.DataFrame, households, fold_attacks, window) -> pd.Series:
    """Household targets for one fold and window: 1 = thief whose attack is in
    ``fold_attacks`` and active on at least one day of ``window`` (onset on or
    before its end; attacks persist after onset), 0 = honest, NaN = excluded
    (other thieves: held-out attack in tuning, other attacks in test)."""
    h = labels.reindex(list(households))
    pos = h.thief & h.attack.isin(list(fold_attacks)) & (h.onset <= pd.Timestamp(window[1]))
    y = pd.Series(np.nan, index=h.index)
    y[~h.thief.astype(bool)] = 0.0
    y[pos.astype(bool)] = 1.0
    return y


def prf(pred: np.ndarray, y: np.ndarray) -> dict:
    """Precision, recall, F1 with Wilson intervals for P and R."""
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p, r = wilson(tp, tp + fp), wilson(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": _f1(p["p"], r["p"])}


def best_threshold(score: pd.Series, y: pd.Series) -> dict:
    """q maximising household-level F1 (flag if score >= q) over the observed
    scores; ties: the lowest q (run sheet §4 tuning step 3)."""
    s = pd.concat([score, y], axis=1, keys=["s", "y"]).dropna()
    if s.y.sum() == 0 or (s.y == 0).sum() == 0:
        raise ValueError("tuning objective undefined: no positives or no negatives (run sheet §6)")
    best = None
    for q in np.unique(s.s.to_numpy()):
        pred = s.s.to_numpy() >= q
        f1 = prf(pred, s.y.to_numpy() == 1)["f1"] or 0.0
        if best is None or f1 > best[1]:
            best = (float(q), f1)
    return {"q": best[0], "f1": best[1], "positives": int(s.y.sum()), "negatives": int((s.y == 0).sum())}


PR_POINTS = 201          # stored points per PR curve (evenly spaced over the thresholds)


def pr_curve(score: pd.Series, y: pd.Series) -> dict:
    """Precision-recall curve and average precision (scikit-learn)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    s = pd.concat([score, y], axis=1, keys=["s", "y"]).dropna()
    p, r, t = precision_recall_curve(s.y.to_numpy(), s.s.to_numpy())
    keep = np.unique(np.linspace(0, len(t) - 1, min(len(t), PR_POINTS)).round().astype(int))
    return {"average_precision": float(average_precision_score(s.y.to_numpy(), s.s.to_numpy())),
            "prevalence": float(s.y.mean()), "points": int(len(t)), "stored_points": int(len(keep)),
            "precision": [float(x) for x in p[keep]], "recall": [float(x) for x in r[keep]],
            "thresholds": [float(x) for x in t[keep]]}


def fold_metrics(score: pd.Series, y: pd.Series, q: float) -> dict:
    s = pd.concat([score, y], axis=1, keys=["s", "y"]).dropna()
    m = prf(s.s.to_numpy() >= q, s.y.to_numpy() == 1)
    return {"q": q, "households_scored": int(len(s)), "positives": int(s.y.sum()), **m,
            "pr_curve": pr_curve(score, y)}


def macro(per_fold: dict, folds) -> dict:
    """Unweighted means of precision, recall, F1 and AP over ``folds``."""
    def mean(f):
        v = [f(per_fold[k]) for k in folds]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    return {"folds": list(folds), "precision": mean(lambda m: m["precision"]["p"]),
            "recall": mean(lambda m: m["recall"]["p"]), "f1": mean(lambda m: m["f1"]),
            "average_precision": mean(lambda m: m["pr_curve"]["average_precision"])}


def by_onset_month(scores: dict[str, pd.Series], q: dict[str, float], labels: pd.DataFrame,
                   groups: dict[str, tuple]) -> dict:
    """Recall by onset month, pooled over the folds of each group, each thief
    scored in the fold that holds its attack out (Wilson intervals)."""
    out = {}
    for g, folds in groups.items():
        rows = []
        for k in folds:
            th = labels[labels.thief.astype(bool) & (labels.attack == k)]
            s = scores[k].reindex(th.index)
            rows.append(pd.DataFrame({"month": th.onset.dt.month, "hit": s >= q[k], "scored": s.notna()}))
        r = pd.concat(rows)
        r = r[r.scored]
        out[g] = {int(m): wilson(int(d.hit.sum()), len(d)) for m, d in r.groupby("month")}
    return out

# --------------------------------------------------------------------------
# SGCC check: Zheng Wide & Deep CNN (run sheet §5)
# --------------------------------------------------------------------------
SGCC_TEST_POOL = YEAR_CACHE_DIR / "sgcc_detector_test.parquet"
# Zheng et al. 2018 Table III, Wide & Deep CNN at an 80 % training ratio
ZHENG_TABLE_III_80 = {"auc": 0.7815, "map_at_100": 0.9503, "map_at_200": 0.9093}


def sgcc_test_flags(path: Path = SGCC_TEST_POOL) -> pd.Series:
    """CONS_NO -> FLAG of the SGCC test pool (scoring only)."""
    d = pd.read_parquet(path, columns=["CONS_NO", "FLAG"])
    return d.set_index("CONS_NO").FLAG.astype(int)


def map_at_n(score: np.ndarray, y: np.ndarray, n: int) -> float:
    """Zheng §V: MAP@N = (1 / r) sum_{i=1..r} P@k_i, k_i the rank of the i-th
    thief among the top N by score, r the number of thieves in the top N."""
    order = np.argsort(-np.asarray(score), kind="stable")[:n]
    hits = np.asarray(y)[order] == 1
    if not hits.any():
        return 0.0
    k = np.flatnonzero(hits) + 1
    return float(np.mean(np.arange(1, len(k) + 1) / k))


def threshold_max_f1(score: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising F1 (flag if score >= t); ties: the lowest t."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, score)
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-300), 0.0)
    return float(t[int(np.flatnonzero(f1 == f1.max())[0])])


def sgcc_metrics(score: pd.Series, flags: pd.Series, threshold: float) -> dict:
    """AUC, MAP@100 / @200, P / R / F1 at ``threshold`` (Wilson) and the PR curve."""
    from sklearn.metrics import roc_auc_score
    s, y = score.align(flags, join="inner")
    s, y = s.to_numpy(), y.to_numpy()
    return {"n": int(len(y)), "thieves": int(y.sum()), "auc": float(roc_auc_score(y, s)),
            "map_at_100": map_at_n(s, y, 100), "map_at_200": map_at_n(s, y, 200),
            "threshold": threshold, **prf(s >= threshold, y == 1),
            "pr_curve": pr_curve(pd.Series(s), pd.Series(y.astype(float)))}


# --------------------------------------------------------------------------
# Thermal engine (plan §3; Stage 3 run sheet §3, §5)
# --------------------------------------------------------------------------

THERMAL_WARMUP = pd.Timestamp("2024-01-01")        # run sheet §3: excluded from every metric
THERMAL_TEST = (pd.Timestamp("2024-05-01"), pd.Timestamp("2024-12-31"))
TRUE_FRAME = YEAR_CACHE_DIR / "meter_frame_true.parquet"


def thermal_true_current_check(theta_h_end_metered: np.ndarray, theta_h_max_metered: np.ndarray,
                               ambient: np.ndarray, cache: Path = YEAR_CACHE_DIR) -> dict:
    """Run sheet §3: each DT once on the true currents (parameter set ``main``);
    the largest hot-spot difference to the metered reference (warm-up day excluded)."""
    from . import thermal as TH
    idx, i_true = TH.read_dt_currents(cache / TRUE_FRAME.name)
    p = TH.dt_params(TH.read_dts(cache / "dts.parquet"), "main")
    r = TH.reference_run(p, TH.load_factor(i_true, TH.rated_current()), ambient)
    keep = np.asarray(idx.normalize() != THERMAL_WARMUP)
    d_end = r["theta_h_end"][keep] - theta_h_end_metered[keep]
    d_max = r["theta_h_max"][keep] - theta_h_max_metered[keep]
    j, n = np.unravel_index(int(np.argmax(np.abs(d_end))), d_end.shape)
    rise = r["theta_h_max"][keep] - np.asarray(ambient)[keep][:, None]
    jr, nr = np.unravel_index(int(np.argmax(rise)), rise.shape)
    return {"max_abs_diff_theta_h_k": float(np.abs(d_end).max()),
            "at": {"timestamp": str(idx[keep][j]), "dt": TH.dt_names()[n]},
            "max_abs_diff_step_peak_k": float(np.abs(d_max).max()),
            "mean_diff_k": float(d_end.mean()),
            "per_dt_max_abs_diff_k": dict(zip(TH.dt_names(), np.abs(d_end).max(axis=0).round(4).tolist())),
            "true_peak_theta_h_c": float(r["theta_h_max"][keep].max()),
            "true_peak_rise_k": {"value": float(rise.max()), "dt": TH.dt_names()[nr],
                                 "timestamp": str(idx[keep][jr])}}


# Step E metrics (run sheet §5), fixed before the test run
THERMAL_HORIZONS = {"1h": 3, "6h": 23, "24h": 95}     # forecast steps 4, 24, 96 (end of each step)
THERMAL_MIN_EVENTS = 10                                # T4: P / R only with >= 10 event DT-days
THERMAL_BOOT_N = 2000
THERMAL_BOOT_SEED = 20261028


def _rmse_mae(e: np.ndarray) -> dict:
    e = np.asarray(e, float)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "n": int(e.size)}


def hot_spot_errors(th: np.ndarray, ref_th: np.ndarray, names) -> dict:
    """theta_h forecast vs reference, [origin, 96, DT]: RMSE / MAE at 1, 6, 24 h and over all 96
    steps, pooled over DTs and origins and per DT."""
    e = th - ref_th
    out = {"pooled": {h: _rmse_mae(e[:, i, :]) for h, i in THERMAL_HORIZONS.items()}}
    out["pooled"]["all_96"] = _rmse_mae(e)
    out["per_dt"] = {n: {**{h: _rmse_mae(e[:, i, j]) for h, i in THERMAL_HORIZONS.items()},
                         "all_96": _rmse_mae(e[:, :, j])} for j, n in enumerate(names)}
    return out


def load_errors(fc: np.ndarray, actual: np.ndarray, names, n_ph: int = 3) -> dict:
    """K_p forecast vs actual (diagnostic), [origin, 96, series]: pooled and per DT (over its phases)."""
    e = fc - actual
    return {"pooled": _rmse_mae(e),
            "per_dt": {n: _rmse_mae(e[:, :, j * n_ph:(j + 1) * n_ph]) for j, n in enumerate(names)}}


def lol_errors(lol_fc: np.ndarray, lol_ref: np.ndarray) -> dict:
    """Daily loss-of-life error |L^ - L| per DT-day (h of normal life per day), [origin, DT]."""
    e = np.abs(lol_fc - lol_ref)
    return {"mean_h": float(e.mean()), "median_h": float(np.median(e)), "p95_h": float(np.percentile(e, 95)),
            "n_dt_days": int(e.size), "reference_mean_h": float(lol_ref.mean())}


def event_metrics(fc_first: np.ndarray, ref_first: np.ndarray) -> dict:
    """One threshold, [origin, DT] first-crossing minutes after 00:00 (NaN = none).
    Event = a DT-day whose reference crosses (run sheet §3)."""
    ev, pred = ~np.isnan(ref_first), ~np.isnan(fc_first)
    det = ev & pred
    lead = ref_first[det] / 60.0
    terr = fc_first[det] - ref_first[det]
    out = {"event_dt_days": int(ev.sum()), "detected": int(det.sum()),
           "forecast_crossing_dt_days": int(pred.sum()), "forecast_only_dt_days": int((pred & ~ev).sum()),
           "lead_time_h": ({"mean": float(lead.mean()), "median": float(np.median(lead)), "min": float(lead.min()),
                            "max": float(lead.max())} if det.any() else None),
           "crossing_time_error_min": ({"mean": float(terr.mean()), "median": float(np.median(terr)),
                                        "mae": float(np.abs(terr).mean())} if det.any() else None)}
    if ev.sum() >= THERMAL_MIN_EVENTS:
        tp, fp = int(det.sum()), int((pred & ~ev).sum())
        out["precision"] = wilson(tp, tp + fp)
        out["recall"] = wilson(tp, int(ev.sum()))
    else:
        out["precision_recall"] = f"not reported: fewer than {THERMAL_MIN_EVENTS} event DT-days (T4)"
    return out


def paired_bootstrap(th: dict, ref_th: np.ndarray, lol: dict, lol_ref: np.ndarray,
                     n: int = THERMAL_BOOT_N, seed: int = THERMAL_BOOT_SEED) -> dict:
    """Plan §4 / run sheet §5: resample test days with replacement (a day carries all 13 DTs);
    RMSE(persistence) - RMSE(LSTM), RMSE(seasonal naive) - RMSE(LSTM) over all 96 steps, and the
    mean daily loss-of-life error difference (persistence - LSTM).  2.5th-97.5th percentiles; the
    LSTM is better only if the whole interval is above zero."""
    sse = {m: ((v - ref_th) ** 2).sum(axis=(1, 2)) for m, v in th.items()}        # per day
    cnt = ref_th.shape[1] * ref_th.shape[2]
    lae = {m: np.abs(v - lol_ref).sum(axis=1) for m, v in lol.items()}             # per day
    n_dt = lol_ref.shape[1]
    days = len(ref_th)
    idx = np.random.default_rng(seed).integers(0, days, (n, days))

    def rmse(m, ix=None):
        s = sse[m] if ix is None else sse[m][ix]
        return np.sqrt(s.sum(axis=-1) / (cnt * s.shape[-1]))

    def lol_mean(m, ix=None):
        s = lae[m] if ix is None else lae[m][ix]
        return s.sum(axis=-1) / (n_dt * s.shape[-1])

    def stat(point, boot):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"point": float(point), "ci95": [float(lo), float(hi)], "lstm_better": bool(lo > 0)}

    return {"resamples": n, "seed": seed, "days": days,
            "rmse_persistence_minus_lstm_k": stat(rmse("persistence") - rmse("lstm"),
                                                  rmse("persistence", idx) - rmse("lstm", idx)),
            "rmse_seasonal_naive_minus_lstm_k": stat(rmse("seasonal_naive") - rmse("lstm"),
                                                     rmse("seasonal_naive", idx) - rmse("lstm", idx)),
            "lol_error_persistence_minus_lstm_h": stat(lol_mean("persistence") - lol_mean("lstm"),
                                                       lol_mean("persistence", idx) - lol_mean("lstm", idx))}
