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
