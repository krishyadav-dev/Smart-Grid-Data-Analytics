"""CP4: fault event set on the CP3 year (brief v2 §2D, X.3.1-X.3.3).

Network checks (asserted ones exit non-zero, brief X.0):
- §2D 3PH hand check at 632: engine without shunts vs hand formula, within 2 %;
- relay margins: Is < 730 A and remote 3PH I/Is >= 1.5 (remote = lowest 3PH bus);
- earth-fault security: max feeder-head |3I0| over the CP3 year < 0.5 x EF pickup;
- every simulated burst's IS 17036 classification = the classification of the
  simulated residual voltage and duration (X.3.1 item 10).
Reported: capacitor cross-check, SLG at 652 with Rf 50 ohm and its EF margin,
SLG at 632/652 with mag0 100 % vs 1e5 %, trip / no-trip counts by type x Rf,
the 80 ms breaker sensitivity, the longest burst.

The 200 faults + 200 quiet windows run on the CP3 population at their own
15-min step (DT loads = sum of household load, OLTC tap from the cached state).
Writes ``results/faults_2024/`` (git-ignored, regenerable from the master seed):
``fault_events.parquet`` (ground truth + results per event),
``meter_classes.parquet`` (classification per event and meter),
``summary.json``.

PowerShell::

    python scripts/run_faults.py                 # ~10 min
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pandapower as pp  # noqa: E402

from src.data_source import config, simulation  # noqa: E402
from src.data_source import faults as F  # noqa: E402
from src.data_source.feeder_model import build_network  # noqa: E402

OUT = ROOT / "results" / "faults_2024"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def network_checks(relay: F.RelaySettings) -> dict:
    net = build_network("B")
    pp.runpp_3ph(net)
    sn = F.build_sequence_network(net)
    v_pre = F.prefault_v012(net, sn)
    c632 = F.check_3ph_632(net)
    cap = F.capacitor_check(net, c632)
    margins = F.relay_margins(net, sn, v_pre, relay)
    res = F.head_residual_from_cache()
    slg50 = F.simulate_fault(net, sn, "652", "SLG", "a", 50.0, v_pre)
    t50, el50 = F.relay_operate(relay, float(slg50.i_head_fault_a.max()), slg50.i_head_residual_a)
    mag0 = {}
    for label, val in (("mag0_100pct", 100.0), ("mag0_1e5pct", 1e5)):
        n = copy.deepcopy(net)
        n.trafo.loc[n.trafo.vector_group == "Dyn", "mag0_percent"] = val
        pp.runpp_3ph(n)
        s = F.build_sequence_network(n)
        vp = F.prefault_v012(n, s)
        mag0[label] = {b: float(F.simulate_fault(n, s, b, "SLG", "a", 0.0, vp).i_fault_ka.max())
                       for b in ("632", "652")}
    return {
        "condition_numbers": {"Y0": sn.cond[0], "Y1": sn.cond[1], "Y2": sn.cond[2]},
        "hand_3ph_632": c632, "capacitor_check": cap, "relay_margins": margins,
        "head_residual_year": {
            **res, "ef_pickup_a": relay.ef_is_a, "limit_a": 0.5 * relay.ef_is_a,
            "label": "model-symmetry check, not evidence of relay security: residual ~0 because "
                     "pandapower lines use transposed (symmetric) sequence impedances while the IEEE 13 "
                     "lines are untransposed, the capacitors are symmetric and every load is behind a Dyn DT",
            "passed": res["max_3i0_a"] < 0.5 * relay.ef_is_a},
        "slg_652_rf50": {"i_fault_a": float(slg50.i_fault_ka.max() * 1e3),
                         "i_relay_residual_a": slg50.i_head_residual_a,
                         "i_relay_phase_max_a": float(slg50.i_head_fault_a.max()),
                         "ef_margin": slg50.i_head_residual_a / relay.ef_is_a,
                         "phase_margin": float(slg50.i_head_fault_a.max()) / relay.is_a,
                         "t_relay_s": t50, "element": el50},
        "slg_mag0_before_after_ka": mag0,
        "dyn11_transfer_max_diff_pu_slg652": F.dyn11_transfer_check(net, sn, slg50),
    }


def run_events(relay: F.RelaySettings, events: pd.DataFrame, progress_every: int = 50):
    pop = simulation.build_population()
    net = pop.net
    pp.runpp_3ph(net)
    state = pd.read_parquet(config.YEAR_CACHE_DIR / "state.parquet", columns=["timestamp", "oltc_tap"])
    taps = dict(zip(pd.DatetimeIndex(state.timestamp), state.oltc_tap))
    f = pop.dt_loads(pd.Timestamp(f"{config.SIMULATION_YEAR}-01-01"))
    ev_rows, meter_rows, longest = [], [], None
    t0 = time.perf_counter()
    for n, ev in enumerate(events.itertuples(), 1):
        step = pd.Timestamp(ev.step)
        p, q = f(0, step)
        F.set_prefault(net, p, q, taps[step])
        sn = F.build_sequence_network(net)
        out = F.run_event(net, sn, ev, relay)
        meters = out.pop("meters")
        row = {**ev._asdict(), **out, "tap": taps[step],
               "fault_time": step + pd.Timedelta(seconds=float(ev.offset_s)),
               "cond_max": max(sn.cond)}
        row.pop("Index", None)
        ev_rows.append(row)
        meter_rows += meters
        n_max = max(m["n_samples"] for m in meters)
        if longest is None or n_max > longest["n_samples"]:
            longest = {"event_id": int(ev.event_id), "n_samples": n_max, "kind": ev.kind,
                       "ftype": ev.ftype, "bus": ev.bus, "rf_ohm": ev.rf_ohm}
        if progress_every and n % progress_every == 0:
            logging.info("events: %d / %d (%.1f s)", n, len(events), time.perf_counter() - t0)
    return pd.DataFrame(ev_rows), pd.DataFrame(meter_rows), longest, time.perf_counter() - t0


def summarise(ev: pd.DataFrame, mc: pd.DataFrame, relay: F.RelaySettings) -> dict:
    fl = ev[ev.kind == "fault"].copy()
    fl["cleared"] = fl.cleared.astype(bool)
    fl["rf_class"] = np.where(fl.rf_model == "arc", "arc", fl.rf_ohm.map(lambda r: f"{r:g}"))
    ev = ev.assign(rf_class=np.where(ev.rf_model == "arc", "arc",
                                     ev.rf_ohm.map(lambda r: "" if pd.isna(r) else f"{r:g}")))
    by_type_rf = {}
    for t in F.FAULT_TYPES:
        for rc in (["arc"] if t in F.ARC_FAULTS else [f"{r:g}" for r in F.RF_OHM]):
            s = fl[(fl.ftype == t) & (fl.rf_class == rc)]
            by_type_rf[f"{t}|{rc}"] = {"trip": int(s.cleared.sum()), "no_trip": int((~s.cleared).sum())}
    arc = fl[fl.rf_model == "arc"]
    arc_r = {t: {"n": int((arc.ftype == t).sum()),
                 "min": float(arc[arc.ftype == t].rf_ohm_effective.min()),
                 "median": float(arc[arc.ftype == t].rf_ohm_effective.median()),
                 "max": float(arc[arc.ftype == t].rf_ohm_effective.max())} for t in F.ARC_FAULTS}
    m = mc.merge(ev[["event_id", "kind", "ftype", "rf_class"]], on="event_id")
    mf = m[m.kind == "fault"]

    def class_counts(df):
        return {f"{t}|{rc}|{c}": int(n)
                for (t, rc, c), n in df.groupby(["ftype", "rf_class", "class_kind"]).size().items()}

    head = mf[mf.role == "feeder_head"]
    worst = mf.loc[mf.groupby("event_id").residual_pct.idxmin()]
    quiet = m[m.kind == "quiet"]
    clr = fl[fl.cleared]
    return {
        "faults": int(len(fl)), "quiet_windows": int((ev.kind == "quiet").sum()),
        "mix_note": "fault type drawn uniformly (balanced by design for classifier evaluation, not field "
                    "fault frequencies, brief X.3.3 item 4); SLG/LLG Rf uniform over {0, 5, 20, 50} ohm "
                    "(Dagenhart, ground faults); 3PH/LL arc resistance (Warrington, NPAG Eq. 11.4 p. 188, "
                    f"arc length {F.ARC_LENGTH_M} m verify)",
        "type_counts": fl.ftype.value_counts().to_dict(), "rf_class_counts": fl.rf_class.value_counts().to_dict(),
        "arc_resistance_ohm": arc_r,
        "bus_counts": fl.bus.value_counts().to_dict(),
        "share_cleared": float(fl.cleared.mean()),
        "trip_no_trip_by_type_rf": by_type_rf,
        "arc_faults_no_trip": int((~arc.cleared).sum()),
        "element_counts": fl.element.value_counts().to_dict(),
        "t_clear_s": {"min": float(clr.t_clear_s.min()), "median": float(clr.t_clear_s.median()),
                      "max": float(clr.t_clear_s.max())} if len(clr) else None,
        "t_clear_s_by_type_rf": {f"{t}|{rc}": {"median": float(g.t_clear_s.median()), "max": float(g.t_clear_s.max())}
                                 for (t, rc), g in clr.groupby(["ftype", "rf_class"])},
        "classification_matches": {
            "label": "binning self-consistency check (classifier vs the bins of the simulated residual "
                     "and duration), not validation against measured events",
            "meter_bursts": int(len(mc)), "matches": int(mc.match.sum()),
            "mismatches": int((~mc.match).sum()), "passed": bool(mc.match.all())},
        "feeder_head_class_by_type_rf": class_counts(head),
        "worst_meter_class_by_type_rf": class_counts(worst),
        "quiet_meter_windows": int(len(quiet)),
        "quiet_meter_windows_no_dip": int((quiet.class_kind == "no_dip").sum()),
        "sensitivity_80ms": {
            "meter_bursts_changed": int(mc.changed_by_80ms.sum()),
            "events_with_any_change": int(mc.groupby("event_id").changed_by_80ms.any().sum()),
            "share_cleared_80ms": float(fl.cleared.mean()),
            "t_clear_80ms_median_s": float(clr.t_clear_80ms_s.median()) if len(clr) else None},
        "dyn11_transfer_max_diff_pu": float(fl.dyn11_transfer_max_diff_pu.max()),
        "cond_max": float(ev.cond_max.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-faults", type=int, default=200)
    ap.add_argument("--n-quiet", type=int, default=200)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.simplefilter("ignore", FutureWarning)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    relay = F.relay_settings(F.max_head_current_a())
    checks = network_checks(relay)
    net0 = build_network("B")
    events = F.make_event_set(net0, args.n_faults, args.n_quiet)
    ev, mc, longest, t_run = run_events(relay, events)
    ev.to_parquet(out / "fault_events.parquet", index=False)
    mc.to_parquet(out / "meter_classes.parquet", index=False)
    summary = {"relay": relay.__dict__, "max_head_load_current_a": relay.is_a / F.IS_FACTOR,
               "breaker_s": F.BREAKER_S, "breaker_sensitivity_s": F.BREAKER_SENS_S,
               "checks": checks, "events": summarise(ev, mc, relay), "longest_burst": longest,
               "run_time_s": round(t_run, 1)}
    asserted = {
        "hand_3ph_632_within_2pct": checks["hand_3ph_632"]["passed"],
        "is_below_730a": checks["relay_margins"]["is_below_ampacity"],
        "remote_margin_ge_1_5": checks["relay_margins"]["margin_ok"],
        "ef_security_residual_lt_half_pickup": checks["head_residual_year"]["passed"],
        "classification_matches_simulated": summary["events"]["classification_matches"]["passed"],
        # CP4 follow-up: with arc resistance, 3PH/LL faults must all trip (X.0: stop if not)
        "no_3ph_ll_no_trip": summary["events"]["arc_faults_no_trip"] == 0,
    }
    summary["asserted"] = asserted
    summary["passed"] = all(asserted.values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json))
    if not summary["passed"]:
        print("CP4 ACCEPTANCE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
