"""Layer 3 fault engine: batch documents, tuning, test, live replay (run sheet Steps C-F).

Subcommands (each writes under ``results/layer3/fault/``):

``docs``   Batch inputs through the Layer 1/2 code paths without the broker
           (:mod:`src.offline_pipeline`): the validated burst documents of all
           400 CP4 events (``results/faults_2024/event_records.npz``), the
           year's validated 15-min stream of the 14 physical meters, and the
           15-min streams of the P2 scenarios.  No engine runs here.
``tune``   Step D: rho*, coincidence window, phase-balance tolerance and delta
           on faults_tune + quiet_tune only; writes config/layer3_params.yaml
           and tuning.json.
``test``   Step E: the frozen engine once over faults_test + quiet_test, the
           year's 15-min stream and the P2 scenarios; refuses to run if the
           parameters are not committed or if the test half was already run.
``live``   Step F: see scripts/run_layer3_live.py.

PowerShell::

    python scripts/run_layer3_fault.py docs
    python scripts/run_layer3_fault.py tune
    python scripts/run_layer3_fault.py test
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

OUT = ROOT / "results" / "layer3" / "fault"
BATCH = OUT / "batch"
RECORDS = ROOT / "results" / "faults_2024" / "event_records.npz"
P2_DIR = ROOT / "results" / "unbalance_scenarios"
PARAMS = ROOT / "config" / "layer3_params.yaml"
TEST_MARKER = OUT / "test_half_opened.json"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return str(o)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json), encoding="utf-8")


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------

def cmd_docs(args) -> int:
    from src.analytics import meters as M
    from src.data_source import meter_server as ms
    from src.offline_pipeline import flatten_doc, run_stream
    BATCH.mkdir(parents=True, exist_ok=True)
    phys = M.physical_meters()
    t0 = time.perf_counter()
    events = ms.load_event_file(RECORDS)
    _, bursts = run_stream(None, events, phys, run_name="layer3-batch-bursts")
    with gzip.open(BATCH / "bursts.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in bursts:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")
    t_b = time.perf_counter() - t0
    logging.info("bursts: %d documents (%.1f s)", len(bursts), t_b)
    info = {"bursts": len(bursts), "bursts_s": round(t_b, 1)}
    if not args.bursts_only:
        t1 = time.perf_counter()
        ids = [m.load_id for m in phys]
        frame = pd.read_parquet(ROOT / "results" / "year_2024" / "meter_frame_metered.parquet",
                                filters=[("load_id", "in", ids)])
        rows = []
        run_stream(frame, None, phys, run_name="layer3-batch-year", on_telemetry=lambda d: rows.append(flatten_doc(d)))
        pd.DataFrame(rows).to_parquet(BATCH / "stream_2024.parquet", index=False)
        info.update({"stream_rows": len(rows), "stream_s": round(time.perf_counter() - t1, 1)})
        del frame, rows
        if (P2_DIR / "scenarios.json").exists():
            (BATCH / "p2").mkdir(exist_ok=True)
            t2 = time.perf_counter()
            for p in sorted((P2_DIR / "frames").glob("*.parquet")):
                fr = pd.read_parquet(p, filters=[("load_id", "in", ids)])
                rows = []
                run_stream(fr, None, phys, run_name=f"layer3-p2-{p.stem}",
                           on_telemetry=lambda d: rows.append(flatten_doc(d)))
                pd.DataFrame(rows).to_parquet(BATCH / "p2" / p.name, index=False)
            info["p2_s"] = round(time.perf_counter() - t2, 1)
    write_json(BATCH / "docs_info.json", info)
    print(json.dumps(info, indent=2))
    return 0


# --------------------------------------------------------------------------
# tune
# --------------------------------------------------------------------------

def cmd_tune(args) -> int:
    from src.analytics import evaluate as V
    if TEST_MARKER.exists():
        print("the test half has been opened: re-tuning is not allowed", file=sys.stderr)
        return 1
    split = V.load_split()
    labels = V.load_labels()
    docs = V.select_docs(V.read_docs(BATCH / "bursts.jsonl.gz"), split["faults_tune"] + split["quiet_tune"])
    t0 = time.perf_counter()
    res = V.tune(docs, labels, split)
    res["tuning_docs"] = len(docs)
    res["run_time_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "tuning.json", res)
    if "stop" in res:
        print("TUNING STOP:", res["stop"], file=sys.stderr)
        return 1
    fz = res["frozen"]
    doc = {
        "frozen": fz,
        "tuned_on": {"faults": "faults_tune (100 events) of config/fault_split.yaml",
                     "quiet": "quiet_tune (100 events) of config/fault_split.yaml",
                     "documents": res["tuning_docs"],
                     "source": "results/layer3/fault/batch/bursts.jsonl.gz (validated burst documents, "
                               "Layer 1/2 code paths without the broker)"},
        "order": ["rho_star", "coincidence_samples", "balance_tol", "delta_pp"],
        "grids": {"rho_star": "1.05 to 3.00, step 0.05", "coincidence_samples": "1 to 10 samples (10 ms)",
                  "balance_tol": "0.05 to 0.50, step 0.05", "delta_pp": "0.25 to 3.00 percentage points, step 0.25"},
        "objectives": {
            "rho_star": "maximise recall of the tuning faults with no IS 17036 detection at the feeder head "
                        "(user decision 27 Oct 2026, replacing 'missed at every meter', an empty set), "
                        "subject to zero events on the 100 tuning quiet windows; tie-break: largest rho*",
            "coincidence_samples": "fewest tuning faults split into more than one event, then fewest events "
                                   "merging two different tuning faults; tie-break: smallest window",
            "balance_tol": "maximise 4-class type accuracy on detected tuning faults; tie-break: middle of "
                           "the best plateau (lower middle if even)",
            "delta_pp": "maximise zone accuracy on detected tuning faults; tie-break: middle of the best "
                        "plateau (lower middle if even)"},
        "plateau_rule": "best plateau = the longest run of consecutive grid values at the maximum (ties: "
                        "the lowest run)",
        "decisions_27_oct_2026": {
            "rho_objective": "tuning faults with no IS 17036 detection at the feeder head",
            "overcurrent_phase": "rho_p > rho*",
            "merging": "open event absorbs: an event stays open while any member meter is disturbed, plus "
                       "one coincidence window; a detection starting while it is open joins it"},
        "results": {"rho_target_faults": len(res["rho_target_faults"]),
                    "rho_found": next(c["found"] for c in res["rho_curve"] if c["rho"] == fz["rho_star"]),
                    "faults_split": next(c["faults_split"] for c in res["window_curve"]
                                         if c["window_samples"] == fz["coincidence_samples"]),
                    "tol_plateau": res["tol_plateau"], "delta_plateau": res["delta_plateau"],
                    "strict_onset_faults_split_tuning": res["strict_onset_faults_split_tuning"]},
        "curves": "results/layer3/fault/tuning.json",
    }
    header = ("# Layer 3 fault engine: frozen parameters (run sheet Step D).  Tuned once on the\n"
              "# tuning half only; written before any test-half run and committed (commit D).\n"
              "# Never re-tuned after the test half is opened.\n")
    PARAMS.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=100), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k.endswith("curve") is False}, indent=2, default=_json))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("docs")
    d.add_argument("--bursts-only", action="store_true")
    sub.add_parser("tune")
    t = sub.add_parser("test")
    t.add_argument("--dry-run-tuning-half", action="store_true",
                   help="same code on the tuning half (no marker, no git check); for checking the runner")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("src.data_source", "pandapower", "sg"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.simplefilter("ignore", FutureWarning)
    if args.cmd == "docs":
        return cmd_docs(args)
    if args.cmd == "tune":
        return cmd_tune(args)
    if args.cmd == "test":
        return cmd_test(args)
    return 2


def _git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cmd_test(args) -> int:
    """Step E: run once.  Guards: the parameters are committed and unchanged,
    and the test half has not been opened before (marker file)."""
    from src.analytics import evaluate as V
    from src.analytics import fault_engine as E
    dry = args.dry_run_tuning_half
    if dry:
        return _evaluate_half(args, "faults_tune", "quiet_tune", OUT / "dry_run_tuning_half.json", None)
    if TEST_MARKER.exists():
        print(f"the test half was already opened ({TEST_MARKER}); Step E runs once", file=sys.stderr)
        return 1
    rel = str(PARAMS.relative_to(ROOT)).replace("\\", "/")
    try:
        _git("ls-files", "--error-unmatch", rel)
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0
    except subprocess.CalledProcessError:
        dirty = True
    if dirty:
        print("config/layer3_params.yaml is not committed unchanged: commit D first", file=sys.stderr)
        return 1
    commit_d = _git("log", "-1", "--format=%H", "--", rel)
    write_json(TEST_MARKER, {"opened_utc": pd.Timestamp.now(tz="UTC").isoformat(), "params_commit": commit_d,
                             "head": _git("rev-parse", "HEAD")})
    return _evaluate_half(args, "faults_test", "quiet_test", OUT / "summary.json", commit_d)


def _evaluate_half(args, f_key: str, q_key: str, out_path: Path, commit_d) -> int:
    from src.analytics import evaluate as V
    from src.analytics import fault_engine as E
    t0 = time.perf_counter()
    params, relay = E.load_params(), E.load_relay()
    split, labels = V.load_split(), V.load_labels()
    ft, qt = split[f_key], split[q_key]
    docs = V.select_docs(V.read_docs(BATCH / "bursts.jsonl.gz"), ft + qt)
    events, an = E.run_batch(docs, params, relay)
    ev = V.events_table(events, labels)
    strict, _ = E.run_batch(docs, params, relay, mode="strict")
    ev_strict = V.events_table(strict, labels)
    facts = V.burst_facts(an, labels)
    first = V.first_alert(ev, ft)
    det = V.detection_metrics(ev, labels, ft, qt)
    flat_alerts = [t for d in docs for t in E.flat_alerts_burst(E.burst_from_doc(d))]
    flat = V.flat_metrics(flat_alerts, labels, ft, qt)
    t_bursts = time.perf_counter() - t0
    # full-year 15-min stream ----------------------------------------------------
    stream = E.stream_frame(pd.read_parquet(BATCH / "stream_2024.parquet"))
    weekly = E.weekly_unbalance(stream)
    flat_stream = E.flat_alerts_stream(stream)
    steps = stream.t_sim.nunique()
    hours = steps * 0.25
    head_id = next(m for m in stream.meter_id.unique() if m == "m10")
    truth = V.head_weekly_truth()
    hw = weekly[weekly.meter_id == head_id].merge(truth, on="week", how="outer", suffixes=("_engine", "_truth"))
    # P2 scenarios ---------------------------------------------------------------
    p2 = {}
    lab_w = pd.read_parquet(P2_DIR / "unbalance_weekly_labels.parquet")
    from src.analytics import meters as M
    lid_of = {m.meter_id: m.load_id for m in M.physical_meters()}
    eng_w = []
    for p in sorted((BATCH / "p2").glob("*.parquet")):
        w = E.weekly_unbalance(E.stream_frame(pd.read_parquet(p)))
        eng_w.append(w.assign(scenario_id=p.stem, load_id=w.meter_id.map(lid_of)))
    eng_w = pd.concat(eng_w, ignore_index=True)
    p2 = V.p2_scores(eng_w[["scenario_id", "load_id", "alarm", "vu_max_pct"]], lab_w)
    with_pos = {k: v for k, v in p2.items() if v["positives"] > 0}
    # overlap of event windows (user addition 2) ---------------------------------
    lw = labels.sort_values("win_start_ms")
    gaps_ms = lw.win_start_ms.to_numpy()[1:] - lw.win_end_ms.to_numpy()[:-1]
    per_strict = ev_strict[ev_strict.matched.isin(set(ft))].matched.value_counts()
    summary = {
        "params": params.__dict__, "params_commit": commit_d, "relay": relay.__dict__,
        "half": {"faults": f_key, "quiet": q_key, "n_faults": len(ft), "n_quiet_windows": len(qt),
                 "documents": len(docs)},
        "detection": det,
        "no_dip": V.no_dip_metrics(first, facts, labels, ft),
        "type": V.type_metrics(first, labels),
        "zone": V.zone_metrics(first, labels),
        "timing": V.timing_metrics(first, labels),
        "splits": {"open_event_rule_faults_split": det["faults_with_more_than_one_alert"],
                   "strict_onset_faults_split_test": int((per_strict > 1).sum()),
                   "strict_onset_events": int(len(ev_strict))},
        "flat_baseline": {"bursts": flat,
                          "stream_alerts": int(len(flat_stream)),
                          "stream_alerts_per_1000h": len(flat_stream) / hours * 1000,
                          "stream_alerts_per_meter": flat_stream.meter_id.value_counts().to_dict()},
        "stream_2024": {"rows": int(len(stream)), "steps": int(steps), "hours": hours,
                        "gap_flagged": int(stream["flags"].str.contains("GAP").sum()),
                        "engine_dip_alarms": 0,
                        "engine_unbalance_alarms_feeder_head": int(weekly[weekly.meter_id == head_id].alarm.sum()),
                        "engine_unbalance_alarms_dt_meters": int(weekly[weekly.meter_id != head_id].alarm.sum()),
                        "engine_alarms_per_1000h_feeder_head": float(weekly[weekly.meter_id == head_id].alarm.sum())
                        / hours * 1000,
                        "unbalance_alarms_per_week_feeder_head": float(weekly[weekly.meter_id == head_id].alarm.mean()),
                        "vu_max_pct_per_meter": weekly.groupby("meter_id").vu_max_pct.max().to_dict()},
        "unbalance_feeder_head_2024": {
            "weeks": int(len(hw)), "truth_weeks_exceeding": int(hw.exceeds.fillna(False).sum()),
            "truth_compliance": V.wilson(int((~hw.exceeds.fillna(False)).sum()), int(len(hw))),
            "engine_weeks_alarm": int(hw.alarm.fillna(False).sum()),
            "agreement_weeks": int((hw.alarm.fillna(False) == hw.exceeds.fillna(False)).sum()),
            "truth_vu_max_pct": float(truth.vu_max_pct.max()),
            "engine_vu_max_pct": float(hw.vu_max_pct_engine.max())},
        "p2": {"per_meter": p2, "meters_with_positives": sorted(with_pos),
               "feeder_head_positives": p2.get("feeder_head_650", {}).get("positives", 0),
               "note": "scenario-level weekly flags; the alarm is not tuned"},
        "event_windows": {"events": int(len(lw)), "min_gap_between_windows_s": float(gaps_ms.min() / 1e3),
                          "overlapping_pairs": int((gaps_ms < 0).sum())},
        "run_time_s": round(time.perf_counter() - t0, 1), "bursts_time_s": round(t_bursts, 1),
    }
    ev.to_parquet(out_path.with_name(out_path.stem + "_events.parquet"), index=False)
    write_json(out_path, summary)
    print(json.dumps({k: summary[k] for k in ("detection", "splits", "event_windows")}, indent=2, default=_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
