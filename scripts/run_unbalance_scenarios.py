"""Layer 3 P2: voltage-unbalance test scenarios (src/data_source/unbalance_scenarios.py).

Generated once, before the fault engine's unbalance results are seen, and not
regenerated afterwards (prompt P2 item 7).  Writes ``results/unbalance_scenarios/``.

Checks (exit 1 if any fails):
(a) s = 0 reproduces the cached 2024 values of the week: OLTC taps identical,
    and every meter value within half a register quantum (0.005 V, 0.0005 A,
    0.0005 kW / kvar; config/register_map.yaml), i.e. the engine sees the
    same telemetry;
(c) every power flow converged (run_timeseries raises on any failure).
Reported (b): VU at the feeder head and every DT meter for each s, and how many
scenarios exceed the IS 17036 weekly limit at each meter.

PowerShell::

    python scripts/run_unbalance_scenarios.py          # ~40 min
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_source import config  # noqa: E402
from src.data_source import unbalance_scenarios as U  # noqa: E402

HALF_QUANTUM = {"voltage_v": 0.005, "load_current_a": 0.0005, "current_a": 0.0005,
                "active_power_kw": 0.0005, "reactive_power_kvar": 0.0005}


def check_baseline(out: Path, meta: dict) -> dict:
    t0, t1 = (pd.Timestamp(x) for x in meta["week"])
    new = pd.read_parquet(out / "baseline_true_frame.parquet")
    old = pd.read_parquet(config.YEAR_CACHE_DIR / "meter_frame_true.parquet",
                          filters=[("timestamp", ">=", t0), ("timestamp", "<=", t1)])
    key = ["timestamp", "load_id", "phase"]
    m = old.merge(new, on=key, suffixes=("_old", "_new"), validate="one_to_one")
    worst = {c: float(np.max(np.abs(m[f"{c}_old"] - m[f"{c}_new"]))) for c in HALF_QUANTUM}
    st_new = pd.read_parquet(out / "states.parquet", filters=[("scenario_id", "==", "s00_base")])
    st_old = pd.read_parquet(config.YEAR_CACHE_DIR / "state.parquet", columns=["timestamp", "oltc_tap"],
                             filters=[("timestamp", ">=", t0), ("timestamp", "<=", t1)])
    taps_equal = bool(np.array_equal(st_old.sort_values("timestamp").oltc_tap.to_numpy(),
                                     st_new.sort_values("timestamp").oltc_tap.to_numpy()))
    return {"rows": int(len(m)), "rows_expected": int(len(old)), "max_abs_diff": worst,
            "tolerance": HALF_QUANTUM, "taps_equal": taps_equal,
            "passed": bool(len(m) == len(old) == len(new) and taps_equal
                           and all(worst[c] <= HALF_QUANTUM[c] for c in HALF_QUANTUM))}


def report(out: Path, meta: dict) -> dict:
    w = pd.read_parquet(out / U.WEEKLY_LABELS_FILE)
    s_of = {s["scenario_id"]: s["s"] for s in meta["scenarios"]}
    w["s"] = w.scenario_id.map(s_of)
    per_s = {}
    for (s, lid), g in w.groupby(["s", "load_id"]):
        per_s.setdefault(f"{s:.2f}", {})[lid] = {
            "vu_max_pct": float(g.vu_max_pct.max()), "vu_p95_max_pct": float(g.vu_p95_pct.max()),
            "scenarios": int(len(g)), "scenarios_exceeding": int(g.exceeds_limit.sum())}
    per_meter = (w[w.s > 0].groupby("load_id")
                 .agg(scenarios=("exceeds_limit", "size"), exceeding=("exceeds_limit", "sum"),
                      vu_max_pct=("vu_max_pct", "max")).reset_index())
    head = w[w.load_id == "feeder_head_650"]
    return {"per_s_per_meter": per_s,
            "per_meter_exceeding": per_meter.to_dict("records"),
            "feeder_head_any_exceeding": bool(head.exceeds_limit.any()),
            "feeder_head_vu_max_pct": float(head.vu_max_pct.max()),
            "meters_with_positives": sorted(per_meter[per_meter.exceeding > 0].load_id.tolist())}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.data_source.stage_b").setLevel(logging.WARNING)
    out = U.OUT_DIR
    t0 = time.perf_counter()
    meta = U.generate(out)
    base = check_baseline(out, meta)
    summary = {"week": meta["week"], "week_choice": meta["week_choice"], "note": meta["note"],
               "n_scenarios": len(meta["scenarios"]), "baseline_check": base,
               "all_power_flows_converged": True, "report": report(out, meta),
               "run_time_s": round(time.perf_counter() - t0, 1)}
    summary["passed"] = base["passed"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "report"}, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary["report"].items() if k != "per_s_per_meter"}, indent=2))
    if not summary["passed"]:
        print("P2 CHECK FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
