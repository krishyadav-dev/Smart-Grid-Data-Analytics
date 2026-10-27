"""Layer 3 theft engine: entry point (Stage 2 run sheet).

Commands (PowerShell)::

    python scripts/run_layer3_ntl.py ltech --inputs real     # L^_tech power flow, ~65 min
    python scripts/run_layer3_ntl.py ltech --inputs ideal    # same on the ideal-meter inputs (N6)
    python scripts/run_layer3_ntl.py balance                 # step C: balance results, scored once

Outputs under ``results/layer3/ntl/`` (``summary.json`` key ``balance``).
Labels are read through :mod:`src.analytics.evaluate` only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import ntl_engine as N  # noqa: E402
from src.data_source.config import YEAR_CACHE_DIR  # noqa: E402

OUT = ROOT / "results" / "layer3" / "ntl"
INPUTS = {
    "real": {"frame": YEAR_CACHE_DIR / "meter_frame_metered.parquet",
             "household_daily": YEAR_CACHE_DIR / "household_daily.parquet"},
    "ideal": {"frame": OUT / "ideal_meter" / "meter_frame_metered.parquet",
              "household_daily": OUT / "ideal_meter" / "household_daily.parquet"},
}

SUMMARY = OUT / "summary.json"
LAMBDA_SWEEP = (0.0, 0.01, 0.02, 0.03, 0.05)      # the existing sweep (plan §2 N3)
LAMBDA_GRID = (0.0, 0.02, 0.04)                   # 3 x 3 true-vs-assumed grid (N3)


def _write_summary(key: str, value) -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    s[key] = value
    SUMMARY.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def _a_minus_ltech(est: dict) -> dict:
    """Validation of L^_tech on balance A (no theft by construction)."""
    out = {}
    for w, r in {"year": V.NTL_YEAR, **V.NTL_WINDOWS}.items():
        m = V._in(est["A"].index, r)
        a, lt, head = est["A"][m], est["ltech_day"].ltech_kw[m], est["head_day"][m]
        res = a - lt
        out[w] = {"days": int(m.sum()), "head_mwh": float(head.sum() / 1e3), "A_mwh": float(a.sum() / 1e3),
                  "ltech_mwh": float(lt.sum() / 1e3), "residual_mwh": float(res.sum() / 1e3),
                  "residual_share_of_head": float(res.sum() / head.sum()),
                  "daily_residual_kwh": {"mean": float(res.mean()), "mae": float(res.abs().mean()),
                                         "min": float(res.min()), "max": float(res.max())}}
    return out


def _run_inputs(name: str, idx: pd.Series, lambdas) -> dict:
    frame = N.read_meter_frame(INPUTS[name]["frame"])
    hh = N.read_household_daily(INPUTS[name]["household_daily"])
    lt = pd.read_parquet(OUT / f"ltech_{name}.parquet")
    return N.balance_estimates(frame, hh, idx, lt, lambdas_assumed=lambdas)


def cmd_balance(args) -> int:
    t0 = time.perf_counter()
    idx = N.read_consumer_index(OUT / "consumer_index.parquet")
    truth = V.theft_truth()
    lams = tuple(sorted(set(LAMBDA_SWEEP) | set(LAMBDA_GRID)))
    est = _run_inputs("real", idx, lams)
    a_val = _a_minus_ltech(est)
    if abs(a_val["year"]["residual_share_of_head"]) > 0.01:            # run sheet §6 stop condition
        print(json.dumps(a_val, indent=2))
        raise SystemExit("STOP: balance A residual exceeds about 1 % of head energy over the year")
    res = {
        "label": "theft energy estimation (h1-h4 truth); test window Jul-Dec, Jan-Jun as reference",
        "energy_identity_test_window": V.ntl_energy_identity(V.NTL_TEST),
        "a_minus_ltech": a_val,
        "ltech_vs_true_losses": V.ltech_vs_true(est["ltech_day"]),
        "offline_estimate": {**est["eoff_info"], "vs_would_have_reported": V.offline_vs_true(est["eoff_day"], truth)},
        "feeder": V.score_feeder({"netted": est["ntl_feeder"], "un_netted_C": est["C"]}, truth),
        "dt": V.score_dt({"netted_lambda_0": est["ntl_dt"][0.0], "un_netted_B": est["B"]}, truth),
        "lambda_sweep_assumed_true_0": {f"{lam:.2f}": V.score_dt({"netted": est["ntl_dt"][lam]}, truth)
                                        for lam in LAMBDA_SWEEP},
        "worked_example": V.worked_example(est, truth),
    }
    # 3 x 3 grid: lambda_true injected into the DT meters (step A inputs), lambda_assumed subtracted
    frame = N.read_meter_frame(INPUTS["real"]["frame"])
    hh = N.read_household_daily(INPUTS["real"]["household_daily"])
    grid = {}
    for lt_ in LAMBDA_GRID:
        e15 = pd.read_parquet(OUT / f"lambda_true_{round(lt_ * 100):d}" / "dt_meter_kwh.parquet")
        g = N.balance_estimates(frame, hh, idx, None, lambdas_assumed=LAMBDA_GRID, dt_energy_15min=e15)
        grid[f"{lt_:.2f}"] = {f"{la:.2f}": V.score_dt({"netted": g["ntl_dt"][la]}, truth) for la in LAMBDA_GRID}
    res["lambda_grid_true_x_assumed"] = grid
    # N6: ideal meters (every gain 1.0), with their own L^_tech run
    ideal = _run_inputs("ideal", idx, (0.0,))
    res["ideal_meter"] = {
        "a_minus_ltech": _a_minus_ltech(ideal),
        "feeder": V.score_feeder({"netted": ideal["ntl_feeder"], "un_netted_C": ideal["C"]}, truth),
        "dt": V.score_dt({"netted_lambda_0": ideal["ntl_dt"][0.0], "un_netted_B": ideal["B"]}, truth),
        "feeder_head_gain_effect": {
            w: {"head_mwh_metered": float(est["head_day"][V._in(est["head_day"].index, r)].sum() / 1e3),
                "head_mwh_ideal": float(ideal["head_day"][V._in(ideal["head_day"].index, r)].sum() / 1e3),
                "difference_mwh": float((est["head_day"] - ideal["head_day"])[V._in(est["head_day"].index, r)].sum()
                                        / 1e3)}
            for w, r in {"year": V.NTL_YEAR, **V.NTL_WINDOWS}.items()},
    }
    # per-day tables for step E (transformer gate) and the report
    pd.DataFrame({"A": est["A"], "C": est["C"], "billed": est["billed"], "ltech": est["ltech_day"].ltech_kw,
                  "eoff": est["eoff_day"], "ntl_feeder": est["ntl_feeder"]}).to_parquet(OUT / "feeder_day.parquet")
    pd.concat({"B": est["B"], "billed_dt": est["billed_dt"], "eoff_dt": est["eoff_dt"],
               "ntl_dt_lambda_0": est["ntl_dt"][0.0]}, axis=1).to_parquet(OUT / "dt_day.parquet")
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    _write_summary("balance", res)
    print(json.dumps({k: res[k] for k in ("a_minus_ltech", "feeder")}, indent=2))
    return 0


def cmd_ltech(args) -> int:
    t0 = time.perf_counter()
    frame = N.read_meter_frame(INPUTS[args.inputs]["frame"])
    lt = N.ltech_run(frame, args.start, args.end, progress_every=960)
    OUT.mkdir(parents=True, exist_ok=True)
    lt.to_parquet(OUT / f"ltech_{args.inputs}.parquet")
    rec = {"inputs": args.inputs, "steps": int(len(lt)), "elapsed_s": round(time.perf_counter() - t0, 1),
           "ltech_mwh": float(lt.ltech_kw.sum() * N.STEP_H / 1e3),
           "line_losses_mwh": float(lt.line_losses_kw.sum() * N.STEP_H / 1e3),
           "dt_losses_mwh": float(lt.dt_losses_kw.sum() * N.STEP_H / 1e3)}
    (OUT / f"ltech_{args.inputs}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ltech", help="L^_tech power flow on the DT-meter loads")
    p.add_argument("--inputs", choices=sorted(INPUTS), default="real")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.set_defaults(func=cmd_ltech)
    p = sub.add_parser("balance", help="step C: balance results (scored once)")
    p.set_defaults(func=cmd_balance)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
