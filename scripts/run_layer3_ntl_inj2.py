"""Layer 3 theft engine, Stage 4 (theft-engine fixes): entry point on injection 2.

Stage 4 run sheet (docs/Layer3_prompt.md).  Injection 2 is a second theft draw
on the same household population (``scripts/reinject_theft.py``); its
theft-dependent files are in ``results/year_2024_inj2/``, everything
theft-independent (meter frames, L^_tech, lambda_true DT inputs, consumer
index) is reused from ``results/year_2024/`` and ``results/layer3/ntl/``.
v1 outputs are never written.

Commands (PowerShell)::

    python scripts/run_layer3_ntl_inj2.py injection     # step A: injection report and identity check
    python scripts/run_layer3_ntl_inj2.py balance       # step C: balance results, scored once

Outputs under ``results/layer3/ntl_inj2/`` (``summary.json``).  Labels are
read through :mod:`src.analytics.evaluate` only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import ntl_engine as N  # noqa: E402
from src.data_source.config import YEAR_CACHE_DIR  # noqa: E402

OUT = ROOT / "results" / "layer3" / "ntl_inj2"
V1_NTL = ROOT / "results" / "layer3" / "ntl"              # theft-independent inputs, read only
INJ2 = ROOT / "results" / "year_2024_inj2"
SUMMARY = OUT / "summary.json"
PROGRESS = OUT / "progress.json"


def _write_summary(key: str, value) -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    s[key] = value
    SUMMARY.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def cmd_injection(args) -> int:
    """Step A report: thieves, onsets, theft-active DT-days, reproduction and
    theft-independence records, and the energy identity on injection 2."""
    t0 = time.perf_counter()
    rep = json.loads((OUT / "reproduction_check.json").read_text(encoding="utf-8"))
    inp = json.loads((OUT / "inputs.json").read_text(encoding="utf-8"))
    if not rep["all_equal"] or not inp["theft_independence_checks"]["passed"]:
        raise SystemExit("STOP: reproduction or theft-independence check failed (run sheet §6)")
    res = {
        "label": "injection 2: second theft draw on the same household population (D1)",
        "summary": V.injection_summary(INJ2),
        "reproduction_check": {k: rep[k] for k in ("all_equal", "all_sha256_equal", "streams")}
        | {"files_compared": [f["file"] for f in rep["files"]]},
        "theft_independence_checks": inp["theft_independence_checks"],
        "energy_identity": {"jul_dec_test": V.ntl_energy_identity(V.NTL_TEST, INJ2, frame_cache=YEAR_CACHE_DIR),
                            "year": V.ntl_energy_identity(V.NTL_YEAR, INJ2, frame_cache=YEAR_CACHE_DIR)},
        "files_regenerated": json.loads((INJ2 / "reinject_info.json").read_text(encoding="utf-8"))["written"],
        "files_reused": json.loads((INJ2 / "reinject_info.json").read_text(encoding="utf-8"))["reused_from_v1"],
        "ideal_meter_households": inp["ideal_meter_households"],
        "elapsed_s": {"reproduction_run": rep["elapsed_s"], "injection_run": inp["elapsed_s"],
                      "report": round(time.perf_counter() - t0, 1)},
    }
    _write_summary("injection", res)
    print(json.dumps({k: res[k] for k in ("summary", "energy_identity")}, indent=2))
    return 0


# --------------------------------------------------------------------------
# Balance engine (steps B-C; run sheet §3)
# --------------------------------------------------------------------------
INPUTS = {
    "real": {"frame": YEAR_CACHE_DIR / "meter_frame_metered.parquet", "household_daily": INJ2 / "household_daily.parquet",
             "ltech": V1_NTL / "ltech_real.parquet"},
    "ideal": {"frame": V1_NTL / "ideal_meter" / "meter_frame_metered.parquet",
              "household_daily": OUT / "ideal_meter" / "household_daily.parquet",
              "ltech": V1_NTL / "ltech_ideal.parquet"},
}
LAMBDA_BAND = (0.0, 0.02, 0.04)                    # A4: band over lambda_assumed, and the 3 x 3 grid (N3)
V1_A_MINUS_LTECH_MWH = -58.089                     # run sheet §3 check (v1, 2024; A carries no theft)


def _consumer_index() -> pd.Series:
    return N.read_consumer_index(V1_NTL / "consumer_index.parquet")


def _estimates(name: str, idx: pd.Series, lambdas=(0.0,)) -> dict:
    i = INPUTS[name]
    return N.balance_estimates(N.read_meter_frame(i["frame"]), N.read_household_daily(i["household_daily"]),
                               idx, pd.read_parquet(i["ltech"]), lambdas_assumed=lambdas, offline="neighbour")


def _daily_stats(s: pd.Series) -> dict:
    return {"mean": float(s.mean()), "mae": float(s.abs().mean()), "min": float(s.min()), "max": float(s.max()),
            "sum_mwh": float(s.sum() / 1e3)}


def _a_residual(est: dict) -> dict:
    """A1: the daily A - L^_tech residual before and after the head-gain correction."""
    out = {}
    for w, r in {"year": V.NTL_YEAR, **V.NTL_WINDOWS}.items():
        m = V._in(est["A"].index, r)
        head = est["head_day"][m]
        out[w] = {"days": int(m.sum()), "head_mwh": float(head.sum() / 1e3),
                  "before": _daily_stats(est["a_minus_ltech"][m]),
                  "after_correction": _daily_stats(est["a_minus_ltech_corrected"][m]),
                  "before_share_of_head": float(est["a_minus_ltech"][m].sum() / head.sum())}
    return out


def _dt_ntl_lambda0(idx: pd.Series) -> pd.DataFrame:
    """Engine NTL^_B per DT-day at lambda_assumed = 0 with the primary (A3)
    offline estimate, from the engine inputs only (no labels); cached for the
    detector's transformer gate and the faithful-rule count."""
    path = OUT / "dt_ntl_lambda0.parquet"
    if not path.exists():
        i = INPUTS["real"]
        est = N.balance_estimates(N.read_meter_frame(i["frame"]), N.read_household_daily(i["household_daily"]),
                                  idx, None, offline="neighbour")
        est["ntl_dt"][0.0].to_parquet(path)
    return pd.read_parquet(path)


def cmd_balance(args) -> int:
    t0 = time.perf_counter()
    if (OUT / "balance_scored.json").exists():
        raise SystemExit("the balance results on injection 2 were already scored (run once)")
    idx = _consumer_index()
    est = _estimates("real", idx, LAMBDA_BAND)
    a_year = float(est["a_minus_ltech"][V._in(est["A"].index, V.NTL_YEAR)].sum() / 1e3)
    v1 = json.loads((V1_NTL / "summary.json").read_text(encoding="utf-8"))["balance"]["a_minus_ltech"]["year"]
    check = {"a_minus_ltech_2024_mwh": a_year, "v1_mwh": v1["residual_mwh"],
             "difference_mwh": a_year - v1["residual_mwh"], "run_sheet_value_mwh": V1_A_MINUS_LTECH_MWH,
             "equal": a_year == v1["residual_mwh"] and round(a_year, 3) == V1_A_MINUS_LTECH_MWH}
    if not check["equal"]:
        print(json.dumps(check, indent=2))
        raise SystemExit("STOP: A - L^_tech differs from v1 (run sheet §6)")
    if abs(a_year * 1e3 / est["head_day"].sum()) > 0.01:
        raise SystemExit("STOP: balance A residual exceeds about 1 % of head energy (Stage 2 run sheet §6)")
    (OUT / "balance_scored.json").write_text(json.dumps({"opened_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                             encoding="utf-8")
    truth = V.theft_truth(INJ2)
    ideal = _estimates("ideal", idx)
    feeder = {"bottom_up_primary": est["bottom_up"], "head_netted_raw": est["ntl_feeder"],
              "head_netted_corrected": est["ntl_feeder_corrected"], "un_netted_C": est["C"],
              "ideal_bottom_up": ideal["bottom_up"], "ideal_head_netted": ideal["ntl_feeder"]}
    med = est["eoff_rules"]["median"]
    est_med_dt = N.per_dt(med["household"], idx, est["dt_day"].columns).reindex(est["dt_day"].index)
    res = {
        "label": "balance engine on injection 2 (h1-h4 truth); test window Jul-Dec, Jan-Jun alongside; "
                 "primary feeder estimate = bottom-up sum_DT (B_DT - E^_offline,DT) at lambda_assumed = 0 (A2), "
                 "primary E^_offline = neighbour ratio (A3); fixed before any injection-2 result",
        "energy_identity_test_window": V.ntl_energy_identity(V.NTL_TEST, INJ2, frame_cache=YEAR_CACHE_DIR),
        "a_minus_ltech_check_vs_v1": check,
        "a1_head_gain": {"window": list(N.GAIN_WINDOW), "window_note": "[our choice of window]",
                         **V.head_gain_vs_true(est["g_hat"]), "a_minus_ltech_daily": _a_residual(est)},
        "ltech_vs_true_losses": V.ltech_vs_true(est["ltech_day"]),
        "feeder": V.score_feeder(feeder, truth),
        "annual_mwh": V.annual_mwh_table(feeder, truth),
        "dt": V.score_dt({"netted_lambda_0": est["ntl_dt"][0.0], "un_netted_B": est["B"],
                          "netted_lambda_0_median_offline_rule": N.netted_dt(est["B"], est["billed_dt"], est_med_dt, 0.0)},
                         truth),
        "dt_lambda_band": {f"{lam:.2f}": V.score_dt({"netted": est["ntl_dt"][lam]}, truth) for lam in LAMBDA_BAND},
        "offline_estimate": {
            "primary": "neighbour ratio (A3)", "baseline": "Stage 2 own-history median",
            "rule_counts": {k: v["info"] for k, v in est["eoff_rules"].items()},
            "vs_would_have_reported": V.offline_households_vs_true(
                {k: v["household"] for k, v in est["eoff_rules"].items()}, INJ2),
            "feeder_day": {k: V.offline_vs_true(v["day"], truth) for k, v in est["eoff_rules"].items()}},
        "worked_example": V.worked_example(est, truth, cache=YEAR_CACHE_DIR),
    }
    # 3 x 3 grid: lambda_true injected into the DT meters (v1 step A inputs, theft-independent)
    frame = N.read_meter_frame(INPUTS["real"]["frame"])
    hh = N.read_household_daily(INPUTS["real"]["household_daily"])
    grid = {}
    for lt_ in LAMBDA_BAND:
        e15 = pd.read_parquet(V1_NTL / f"lambda_true_{round(lt_ * 100):d}" / "dt_meter_kwh.parquet")
        g = N.balance_estimates(frame, hh, idx, None, lambdas_assumed=LAMBDA_BAND, dt_energy_15min=e15,
                                offline="neighbour")
        grid[f"{lt_:.2f}"] = {f"{la:.2f}": V.score_dt({"netted": g["ntl_dt"][la]}, truth) for la in LAMBDA_BAND}
    res["lambda_grid_true_x_assumed"] = grid
    # N6: ideal meters (every gain 1.0), with their own L^_tech run (v1, theft-independent)
    res["ideal_meter"] = {
        "a_minus_ltech": _a_residual(ideal), "g_hat": ideal["g_hat"],
        "feeder": V.score_feeder({"bottom_up": ideal["bottom_up"], "head_netted": ideal["ntl_feeder"],
                                  "un_netted_C": ideal["C"]}, truth),
        "dt": V.score_dt({"netted_lambda_0": ideal["ntl_dt"][0.0], "un_netted_B": ideal["B"]}, truth),
        "feeder_head_gain_effect": {
            w: {"head_mwh_metered": float(est["head_day"][V._in(est["head_day"].index, r)].sum() / 1e3),
                "head_mwh_ideal": float(ideal["head_day"][V._in(ideal["head_day"].index, r)].sum() / 1e3),
                "difference_mwh": float((est["head_day"] - ideal["head_day"])[V._in(est["head_day"].index, r)].sum()
                                        / 1e3)}
            for w, r in {"year": V.NTL_YEAR, **V.NTL_WINDOWS}.items()},
    }
    pd.DataFrame({"A": est["A"], "C": est["C"], "billed": est["billed"], "ltech": est["ltech_day"].ltech_kw,
                  "eoff": est["eoff_day"], "head_corrected": est["head_corrected_day"],
                  "ntl_feeder": est["ntl_feeder"], "ntl_feeder_corrected": est["ntl_feeder_corrected"],
                  "bottom_up": est["bottom_up"]}).to_parquet(OUT / "feeder_day.parquet")
    pd.concat({"B": est["B"], "billed_dt": est["billed_dt"], "eoff_dt": est["eoff_dt"],
               "ntl_dt_lambda_0": est["ntl_dt"][0.0]}, axis=1).to_parquet(OUT / "dt_day.parquet")
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    _write_summary("balance", res)
    print(json.dumps({k: res[k] for k in ("a_minus_ltech_check_vs_v1", "annual_mwh", "feeder")}, indent=2))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("injection", help="step A: injection report").set_defaults(func=cmd_injection)
    sub.add_parser("balance", help="step C: balance results on injection 2, scored once").set_defaults(func=cmd_balance)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
