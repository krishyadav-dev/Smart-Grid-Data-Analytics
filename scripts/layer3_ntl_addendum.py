"""Checkpoint 2 addendum H: post-hoc, descriptive analyses after Checkpoint 2.

No re-tuning and no re-run of anything tuned.  Every item is labelled
"post-hoc, after Checkpoint 2".  Labels are read through
:mod:`src.analytics.evaluate` only.  Writes ``summary.json`` key ``addendum_h``.

1. Bottom-up feeder estimate: per day, sum over DTs of (B_DT - E^_offline,DT)
   at lambda = 0 (no head meter, no L^_tech), bias / MAE against true energy
   theft, Jul-Dec and Jan-Jun, real and ideal meters, next to the head-based
   netted estimate.
2. Jokar diagnostic: share of days flagged by calendar month, honest
   households and thieves before onset, per fold and pooled.  Jan-Jun: the
   tuning-phase models (trained Jan-Apr with the frozen (C, gamma); refitted
   deterministically, checked by reproducing every frozen q from their
   May-Jun household scores; Jan-Apr days are their own training days).
   Jul-Dec: the test-phase flags saved by step E (not re-run).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import ntl_engine as N  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_layer3_ntl as R  # noqa: E402

LABEL = "post-hoc, after Checkpoint 2"
LOO = tuple(f for f in N.FOLDS if f != "circular")


# --------------------------------------------------------------------------
# 1. Bottom-up feeder estimate
# --------------------------------------------------------------------------

def bottom_up(truth: dict) -> dict:
    idx = N.read_consumer_index(R.OUT / "consumer_index.parquet")
    out = {"label": LABEL, "definition": "sum over DTs of (B_DT - E^_offline,DT) at lambda = 0; "
                                         "uses the DT meters only (no head meter, no L^_tech)"}
    for name in ("real", "ideal"):
        est = N.balance_estimates(N.read_meter_frame(R.INPUTS[name]["frame"]),
                                  N.read_household_daily(R.INPUTS[name]["household_daily"]), idx,
                                  pd.read_parquet(R.OUT / f"ltech_{name}.parquet"))
        bu = est["ntl_dt"][0.0].sum(axis=1)
        out[f"{name}_meters"] = V.score_feeder({"bottom_up": bu, "head_netted": est["ntl_feeder"],
                                                "un_netted_C": est["C"]}, truth)
        out[f"{name}_meters"]["year_mwh"] = {"bottom_up": float(bu.sum() / 1e3),
                                             "head_netted": float(est["ntl_feeder"].sum() / 1e3),
                                             "true_energy_theft": float(truth["feeder_theft"].sum() / 1e3)}
    return out


# --------------------------------------------------------------------------
# 2. Jokar diagnostic
# --------------------------------------------------------------------------

def _tuning_phase_job(rows, xs, days, params):
    """Tuning-phase models (Jan-Apr training, frozen (C, gamma)) scored on every
    online Jan-Jun day: flags (-1 offline / not scored)."""
    from threadpoolctl import threadpool_limits
    jan_jun = N.window_mask(days, ("2024-01-01", "2024-06-30"))
    out = []
    with threadpool_limits(1):
        for r, x in zip(rows, xs):
            p = N.prepare_household(x, days, r, "validation")
            if p is None:
                out.append(None)
                continue
            online = jan_jun & ~np.isnan(x).any(axis=1)
            z = N.scale(x[online].astype(float), p["lo"], p["hi"])
            res = {}
            for f in N.FOLDS:
                zm = np.vstack([p["zm"][a] for a in N.FOLDS[f]])
                svm, kp = N.fit_svm(p["zb"], p["lb"], zm, params[f]["C"], params[f]["gamma"])
                fl = np.full(int(jan_jun.sum()), -1, np.int8)
                fl[online[jan_jun]] = N.malicious_score(svm, kp, z)[1]
                res[f] = fl
            out.append(res)
    return out


def _monthly(flags: np.ndarray, days: pd.DatetimeIndex, mask: np.ndarray) -> dict:
    """Flagged / online days by calendar month over the households x days cells in ``mask``."""
    on = (flags >= 0) & mask
    hit = (flags == 1) & mask
    m = days.month.to_numpy()
    return {int(k): V.wilson(int(hit[:, m == k].sum()), int(on[:, m == k].sum())) for k in np.unique(m)}


def jokar_diagnostic() -> dict:
    t0 = time.perf_counter()
    params = yaml.safe_load(R.PARAMS.read_text(encoding="utf-8"))["folds"]
    ids, days, X = N.ami_hourly(R.AMI_DIR)
    tun = R._parallel(_tuning_phase_job, list(range(len(ids))), X, days, params)
    jan_jun = days[N.window_mask(days, ("2024-01-01", "2024-06-30"))]
    test = np.load(R.OUT / "jokar_test_flags.npz")
    if list(test["ids"]) != ids:
        raise SystemExit("jokar_test_flags.npz households differ")
    jul_dec = pd.date_range("2024-07-01", "2024-12-31")
    all_days = jan_jun.append(jul_dec)
    labels = V.load_household_labels().reindex(ids)
    honest = ~labels.thief.astype(bool).to_numpy()
    onset = labels.onset.to_numpy()

    # reproduction check: May-Jun household scores of the refitted models give every frozen q
    mj = np.asarray((jan_jun >= "2024-05-01"))
    q_check = {}
    for f in N.FOLDS:
        s = pd.Series({h: N.household_score(t[f][mj]) for h, t in zip(ids, tun) if t is not None})
        y = V.fold_targets(V.load_household_labels(), ids, N.FOLDS[f], ("2024-05-01", "2024-06-30"))
        q_check[f] = {"q_frozen": params[f]["q"], "q_from_refit": V.best_threshold(s, y)["q"]}
        if q_check[f]["q_frozen"] != q_check[f]["q_from_refit"]:
            raise SystemExit(f"refitted tuning-phase models do not reproduce q: {q_check}")

    flags = {}
    for f in N.FOLDS:
        a = np.stack([t[f] if t is not None else np.full(len(jan_jun), -1, np.int8) for t in tun])
        b = test[f].copy()
        b[b < -1] = -1                                           # skipped households
        flags[f] = np.concatenate([a, b], axis=1)                # households x 366 days
    pre_onset = (~honest)[:, None] & (all_days.values[None, :] < onset[:, None])
    honest_cells = np.repeat(honest[:, None], len(all_days), axis=1)
    out = {"label": LABEL,
           "models": "Jan-Jun: tuning-phase models (trained Jan-Apr, frozen C/gamma; Jan-Apr are their own "
                     "training days, May-Jun out of sample); Jul-Dec: test-phase flags of step E",
           "note": "shares of household-days; days of one household are not independent, so the Wilson "
                   "intervals are narrower than the uncertainty",
           "q_reproduction_check": q_check, "honest": {}, "thieves_before_onset": {}}
    for group, cells in (("honest", honest_cells), ("thieves_before_onset", pre_onset)):
        for f in N.FOLDS:
            out[group][f] = _monthly(flags[f], all_days, cells)
        pooled = np.concatenate([flags[f] for f in LOO], axis=0)
        out[group]["pooled_six_folds"] = _monthly(pooled, all_days, np.concatenate([cells] * len(LOO), axis=0))
    out["households"] = {"honest": int(honest.sum()), "thieves": int((~honest).sum()),
                         "thieves_with_pre_onset_days": int(pre_onset.any(axis=1).sum())}
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> int:
    t0 = time.perf_counter()
    truth = V.theft_truth()
    res = {"label": LABEL, "bottom_up_feeder": bottom_up(truth), "jokar_flag_share_by_month": jokar_diagnostic()}
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    R._write_summary("addendum_h", res)
    b = res["bottom_up_feeder"]
    print(json.dumps({k: b[k] for k in ("real_meters", "ideal_meters")}, indent=1))
    d = res["jokar_flag_share_by_month"]
    for g in ("honest", "thieves_before_onset"):
        print(g, {m: round(v["p"], 3) if v["p"] is not None else None for m, v in d[g]["pooled_six_folds"].items()})
    print(d["q_reproduction_check"], d["households"], res["elapsed_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
