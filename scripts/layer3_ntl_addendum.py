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

Addendum J (post-hoc, after Checkpoint 4, descriptive; injection 2): does the
per-customer detector fail because honest days of one household-month are one
intraday shape scaled by daily energy?  J1 shape similarity within and across
months (simulated honest households, and the raw eMARC data as the
reference); J2 how close each synthetic attack h1-h6 stays to its source day;
J3 which honest days the saved Stage 4 flags hit.  Nothing refitted or
rescored.  Writes ``results/layer3/ntl_inj2/summary.json`` key ``addendum_j``.

    python scripts/layer3_ntl_addendum.py h    # addendum H (v1, Checkpoint 2)
    python scripts/layer3_ntl_addendum.py j    # addendum J (injection 2, Checkpoint 4)
"""

from __future__ import annotations

import argparse
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


def main_h() -> int:
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


# --------------------------------------------------------------------------
# Addendum J (post-hoc, after Checkpoint 4, descriptive)
# --------------------------------------------------------------------------
LABEL_J = "post-hoc, after Checkpoint 4, descriptive"
MIN_MONTH_DAYS = 15        # household-months need >= 15 days: the eMARC shape-library rule (brief v2 X.3.3)
J2_ATTACK_SEED = 20261031  # [our choice] (addendum J instruction)
EMARC_LOCKDOWN = ("2020-03-01", "2020-06-30")   # COVID lockdown months, as excluded by the shape library


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis; NaN where either side is constant
    (tested exactly, max == min: a repeated mean leaves rounding residue after
    centring, which would give an arbitrary correlation)."""
    flat = (np.ptp(a, axis=-1) == 0) | (np.ptp(b, axis=-1) == 0)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    ok = ~flat & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, (a * b).sum(axis=-1) / np.where(ok, den, 1.0), np.nan)


def _dist(x) -> dict:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90))}


def shape_similarity(groups) -> dict:
    """J1 statistics.  ``groups`` yields (unit, (year, month), days x 96 array of
    complete days with positive energy).  Within a household-month: each day's
    profile normalised to sum 1 against the household-month mean profile;
    across months: consecutive months' mean profiles (Jan-Feb .. Nov-Dec)."""
    mean_c, median_c, means = [], [], {}
    small = undefined_months = undefined_days = days_used = 0
    for unit, ym, x in groups:
        if len(x) < MIN_MONTH_DAYS:
            small += 1
            continue
        p = x / x.sum(axis=1, keepdims=True)
        m = p.mean(axis=0)
        c = _row_corr(p, m[None, :])
        days_used += len(c)
        undefined_days += int(np.isnan(c).sum())
        if np.isnan(c).all():           # every day (or the month mean) is a flat profile
            undefined_months += 1
            continue
        mean_c.append(np.nanmean(c))
        median_c.append(np.nanmedian(c))
        means[unit, ym] = m
    pairs = {f"{m}-{m + 1}": [] for m in range(1, 12)}
    for (unit, (y, m)), prof in means.items():
        nxt = means.get((unit, (y, m + 1)))
        if m < 12 and nxt is not None:
            pairs[f"{m}-{m + 1}"].append(float(_row_corr(prof, nxt)))
    across = [c for v in pairs.values() for c in v]
    median_c = np.asarray(median_c)
    return {
        "household_months": int(len(mean_c)), "household_months_below_min_days": small,
        "household_months_corr_undefined": undefined_months,
        "days_in_qualifying_months": days_used, "days_corr_undefined_flat_profile": undefined_days,
        "within_month_mean_corr": _dist(mean_c),
        "share_household_months_median_corr_ge_0_999": float((median_c >= 0.999).mean()) if len(median_c) else None,
        "across_consecutive_months_corr": {"pooled": _dist(across),
                                           "by_pair": {k: _dist(v) for k, v in pairs.items()}},
    }


def _sim_groups(X: np.ndarray, days: pd.DatetimeIndex, rows):
    """Simulated honest households: online days (all 96 samples) with positive energy."""
    months = days.month.to_numpy()
    for m in range(1, 13):
        xm = X[rows][:, months == m].astype(float)
        e = xm.sum(axis=2)
        ok = ~np.isnan(xm).any(axis=2) & (e > 0)
        for i, r in enumerate(rows):
            yield r, (2024, m), xm[i][ok[i]]


def emarc_reference() -> tuple[dict, dict]:
    """The raw eMARC 15-min data: mainline (whole-household) deployments, complete
    days (96 blocks), positive energy, 2018-01-01 .. 2020-02-29 (the file ends
    2020-06-30; the lockdown months 2020-03 .. 2020-06 are excluded), grouped by
    deployment and year-month."""
    import pyarrow.csv as pc

    from src.data_source import config
    from src.data_source import emarc as E
    d = config.EMARC_DIR
    meta = pd.read_csv(d / E.DAILY_FILE, usecols=["household_id", "Deployment type", "deployment_id"]
                       ).drop_duplicates("deployment_id")
    main = set(meta.loc[meta["Deployment type"].str.lower() == "mainline", "deployment_id"])
    tb = pc.read_csv(d / E.BLOCKS_FILE).to_pandas()
    info = {"file_rows": int(len(tb)), "deployments_in_file": int(tb.deployment_id.nunique()),
            "mainline_deployments_in_metadata": len(main)}
    tb = tb[tb.deployment_id.isin(main)]
    tb["date"] = pd.to_datetime(tb["date"], format="%m/%d/%Y")
    info |= {"first_date": str(tb.date.min().date()), "last_date": str(tb.date.max().date())}
    n_blocks = tb.groupby(["deployment_id", "date"]).block.transform("nunique")
    tb = tb[n_blocks == E.STEPS].sort_values(["deployment_id", "date", "block"])
    dd = tb[["deployment_id", "date"]].iloc[::E.STEPS].reset_index(drop=True)
    kw = tb["Load (kW)"].to_numpy(float).reshape(-1, E.STEPS)
    lock = ((dd.date >= EMARC_LOCKDOWN[0]) & (dd.date <= EMARC_LOCKDOWN[1])).to_numpy()
    zero = kw.sum(axis=1) <= 0
    info |= {"complete_days": int(len(dd)), "complete_days_in_lockdown_excluded": int(lock.sum()),
             "complete_days_zero_energy_excluded": int((zero & ~lock).sum())}
    keep = ~lock & ~zero
    dd, kw = dd[keep].reset_index(drop=True), kw[keep]
    info |= {"days_used": int(len(dd)), "deployments_used": int(dd.deployment_id.nunique()),
             "window_used": [str(dd.date.min().date()), str(dd.date.max().date())]}
    ym = list(zip(dd.date.dt.year, dd.date.dt.month))
    idx = pd.Series(range(len(dd))).groupby([dd.deployment_id, [y for y, _ in ym], [m for _, m in ym]]).indices
    groups = ((dep, (int(y), int(m)), kw[np.asarray(ix)]) for (dep, y, m), ix in idx.items())
    return shape_similarity(groups), info


def attack_closeness(X: np.ndarray, days: pd.DatetimeIndex, honest_rows: list[int]) -> dict:
    """J2: 300 honest households (the tuning-sample rule restricted to honest
    households: >= 30 online January-April days, rng 20261027); each one's
    January-April online days (positive energy) through apply_attack h1-h6 with
    one generator, rng 20261031, households in sorted order, attacks h1..h6."""
    from src.data_source import theft_injector as T
    jan_apr = N.window_mask(days, ("2024-01-01", "2024-04-30"))
    eligible = [r for r in honest_rows if (~np.isnan(X[r]).any(axis=1) & jan_apr).sum() >= N.MIN_TRAIN_DAYS]
    sample = sorted(np.random.default_rng(N.JOKAR_SEED).choice(eligible, N.N_TUNE_HOUSEHOLDS,
                                                               replace=False).tolist())
    rng = np.random.default_rng(J2_ATTACK_SEED)
    months = days.month.to_numpy()
    acc = {a: {"corr": [], "inside": []} for a in T.ATTACKS}
    for r in sample:
        x = X[r].astype(float)
        ok = jan_apr & ~np.isnan(x).any(axis=1) & (np.nan_to_num(x).sum(axis=1) > 0)
        src, mon = x[ok], months[ok]
        e = src.sum(axis=1)
        lo = np.array([np.percentile(e[mon == m], 5) for m in mon])
        hi = np.array([np.percentile(e[mon == m], 95) for m in mon])
        for a in T.ATTACKS:
            syn, _ = T.apply_attack(a, src, rng)
            acc[a]["corr"].append(_row_corr(syn, src))     # scale-free: same as on sum-1 profiles
            es = syn.sum(axis=1)
            acc[a]["inside"].append((es >= lo) & (es <= hi))
    out = {"households": len(sample), "eligible_honest_households": len(eligible), "attacks": {}}
    for a, v in acc.items():
        c, ins = np.concatenate(v["corr"]), np.concatenate(v["inside"])
        out["attacks"][a] = {
            "days": int(len(c)), "corr_undefined_days": int(np.isnan(c).sum()),
            "median_corr": float(np.nanmedian(c)) if (~np.isnan(c)).any() else None,
            "share_energy_inside_p5_p95": float(ins.mean()),
            "share_corr_ge_0_999_and_energy_inside": float(((c >= 0.999) & ins).mean()),
        }
    out["note"] = ("corr: Pearson of the synthetic day's sum-1 profile with its source day's; undefined "
                   "(counted, never 'both') for a constant synthetic day (h5 always; h2 when the whole day "
                   "is switched off). Energy range: the household's own January-April online days of the "
                   "same calendar month, 5th-95th percentile (numpy linear), bounds inclusive")
    return out


def _deciles(energy: np.ndarray, valid: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Within each household-month, rank the valid days by daily energy (ties by
    date order) and map rank r of n to decile floor(10 r / n) + 1; 0 elsewhere
    [our choice of decile rule]."""
    dec = np.zeros(energy.shape, np.int8)
    for m in np.unique(months):
        cols = np.flatnonzero(months == m)
        for i in range(len(energy)):
            c = cols[valid[i, cols]]
            if not len(c):
                continue
            order = c[np.argsort(energy[i, c], kind="stable")]
            dec[i, order] = (10 * np.arange(len(c)) // len(c) + 1).astype(np.int8)
    return dec


def flagged_honest_days(X: np.ndarray, ids: list, days: pd.DatetimeIndex, honest: np.ndarray) -> dict:
    """J3 from the saved Stage 4 flags (opened read-only after the sha256 check):
    honest households, both variants, pooled over the six leave-one-attack-out
    folds; flagged share by within-household-month energy decile and on days
    1-7 of each month against the rest.  Test: July-December; validation:
    May-June (the validation models' out-of-sample months)."""
    import run_layer3_ntl_inj2 as R2
    rec = json.loads(R2.RECORD.read_text(encoding="utf-8"))
    for name in R2.PHASE_FILES:
        if R2._sha256(R2.OUT / name) != rec["converted"][name]["new_sha256"]:
            raise SystemExit(f"{name}: sha256 differs from scores_of_record.json; stop")
    from src.analytics import ntl_detector as D
    out = {"sha256_verified": {n: rec["converted"][n]["new_sha256"] for n in R2.PHASE_FILES},
           "folds_pooled": list(D.LOO_FOLDS),
           "note": "shares of household-days; days of one household are not independent, so the Wilson "
                   "intervals are narrower than the uncertainty"}
    col = {d: k for k, d in enumerate(days)}
    for phase, name, window in (("test_jul_dec", "jokar_test.npz", ("2024-07-01", "2024-12-31")),
                                ("validation_may_jun", "jokar_validation.npz", ("2024-05-01", "2024-06-30"))):
        p_ids, p_days, arr = R2._load_phase(R2.OUT / name)
        if p_ids != ids:
            raise SystemExit(f"{name}: households differ from the AMI export")
        sel = N.window_mask(p_days, window)
        w_days = p_days[sel]
        energy = X[:, [col[d] for d in w_days]].sum(axis=2)
        months = w_days.month.to_numpy()
        first7 = np.asarray(w_days.day <= 7)
        res = {"days": int(sel.sum())}
        for v in D.VARIANTS:
            fl = np.concatenate([arr[v, f]["flag"][:, sel][honest] for f in D.LOO_FOLDS], axis=0)
            valid = fl[: honest.sum()] >= 0
            if not all(np.array_equal(valid, (arr[v, f]["flag"][:, sel][honest] >= 0)) for f in D.LOO_FOLDS):
                raise SystemExit("online days differ between folds")
            dec = np.tile(_deciles(energy[honest], valid, months), (len(D.LOO_FOLDS), 1))
            on, hit = fl >= 0, fl == 1
            f7 = np.broadcast_to(first7, fl.shape)
            res[v] = {
                "by_energy_decile": {int(k): V.wilson(int((hit & (dec == k)).sum()), int((on & (dec == k)).sum()))
                                     for k in range(1, 11)},
                "days_1_to_7": V.wilson(int((hit & f7).sum()), int((on & f7).sum())),
                "days_8_to_end": V.wilson(int((hit & ~f7).sum()), int((on & ~f7).sum())),
                "all_online": V.wilson(int(hit.sum()), int(on.sum())),
            }
        out[phase] = res
    return out


def main_j() -> int:
    import run_layer3_ntl_inj2 as R2
    from src.analytics import ntl_detector as D
    t0 = time.perf_counter()
    ids, days, X = D.ami_vectors(R2.AMI, 96)
    labels = V.load_household_labels(R2.INJ2).reindex(ids)
    honest = (labels.attack == "").to_numpy()
    rows = np.flatnonzero(honest).tolist()
    t = {}
    sim = shape_similarity(_sim_groups(X, days, rows))
    t["j1_simulated"] = round(time.perf_counter() - t0, 1)
    emarc, emarc_info = emarc_reference()
    t["j1_emarc"] = round(time.perf_counter() - t0 - sum(t.values()), 1)
    j2 = attack_closeness(X, days, rows)
    t["j2"] = round(time.perf_counter() - t0 - sum(t.values()), 1)
    j3 = flagged_honest_days(X, ids, days, honest)
    t["j3"] = round(time.perf_counter() - t0 - sum(t.values()), 1)
    res = {
        "label": LABEL_J,
        "hypothesis": "honest days of one household-month are one intraday shape scaled by daily energy, so "
                      "(i) the honest shape changes in steps at month boundaries and (ii) h1 (alpha x) yields "
                      "days indistinguishable from honest low-energy days",
        "data": "injection 2 (results/year_2024_inj2), honest households (attack == ''), 96 samples a day, "
                "online days (all 96 samples) with positive energy",
        "honest_households": int(honest.sum()),
        "min_days_per_household_month": MIN_MONTH_DAYS,
        "j1_shape_similarity": {"simulated_honest_2024": sim, "emarc_raw_reference": emarc,
                                "emarc_handling": emarc_info},
        "j2_attack_closeness": j2,
        "j3_flagged_honest_days": j3,
        "elapsed_s": {**t, "total": round(time.perf_counter() - t0, 1)},
    }
    R2._write_summary("addendum_j", res)
    print(json.dumps(res, indent=1, default=str)[:6000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("addendum", choices=("h", "j"))
    return {"h": main_h, "j": main_j}[ap.parse_args(argv).addendum]()


if __name__ == "__main__":
    sys.exit(main())
