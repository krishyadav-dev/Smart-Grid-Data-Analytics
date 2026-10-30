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
    python scripts/run_layer3_ntl_inj2.py jokar-grid --samples 24                     # step D: grid AUC
    python scripts/run_layer3_ntl_inj2.py jokar-grid --samples 96 --variants static   # step D: B9b
    python scripts/run_layer3_ntl_inj2.py jokar-tune    # step D: freeze config/layer3_ntl_params_inj2.yaml
    python scripts/run_layer3_ntl_inj2.py jokar-test    # step E: July-December, run once
    python scripts/run_layer3_ntl_inj2.py jokar-convert-dates   # step E recovery (30 Oct), once
    python scripts/run_layer3_ntl_inj2.py jokar-metrics         # step E recovery: metrics from saved scores, once

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
from src.analytics import ntl_detector as D  # noqa: E402
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


# --------------------------------------------------------------------------
# Jokar detector (steps D-E; run sheet §4)
# --------------------------------------------------------------------------
PARAMS = ROOT / "config" / "layer3_ntl_params_inj2.yaml"
AMI = INJ2 / "ami"
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
WINDOWS = {"validation": ("2024-05-01", "2024-06-30"), "test": ("2024-07-01", "2024-12-31")}


def _tuning_sample(ids, days, X) -> list[int]:
    """The Stage 2 sample rule: 300 households with >= 30 online January-April
    days, rng 20261027."""
    tr = N.window_mask(days, D.PHASES["validation"][0])
    eligible = [r for r in range(len(ids)) if (~np.isnan(X[r]).any(axis=1) & tr).sum() >= N.MIN_TRAIN_DAYS]
    return sorted(np.random.default_rng(N.JOKAR_SEED).choice(eligible, N.N_TUNE_HOUSEHOLDS, replace=False).tolist())


def _grid_job(cid, rows, xs, days, variants):
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(1):
        for r, x in zip(rows, xs):
            p = D.prepare(x, days, r, "validation")
            out.append(None if p is None else {f: D.grid_auc(p, f, D.GRID, variants) for f in D.FOLDS})
    return cid, out


def _run_chunks(fn, rows, X, days, parts: Path, *a, chunk: int):
    """Run ``fn`` over household chunks in parallel; each finished chunk is
    saved under ``parts`` and skipped on a restart (an interruption loses at
    most the chunks in flight)."""
    import pickle

    from joblib import Parallel, delayed
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    todo = [k for k in range(len(chunks)) if not (parts / f"{k:05d}.pkl").exists()]
    logging.info("%s: %d chunks, %d to do", parts.name, len(chunks), len(todo))
    gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered", verbose=5)(
        delayed(fn)(k, chunks[k], [X[r] for r in chunks[k]], days, *a) for k in todo)
    for k, res in gen:
        (parts / f"{k:05d}.pkl").write_bytes(pickle.dumps(res))
    out = []
    for k in range(len(chunks)):
        out += pickle.loads((parts / f"{k:05d}.pkl").read_bytes())
    return out


def cmd_jokar_grid(args) -> int:
    """Step D: per-household day-level AUC over the B9a grid on the 300-household
    tuning sample, January-June only (synthetic objective; no labels)."""
    t0 = time.perf_counter()
    variants = tuple(args.variants.split(","))
    ids, days, X = D.ami_vectors(AMI, args.samples)
    sample = _tuning_sample(ids, days, X)
    tag = f"{args.samples}_{'_'.join(variants)}"
    res = _run_chunks(_grid_job, sample, X, days, OUT / f"grid_parts_{tag}", variants, chunk=2)
    if any(r is None for r in res):
        raise SystemExit("a tuning-sample household has fewer than 30 training days")
    arrays = {f"auc_{v}": np.array([[r[f][v] for f in D.FOLDS] for r in res]) for v in variants}
    np.savez_compressed(OUT / f"jokar_grid_auc_{tag}.npz", households=np.array([ids[r] for r in sample]),
                        folds=np.array(list(D.FOLDS)), grid=np.array(D.GRID), samples=args.samples,
                        elapsed_s=time.perf_counter() - t0, **arrays)
    print(f"grid {tag}: {[a.shape for a in arrays.values()]}, {time.perf_counter() - t0:.0f} s")
    return 0


def _load_grid(samples: int, variant: str) -> tuple[np.ndarray, list]:
    """(auc (households, folds, grid), households) for one resolution and variant."""
    for path in sorted(OUT.glob(f"jokar_grid_auc_{samples}_*.npz")):
        f = np.load(path)
        if f"auc_{variant}" in f:
            if list(f["folds"]) != list(D.FOLDS) or [tuple(x) for x in f["grid"]] != list(D.GRID):
                raise SystemExit(f"{path.name}: folds or grid differ")
            return f[f"auc_{variant}"], list(f["households"])
    raise SystemExit(f"no grid AUC for {samples} samples, variant {variant}: run jokar-grid first")


def _phase_job(cid, rows, xs, days, phase, params):
    """Every fold and variant for one phase: per-day scores, flags and B1 adds."""
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(1):
        for r, x in zip(rows, xs):
            p = D.prepare(x, days, r, phase)
            if p is None:
                out.append(None)
                continue
            res = {"k_star": p["base"]["cl"]["k_star"], "k_prime": p["base"]["cl"]["k_prime"],
                   "n_train_days": p["n_train_days"], "runs": {}}
            for v in D.VARIANTS:
                for f in D.FOLDS:
                    q = params[v][f]
                    res["runs"][v, f] = D.run_fold(p, D.FOLDS[f], q["C"], q["gamma"], v)
            out.append(res)
    return cid, out


def _stack(scored, days_out: int) -> dict:
    """{(variant, fold): {"flag", "score", "added"}} arrays (households, days); skipped households: flag -2."""
    out = {}
    for v in D.VARIANTS:
        for f in D.FOLDS:
            fl = np.full((len(scored), days_out), -2, np.int8)
            sc = np.full((len(scored), days_out), np.nan, np.float32)
            ad = np.zeros((len(scored), days_out), bool)
            for i, s in enumerate(scored):
                if s is not None:
                    r = s["runs"][v, f]
                    fl[i], sc[i], ad[i] = r["flag"], r["score"], r["added"]
            out[v, f] = {"flag": fl, "score": sc, "added": ad}
    return out


def _gate(ids, window_days: pd.DatetimeIndex, idx: pd.Series) -> np.ndarray:
    """(households, days) transformer gate (Jokar §IV-B step 1): the DT's engine
    NTL^_B (lambda_assumed = 0) of that day above zero."""
    d = _dt_ntl_lambda0(idx).reindex(window_days)
    col = {n: k for k, n in enumerate(d.columns)}
    return (d > 0).to_numpy()[:, [col[idx[h]] for h in ids]].T


def _household_scores(flag: np.ndarray, ids, sel: np.ndarray, gate: Optional[np.ndarray] = None) -> pd.Series:
    """Household score over the days ``sel``: share of online days flagged (gated: and gate open)."""
    f = flag[:, sel]
    g = None if gate is None else gate
    v = {h: N.household_score(f[i], None if g is None else g[i]) for i, h in enumerate(ids) if f[i, 0] != -2}
    return pd.Series(v, dtype=float)


def _faithful_counts(arr: dict, days: pd.DatetimeIndex, gate: np.ndarray, add_months) -> dict:
    """B1 report: per month M - 1 whose days B1 adds, the days added by the rule
    used (benign by the SVM) and the days the faithful gated rule (Jokar step 4:
    benign by the SVM and NTL^_B <= 0 on that DT-day) would have added, per fold
    and pooled over the six leave-one-attack-out folds (``update`` variant)."""
    out = {}
    for f in D.FOLDS:
        a = arr["update", f]
        out[f] = {}
        for m in add_months:
            sel = np.asarray(days.month == m)
            ben = a["flag"][:, sel] == 0
            out[f][int(m)] = {"added_b1": int(a["added"][:, sel].sum()), "benign_by_svm": int(ben.sum()),
                              "faithful_gated_rule": int((ben & ~gate[:, sel]).sum())}
    out["pooled_loo"] = {int(m): {k: sum(out[f][int(m)][k] for f in D.LOO_FOLDS)
                                  for k in ("added_b1", "benign_by_svm", "faithful_gated_rule")} for m in add_months}
    return out


def _kmeans_table(scored) -> dict:
    ok = [s for s in scored if s is not None]
    return {"k_prime": {int(k): int(n) for k, n in zip(*np.unique([s["k_prime"] for s in ok], return_counts=True))},
            "k_star": {int(k): int(n) for k, n in zip(*np.unique([s["k_star"] for s in ok], return_counts=True))},
            "households": len(ok), "skipped_fewer_than_min_train_days": int(sum(s is None for s in scored))}


def _k_prime_after_updates(scored) -> dict:
    """k' of the update variant's models per month, pooled over households and folds."""
    out: dict = {}
    for s in scored:
        if s is None:
            continue
        for f in D.FOLDS:
            for m, k in s["runs"]["update", f]["k_prime"].items():
                out.setdefault(int(m), {}).setdefault(int(k), 0)
                out[int(m)][int(k)] += 1
    return out


def _save_phase(path: Path, ids, days, arr: dict) -> None:
    # days as fixed-width text: strftime returns an object Index, which np.savez stores
    # pickled and np.load (allow_pickle=False, as in _load_phase) refuses
    np.savez_compressed(path, ids=np.array(ids), days=np.asarray(days.strftime("%Y-%m-%d"), dtype="U10"),
                        **{f"{k}__{v}__{f}": a[k] for (v, f), a in arr.items() for k in ("flag", "score", "added")})


def _load_phase(path: Path) -> tuple[list, pd.DatetimeIndex, dict]:
    z = np.load(path)
    arr: dict = {}
    for name in z.files:
        if "__" in name:
            k, v, f = name.split("__")
            arr.setdefault((v, f), {})[k] = z[name]
    return list(z["ids"]), pd.DatetimeIndex(z["days"]), arr


def cmd_jokar_tune(args) -> int:
    """Step D: B9b resolution, (C, gamma) per fold and variant, validation run
    (train January-April, score May-June; update at the start of June), q per
    fold and variant; writes config/layer3_ntl_params_inj2.yaml (freeze)."""
    import yaml
    t0 = time.perf_counter()
    if PARAMS.exists():
        raise SystemExit(f"{PARAMS.name} exists: parameters are frozen (tune once)")
    # B9b: each resolution's static objective at each fold's best grid point
    best = {}
    for s in D.RESOLUTIONS:
        auc, _ = _load_grid(s, "static")
        best[s] = {f: N.select_svm_params(auc[:, j, :], list(D.GRID))[2] for j, f in enumerate(D.FOLDS)}
    choice = D.resolution_choice(best)
    choice["average_over_all_seven_folds"] = {s: float(np.mean(list(best[s].values()))) for s in best}
    choice["best_mean_auc_per_fold"] = best
    samples = choice["samples_per_day"]
    params, picks, grid_table = {}, {}, {}
    for v in D.VARIANTS:
        auc, sample_ids = _load_grid(samples, v)
        params[v], picks[v], grid_table[v] = {}, {}, {}
        for j, f in enumerate(D.FOLDS):
            c, g, m = N.select_svm_params(auc[:, j, :], list(D.GRID))
            params[v][f] = {"C": c, "gamma": g, "mean_auc": m}
            picks[v][f] = {"C": c, "gamma": g, "mean_auc": m, "on_edge": D.edge(c, g)}
            grid_table[v][f] = {f"C={cc:g},gamma={gg:g}": float(x)
                                for (cc, gg), x in zip(D.GRID, np.nanmean(auc[:, j, :], axis=0))}
    ids, days, X = D.ami_vectors(AMI, samples)
    t_fit = time.perf_counter()
    scored = _run_chunks(_phase_job, list(range(len(ids))), X, days, OUT / f"validation_parts_{samples}",
                         "validation", params, chunk=20)
    t_fit = time.perf_counter() - t_fit
    out_days = days[N.window_mask(days, D.PHASES["validation"][2])]
    arr = _stack(scored, len(out_days))
    _save_phase(OUT / "jokar_validation.npz", ids, out_days, arr)
    labels = V.load_household_labels(INJ2)
    idx = _consumer_index()
    sel = np.asarray(out_days >= pd.Timestamp(WINDOWS["validation"][0]))
    gate = _gate(ids, out_days, idx)
    thresholds = {}
    for v in D.VARIANTS:
        thresholds[v] = {}
        for f in D.FOLDS:
            y = V.fold_targets(labels, ids, D.FOLDS[f], WINDOWS["validation"])
            fl = arr[v, f]["flag"]
            thresholds[v][f] = {"ungated": V.best_threshold(_household_scores(fl, ids, sel), y),
                                "gated": V.best_threshold(_household_scores(fl, ids, sel, gate[:, sel]), y)}
            params[v][f]["q"] = thresholds[v][f]["ungated"]["q"]
            params[v][f]["q_gated"] = thresholds[v][f]["gated"]["q"]
    frozen = {
        "label": "Stage 4 (theft-engine fixes), injection 2; supersedes nothing (v1: config/layer3_ntl_params.yaml)",
        "tuned_on": {"data": "injection 2 (results/year_2024_inj2), January-June 2024 only",
                     "inner_split": "initial training January-April, validation May-June; update variant: B1 at "
                                    "the start of June with May's benign-classified days",
                     "svm_params": f"{N.N_TUNE_HOUSEHOLDS} households, rng {N.JOKAR_SEED}; mean per-household day-level "
                                   "ROC AUC, May-June days vs synthetic attacked versions (fold's attacks)",
                     "threshold": "household-level F1 on May-June; positives: thieves of the fold's training attacks "
                                  "active on >= 1 May-June day; negatives: honest households; ties: lowest q"},
        "samples_per_day": samples,
        "resolution_choice_b9b": {"rule": "average over the six leave-one-attack-out folds of each fold's best mean "
                                          "AUC (static objective); 96 only if higher, ties keep 24",
                                  "average_over_folds": {int(k): v for k, v in choice["average_over_folds"].items()}},
        "variants": {"primary": "update", "ablation": "static"},
        "folds": params,
        "b1_rule": "update: at the start of each scored month M, the online days of M - 1 the current model "
                   "classified benign join the benign set; their attack patterns (fold's training attacks, child "
                   f"stream {D.UPDATE_STREAM} + month of rng {N.JOKAR_SEED}) join the attack set; scaling, k-means "
                   "rule and SVC (frozen C, gamma) refitted [our choice: Jokar step 4 without its DT condition]",
        "day_score": "SVC decision_function (ovr), malicious class column",
        "household_score": "share of online scored days flagged (predicted class = malicious)",
        "flag_rule": "household flagged if score >= q",
        "kmeans": {"k_range": list(N.K_RANGE), "rule": "argmax mean silhouette; ties: smaller k",
                   "min_cluster_share": N.MIN_CLUSTER_SHARE, "random_state": N.JOKAR_SEED},
        "min_train_days": N.MIN_TRAIN_DAYS, "seed": N.JOKAR_SEED,
        "grid": {"C": list(D.GRID_C), "gamma": list(D.GRID_GAMMA), "rule": "B9a; edge picks reported, not extended"},
        "gate": "transformer gate = engine NTL^_B (lambda_assumed = 0, A3 offline estimate) > 0; gated uses q_gated",
        "primary": "ungated (q) unless the gate is open on 95 % of DT-days or fewer",
    }
    PARAMS.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    rec = {"params": params, "picks": picks, "resolution_b9b": choice, "grid_mean_auc": grid_table,
           "thresholds": thresholds, "tuning_sample_households": sample_ids,
           "kmeans_validation_phase": _kmeans_table(scored),
           "k_prime_update_by_month": _k_prime_after_updates(scored),
           "faithful_rule_counts_validation": _faithful_counts(arr, out_days, gate, [5]),
           "gate_open_may_jun_dt_days": V.wilson(
               int((_dt_ntl_lambda0(idx).reindex(pd.date_range(*WINDOWS["validation"])) > 0).to_numpy().sum()),
               int(_dt_ntl_lambda0(idx).reindex(pd.date_range(*WINDOWS["validation"])).size)),
           "elapsed_s": {"validation_run": round(t_fit, 1), "tune_total": round(time.perf_counter() - t0, 1),
                         "grids": {p.name: float(np.load(p)["elapsed_s"]) for p in sorted(OUT.glob("jokar_grid_auc_*.npz"))}}}
    (OUT / "jokar_tuning.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"resolution": {k: choice[k] for k in ("average_over_folds", "samples_per_day")},
                      "picks": picks, "q": {v: {f: params[v][f]["q"] for f in D.FOLDS} for v in D.VARIANTS},
                      "elapsed_s": rec["elapsed_s"]}, indent=2, default=str))
    return 0


B8_NOTE = ("h5/h6 shift load without reducing energy, so neither the balance nor an energy-theft label applies "
           "under a flat tariff; reported separately (B8)")


def _pooled_month(per_fold: dict, folds) -> dict:
    months = sorted({m for f in folds for m in per_fold[f]})
    return {m: V.wilson(sum(per_fold[f][m]["k"] for f in folds if m in per_fold[f]),
                        sum(per_fold[f][m]["n"] for f in folds if m in per_fold[f])) for m in months}


def cmd_jokar_test(args) -> int:
    """Step E: July-December on injection 2, scored once with the frozen settings."""
    import hashlib
    import subprocess

    import yaml
    t0 = time.perf_counter()
    if not PARAMS.exists():
        raise SystemExit(f"{PARAMS.name} missing: run jokar-tune and commit it first (step D)")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PARAMS)], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PARAMS)], cwd=ROOT)
    if tracked.returncode or dirty.returncode:
        raise SystemExit(f"{PARAMS.name} is not committed as it is: commit D must precede the test run")
    if (OUT / "jokar_test_opened.json").exists():
        raise SystemExit("the July-December window was already scored (run once)")
    frozen = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    params, samples = frozen["folds"], frozen["samples_per_day"]
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "jokar_test_opened.json").write_text(json.dumps({
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "head_commit": head,
        "params_sha256": hashlib.sha256(PARAMS.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    ids, days, X = D.ami_vectors(AMI, samples)
    t_fit = time.perf_counter()
    scored = _run_chunks(_phase_job, list(range(len(ids))), X, days, OUT / f"test_parts_{samples}", "test",
                         params, chunk=20)
    t_fit = time.perf_counter() - t_fit
    out_days = days[N.window_mask(days, D.PHASES["test"][2])]
    arr = _stack(scored, len(out_days))
    _save_phase(OUT / "jokar_test.npz", ids, out_days, arr)
    res = _jokar_results(ids, out_days, scored, arr, params, samples)
    res["elapsed_s"] = {"fit_and_score": round(t_fit, 1), "total": round(time.perf_counter() - t0, 1)}
    _write_summary("jokar", res)
    _print_headline(res)
    return 0


def _jokar_results(ids, out_days, scored, arr: dict, params: dict, samples: int) -> dict:
    """Step E metrics and diagnostics from the July-December scores (moved unchanged
    out of cmd_jokar_test so that jokar-metrics computes the same from saved scores)."""
    labels = V.load_household_labels(INJ2)
    idx = _consumer_index()
    gate = _gate(ids, out_days, idx)
    d_ntl = _dt_ntl_lambda0(idx).reindex(out_days)
    gate_rate = V.wilson(int((d_ntl > 0).to_numpy().sum()), int(d_ntl.size))
    allsel = np.ones(len(out_days), bool)
    loo = list(D.LOO_FOLDS)
    res = {"label": "Jokar SVM on injection 2, leave-one-attack-out; July-December scored once with the frozen "
                    "settings; update (B1) primary, static the ablation",
           "params_file": str(PARAMS.relative_to(ROOT)), "samples_per_day": samples,
           "gate_open_dt_days": gate_rate, "kmeans_test_phase": _kmeans_table(scored),
           "k_prime_update_by_month": _k_prime_after_updates(scored),
           "faithful_rule_counts": _faithful_counts(arr, out_days, gate, [7, 8, 9, 10, 11]),
           "b8_note": B8_NOTE}
    for v in D.VARIANTS:
        res[v] = {}
        for gname, g, qkey in (("ungated", None, "q"), ("gated", gate, "q_gated")):
            per_fold, scores, qs = {}, {}, {}
            for f in D.FOLDS:
                s = _household_scores(arr[v, f]["flag"], ids, allsel, g)
                y = V.fold_targets(labels, ids, D.FOLDS[f] if f == "circular" else (f,), WINDOWS["test"])
                per_fold[f] = V.fold_metrics(s, y, params[v][f][qkey])
                scores[f], qs[f] = s, params[v][f][qkey]
            res[v][gname] = {
                "per_fold": per_fold,
                "macro_energy_theft_h1_h4": V.macro(per_fold, V.ENERGY_THEFT),
                "macro_tariff_shift_h5_h6": {**V.macro(per_fold, V.TARIFF_SHIFT), "note": B8_NOTE},
                "macro_all_six": V.macro(per_fold, tuple(loo)),
                "circular_reference": {"label": "circular: graded on its own generator; upper bound only",
                                       **{k: per_fold["circular"][k] for k in ("precision", "recall", "f1")},
                                       "average_precision": per_fold["circular"]["pr_curve"]["average_precision"]},
                "recall_by_onset_month": V.by_onset_month(scores, qs, labels, {
                    "energy_theft_h1_h4": V.ENERGY_THEFT, "tariff_shift_h5_h6": V.TARIFF_SHIFT,
                    "all_six": tuple(loo)}),
            }
        pb = {f: V.precision_at_budgets(arr[v, f]["flag"], arr[v, f]["score"], out_days, ids, labels, (f,))
              for f in loo}
        res[v]["precision_at_budgets"] = {
            "per_fold": pb,
            "macro_energy_theft_h1_h4": {
                **{k: float(np.mean([pb[f]["pooled"][k]["p"] for f in V.ENERGY_THEFT]))
                   for k in pb["h1"]["pooled"] if k.startswith("precision_at")},
                "prevalence": float(np.mean([pb[f]["pooled"]["prevalence"] for f in V.ENERGY_THEFT]))}}
        rm = {f: V.recall_by_month(arr[v, f]["flag"], arr[v, f]["score"], out_days, ids, labels, (f,),
                                   params[v][f]["q"]) for f in loo}
        res[v]["recall_by_month"] = {"per_fold": rm, "pooled_energy_theft_h1_h4": _pooled_month(rm, V.ENERGY_THEFT),
                                     "pooled_tariff_shift_h5_h6": _pooled_month(rm, V.TARIFF_SHIFT)}
    # drift diagnostic: honest households' flagged-day share by month, pooled over the six LOO folds;
    # January-June from the validation phase (January-April = its own training days), July-December from the test
    v_ids, v_days, v_arr = _load_phase(OUT / "jokar_validation.npz")
    if v_ids != ids:
        raise SystemExit("validation and test households differ")
    all_days = v_days.append(out_days)
    res["drift_honest_flag_share_by_month"] = {
        v: V.honest_flag_share_by_month({f: np.hstack([v_arr[v, f]["flag"], arr[v, f]["flag"]]) for f in loo},
                                         all_days, ids, labels) for v in D.VARIANTS}
    res["drift_honest_flag_share_by_month"]["note"] = ("pooled over the six leave-one-attack-out folds; January-April "
                                                       "are the validation models' own training days")
    res["absorption_update"] = V.absorption({f: arr["update", f]["flag"] for f in loo},
                                            {f: arr["update", f]["added"] for f in loo}, out_days, ids, labels)
    res["primary_gate"] = "ungated" if (gate_rate["p"] or 0) > 0.95 else "gated"
    return res


def _print_headline(res: dict) -> None:
    print(json.dumps({v: {k: res[v]["ungated"][k] for k in ("macro_energy_theft_h1_h4", "macro_tariff_shift_h5_h6",
                                                              "macro_all_six")} for v in D.VARIANTS}
                     | {"gate": res["gate_open_dt_days"], "elapsed_s": res["elapsed_s"]}, indent=2, default=str))


# --------------------------------------------------------------------------
# Step E recovery (approved 30 Oct): the 07:27 jokar-test run scored every
# household, then stopped at the drift diagnostic because _save_phase had
# stored the days as a pickled object array.  The two phase files are
# converted once to fixed-width text and the metrics computed from the saved
# scores; nothing is refitted.
# --------------------------------------------------------------------------
RECORD = OUT / "scores_of_record.json"          # sha256 of the saved scores, taken before any code change
PHASE_FILES = ("jokar_validation.npz", "jokar_test.npz")


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_record(names) -> None:
    """Refuse unless every named file still has the hash of record (converted
    phase files: their hash after the conversion)."""
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    conv = rec.get("converted", {})
    for name in names:
        want = conv[name]["new_sha256"] if name in conv else rec["files"][name]["sha256"]
        if _sha256(OUT / name) != want:
            raise SystemExit(f"{name}: sha256 differs from the record; stop")


def cmd_jokar_convert_dates(args) -> int:
    """Rewrite the two phase files with the days as fixed-width text, once; every
    array is checked equal before the old file is replaced (old and new sha256
    recorded)."""
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    if "converted" in rec:
        raise SystemExit("phase files already converted (run once)")
    _check_record(PHASE_FILES)
    conv = {}
    for name in PHASE_FILES:
        path, tmp = OUT / name, OUT / f"{name}.tmp.npz"
        old = np.load(path, allow_pickle=True)                 # our own file; the pickled array is "days"
        arrays = {k: old[k] for k in old.files}
        if arrays["days"].dtype != object:
            raise SystemExit(f"{name}: days are not an object array; nothing to convert")
        arrays["days"] = np.asarray(arrays["days"], dtype="U10")
        np.savez_compressed(tmp, **arrays)
        new = np.load(tmp)                                     # reads without pickle
        if new.files != old.files or new["days"].dtype != np.dtype("U10"):
            raise SystemExit(f"{name}: converted file differs in layout")
        for k in old.files:
            a, b = old[k], new[k]
            same = (list(a) == list(b)) if k == "days" else (
                a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
            if not same:
                raise SystemExit(f"{name}: array {k} changed in the conversion")
        old.close()
        new.close()
        conv[name] = {"old_sha256": rec["files"][name]["sha256"], "new_sha256": _sha256(tmp),
                      "arrays_checked_equal": len(old.files), "days_dtype": "object -> <U10"}
        tmp.replace(path)
    rec["converted"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **conv}
    RECORD.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec["converted"], indent=2))
    return 0


def _load_parts(parts: Path, n_households: int) -> list:
    """The saved per-household results, in order; refuses (never fits) if a chunk is missing."""
    import pickle
    files = sorted(parts.glob("*.pkl"))
    if [p.name for p in files] != [f"{k:05d}.pkl" for k in range(len(files))]:
        raise SystemExit(f"{parts.name}: chunks are not contiguous")
    out = []
    for p in files:
        out += pickle.loads(p.read_bytes())
    if len(out) != n_households:
        raise SystemExit(f"{parts.name}: {len(out)} households saved, {n_households} expected")
    return out


def cmd_jokar_metrics(args) -> int:
    """Step E metrics from the saved scores of the 07:27 jokar-test run, once:
    retrains nothing; refuses unless the frozen settings are the ones the test
    run opened with and every score file has its hash of record."""
    import subprocess

    import yaml
    t0 = time.perf_counter()
    opened = OUT / "jokar_test_opened.json"
    if not opened.exists():
        raise SystemExit("jokar_test_opened.json missing: there is no scored test run to compute metrics from")
    if (OUT / "jokar_metrics_opened.json").exists():
        raise SystemExit("metrics were already computed from the saved scores (run once)")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PARAMS)], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PARAMS)], cwd=ROOT)
    if tracked.returncode or dirty.returncode:
        raise SystemExit(f"{PARAMS.name} is not committed as it is")
    test_run = json.loads(opened.read_text(encoding="utf-8"))
    if _sha256(PARAMS) != test_run["params_sha256"]:
        raise SystemExit(f"{PARAMS.name} differs from the settings the test run opened with")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    if "converted" not in rec:
        raise SystemExit("run jokar-convert-dates first")
    _check_record(list(rec["files"]))                          # immediately before computing metrics
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "jokar_metrics_opened.json").write_text(json.dumps({
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "head_commit": head,
        "params_sha256": test_run["params_sha256"], "scores_of_record_verified": len(rec["files"]),
        "from_test_run": test_run}, indent=2), encoding="utf-8")
    frozen = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    params, samples = frozen["folds"], frozen["samples_per_day"]
    ids, out_days, arr = _load_phase(OUT / "jokar_test.npz")
    scored = _load_parts(OUT / f"test_parts_{samples}", len(ids))
    stacked = _stack(scored, len(out_days))
    for key, a in arr.items():
        for k in ("flag", "score", "added"):
            if not np.array_equal(a[k], stacked[key][k], equal_nan=k == "score"):
                raise SystemExit(f"jokar_test.npz differs from the saved chunks at {key} {k}")
    res = _jokar_results(ids, out_days, scored, arr, params, samples)
    res["computed_from_saved_scores"] = {
        "note": "metrics computed from the saved scores of the 07:27 jokar-test run after the loader fix "
                "(that run stopped at the drift diagnostic before writing any metric); nothing refitted",
        "test_run": test_run, "scores_of_record": RECORD.name, "digest_of_file_hashes": rec["digest_of_file_hashes"]}
    res["elapsed_s"] = {"fit_and_score_wall_clock_07_27_run": round(
        (pd.Timestamp(rec["files"]["jokar_test.npz"]["mtime"]) - pd.Timestamp(test_run["opened_at"])).total_seconds(), 1),
        "metrics_only": round(time.perf_counter() - t0, 1)}
    _write_summary("jokar", res)
    _print_headline(res)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("injection", help="step A: injection report").set_defaults(func=cmd_injection)
    sub.add_parser("balance", help="step C: balance results on injection 2, scored once").set_defaults(func=cmd_balance)
    p = sub.add_parser("jokar-grid", help="step D: grid AUC on the tuning sample")
    p.add_argument("--samples", type=int, choices=D.RESOLUTIONS, required=True)
    p.add_argument("--variants", default="static,update")
    p.set_defaults(func=cmd_jokar_grid)
    sub.add_parser("jokar-tune", help="step D: choose and freeze the detector settings").set_defaults(func=cmd_jokar_tune)
    sub.add_parser("jokar-test", help="step E: score July-December once").set_defaults(func=cmd_jokar_test)
    sub.add_parser("jokar-convert-dates", help="step E recovery: phase files' days to fixed-width text, once"
                   ).set_defaults(func=cmd_jokar_convert_dates)
    sub.add_parser("jokar-metrics", help="step E recovery: metrics from the saved test scores, once"
                   ).set_defaults(func=cmd_jokar_metrics)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
