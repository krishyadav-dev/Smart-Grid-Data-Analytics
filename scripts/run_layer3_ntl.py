"""Layer 3 theft engine: entry point (Stage 2 run sheet).

Commands (PowerShell)::

    python scripts/run_layer3_ntl.py ltech --inputs real     # L^_tech power flow, ~65 min
    python scripts/run_layer3_ntl.py ltech --inputs ideal    # same on the ideal-meter inputs (N6)
    python scripts/run_layer3_ntl.py balance                 # step C: balance results, scored once
    python scripts/run_layer3_ntl.py jokar-grid              # step D: (C, gamma) grid AUC, 300 households
    python scripts/run_layer3_ntl.py jokar-tune              # step D: (C, gamma) and q per fold -> freeze
    python scripts/run_layer3_ntl.py jokar-test              # step E: July-December, run once

Outputs under ``results/layer3/ntl/`` (``summary.json`` key ``balance``).
Labels are read through :mod:`src.analytics.evaluate` only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
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


# --------------------------------------------------------------------------
# Jokar detector: tuning (step D) and test (step E)
# --------------------------------------------------------------------------
PARAMS = ROOT / "config" / "layer3_ntl_params.yaml"
AMI_DIR = YEAR_CACHE_DIR / "ami"
GRID = [(c, g) for c in N.GRID_C for g in N.GRID_GAMMA]
GRID_FIRST = [(c, g) for c in N.GRID_C_FIRST for g in N.GRID_GAMMA_FIRST]
N_JOBS = max(1, (os.cpu_count() or 2) - 2)


def _chunks(n: int, size: int):
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def _grid_job(rows, xs, days, grid):
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(1):
        for r, x in zip(rows, xs):
            p = N.prepare_household(x, days, r, "validation")
            out.append(None if p is None else np.array([N.grid_auc(p, r, f, grid) for f in N.FOLDS]))
    return out


def _score_job(rows, xs, days, phase, params):
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(1):
        for r, x in zip(rows, xs):
            p = N.prepare_household(x, days, r, phase)
            if p is None:
                out.append(None)
                continue
            res = {f: N.score_household(p, N.FOLDS[f], params[f]["C"], params[f]["gamma"]) for f in N.FOLDS}
            out.append({"k_star": p["clusters"]["k_star"], "k_prime": p["clusters"]["k_prime"],
                        "n_train_days": p["n_train_days"], "folds": res})
    return out


def _parallel(fn, rows, X, days, *a, n_jobs=None, chunk=40):
    from joblib import Parallel, delayed
    jobs = _chunks(len(rows), chunk)
    res = Parallel(n_jobs=n_jobs or N_JOBS, verbose=5)(
        delayed(fn)([rows[i] for i in c], [X[rows[i]] for i in c], days, *a) for c in jobs)
    return [x for r in res for x in r]


def _dt_ntl_lambda0() -> pd.DataFrame:
    """Engine NTL^_B per DT-day at lambda_assumed = 0 (days x DT), computed from
    the engine inputs only (no labels); cached in dt_ntl_lambda0.parquet."""
    path = OUT / "dt_ntl_lambda0.parquet"
    if not path.exists():
        idx = N.read_consumer_index(OUT / "consumer_index.parquet")
        est = N.balance_estimates(N.read_meter_frame(INPUTS["real"]["frame"]),
                                  N.read_household_daily(INPUTS["real"]["household_daily"]), idx, None)
        est["ntl_dt"][0.0].to_parquet(path)
    return pd.read_parquet(path)


def _gate(ids, days_window, idx: pd.Series) -> np.ndarray:
    """(households, days) transformer gate (Jokar §IV-B step 1): the DT's
    NTL^_B (lambda_assumed = 0) of that day above zero."""
    d = _dt_ntl_lambda0().reindex(pd.DatetimeIndex(days_window))
    open_ = (d > 0).to_numpy()                                      # (days, DT)
    col = {n: k for k, n in enumerate(d.columns)}
    return open_[:, [col[idx[h]] for h in ids]].T


def _gate_rate(days_window) -> dict:
    """Share of DT-days in the window with the gate open (Wilson)."""
    d = _dt_ntl_lambda0().reindex(pd.DatetimeIndex(days_window))
    return V.wilson(int((d > 0).to_numpy().sum()), int(d.size))


def _household_scores(scored, ids, fold, gate=None) -> pd.Series:
    v = {}
    for k, (h, s) in enumerate(zip(ids, scored)):
        if s is not None:
            v[h] = N.household_score(s["folds"][fold][1], None if gate is None else gate[k])
    return pd.Series(v, dtype=float)


def _kmeans_table(scored) -> dict:
    ok = [s for s in scored if s is not None]
    return {"k_prime": {int(k): int(n) for k, n in zip(*np.unique([s["k_prime"] for s in ok], return_counts=True))},
            "k_star": {int(k): int(n) for k, n in zip(*np.unique([s["k_star"] for s in ok], return_counts=True))},
            "households": len(ok), "skipped_fewer_than_min_train_days": int(sum(s is None for s in scored))}


def cmd_jokar_grid(args) -> int:
    """Tuning step 2: per-household day-level AUC over the (C, gamma) grid on a
    seeded sample of 300 households (synthetic objective; no labels, Jan-Jun only).

    Grid amendment (27 Oct, before D was frozen): AUCs already computed on the
    first grid (``jokar_grid_auc_first.npz``) are reused; only the new points
    are fitted.  Every fit and synthetic set is seeded per household and fold,
    so a reused value equals a refit (checked on a few households)."""
    t0 = time.perf_counter()
    ids, days, X = N.ami_hourly(AMI_DIR)
    eligible = [r for r in range(len(ids)) if (~np.isnan(X[r]).any(axis=1)
                & N.window_mask(days, N.PHASES_JOKAR["validation"][0])).sum() >= N.MIN_TRAIN_DAYS]
    sample = sorted(np.random.default_rng(N.JOKAR_SEED).choice(eligible, N.N_TUNE_HOUSEHOLDS,
                                                                 replace=False).tolist())
    first_path = OUT / "jokar_grid_auc_first.npz"
    if not first_path.exists() and (OUT / "jokar_grid_auc.npz").exists():
        (OUT / "jokar_grid_auc.npz").rename(first_path)
    old = {}
    rec = {}
    if first_path.exists():
        f = np.load(first_path)
        if list(f["households"]) != [ids[r] for r in sample] or list(f["folds"]) != list(N.FOLDS):
            raise SystemExit("first-grid file: different sample or folds")
        for k, pt in enumerate(map(tuple, f["grid"])):
            old[pt] = f["auc"][:, :, k]
        # reuse check: refit two first-grid points on three households
        chk_pts = [GRID_FIRST[0], GRID_FIRST[-1]]
        chk = np.stack(_parallel(_grid_job, sample[:3], X, days, chk_pts, n_jobs=3, chunk=1))
        ref = np.stack([old[pt][:3] for pt in chk_pts], axis=2)
        rec["reuse_check_max_abs_diff"] = float(np.abs(chk - ref).max())
        if rec["reuse_check_max_abs_diff"] > 1e-12:
            raise SystemExit(f"reused AUCs do not reproduce: {rec}")
    new_pts = [pt for pt in GRID if pt not in old]
    new = np.stack(_parallel(_grid_job, sample, X, days, new_pts)) if new_pts else None
    auc = np.empty((len(sample), len(N.FOLDS), len(GRID)))
    for k, pt in enumerate(GRID):
        auc[:, :, k] = old[pt] if pt in old else new[:, :, new_pts.index(pt)]
    elapsed = time.perf_counter() - t0 + (float(np.load(first_path)["elapsed_s"]) if first_path.exists() else 0.0)
    np.savez_compressed(OUT / "jokar_grid_auc.npz", auc=auc, households=np.array([ids[r] for r in sample]),
                        folds=np.array(list(N.FOLDS)), grid=np.array(GRID), elapsed_s=elapsed,
                        reused_points=len(GRID) - len(new_pts), new_points=len(new_pts))
    print(f"grid AUC {auc.shape}: {len(GRID) - len(new_pts)} points reused, {len(new_pts)} new, "
          f"{time.perf_counter() - t0:.0f} s this run; {rec}")
    return 0


def cmd_jokar_tune(args) -> int:
    import yaml
    t0 = time.perf_counter()
    if PARAMS.exists():
        raise SystemExit(f"{PARAMS.name} exists: parameters are frozen (tune once)")
    gf = np.load(OUT / "jokar_grid_auc.npz")
    if list(gf["folds"]) != list(N.FOLDS) or [tuple(x) for x in gf["grid"]] != GRID:
        raise SystemExit("jokar_grid_auc.npz does not match the folds or the grid")
    auc, sample_ids, t_grid = gf["auc"], list(gf["households"]), float(gf["elapsed_s"])
    ids, days, X = N.ami_hourly(AMI_DIR)
    rows = list(range(len(ids)))
    params, grid_table, picks = {}, {}, {}

    def edge(c, g, cs, gs):
        return [s for s, on in (("C_low", c == min(cs)), ("C_high", c == max(cs)),
                                ("gamma_low", g == min(gs)), ("gamma_high", g == max(gs))) if on]
    first_idx = [GRID.index(pt) for pt in GRID_FIRST]
    for j, f in enumerate(N.FOLDS):
        c, g, m = N.select_svm_params(auc[:, j, :], GRID)
        c1, g1, m1 = N.select_svm_params(auc[:, j, first_idx], GRID_FIRST)
        params[f] = {"C": c, "gamma": g, "mean_auc": m}
        picks[f] = {"extended_grid": {"C": c, "gamma": g, "mean_auc": m,
                                      "on_edge": edge(c, g, N.GRID_C, N.GRID_GAMMA)},
                    "first_grid": {"C": c1, "gamma": g1, "mean_auc": m1,
                                   "on_edge": edge(c1, g1, N.GRID_C_FIRST, N.GRID_GAMMA_FIRST)}}
        grid_table[f] = {f"C={cc:g},gamma={gg:g}": float(v) for (cc, gg), v in zip(GRID, np.nanmean(auc[:, j, :], axis=0))}
    # 2. validation scores (train Jan-Apr, score May-Jun) for every household
    scored = _parallel(_score_job, rows, X, days, "validation", params)
    labels = V.load_household_labels()
    idx = N.read_consumer_index(OUT / "consumer_index.parquet")
    sc_w = N.PHASES_JOKAR["validation"][1]
    gate = _gate(ids, pd.date_range(*sc_w), idx)
    thresholds = {}
    for f in N.FOLDS:
        y = V.fold_targets(labels, ids, N.FOLDS[f], sc_w)
        thresholds[f] = {"ungated": V.best_threshold(_household_scores(scored, ids, f), y),
                         "gated": V.best_threshold(_household_scores(scored, ids, f, gate), y)}
        params[f]["q"] = thresholds[f]["ungated"]["q"]
        params[f]["q_gated"] = thresholds[f]["gated"]["q"]
    frozen = {
        "tuned_on": {"window": "2024-01-01..2024-06-30 (inner split: train Jan-Apr, validate May-Jun)",
                     "svm_params": f"{N.N_TUNE_HOUSEHOLDS} households, rng {N.JOKAR_SEED}; mean per-household "
                                   "day-level ROC AUC, May-Jun days vs synthetic attacked versions (fold's attacks)",
                     "threshold": "household-level F1 on May-Jun; positives: thieves of the fold's training "
                                  "attacks active on >= 1 May-Jun day; negatives: honest households"},
        "folds": params,
        "day_score": "SVC decision_function (ovr), malicious class column [our choice, user decision 27 Oct]",
        "household_score": "share of online scored days flagged (predicted class = malicious)",
        "flag_rule": "household flagged if score >= q",
        "kmeans": {"k_range": list(N.K_RANGE), "rule": "argmax mean silhouette; ties: smaller k",
                   "min_cluster_share": N.MIN_CLUSTER_SHARE, "random_state": N.JOKAR_SEED},
        "min_train_days": N.MIN_TRAIN_DAYS, "seed": N.JOKAR_SEED,
        "grid": {"C": list(N.GRID_C), "gamma": list(N.GRID_GAMMA),
                 "amended": "27 Oct before freezing (user decision, tuning data only); first grid C "
                            f"{list(N.GRID_C_FIRST)}, gamma {list(N.GRID_GAMMA_FIRST)}; first-grid AUCs reused"},
        "primary": "ungated (q); gated uses q_gated",
    }
    PARAMS.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    rec = {"params": params, "picks_first_vs_extended_grid": picks, "grid_mean_auc": grid_table,
           "thresholds": thresholds,
           "tuning_sample_households": sample_ids,
           "kmeans_validation_phase": _kmeans_table(scored),
           "gate_open_share_may_jun_household_days": float(gate.mean()),
           "gate_open_may_jun_dt_days": _gate_rate(pd.date_range(*sc_w)),
           "elapsed_s": {"grid": round(t_grid, 1), "tune": round(time.perf_counter() - t0, 1)}}
    (OUT / "jokar_tuning.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({"picks": picks, "params": params,
                      **{k: rec[k] for k in ("kmeans_validation_phase", "gate_open_may_jun_dt_days", "elapsed_s")}},
                     indent=2))
    return 0


def cmd_jokar_test(args) -> int:
    import yaml
    t0 = time.perf_counter()
    if not PARAMS.exists():
        raise SystemExit(f"{PARAMS.name} missing: run jokar-tune and commit it first (step D)")
    if (OUT / "jokar_test_opened.json").exists():
        raise SystemExit("the July-December window was already scored (run once)")
    frozen = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    params = frozen["folds"]
    (OUT / "jokar_test_opened.json").write_text(json.dumps({
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "params_sha256": __import__("hashlib").sha256(
            PARAMS.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    ids, days, X = N.ami_hourly(AMI_DIR)
    rows = list(range(len(ids)))
    scored = _parallel(_score_job, rows, X, days, "test", params)
    t_fit = time.perf_counter() - t0
    labels = V.load_household_labels()
    idx = N.read_consumer_index(OUT / "consumer_index.parquet")
    sc_w = N.PHASES_JOKAR["test"][1]
    test_days = pd.date_range(*sc_w)
    gate = _gate(ids, test_days, idx)
    gate_rate = _gate_rate(test_days)
    res = {"label": "Jokar SVM, leave-one-attack-out; July-December scored once with the frozen settings",
           "params_file": str(PARAMS.relative_to(ROOT)), "gate_open_dt_days": gate_rate,
           "kmeans_test_phase": _kmeans_table(scored)}
    for variant, g, qkey in (("ungated", None, "q"), ("gated", gate, "q_gated")):
        per_fold, scores, qs = {}, {}, {}
        for f in N.FOLDS:
            s = _household_scores(scored, ids, f, g)
            y = V.fold_targets(labels, ids, N.FOLDS[f] if f == "circular" else (f,), sc_w)
            per_fold[f] = V.fold_metrics(s, y, params[f][qkey])
            scores[f], qs[f] = s, params[f][qkey]
        res[variant] = {
            "per_fold": per_fold,
            "macro_energy_theft_h1_h4": V.macro(per_fold, V.ENERGY_THEFT),
            "macro_tariff_shift_h5_h6": V.macro(per_fold, V.TARIFF_SHIFT),
            "macro_all_six": V.macro(per_fold, tuple(a for a in N.FOLDS if a != "circular")),
            "circular_reference": {"label": "circular: graded on its own generator; upper bound only",
                                   **{k: per_fold["circular"][k] for k in ("precision", "recall", "f1")},
                                   "average_precision": per_fold["circular"]["pr_curve"]["average_precision"]},
            "recall_by_onset_month": V.by_onset_month(scores, qs, labels, {
                "energy_theft_h1_h4": V.ENERGY_THEFT, "tariff_shift_h5_h6": V.TARIFF_SHIFT,
                "all_six": tuple(a for a in N.FOLDS if a != "circular")}),
        }
    res["primary"] = "ungated" if (gate_rate["p"] or 0) > 0.95 else "gated"
    res["elapsed_s"] = {"fit_and_score": round(t_fit, 1), "total": round(time.perf_counter() - t0, 1)}
    np.savez_compressed(OUT / "jokar_test_flags.npz", ids=np.array(ids), **{
        f: np.stack([s["folds"][f][1] if s is not None else np.full(len(test_days), -2, np.int8) for s in scored])
        for f in N.FOLDS})
    _write_summary("jokar", res)
    print(json.dumps({v: {k: res[v][k] for k in ("macro_energy_theft_h1_h4", "macro_tariff_shift_h5_h6",
                                                 "macro_all_six", "circular_reference")}
                      for v in ("ungated", "gated")} | {"gate": gate_rate, "primary": res["primary"]}, indent=2))
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
    for name, fn, h in (("jokar-grid", cmd_jokar_grid, "step D: (C, gamma) grid AUC"),
                        ("jokar-tune", cmd_jokar_tune, "step D: tune and freeze config/layer3_ntl_params.yaml"),
                        ("jokar-test", cmd_jokar_test, "step E: score July-December once")):
        sub.add_parser(name, help=h).set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
