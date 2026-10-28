"""Layer 3 thermal engine: entry point (Stage 3 run sheet).

Commands (PowerShell)::

    python scripts/run_layer3_thermal.py annex-e        # step A: acceptance test + step-size table
    python scripts/run_layer3_thermal.py reference      # step B: reference trajectories, counts, correlation
    python scripts/run_layer3_thermal.py select         # step C: LSTM size on April origins -> freeze
    python scripts/run_layer3_thermal.py test           # step D: walk-forward test run, once

Outputs under ``results/layer3/thermal/`` (``summary.json``).
Labels are read through :mod:`src.analytics.evaluate` only.
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
from src.analytics import forecast as F  # noqa: E402
from src.analytics import thermal as T  # noqa: E402
from src.data_source.config import YEAR_CACHE_DIR  # noqa: E402

OUT = ROOT / "results" / "layer3" / "thermal"
PARAMS = ROOT / "config" / "layer3_thermal_params.yaml"
FRAME = YEAR_CACHE_DIR / "meter_frame_metered.parquet"
DTS = YEAR_CACHE_DIR / "dts.parquet"
SUMMARY = OUT / "summary.json"
ANNEX_E_DT_TABLE = (1.0, 0.5, 0.25, 0.1, 0.01)     # the step-size table of the 28 Oct stop


def _write_summary(key: str, value) -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    s[key] = value
    SUMMARY.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def _annex_e_exact() -> dict:
    """The same equations solved exactly over each constant-load interval (the Dt -> 0 limit)."""
    p, ta = T.annex_e_params(), T.ANNEX_E_THETA_A
    s = T.steady_state(p, T.ANNEX_E_K[0], T.ANNEX_E_K[0], ta)
    taus = (p.k11 * p.tau_o, p.k22 * p.tau_w, p.tau_o / p.k22)
    out, x = {}, (s.theta_o, s.dh1, s.dh2)
    for k, dur, t_end in ((T.ANNEX_E_K[1], T.ANNEX_E_STEP_MIN, 30),
                          (T.ANNEX_E_K[2], T.ANNEX_E_END_MIN - T.ANNEX_E_STEP_MIN, 1440)):
        oil, h1, h2 = T.ultimate(p, k, k)
        x = tuple(u + (x0 - u) * np.exp(-dur / tau) for x0, u, tau in zip(x, (ta + oil, h1, h2), taus))
        out[t_end] = round(float(x[0] + x[1] - x[2]), 3)
    return out


def cmd_annex_e(args) -> int:
    t0 = time.perf_counter()
    res = T.annex_e_example()
    table = {str(dt): {t: round(v, 3) for t, v in T.annex_e_example(dt=dt)["theta_h_c"].items()}
             for dt in ANNEX_E_DT_TABLE}
    table["exact_solution"] = _annex_e_exact()
    out = {"dt_min": T.DT_MIN, "substeps_per_15min": T.SUBSTEPS, "expected_theta_h_c": T.ANNEX_E_EXPECTED,
           "result": res, "passes": T.annex_e_passes(res),
           "theta_h_by_dt": table,
           "note": "the run sheet's Dt = 1 min fails at t = 30 (114.391 -> 114.4); user decision 28 Oct: "
                   "Dt = 0.1 min with the same Cl. 8.2.3 difference equations (150 sub-steps per 15-min step)",
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    _write_summary("acceptance", out)
    print(json.dumps(out, indent=1))
    if not out["passes"]:
        print("STOP: Annex E acceptance test failed")
        return 1
    return 0


# --------------------------------------------------------------------------
# Step B: reference trajectories
# --------------------------------------------------------------------------

def load_inputs() -> dict:
    """Metered K_p [T, DT, phase], ambient [T], timestamps, days, DT table."""
    idx, cur = T.read_dt_currents(FRAME)
    ta = T.read_ambient()
    if not ta.index.equals(idx):
        raise SystemExit("ERA5 timestamps differ from the meter frame")
    days = pd.DatetimeIndex(idx[::T.STEPS_PER_DAY])
    return {"idx": idx, "k": T.load_factor(cur, T.rated_current()), "ta": ta.to_numpy(float),
            "days": days, "dts": T.read_dts(DTS), "names": T.dt_names()}


def reference_path(ps: str) -> Path:
    return OUT / f"reference_{ps}.npz"


def save_reference(ps: str, r: dict, days: pd.DatetimeIndex) -> None:
    arrays = {k: v for k, v in r.items() if k != "cross_min"}
    arrays.update({f"cross_min_{int(t)}": v for t, v in r["cross_min"].items()})
    np.savez_compressed(reference_path(ps), days=days.values.astype("datetime64[D]").astype(str), **arrays)


def load_reference(ps: str) -> dict:
    z = np.load(reference_path(ps))
    r = {k: z[k] for k in ("theta_h_end", "theta_h_max", "lol_h", "theta_o", "dh1", "dh2")}
    r["cross_min"] = {t: z[f"cross_min_{int(t)}"] for t in T.THRESHOLDS_C}
    return r


def _window_masks(days: pd.DatetimeIndex) -> dict:
    return {"2024": np.asarray(days != V.THERMAL_WARMUP),
            "test_window": np.asarray((days >= V.THERMAL_TEST[0]) & (days <= V.THERMAL_TEST[1]))}


def reference_report(r: dict, inp: dict) -> dict:
    idx, days, names, ta = inp["idx"], inp["days"], inp["names"], inp["ta"]
    ds = T.day_summary(r)
    win = _window_masks(days)
    step_keep = np.repeat(win["2024"], T.STEPS_PER_DAY)
    th = np.where(step_keep[:, None], r["theta_h_max"], -np.inf)
    rise = np.where(step_keep[:, None], r["theta_h_max"] - ta[:, None], -np.inf)
    per_dt = {}
    for n, name in enumerate(names):
        jp, jr = int(np.argmax(th[:, n])), int(np.argmax(rise[:, n]))
        events = {w: {str(int(t)): int((~np.isnan(ds["first_cross_min"][t][m, n])).sum())
                      for t in T.THRESHOLDS_C} for w, m in win.items()}
        lol = {w: {"mean_h_per_day": float(ds["lol_h"][m, n].mean()),
                   "max_h_per_day": float(ds["lol_h"][m, n].max()),
                   "max_day": str(days[m][int(np.argmax(ds["lol_h"][m, n]))].date())} for w, m in win.items()}
        per_dt[name] = {"peak_theta_h_c": float(th[jp, n]), "peak_at_step": str(idx[jp]),
                        "peak_rise_k": float(rise[jr, n]), "peak_rise_at_step": str(idx[jr]),
                        "ambient_at_peak_rise_c": float(ta[jr]),
                        "event_dt_days": events, "loss_of_life": lol}
    totals = {w: {str(int(t)): int(sum(v["event_dt_days"][w][str(int(t))] for v in per_dt.values()))
                  for t in T.THRESHOLDS_C} for w in win}
    lol_all = {w: {"mean_h_per_day": float(ds["lol_h"][m].mean()), "max_h_per_day": float(ds["lol_h"][m].max())}
               for w, m in win.items()}
    jmax = max(per_dt, key=lambda d: per_dt[d]["peak_theta_h_c"])
    jrise = max(per_dt, key=lambda d: per_dt[d]["peak_rise_k"])
    return {"per_dt": per_dt, "event_dt_days_total": totals, "loss_of_life_all_dts": lol_all,
            "peak_theta_h": {"dt": jmax, "value_c": per_dt[jmax]["peak_theta_h_c"],
                             "at_step": per_dt[jmax]["peak_at_step"]},
            "peak_rise": {"dt": jrise, "value_k": per_dt[jrise]["peak_rise_k"],
                          "at_step": per_dt[jrise]["peak_rise_at_step"]}}


def load_temperature_correlation(inp: dict) -> dict:
    """T5: Pearson r of daily mean K_oil and daily mean ambient, per DT and pooled, 2024 (warm-up excluded)."""
    keep = _window_masks(inp["days"])["2024"]
    ko = T.per_day(T.k_oil(inp["k"])).mean(axis=1)[keep]          # [day, DT]
    ta = T.per_day(inp["ta"]).mean(axis=1)[keep]                   # [day]
    per = {n: float(np.corrcoef(ko[:, i], ta)[0, 1]) for i, n in enumerate(inp["names"])}
    pooled = float(np.corrcoef(ko.ravel(), np.repeat(ta, ko.shape[1]))[0, 1])
    return {"per_dt": per, "pooled": pooled, "days": int(keep.sum()),
            "definition": "Pearson r, daily mean K_oil vs daily mean ERA5 ambient, 2024-01-02..2024-12-31"}


def cmd_reference(args) -> int:
    t0 = time.perf_counter()
    inp = load_inputs()
    out = {"definition": {"reference_load": "metered DT currents [our choice; plan 'actual load']",
                          "initial_state": "steady state at the 2024-01-01 00:00 load and ambient",
                          "warm_up": "2024-01-01 excluded from every metric [our choice; 3 tau_o ~ 9 h]",
                          "event": "a DT-day whose reference theta_h exceeds the threshold at any sub-step "
                                   "(Dt = 0.1 min) [our choice]",
                          "peak_at_step": "start of the 15-min step holding the peak sub-step",
                          "test_window": "2024-05-01..2024-12-31"}}
    for ps in T.PARAM_SETS:
        r = T.reference_run(T.dt_params(inp["dts"], ps), inp["k"], inp["ta"])
        bad = [k for k, v in r.items() if k != "cross_min" and np.isnan(v).any()]
        if bad:
            print(f"STOP: NaN in the {ps} reference trajectory ({bad})")
            return 1
        save_reference(ps, r, inp["days"])
        out[ps] = reference_report(r, inp)
        if ps == "main":
            out["true_current_check"] = V.thermal_true_current_check(r["theta_h_end"], r["theta_h_max"], inp["ta"])
        print(ps, json.dumps({"peak": out[ps]["peak_theta_h"], "rise": out[ps]["peak_rise"],
                              "events": out[ps]["event_dt_days_total"]}))
    out["load_temperature_correlation"] = load_temperature_correlation(inp)
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    _write_summary("reference", out)
    print(json.dumps({"true_current_check": {k: out["true_current_check"][k] for k in
                                             ("max_abs_diff_theta_h_k", "at", "true_peak_rise_k")},
                      "corr": out["load_temperature_correlation"]["pooled"], "elapsed_s": out["elapsed_s"]}))
    return 0


# --------------------------------------------------------------------------
# Step C: LSTM size on April origins, then freeze
# --------------------------------------------------------------------------

def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forecast_inputs(inp: dict) -> dict:
    kd = F.daily_series(inp["k"])
    return {"kd": kd, "n_dt": len(inp["names"]), "n_series": kd.shape[2],
            "feats": F.features(kd, inp["days"], len(inp["names"]))}


def origin_days(inp: dict, window: tuple) -> np.ndarray:
    return np.arange(F.day_index(inp["days"], window[0]), F.day_index(inp["days"], window[1]) + 1)


def thermal_from_forecast(fc: np.ndarray, origins: np.ndarray, inp: dict, ref: dict, ps: str) -> dict:
    """Run clipped forecasts [origin, 96, series] through parameter set ``ps``
    from the reference state at 00:00 of each origin, with the actual ambient."""
    s0 = T.state_at(ref, origins * T.STEPS_PER_DAY - 1)
    ta = inp["ta"].reshape(-1, T.STEPS_PER_DAY)[origins]
    return T.forecast_run(T.dt_params(inp["dts"], ps), F.to_dt_phase(fc, len(inp["names"])), ta, s0)


def hot_spot_rmse_per_dt(th: np.ndarray, ref_th: np.ndarray) -> np.ndarray:
    """RMSE over origins x 96 steps, per DT ([origin, 96, DT] inputs)."""
    return np.sqrt(np.mean((th - ref_th) ** 2, axis=(0, 1)))


def train_seeds(win, fi: dict, inp: dict, rnd: tuple, layers: int, units: int, origins: np.ndarray,
                tag: str) -> tuple[dict, dict]:
    """Train the three seeds for one round; clipped K_p forecasts per seed and logs."""
    trp = F.samples(rnd[0], inp["days"], fi["n_series"])
    vap = F.samples(rnd[1], inp["days"], fi["n_series"])
    fcs, logs = {}, {}
    for seed in F.SEEDS:
        model, log = F.train(win, trp, vap, layers, units, seed)
        raw = F.forecast_origins(model, win, origins, fi["n_series"])
        fcs[seed], log["clipped"] = F.clip(F.compose(raw, fi["kd"], origins, win.variant))
        logs[seed] = log
        print(f"  {tag} seed {seed}: best val {log['best_val_mse']:.6f} at epoch {log['best_epoch']} "
              f"({log['epochs']} epochs, {log['seconds']} s, {log['clipped']} clipped)", flush=True)
    return fcs, logs


SEED_KEYS = ("best_val_mse", "best_epoch", "epochs", "seconds", "clipped", "n_train", "n_val")


def _explained_variance(k_mse: float, mse_train_mean: float) -> float:
    """1 - MSE(forecast) / MSE(per-series training-mean forecast), April origins."""
    return 1.0 - k_mse / mse_train_mean


def _reused_direct(prev: dict, mse_train_mean: float) -> dict:
    """The six direct candidates of the first selection run (28 Oct), not re-run."""
    if "candidates" in prev:
        return {t: v for t, v in prev["candidates"].items() if v["variant"] == "direct"}
    out = {}
    for tag, v in prev.get("sizes", {}).items():
        row = {k: v[k] for k in ("layers", "units", "n_parameters", "mean_hot_spot_rmse_k",
                                 "per_dt_hot_spot_rmse_k", "k_mse_seed_mean")}
        row["variant"] = "direct"
        row["source"] = "first selection run, 28 Oct (before the amendment); not re-run"
        row["seeds"] = {}
        for seed, s in v["seeds"].items():
            ev = _explained_variance(s["k_mse"], mse_train_mean)
            row["seeds"][seed] = {"mean_hot_spot_rmse_k": s["mean_hot_spot_rmse_k"], "k_mse": s["k_mse"],
                                  "explained_variance": ev,
                                  "origins_worse_than_persistence": s["origins_worse_than_persistence"],
                                  "diverged": (not np.isfinite(s["best_val_mse"])) or ev <= 0,
                                  **{k: s[k] for k in SEED_KEYS}}
        row["explained_variance_seed_mean"] = _explained_variance(row["k_mse_seed_mean"], mse_train_mean)
        out[f"direct {tag}"] = row
    return out


def cmd_select(args) -> int:
    if PARAMS.exists():
        print(f"STOP: {PARAMS.name} exists; the LSTM size is frozen (tune once)")
        return 1
    t0 = time.perf_counter()
    inp = load_inputs()
    fi = forecast_inputs(inp)
    rnd = F.training_round(F.SELECTION_ORIGINS[0])
    assert rnd == (F.SELECTION_TRAIN, F.SELECTION_VAL), rnd
    origins = origin_days(inp, F.SELECTION_ORIGINS)
    ref = load_reference("main")
    ref_th = T.per_day(ref["theta_h_end"])[origins]                     # [origin, 96, DT]
    actual = fi["kd"][origins]
    train_days = np.unique(F.samples(rnd[0], inp["days"], 1)[:, 1])
    train_mean = fi["kd"][train_days].mean(axis=(0, 1))                 # per series
    mse_train_mean = float(((actual - train_mean) ** 2).mean())
    base = {"persistence": F.persistence(fi["kd"], origins), "seasonal_naive": F.seasonal_naive(fi["kd"], origins)}
    pers_mse = ((base["persistence"] - actual) ** 2).mean(axis=(1, 2))
    baselines = {}
    for name, fc in base.items():
        th = thermal_from_forecast(fc, origins, inp, ref, "main")["theta_h_end"]
        r = hot_spot_rmse_per_dt(th, ref_th)
        k_mse = float(((fc - actual) ** 2).mean())
        baselines[name] = {"mean_hot_spot_rmse_k": float(r.mean()), "k_mse": k_mse,
                           "explained_variance": _explained_variance(k_mse, mse_train_mean),
                           "per_dt_hot_spot_rmse_k": dict(zip(inp["names"], r.round(4).tolist()))}
    prev = json.loads(SUMMARY.read_text(encoding="utf-8")).get("lstm_selection", {}) if SUMMARY.exists() else {}
    cands = _reused_direct(prev, mse_train_mean)
    if len(cands) != len(F.GRID_LAYERS) * len(F.GRID_UNITS):
        raise SystemExit(f"expected the 6 direct candidates of the first run, found {len(cands)}")
    win = F.Windows(fi["feats"], fi["kd"], _device(), variant="residual")
    for layers in F.GRID_LAYERS:
        for units in F.GRID_UNITS:
            tag = f"residual {layers}x{units}"
            fcs, logs = train_seeds(win, fi, inp, rnd, layers, units, origins, tag)
            seeds = {}
            for seed, fc in fcs.items():
                mse_o = ((fc - actual) ** 2).mean(axis=(1, 2))
                ev = _explained_variance(float(mse_o.mean()), mse_train_mean)
                th = thermal_from_forecast(fc, origins, inp, ref, "main")["theta_h_end"]
                seeds[str(seed)] = {"mean_hot_spot_rmse_k": float(hot_spot_rmse_per_dt(th, ref_th).mean()),
                                    "k_mse": float(mse_o.mean()), "explained_variance": ev,
                                    "origins_worse_than_persistence": int((mse_o > pers_mse).sum()),
                                    "diverged": (not np.isfinite(logs[seed]["best_val_mse"])) or ev <= 0,
                                    **{k: logs[seed][k] for k in SEED_KEYS}}
            mean_fc = np.mean([fcs[s] for s in F.SEEDS], axis=0)
            th = thermal_from_forecast(mean_fc, origins, inp, ref, "main")["theta_h_end"]
            r = hot_spot_rmse_per_dt(th, ref_th)
            k_mse = float(((mean_fc - actual) ** 2).mean())
            cands[tag] = {"variant": "residual", "layers": layers, "units": units,
                          "n_parameters": F.n_parameters(F.build_model(fi["feats"].shape[2], layers, units)),
                          "mean_hot_spot_rmse_k": float(r.mean()),
                          "per_dt_hot_spot_rmse_k": dict(zip(inp["names"], r.round(4).tolist())),
                          "k_mse_seed_mean": k_mse,
                          "explained_variance_seed_mean": _explained_variance(k_mse, mse_train_mean),
                          "seeds": seeds}
            print(f"{tag}: mean hot-spot RMSE {r.mean():.4f} K", flush=True)
    diverged = [f"{t} seed {s}" for t, c in cands.items() for s, v in c["seeds"].items() if v["diverged"]]
    out = {"criterion": "lowest mean over the 13 DTs of the hot-spot RMSE over all April origins x 96 steps "
                        "(seed-mean forecast, main parameter set) across the 12 candidates "
                        "(6 direct + 6 residual); tie-break fewer parameters",
           "divergence_check": "a seed has diverged if its best validation loss is NaN or its April explained "
                               "variance of K_p is <= 0 (user decision 28 Oct; worse than persistence is a "
                               "finding, not a stop)",
           "explained_variance": "1 - MSE(forecast) / MSE(per-series mean K_p over the training target days), "
                                 "pooled over April origins x 96 steps x 39 series",
           "train_target_days": [str(F.SELECTION_TRAIN[0].date()), str(F.SELECTION_TRAIN[1].date())],
           "first_target_day": str(inp["days"][train_days[0]].date()),
           "early_stopping_days": [str(F.SELECTION_VAL[0].date()), str(F.SELECTION_VAL[1].date())],
           "origins": [str(F.SELECTION_ORIGINS[0].date()), str(F.SELECTION_ORIGINS[1].date())],
           "mse_training_mean_forecast": mse_train_mean,
           "device": str(win.device), "candidates": cands, "baselines_april": baselines, "diverged": diverged,
           "first_run_stop": "first run (direct only) stopped on the run sheet's check: every seed's K_p MSE "
                             "above persistence on all 30 April origins; check redefined by the user 28 Oct"}
    if diverged:
        out["elapsed_s"] = round(time.perf_counter() - t0, 1)
        _write_summary("lstm_selection", out)
        print(f"STOP: diverged seeds {diverged}")
        return 1
    best = min(cands, key=lambda t: (round(cands[t]["mean_hot_spot_rmse_k"], 12), cands[t]["n_parameters"]))
    out["chosen"] = best
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    _write_summary("lstm_selection", out)
    freeze(cands[best], out)
    print(f"chosen {best}; frozen in {PARAMS.relative_to(ROOT)}; {out['elapsed_s']} s")
    return 0


def freeze(size: dict, sel: dict) -> None:
    d = {"lstm": {"variant": size["variant"], "layers": size["layers"], "units": size["units"],
                  "dropout": F.DROPOUT,
                  "n_parameters": size["n_parameters"], "lookback_steps": F.LOOKBACK, "horizon_steps": F.HORIZON,
                  "april_mean_hot_spot_rmse_k": round(size["mean_hot_spot_rmse_k"], 6)},
         "training": {"loss": "MSE on the variant's target (direct: K_p; residual: K_p(D) - K_p(D-1), equal to the K_p error before clipping)", "optimizer": "adam", "lr": F.LR, "batch": F.BATCH,
                      "max_epochs": F.MAX_EPOCHS, "patience": F.PATIENCE, "val_days": F.VAL_DAYS,
                      "gap_days": F.GAP_DAYS, "data_start": str(F.DATA_START.date()),
                      "negative_forecasts": "clipped at 0"},
         "seeds": list(F.SEEDS),
         "grid": {"variants": list(F.VARIANTS), "layers": list(F.GRID_LAYERS), "units": list(F.GRID_UNITS)},
         "tuned_on": {"window": f"train {sel['train_target_days'][0]}..{sel['train_target_days'][1]} "
                                f"(first target {sel['first_target_day']}), early stopping "
                                f"{sel['early_stopping_days'][0]}..{sel['early_stopping_days'][1]}, "
                                f"origins {sel['origins'][0]}..{sel['origins'][1]}",
                      "criterion": sel["criterion"]},
         "walk_forward": {"retrain": "monthly, at the first origin of each month",
                          "test_origins": f"{F.TEST_ORIGINS[0].date()}..{F.TEST_ORIGINS[1].date()}"},
         "thermal": {"dt_min": T.DT_MIN, "substeps": T.SUBSTEPS, "param_sets": list(T.PARAM_SETS)}}
    PARAMS.write_text("# Stage 3 step C: frozen before any May-December forecast (run sheet §4)\n"
                      + yaml.safe_dump(d, sort_keys=False), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# Step D: walk-forward test run, once
# --------------------------------------------------------------------------

FORECASTS_TEST = OUT / "forecasts_test.npz"


def trajectories_path(ps: str) -> Path:
    return OUT / f"trajectories_test_{ps}.npz"


def frozen() -> dict:
    return yaml.safe_load(PARAMS.read_text(encoding="utf-8"))


def cmd_test(args) -> int:
    if FORECASTS_TEST.exists():
        print(f"STOP: {FORECASTS_TEST.name} exists; the test run is done once")
        return 1
    t0 = time.perf_counter()
    cfg = frozen()["lstm"]
    inp = load_inputs()
    fi = forecast_inputs(inp)
    win = F.Windows(fi["feats"], fi["kd"], _device(), variant=cfg["variant"])
    origins = origin_days(inp, F.TEST_ORIGINS)
    o_days = inp["days"][origins]
    lstm = {s: np.full((len(origins), F.HORIZON, fi["n_series"]), np.nan) for s in F.SEEDS}
    rounds = {}
    for month in sorted(set(o_days.month)):
        sel = np.flatnonzero(o_days.month == month)
        first = o_days[sel[0]]
        rnd = F.training_round(first)
        fcs, logs = train_seeds(win, fi, inp, rnd, cfg["layers"], cfg["units"], origins[sel],
                                f"{first:%Y-%m} {cfg['variant']} {cfg['layers']}x{cfg['units']}")
        for s in F.SEEDS:
            lstm[s][sel] = fcs[s]
        rounds[f"{first:%Y-%m}"] = {"first_origin": str(first.date()), "last_origin": str(o_days[sel[-1]].date()),
                                    "origins": int(len(sel)),
                                    "train_target_days": [str(rnd[0][0].date()), str(rnd[0][1].date())],
                                    "early_stopping_days": [str(rnd[1][0].date()), str(rnd[1][1].date())],
                                    "seeds": {str(s): {k: logs[s][k] for k in SEED_KEYS} for s in F.SEEDS}}
    fc = {"persistence": F.persistence(fi["kd"], origins), "seasonal_naive": F.seasonal_naive(fi["kd"], origins),
          "lstm": np.mean([lstm[s] for s in F.SEEDS], axis=0),
          **{f"lstm_seed_{s}": lstm[s] for s in F.SEEDS}}
    bad = [m for m, v in fc.items() if np.isnan(v).any()]
    if bad:
        print(f"STOP: NaN forecasts {bad}")
        return 1
    np.savez_compressed(FORECASTS_TEST, origins=origins, **fc)
    for ps in T.PARAM_SETS:
        ref = load_reference(ps)
        arrays = {}
        for m, v in fc.items():
            r = thermal_from_forecast(v, origins, inp, ref, ps)
            for k in ("theta_h_end", "theta_h_max", "lol_h"):
                arrays[f"{m}__{k}"] = r[k]
            for thr, c in r["cross_min"].items():
                arrays[f"{m}__cross_min_{int(thr)}"] = c
        bad = [k for k, v in arrays.items() if "cross_min" not in k and np.isnan(v).any()]
        if bad:
            print(f"STOP: NaN forecast trajectories {bad[:5]}")
            return 1
        np.savez_compressed(trajectories_path(ps), origins=origins, **arrays)
    out = {"frozen": cfg, "origins": [str(o_days[0].date()), str(o_days[-1].date())], "n_origins": int(len(origins)),
           "rounds": rounds, "device": str(win.device),
           "clipped_lstm_values": {str(s): int(sum(r["seeds"][str(s)]["clipped"] for r in rounds.values()))
                                   for s in F.SEEDS},
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    _write_summary("test_run", out)
    print(f"test run done: {len(origins)} origins, {out['elapsed_s']} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annex-e").set_defaults(fn=cmd_annex_e)
    sub.add_parser("reference").set_defaults(fn=cmd_reference)
    sub.add_parser("select").set_defaults(fn=cmd_select)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
