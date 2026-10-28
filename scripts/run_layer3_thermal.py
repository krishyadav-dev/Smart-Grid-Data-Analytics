"""Layer 3 thermal engine: entry point (Stage 3 run sheet).

Commands (PowerShell)::

    python scripts/run_layer3_thermal.py annex-e        # step A: acceptance test + step-size table
    python scripts/run_layer3_thermal.py reference      # step B: reference trajectories, counts, correlation

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

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import thermal as T  # noqa: E402
from src.data_source.config import YEAR_CACHE_DIR  # noqa: E402

OUT = ROOT / "results" / "layer3" / "thermal"
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annex-e").set_defaults(fn=cmd_annex_e)
    sub.add_parser("reference").set_defaults(fn=cmd_reference)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
