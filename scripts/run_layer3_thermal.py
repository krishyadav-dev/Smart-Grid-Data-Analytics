"""Layer 3 thermal engine: entry point (Stage 3 run sheet).

Commands (PowerShell)::

    python scripts/run_layer3_thermal.py annex-e        # step A: acceptance test + step-size table

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

from src.analytics import thermal as T  # noqa: E402

OUT = ROOT / "results" / "layer3" / "thermal"
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annex-e").set_defaults(fn=cmd_annex_e)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
