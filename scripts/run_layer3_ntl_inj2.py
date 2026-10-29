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


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("injection", help="step A: injection report").set_defaults(func=cmd_injection)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
