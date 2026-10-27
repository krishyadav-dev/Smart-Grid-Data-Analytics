"""Theft-engine scenario inputs (Layer 3 Stage 2 run sheet §2, step A).

Writes ``results/layer3/ntl/``: ``lambda_true_{0,2,4}/``, ``ideal_meter/``,
``consumer_index.parquet`` and the check record ``inputs.json``
(:mod:`src.data_source.ntl_inputs`).

PowerShell::

    python scripts/build_ntl_inputs.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_source import ntl_inputs  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    rec = ntl_inputs.build()
    rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (ntl_inputs.OUT_DIR / "inputs.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
