"""LV-loss sweep over the cached year: un-netted DT-level baseline.

For lambda in {0, 1, 2, 3, 5} %: per DT-day dE = E_DT - sum(E_consumer)
against the true theft energy (bias, MAE, correlation); PR / AUC-PR only
where the share of DT-days with active theft is below 90 %.  This is the
un-netted DT-level baseline of the Evaluation Plan, not the Layer 3 engine.
Reported, not asserted (brief v2 X.3.1 item 6): always exits 0.  Every
metric for three label sets: all attacks / without h6 / without h5 + h6.
Writes ``results/year_2024/lambda_sweep/``.

PowerShell::

    python scripts/lambda_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_source import config, simulation  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(config.YEAR_CACHE_DIR))
    args = ap.parse_args(argv)
    res = simulation.lambda_sweep(Path(args.cache))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
