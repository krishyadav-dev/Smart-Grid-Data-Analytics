"""Validate Stage A (IEEE 13-node as published) against Kersting's results.

Prints totals, phase-average and per-phase voltage tables, and writes
results/stage_a_validation.csv (per-phase table) and
results/stage_a_validation_totals.csv.  The published numbers come from
src/data_source/ieee13_data.py, whose source is
docs/References/_text/IEEE13/IEEE 13 Node Test Feeder.txt.

    python scripts/validate_stage_a.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data_source.stage_a_validation import validate_stage_a  # noqa: E402


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.WARNING)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    r = validate_stage_a()
    print(f"Load-model iteration: {r['n_power_flows']} power flows\n")
    print("Totals (model - published):")
    print(r["totals"].round(3).to_string(index=False))
    print("\nPhase-average voltage magnitude per bus:")
    print(r["bus_avg"].round(4).to_string(index=False))
    print("\nPer-phase voltages (not asserted):")
    print(r["per_phase"].round(4).to_string(index=False))
    print("\nCapacitor kvar per phase (675 + 611):")
    print(r["capacitors"].round(2).to_string(index=False))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    r["per_phase"].to_csv(out / "stage_a_validation.csv", index=False)
    r["totals"].to_csv(out / "stage_a_validation_totals.csv", index=False)
    print(f"\nwrote {out / 'stage_a_validation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
