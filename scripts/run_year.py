"""Run the simulated year once, cache it and check the CP3 year acceptance.

Writes ``results/year_2024/`` (git-ignored, regenerable from the master seed):
meter frames (true and metered), state frame with DT loading, households
(ground truth and daily table), the monthly AMI export, the SGCC detector
pools and ``run_info.json`` / ``acceptance.json``.  Exits non-zero if an
asserted check fails (brief v2 X.3 item 4 as revised by X.3.1 / X.3.2):
(a) IS 17036 +/- 10 % on every 11 kV bus-phase at every step;
(b) every DT phase <= 1.5 pu at every step (IS 2026-7 Table 4, normal cyclic
loading of distribution transformers);
(c) total DT loss = IS 1180 P0 v^2 + Pk (K0^2 + K1^2 + K2^2) within 5 % at
each DT's peak step.  The lambda sweep (scripts/lambda_sweep.py) is separate
and reported only.

PowerShell::

    python scripts/run_year.py                                # full 2024 (~1.4 h)
    python scripts/run_year.py --start 2024-06-24 --end "2024-06-25 23:45" --out results/tmp
    python scripts/run_year.py --check-only                   # re-check an existing cache
    python scripts/run_year.py --precheck-only                # population + design-peak DT check, no power flow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data_source import config, simulation  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=f"{config.SIMULATION_YEAR}-01-01")
    ap.add_argument("--end", default=f"{config.SIMULATION_YEAR}-12-31 23:45")
    ap.add_argument("--out", default=str(config.YEAR_CACHE_DIR))
    ap.add_argument("--sgcc-path", default=None)
    ap.add_argument("--no-ami", action="store_true", help="skip the AMI export")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--precheck-only", action="store_true",
                    help="build the population, print s_design, N and the design-peak DT loading")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.simplefilter("ignore", FutureWarning)     # pandapower fillna downcasting
    out = Path(args.out)
    if args.precheck_only:
        pop = simulation.build_population(args.start, args.end, args.sgcc_path)
        print(json.dumps({k: pop.info[k] for k in (
            "h_bar_kw", "design_peak", "counts", "households", "thieves_per_attack",
            "feeder_mostly_zero", "mostly_zero_per_dt", "emarc", "emarc_households_used_by_feeder")}, indent=2, default=simulation._json))
        print(simulation.design_precheck(pop).to_string(index=False))
        return 0
    if not args.check_only:
        info = simulation.run_year(out, args.start, args.end, args.sgcc_path,
                                   write_ami=not args.no_ami)
        logging.info("run info: %s", {k: info[k] for k in (
            "steps", "time_population_s", "time_power_flow_s", "s_per_step",
            "time_total_s", "peak_memory_mb")})
    state = pd.read_parquet(out / "state.parquet")
    dts = pd.read_parquet(out / "dts.parquet")
    acc = simulation.year_acceptance(state, dts)
    acc["nominal_dt_table"] = simulation.nominal_dt_table().to_dict("records")
    info_path = out / "run_info.json"
    run_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else None
    acc["shape_validation"] = simulation.shape_validation(state, run_info)
    acc["vuf_design_peak"] = {"before_x3_ieee_phasing": simulation.nominal_vuf("ieee"),
                              "after_x3_phasing": simulation.nominal_vuf("x3")}
    (out / "acceptance.json").write_text(json.dumps(acc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in acc.items() if k not in ("dt_loading", "nominal_dt_table", "dt_losses")},
                     indent=2))
    print(pd.DataFrame(acc["dt_loading"]).to_string(index=False))
    print(pd.DataFrame(acc["dt_losses"]["rows"]).to_string(index=False))
    print(pd.DataFrame(acc["nominal_dt_table"]).to_string(index=False))
    if not acc["passed"]:
        print("YEAR ACCEPTANCE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
