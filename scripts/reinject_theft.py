"""Second theft injection (Layer 3 Stage 4 run sheet §2, step A; D1).

Rebuilds the household population as ``run_year.py`` does, draws the theft
again on ``config.rng("theft.pick/<tag>")`` / ``config.rng("theft.attack/<tag>")``
and writes only the theft-dependent files
(:mod:`src.data_source.theft_reinjection`).

- ``--tag inj2``: injection 2; writes ``--out`` (default
  ``results/year_2024_inj2/``) and the ideal-meter household inputs
  (``results/layer3/ntl_inj2/ideal_meter/``); records the theft-independence
  checks in ``results/layer3/ntl_inj2/inputs.json``.
- ``--tag ""``: the v1 stream names; the files must equal the v1 files
  exactly (reproduction check, ``results/layer3/ntl_inj2/reproduction_check.json``);
  exits non-zero if they do not.

PowerShell::

    python scripts/reinject_theft.py --tag '""' --out results/year_2024_repro/
    python scripts/reinject_theft.py --tag inj2 --out results/year_2024_inj2/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_source import simulation as S  # noqa: E402
from src.data_source import theft_reinjection as T  # noqa: E402

NTL_INJ2 = ROOT / "results" / "layer3" / "ntl_inj2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help='stream tag; "" (or \'""\') = the v1 streams')
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the reproduction files after the check")
    args = ap.parse_args(argv)
    tag = "" if args.tag in ("", '""', "''") else args.tag
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.simplefilter("ignore", FutureWarning)
    t0 = time.perf_counter()
    default = ROOT / "results" / ("year_2024_repro" if not tag else f"year_2024_{tag}")
    out = (Path(args.out) if args.out else default).resolve()
    ideal_out = out / "ideal_meter" if not tag else NTL_INJ2 / "ideal_meter"
    NTL_INJ2.mkdir(parents=True, exist_ok=True)
    pop = S.build_population()
    t_pop = time.perf_counter() - t0
    theft = T.redraw(pop, tag)
    info = T.write(pop, theft, out, ideal_out, tag)
    if not tag:
        rec = T.reproduction_check(out, ideal_out)
        rec.update({"tag": "", "streams": list(T.stream_names("")), "out": str(out.relative_to(ROOT)),
                    "elapsed_s": round(time.perf_counter() - t0, 1)})
        (NTL_INJ2 / "reproduction_check.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in rec.items() if k != "files"}, indent=2))
        if not rec["all_equal"]:
            print(json.dumps(rec["files"], indent=2), file=sys.stderr)
            print("STOP: the v1 stream names do not reproduce the v1 theft-dependent files", file=sys.stderr)
            return 1
        if not args.keep:
            shutil.rmtree(out)
        return 0
    checks = T.theft_independence_checks(pop, theft)
    rec = {"tag": tag, "out": str(out.relative_to(ROOT)),
           "ideal_meter_households": str((ideal_out / "household_daily.parquet").relative_to(ROOT)),
           "theft_independence_checks": checks,
           "time_population_s": round(t_pop, 1), "write_s": info["write_s"],
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    (NTL_INJ2 / "inputs.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not checks["passed"]:
        print("STOP: a reused v1 input is not shown to be theft-independent", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
