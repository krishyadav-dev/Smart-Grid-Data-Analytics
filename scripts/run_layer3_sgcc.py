"""Layer 3 SGCC check: Zheng et al. 2018 Wide & Deep CNN (Stage 2 run sheet §5, step F).

Trains on the SGCC training pool (10 % stratified validation for early
stopping and the operating threshold), three seeds, then scores the test pool
once.  Writes ``results/layer3/sgcc/summary.json``.  Test-pool labels are read
through :mod:`src.analytics.evaluate` only.

PowerShell::

    python scripts/run_layer3_sgcc.py
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import zheng_cnn as Z  # noqa: E402
from src.data_source.config import YEAR_CACHE_DIR  # noqa: E402

OUT = ROOT / "results" / "layer3" / "sgcc"
SEEDS = (Z.SEED, Z.SEED + 1, Z.SEED + 2)        # run sheet §5


def main() -> int:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("sgcc").info
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "test_opened.json").exists():
        raise SystemExit("the SGCC test pool was already scored (used once)")
    _, x, flag = Z.load_pool(YEAR_CACHE_DIR / "sgcc_detector_train.parquet")
    tr, va = Z.stratified_split(flag)
    test_ids, x_test, _ = Z.load_pool(YEAR_CACHE_DIR / "sgcc_detector_test.parquet")   # FLAG unused here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = {}
    for seed in SEEDS:
        model, info = Z.train(x[tr], flag[tr], x[va], flag[va], seed, device, log)
        sv = Z.predict(model, x[va], device)
        thr = V.threshold_max_f1(sv, flag[va])
        runs[seed] = {"train": info, "threshold_from_validation": thr,
                      "test_scores": Z.predict(model, x_test, device)}
    (OUT / "test_opened.json").write_text(json.dumps({"opened_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                          encoding="utf-8")
    flags_test = V.sgcc_test_flags()
    res = {"label": "Zheng et al. 2018 Wide & Deep CNN on the real SGCC labels (non-circular check); "
                    "test pool scored once",
           "pools": {"train": int(len(tr)), "validation": int(len(va)), "validation_thieves": int(flag[va].sum()),
                     "train_thieves": int(flag[tr].sum()), "test": int(len(x_test))},
           "preprocessing": "pools as written: Zheng Eq. (1)-(3) already applied (user decision 27 Oct)",
           "zheng_table_iii_80pct": V.ZHENG_TABLE_III_80, "device": device, "seeds": {}}
    for seed, r in runs.items():
        s = pd.Series(r["test_scores"], index=test_ids)
        m = V.sgcc_metrics(s, flags_test, r["threshold_from_validation"])
        res["seeds"][str(seed)] = {"train": {k: v for k, v in r["train"].items()},
                                   "test": m}
    keys = ("auc", "map_at_100", "map_at_200", "f1")
    vals = {k: [res["seeds"][str(s)]["test"][k] for s in SEEDS] for k in keys}
    vals["precision"] = [res["seeds"][str(s)]["test"]["precision"]["p"] for s in SEEDS]
    vals["recall"] = [res["seeds"][str(s)]["test"]["recall"]["p"] for s in SEEDS]
    res["mean_and_range"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                             for k, v in vals.items()}
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"mean_and_range": res["mean_and_range"], "table_iii": V.ZHENG_TABLE_III_80,
                      "elapsed_s": res["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
