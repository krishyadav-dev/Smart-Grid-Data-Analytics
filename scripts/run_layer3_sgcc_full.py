"""Layer 3 SGCC check under Zheng's protocol (Stage 4 run sheet §5, step F; C1).

The full SGCC file (``data/fallback/SGCC_theft_detecton_data/data set.csv``),
Zheng §IV-A Eqs. (1)-(3) applied to the raw series per consumer, a random
80 % / 20 % split (rng 20261030 [our choice]), no validation set, a fixed 14
epochs [our choice], three seeds (20261030, +1, +2); the test set is scored
once.  Writes ``results/layer3/sgcc_full/summary.json``.  Test labels are read
through :mod:`src.analytics.evaluate` only; no operating point is reported.

PowerShell::

    python scripts/run_layer3_sgcc_full.py
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

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import zheng_cnn as Z  # noqa: E402
from src.data_source import sgcc  # noqa: E402

OUT = ROOT / "results" / "layer3" / "sgcc_full"
SEEDS = (Z.FULL_SEED, Z.FULL_SEED + 1, Z.FULL_SEED + 2)        # run sheet §5


def main() -> int:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("sgcc_full").info
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "test_opened.json").exists():
        raise SystemExit("the full-SGCC test set was already scored (used once)")
    m = sgcc.load_matrix()
    x = sgcc.zheng_preprocess(m.kwh).astype(np.float32)          # Eqs. (1)-(3) on the raw series
    if np.isnan(x).any() or x.min() < 0.0 or x.max() > 1.0:
        raise SystemExit("preprocessed series not in [0, 1] without NaN")
    tr, te = Z.random_split(len(x))
    y_tr = m.flag[tr].astype(np.int64)                           # training labels (the model needs them)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs, t_seed = {}, {}
    for seed in SEEDS:
        ts = time.perf_counter()
        model, info = Z.train_fixed(x[tr], y_tr, seed, device=device, log=log)
        runs[seed] = {"train": info, "test_scores": Z.predict(model, x[te], device)}
        t_seed[seed] = round(time.perf_counter() - ts, 1)
    (OUT / "test_opened.json").write_text(json.dumps({"opened_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                          encoding="utf-8")
    flags = V.sgcc_full_flags()
    if len(flags) != len(x):
        raise SystemExit("FLAG column and matrix rows differ")
    res = {"label": "Zheng et al. 2018 Wide & Deep CNN under Zheng's protocol on the full SGCC set (C1); "
                    "test set scored once; no validation set, so no operating point",
           "data": {"file": "data/fallback/SGCC_theft_detecton_data/data set.csv", "consumers": int(len(x)),
                    "days": int(x.shape[1]), "first_day": str(m.dates[0].date()), "last_day": str(m.dates[-1].date())},
           "preprocessing": "Zheng §IV-A Eqs. (1)-(3) per consumer on the raw series (sgcc.zheng_preprocess)",
           "split": {"rule": "random permutation, rng 20261030; first 20 % = test [our choice of procedure]",
                     "train": int(len(tr)), "test": int(len(te)),
                     "train_thieves": int(flags[tr].sum()), "test_thieves": int(flags[te].sum()),
                     "train_thief_share": float(flags[tr].mean()), "test_thief_share": float(flags[te].mean()),
                     "all_thief_share": float(flags.mean())},
           "training": {"epochs": Z.FULL_EPOCHS, "optimizer": "Adam", "lr": Z.LR, "batch": Z.BATCH,
                        "validation_set": None, "seeds": list(SEEDS)},
           "zheng_table_iii_80pct": V.ZHENG_TABLE_III_80, "device": device, "seeds": {}}
    for seed, r in runs.items():
        res["seeds"][str(seed)] = {"train": r["train"], "elapsed_s": t_seed[seed],
                                   "test": V.sgcc_rank_metrics(r["test_scores"], flags[te])}
    keys = ("auc", "map_at_100", "map_at_200")
    vals = {k: [res["seeds"][str(s)]["test"][k] for s in SEEDS] for k in keys}
    res["mean_and_range"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                             for k, v in vals.items()}
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "test_scores.npz", test_rows=te,
                        **{f"seed_{s}": runs[s]["test_scores"] for s in SEEDS})
    print(json.dumps({"split": res["split"], "mean_and_range": res["mean_and_range"],
                      "table_iii": V.ZHENG_TABLE_III_80, "elapsed_s": res["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
