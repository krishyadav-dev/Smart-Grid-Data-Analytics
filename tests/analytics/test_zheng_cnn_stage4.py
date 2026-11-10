"""Zheng CNN under Zheng's protocol on the full SGCC set (Stage 4 run sheet §5, step F; C1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics import evaluate as V
from src.analytics import zheng_cnn as Z


def test_constants():
    assert Z.FULL_SEED == 20261030 and Z.FULL_TEST_SHARE == 0.20
    assert Z.FULL_EPOCHS == round(np.mean([17, 19, 5]))           # the Stage 2 seeds' best epochs


def test_random_split():
    tr, te = Z.random_split(42372)
    assert len(te) == round(0.2 * 42372) and len(tr) + len(te) == 42372
    assert not np.intersect1d(tr, te).size and np.array_equal(np.sort(np.r_[tr, te]), np.arange(42372))
    tr2, te2 = Z.random_split(42372)
    np.testing.assert_array_equal(te, te2)                         # seeded
    assert not np.array_equal(te, Z.random_split(42372, seed=1)[1])


def test_train_fixed_runs_the_given_epochs():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (64, 30)).astype(np.float32)
    y = (rng.uniform(size=64) < 0.3).astype(np.int64)
    model, info = Z.train_fixed(x, y, 7, epochs=2, device="cpu", log=lambda *_: None)
    assert info["epochs"] == 2 and len(info["history"]) == 2
    s = Z.predict(model, x, "cpu")
    assert s.shape == (64,) and ((s >= 0) & (s <= 1)).all()
    _, info2 = Z.train_fixed(x, y, 7, epochs=2, device="cpu", log=lambda *_: None)
    assert info2["history"][-1]["train_loss"] == pytest.approx(info["history"][-1]["train_loss"])
    del torch


def test_sgcc_rank_metrics():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    m = V.sgcc_rank_metrics(s, y)
    assert m["thieves"] == 3 and m["map_at_100"] == pytest.approx((1 / 1 + 2 / 3 + 3 / 6) / 3)
    assert 0 <= m["auc"] <= 1 and "precision" not in m           # no operating point (no validation set)


def test_full_file_matches_the_references():
    from src.data_source.config import SGCC_FALLBACK_PATH
    if not SGCC_FALLBACK_PATH.exists():
        pytest.skip("full SGCC file absent")
    f = V.sgcc_full_flags()
    assert len(f) == 42372 and int(f.sum()) == 3615                # References README, Zheng §V-A
