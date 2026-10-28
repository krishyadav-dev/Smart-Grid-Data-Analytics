"""Zheng Wide & Deep CNN and the SGCC scoring (Stage 2 run sheet §5, step F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import zheng_cnn as Z  # noqa: E402


def test_constants():
    assert (Z.ALPHA, Z.GAMMA_FILTERS, Z.BETA, Z.R_LAYERS) == (90, 90, 60, 3)
    assert (Z.LR, Z.BATCH, Z.MAX_EPOCHS, Z.PATIENCE, Z.SEED, Z.VAL_SHARE) == (1e-3, 128, 30, 5, 20261027, 0.10)


def test_weekly_reshape_pads_at_end():
    x = torch.arange(1, 1035, dtype=torch.float32)[None, :]
    w = Z.weekly(x, 1034)
    assert w.shape == (1, 1, 148, 7)
    assert w[0, 0, 0].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert w[0, 0, 147].tolist() == [1030, 1031, 1032, 1033, 1034, 0, 0]


def test_gconv_is_g1_plus_g2():
    """The effective kernel M K + K M equals the filter applied to g1(B) + g2(B)."""
    torch.manual_seed(0)
    conv = Z.build_model(1034).conv[0]
    x = torch.randn(1, 1, 6, 7)
    with torch.no_grad():
        y = conv(x)
    w, b = conv.weight[0, 0].detach().numpy(), float(conv.bias[0].detach())
    xp = np.pad(x[0, 0].numpy(), 1)
    for r, c in ((0, 0), (2, 3), (5, 6)):
        B = xp[r:r + 3, c:c + 3]
        g1 = np.array([2 * B[i] - B[[j for j in range(3) if j != i]].sum(0) for i in range(3)])
        g2 = np.array([2 * B[:, i] - B[:, [j for j in range(3) if j != i]].sum(1) for i in range(3)]).T
        assert float(y[0, 0, r, c]) == pytest.approx(float((w * (g1 + g2)).sum() + b), abs=1e-5)


def test_model_shapes():
    m = Z.build_model(1034)
    out = m(torch.rand(4, 1034))
    assert out.shape == (4,)
    assert m.wide[0].out_features == 90 and m.deep_fc[0].out_features == 60 and m.out.in_features == 150
    assert sum(isinstance(l, torch.nn.Conv2d) for l in m.conv) == 3


def test_stratified_split():
    flag = np.array([1] * 100 + [0] * 900)
    tr, va = Z.stratified_split(flag)
    assert len(va) == 100 and flag[va].sum() == 10
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == 1000
    tr2, va2 = Z.stratified_split(flag)
    assert np.array_equal(va, va2)


def test_load_pool_rejects_raw(tmp_path):
    pd.DataFrame({"CONS_NO": ["a"], "FLAG": [0], "2014-01-01": [5.0]}).to_parquet(tmp_path / "p.parquet")
    with pytest.raises(ValueError):
        Z.load_pool(tmp_path / "p.parquet")


def test_map_at_n():
    # ranks of thieves among the top 5: 1, 3, 4 -> (1/1 + 2/3 + 3/4) / 3
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y = np.array([1, 0, 1, 1, 0, 1])
    assert V.map_at_n(s, y, 5) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    assert V.map_at_n(s, np.zeros(6), 5) == 0.0


def test_threshold_max_f1():
    s = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([1, 1, 0, 1])
    t = V.threshold_max_f1(s, y)
    assert t == pytest.approx(0.2)            # all flagged: P 0.75, R 1 -> F1 0.857 > 0.8
