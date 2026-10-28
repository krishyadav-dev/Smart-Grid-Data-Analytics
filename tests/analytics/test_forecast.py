"""Day-ahead load forecasts for the thermal engine (Stage 3 run sheet §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import forecast as F

DAYS = pd.date_range("2024-01-01", "2024-12-31")


def _kd(n_days=20, n_series=6):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (n_days, 96, n_series))


def test_constants():
    assert (F.LOOKBACK, F.HORIZON) == (672, 96)
    assert F.SEEDS == (20261028, 20261029, 20261030)
    assert (F.GRID_LAYERS, F.GRID_UNITS, F.DROPOUT) == ((1, 2), (32, 64, 128), 0.1)
    assert (F.LR, F.BATCH, F.MAX_EPOCHS, F.PATIENCE) == (1e-3, 256, 50, 5)


def test_baselines():
    kd = _kd()
    o = np.array([7, 10])
    np.testing.assert_array_equal(F.persistence(kd, o), kd[[6, 9]])
    np.testing.assert_array_equal(F.seasonal_naive(kd, o), kd[[0, 3]])


def test_series_layout():
    k = np.arange(2 * 96 * 13 * 3, dtype=float).reshape(2 * 96, 13, 3)
    kd = F.daily_series(k)
    assert kd.shape == (2, 96, 39)
    assert kd[1, 5, 3 * 4 + 2] == k[96 + 5, 4, 2]
    np.testing.assert_array_equal(F.to_dt_phase(kd, 13), k.reshape(2, 96, 13, 3))


def test_training_rounds():
    """April selection: train to 2024-03-17, early-stop 03-18..03-31; first target day 2024-01-09."""
    assert F.training_round(pd.Timestamp("2024-04-01")) == (F.SELECTION_TRAIN, F.SELECTION_VAL)
    tr, va = F.training_round(pd.Timestamp("2024-06-01"))
    assert tr == (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-05-17"))
    assert va == (pd.Timestamp("2024-05-18"), pd.Timestamp("2024-05-31"))
    p = F.samples(F.SELECTION_TRAIN, DAYS, 39)
    assert DAYS[p[:, 1].min()] == pd.Timestamp("2024-01-09") and DAYS[p[:, 1].max()] == pd.Timestamp("2024-03-17")
    assert len(p) == 39 * 69
    v = F.samples(F.SELECTION_VAL, DAYS, 39)
    assert len(v) == 39 * 14 and not set(v[:, 1]) & set(p[:, 1])


def test_features():
    kd = _kd(n_days=14, n_series=6)
    days = DAYS[:14]
    x = F.features(kd, days, n_dt=2)
    assert x.shape == (6, 14 * 96, 1 + 2 + 7 + 2 + 3)
    assert x[4, 96 + 10, 0] == pytest.approx(kd[1, 10, 4])
    assert x[0, 24, 1] == pytest.approx(1.0) and abs(x[0, 24, 2]) < 1e-6        # 06:00 -> sin = 1
    assert x[0, 0, 3 + days[0].weekday()] == 1.0 and x[0, 0, 3:10].sum() == 1.0
    assert list(x[4, 0, 10:]) == [0, 1, 0, 1, 0]                                 # series 4 = DT 1, phase b


def test_residual_target_and_compose():
    kd = _kd(n_days=10, n_series=3)
    r = F.target(kd, "residual")
    np.testing.assert_allclose(r[5], kd[5] - kd[4])
    assert (r[0] == 0).all() and F.target(kd, "direct") is kd
    o = np.array([8, 9])
    np.testing.assert_allclose(F.compose(r[o], kd, o, "residual"), kd[o])         # perfect residual -> actual
    np.testing.assert_array_equal(F.compose(kd[o], kd, o, "direct"), kd[o])
    with pytest.raises(ValueError):
        F.target(kd, "other")


def test_clip():
    c, n = F.clip(np.array([-0.1, 0.0, 0.3, -2.0]))
    assert n == 2 and (c >= 0).all() and c[2] == 0.3


torch = pytest.importorskip("torch")


def test_windows_and_model_shapes():
    kd = _kd(n_days=10, n_series=6)
    x = F.features(kd, DAYS[:10], n_dt=2)
    win = F.Windows(x, kd, torch.device("cpu"))
    xb, yb = win.batch(np.array([[3, 8], [0, 7]]))
    assert xb.shape == (2, 672, x.shape[2]) and yb.shape == (2, 96)
    np.testing.assert_allclose(xb[0, -1, 0].item(), kd[7, 95, 3], rtol=1e-6)
    np.testing.assert_allclose(yb[1].numpy(), kd[7, :, 0], rtol=1e-6)
    for layers in F.GRID_LAYERS:
        m = F.build_model(x.shape[2], layers, 32)
        assert m(xb).shape == (2, 96)
    assert F.n_parameters(F.build_model(15, 1, 32)) == 4 * 32 * (15 + 32 + 2) + 32 * 96 + 96


def test_training_is_seeded():
    kd = _kd(n_days=10, n_series=4)
    x = F.features(kd, DAYS[:10], n_dt=2, n_ph=2)
    win = F.Windows(x, kd, torch.device("cpu"))
    trp, vap = np.array([[s, 7] for s in range(4)] + [[s, 8] for s in range(4)]), np.array([[0, 9], [1, 9]])
    old = F.MAX_EPOCHS
    F.MAX_EPOCHS = 2
    try:
        a = F.forecast_origins(F.train(win, trp, vap, 1, 32, 7)[0], win, np.array([9]), 4)
        b = F.forecast_origins(F.train(win, trp, vap, 1, 32, 7)[0], win, np.array([9]), 4)
    finally:
        F.MAX_EPOCHS = old
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    assert a.shape == (1, 96, 4)
