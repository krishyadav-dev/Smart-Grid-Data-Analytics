"""Addendum J helpers (post-hoc, after Checkpoint 4, descriptive)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("layer3_ntl_addendum", ROOT / "scripts" / "layer3_ntl_addendum.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def test_row_corr():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(5, 96))
    np.testing.assert_allclose(A._row_corr(x, 3 * x), 1.0)          # scale-free
    np.testing.assert_allclose(A._row_corr(x, -x), -1.0)
    c = A._row_corr(x, np.ones_like(x))                              # constant side: undefined
    assert np.isnan(c).all()
    h5 = np.repeat(x.mean(axis=1, keepdims=True), 96, axis=1)        # attack h5: the day's mean, repeated
    assert np.isnan(A._row_corr(h5, x)).all()


def test_shape_similarity_scaled_days_and_month_step():
    rng = np.random.default_rng(1)
    shape_a, shape_b = rng.uniform(size=96), rng.uniform(size=96)
    jan = rng.uniform(1, 5, size=(20, 1)) * shape_a                  # one shape scaled by energy
    feb = rng.uniform(1, 5, size=(20, 1)) * shape_b
    mar = rng.uniform(1, 5, size=(10, 1)) * shape_b                  # below the 15-day rule
    out = A.shape_similarity([("u", (2024, 1), jan), ("u", (2024, 2), feb), ("u", (2024, 3), mar)])
    assert out["household_months"] == 2 and out["household_months_below_min_days"] == 1
    assert out["within_month_mean_corr"]["median"] > 0.999999
    assert out["share_household_months_median_corr_ge_0_999"] == 1.0
    jf = out["across_consecutive_months_corr"]["by_pair"]["1-2"]
    assert jf["n"] == 1 and abs(jf["median"] - float(A._row_corr(shape_a, shape_b))) < 1e-12
    assert out["across_consecutive_months_corr"]["by_pair"]["2-3"]["n"] == 0


def test_deciles():
    energy = np.array([[5.0, 1.0, 3.0, 2.0, 4.0, 9.0, 9.0, 0.5, 7.0, 8.0, 6.0]])
    valid = np.ones_like(energy, bool)
    valid[0, -1] = False
    months = np.ones(energy.shape[1], int)
    dec = A._deciles(energy, valid, months)[0]
    assert dec[-1] == 0                                              # not a valid day
    assert dec[7] == 1 and dec[1] == 2                               # lowest energy -> decile 1
    assert dec[5] == 10 - 1 and dec[6] == 10                         # ties kept in date order
    assert sorted(dec[:-1].tolist()) == list(range(1, 11))
