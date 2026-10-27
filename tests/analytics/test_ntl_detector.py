"""Theft engine, Jokar consumer detector (Stage 2 run sheet §4, step D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import evaluate as V
from src.analytics import ntl_engine as N
from src.data_source.theft_injector import ATTACKS


def test_folds():
    assert list(N.FOLDS) == [*ATTACKS, "circular"]
    for a in ATTACKS:
        assert a not in N.FOLDS[a] and len(N.FOLDS[a]) == 5
    assert N.FOLDS["circular"] == ATTACKS


def test_grids_and_constants():
    assert N.GRID_C == tuple(2.0 ** k for k in (-6, -4, -2, 0, 2, 4, 6, 8))
    assert N.GRID_GAMMA == tuple(2.0 ** k for k in (-13, -11, -9, -7, -5, -3, -1, 1))
    assert set(N.GRID_C_FIRST) <= set(N.GRID_C) and set(N.GRID_GAMMA_FIRST) <= set(N.GRID_GAMMA)
    assert N.K_RANGE == (2, 3, 4) and N.MIN_CLUSTER_SHARE == 0.05 and N.MIN_TRAIN_DAYS == 30
    assert N.JOKAR_SEED == 20261027


def test_ami_hourly(tmp_path):
    ts = pd.date_range("2024-03-01", periods=2 * 96, freq="15min")
    a = np.arange(len(ts), dtype=float)
    b = np.ones(len(ts))
    b[100] = np.nan                                     # one missing sample on day 2
    pd.DataFrame({"timestamp": ts, "hh_1": pd.array(a, dtype="UInt16"),
                  "hh_2": pd.array(np.where(np.isnan(b), None, b), dtype="UInt16")}).to_parquet(
        tmp_path / "ami_2024-03.parquet", index=False)
    ids, days, x = N.ami_hourly(tmp_path)
    assert ids == ["hh_1", "hh_2"] and list(days) == list(pd.date_range("2024-03-01", periods=2))
    assert x.shape == (2, 2, 24)
    np.testing.assert_allclose(x[0, 0, 0], (0 + 1 + 2 + 3) / 1e3)
    assert np.isnan(x[1, 1]).sum() == 1 and not np.isnan(x[1, 0]).any()


def test_scale():
    x = np.array([[0.0, 5.0, 2.0], [2.0, 5.0, 4.0]])
    lo, hi = N.scale_fit(x)
    z = N.scale(x, lo, hi)
    np.testing.assert_allclose(z, [[-1, 0, -1], [1, 0, 1]])
    np.testing.assert_allclose(N.scale(np.array([[1.0, 9.0, 6.0]]), lo, hi), [[0, 0, 3]])


def test_benign_clusters():
    rng = np.random.default_rng(1)
    z = np.vstack([rng.normal(-0.5, 0.02, (60, 24)), rng.normal(0.5, 0.02, (57, 24)),
                   rng.normal(0.0, 0.02, (3, 24))])           # third group below 5 %
    cl = N.benign_clusters(z)
    assert cl["k_star"] in (2, 3)
    assert cl["k_prime"] == 2 and cl["keep"].sum() == 117
    assert sorted(np.unique(cl["labels"])) == [0, 1]
    flat = N.benign_clusters(np.zeros((40, 24)))               # silhouette undefined
    assert flat["k_star"] == 1 and flat["k_prime"] == 1 and flat["keep"].all()


def test_fit_svm_oversamples_and_scores():
    rng = np.random.default_rng(2)
    zb = rng.normal(0.5, 0.05, (20, 24))
    lb = np.array([0] * 12 + [1] * 8)
    zm = rng.normal(-0.5, 0.05, (100, 24))
    svm, kp = N.fit_svm(zb, lb, zm, 4.0, 2 ** -3)
    assert kp == 2 and list(svm.classes_) == [0, 1, 2]
    n = np.bincount(np.r_[np.resize(np.arange(20), 100)], minlength=20)
    assert n.sum() == 100 and n.max() - n.min() <= 1          # replicated evenly to the malicious count
    s, f = N.malicious_score(svm, kp, np.vstack([zb[:3], zm[:3]]))
    assert list(f) == [False] * 3 + [True] * 3 and (s[3:] > s[:3].max()).all()
    svm2, kp2 = N.fit_svm(zb, np.zeros(20, int), zm, 4.0, 2 ** -3)    # k' = 1: binary
    s2, f2 = N.malicious_score(svm2, kp2, np.vstack([zb[:2], zm[:2]]))
    assert list(f2) == [False, False, True, True] and s2[2:].min() > s2[:2].max()


def test_select_svm_params_tie_break():
    grid = [(c, g) for c in (1.0, 4.0) for g in (0.1, 0.5)]
    auc = np.array([[0.8, 0.9, 0.9, 0.7], [0.8, 0.9, 0.9, np.nan]])
    assert N.select_svm_params(auc, grid)[:2] == (1.0, 0.5)


def test_household_score():
    f = np.array([1, 0, -1, 1], np.int8)
    assert N.household_score(f) == pytest.approx(2 / 3)
    assert N.household_score(f, np.array([True, True, True, False])) == pytest.approx(1 / 3)
    assert np.isnan(N.household_score(np.array([-1, -1], np.int8)))


def test_prepare_and_score_household():
    days = pd.date_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(3)
    shape = np.sin(np.linspace(0, 2 * np.pi, 24)) + 1.5
    x = (shape[None, :] * rng.uniform(0.8, 1.2, (len(days), 1))).astype(np.float32)
    x[10] = np.nan
    p = N.prepare_household(x, days, 0, "validation")
    assert p["n_train_days"] == 120                            # Jan-Apr 121 days, one offline
    s, f = N.score_household(p, N.FOLDS["h1"], 4.0, 2 ** -3)
    assert len(s) == 61 and (f >= 0).all()
    auc = N.grid_auc(p, 0, "h1", [(4.0, 2 ** -3)])
    assert 0.5 <= auc[0] <= 1.0
    # determinism: same row, same phase -> same synthetic set
    p2 = N.prepare_household(x, days, 0, "validation")
    np.testing.assert_array_equal(p["zm"]["h3"], p2["zm"]["h3"])
    short = x.copy()
    short[:110] = np.nan
    assert N.prepare_household(short, days, 0, "validation") is None


def test_best_threshold_and_targets():
    s = pd.Series({"a": 0.9, "b": 0.6, "c": 0.6, "d": 0.1, "e": 0.0})
    y = pd.Series({"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": np.nan})
    r = V.best_threshold(s, y)
    assert r["q"] == pytest.approx(0.6) and r["f1"] == pytest.approx(0.8)
    assert r["positives"] == 2 and r["negatives"] == 2
    with pytest.raises(ValueError):
        V.best_threshold(s, pd.Series(0.0, index=s.index))
    lab = pd.DataFrame({"thief": [True, True, True, False], "attack": ["h1", "h2", "h1", ""],
                        "onset": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-08-01", None])},
                       index=["t1", "t2", "t3", "h"])
    y = V.fold_targets(lab, lab.index, ("h1",), ("2024-05-01", "2024-06-30"))
    assert y["t1"] == 1 and np.isnan(y["t2"]) and np.isnan(y["t3"]) and y["h"] == 0
