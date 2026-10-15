"""
SGCC matrix, preprocessing, calendar and pools (Part 2C.4, plan CP3).

Zheng et al. 2018 (ALL_TII-17-2030.pdf): Eq. (1) PDF p. 3, Eq. (2)-(3) PDF
p. 4, §V-A p. 6 (42 372 customers, 3 615 thieves), §V-C p. 7 (80 : 20).
"""

import numpy as np
import pandas as pd
import pytest

from src.data_source import config, sgcc

nan = np.nan


def test_zheng_eq1_missing_values():
    x = np.array([[1.0, nan, 3.0, nan, nan, 6.0, nan]], np.float32)
    assert np.allclose(sgcc.zheng_eq1(x), [[1, 2, 3, 0, 0, 6, 0]])


def test_zheng_eq2_positive_outliers_only():
    x = np.array([[1.0, 1, 1, 1, 1, 1, 1, 1, 1, 20.0, -5.0]])
    thr = x.mean() + 2 * x.std()
    out = sgcc.zheng_eq2(x)
    assert out[0, 9] == pytest.approx(thr)
    assert out[0, 10] == -5.0                       # negative side untouched
    assert np.allclose(out[0, :9], 1.0)


def test_feeder_eq2_leave_one_out_unmasks_single_outlier():
    """Brief v2 X.3.3: feeder pool only.  Zheng Eq. (2) lets a single extreme
    day mask itself (it inflates its own std); leave-one-out clips it against
    the other days.  Normal days and the negative side are unchanged."""
    x = np.array([[14.0, 15.8, 15.4, 15.6, 14.8, 13442.3, 15.0, 14.2, nan, -1.0]])
    zheng = sgcc.zheng_eq2(x)
    loo = sgcc.zheng_eq2_loo(x)
    others = np.delete(x[0], [5, 8])
    thr = others.mean() + 2 * others.std()
    assert zheng[0, 5] > 1000                              # masked: survives Zheng
    assert loo[0, 5] == pytest.approx(thr)                 # clipped against the other days
    assert np.allclose(loo[0, [0, 1, 2, 3, 4, 6, 7]], x[0, [0, 1, 2, 3, 4, 6, 7]])
    assert np.isnan(loo[0, 8]) and loo[0, 9] == -1.0


def test_zheng_eq3_min_max():
    x = np.array([[2.0, 4.0, 6.0], [5.0, 5.0, 5.0]])
    assert np.allclose(sgcc.zheng_eq3(x), [[0, 0.5, 1], [0, 0, 0]])


def test_zheng_preprocess_order():
    x = np.array([[1.0, nan, 3.0, 2.0, 50.0, 2.0, 1.0, 2.0]])
    expected = sgcc.zheng_eq3(sgcc.zheng_eq2(sgcc.zheng_eq1(x)))
    assert np.allclose(sgcc.zheng_preprocess(x), expected)


def test_feeder_rule_keeps_long_gaps_and_zeros():
    x = np.array([[0.0, 2.0, nan, 4.0, nan, nan, 5.0, 0.0, 3.0, 3.0]])
    out = sgcc.feeder_preprocess(x)
    assert out[0, 2] == pytest.approx(3.0)          # isolated gap interpolated
    assert np.isnan(out[0, 4]) and np.isnan(out[0, 5])   # run stays missing (no zero-fill)
    assert out[0, 0] == 0.0 and out[0, 7] == 0.0    # genuine zeros kept


def test_calendar_constant_offset_keeps_weekdays():
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    mapped = sgcc.map_to_sgcc(days)
    assert mapped[0] == pd.Timestamp("2015-01-05")
    assert mapped[-1] == pd.Timestamp("2016-01-05")      # 2024-12-31 = day 365
    assert (mapped.dayofweek == days.dayofweek).all()
    assert (np.diff(mapped.values).astype("timedelta64[D]").astype(int) == 1).all()


def _synthetic(n=400, d=60, seed=3):
    rng = np.random.default_rng(seed)
    kwh = rng.gamma(2.0, 3.0, size=(n, d)).astype(np.float32)
    kwh[rng.random((n, d)) < 0.02] = np.nan
    kwh[:20, :10] = np.nan                                   # incomplete consumers
    flag = (rng.random(n) < 0.1).astype(np.int8)
    dates = pd.date_range("2015-01-05", periods=d, freq="D")
    return sgcc.SGCCMatrix(np.array([f"C{i:04d}" for i in range(n)]), flag, dates, kwh)


def test_eligibility_filters_in_order():
    m = _synthetic()
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    el = sgcc.eligibility(m, days)
    flag0_complete = (m.flag == 0) & (el.completeness >= sgcc.COMPLETENESS_MIN)
    assert not el.eligible[:20].any()                        # < 95 % complete
    assert (m.flag[el.eligible] == 0).all()
    assert el.top_threshold_kwh == pytest.approx(np.percentile(el.median_kwh[flag0_complete], 95))
    assert (el.median_kwh[el.eligible] <= el.top_threshold_kwh).all()
    assert el.report["excluded_top5"] == int((flag0_complete & ~el.eligible).sum())


def test_pools_disjoint_cover_and_stratified():
    m = _synthetic()
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    el = sgcc.eligibility(m, days)
    pools = sgcc.build_pools(m, el, 100, config.rng("test.feeder"), config.rng("test.split"))
    pools.assert_valid(m)
    f, tr, te = (set(p) for p in (pools.feeder, pools.detector_train, pools.detector_test))
    assert not (f & tr) and not (f & te) and not (tr & te)
    assert (m.flag[pools.feeder] == 0).all()
    rest = np.setdiff1d(np.arange(len(m.flag)), pools.feeder)
    for flag in (0, 1):
        n = int((m.flag[rest] == flag).sum())
        assert int((m.flag[pools.detector_test] == flag).sum()) == round(0.2 * n)


def test_pools_too_small_raises():
    m = _synthetic(n=50)
    el = sgcc.eligibility(m, pd.date_range("2024-01-01", periods=30, freq="D"))
    with pytest.raises(ValueError, match="eligible"):
        sgcc.build_pools(m, el, 10_000, config.rng("a"), config.rng("b"))


def test_named_streams_reproducible():
    a = config.rng("x").random(3)
    assert np.array_equal(a, config.rng("x").random(3))
    assert not np.array_equal(a, config.rng("y").random(3))


@pytest.fixture(scope="module")
def full():
    assert config.SGCC_FALLBACK_PATH.exists(), "full SGCC file required"
    return sgcc.load_matrix()


def test_full_file_matches_zheng(full):
    """Zheng §V-A: 42 372 customers, 3 615 thieves -> p = 8.53 %."""
    assert len(full.flag) == 42_372
    assert int(full.flag.sum()) == 3_615
    assert full.theft_prevalence == pytest.approx(3615 / 42372)
    assert full.dates[0] == pd.Timestamp("2014-01-01")
    assert full.dates[-1] == pd.Timestamp("2016-10-31")
    assert full.dates.is_monotonic_increasing
    # Zheng states 1 035 days; the file has one date column fewer (reported)
    print(f"\nSGCC file: {len(full.dates)} date columns (Zheng §V-A: 1 035 days)")
    assert full.kwh.dtype == np.float32


def test_year_maps_inside_sgcc(full):
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    cols = full.columns_for(days)
    assert len(cols) == 366 and (np.diff(cols) == 1).all()
