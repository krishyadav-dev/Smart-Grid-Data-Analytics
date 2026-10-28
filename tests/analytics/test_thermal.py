"""IS 2026-7 thermal model (Stage 3 run sheet §2, step A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import thermal as T
from src.data_source.config import YEAR_CACHE_DIR

DTS = YEAR_CACHE_DIR / "dts.parquet"


def test_annex_e_acceptance():
    """Annex E worked example: 114.2 degC at t = 30 min and 75.2 degC at t = 1 440 min."""
    r = T.annex_e_example()
    assert round(r["theta_h_c"][30], 1) == 114.2
    assert round(r["theta_h_c"][1440], 1) == 75.2
    assert T.annex_e_passes(r)
    # Annex E initial values: Dtheta_oi 38.7 K, Dtheta_hi 16.5 K
    assert round(r["dtheta_oi_k"], 1) == 38.7 and round(r["dtheta_hi_k"], 1) == 16.5


def test_time_step():
    """Dt = 0.1 min, 150 sub-steps per 15-min step, below half the smallest time constant."""
    assert T.DT_MIN * T.SUBSTEPS == pytest.approx(T.STEP_MIN)
    assert T.DT_MIN < T.TABLE5_ONAN["tau_w"] / 2
    # the run sheet's 1 min misses the acceptance point (the reason for 0.1 min)
    assert round(T.annex_e_example(dt=1.0)["theta_h_c"][30], 1) == 114.4


def _onan(R=5.0, d_or=55.0, d_hr=23.0):
    return T.ThermalParams(**T.TABLE5_ONAN, R=np.float64(R), dtheta_or=np.float64(d_or),
                           dtheta_hr=np.float64(d_hr))


@pytest.mark.parametrize("params", [_onan(), T.annex_e_params()], ids=["onan", "annex_e_of"])
def test_closed_form_matches_substep_loop(params):
    rng = np.random.default_rng(0)
    ko = rng.uniform(0.2, 1.5, (6, 3))
    kh = ko * rng.uniform(1.0, 1.2, (6, 3))
    ta = rng.uniform(5, 45, (6, 3))
    s0 = T.steady_state(params, 0.5, 0.6, 20.0)
    r = T.simulate(params, ko, kh, ta, s0, substeps=150, dt=0.1)
    loop = T.simulate_loop(params, ko, kh, ta, s0, substeps=150, dt=0.1)
    np.testing.assert_allclose(r["theta_h_end"], loop[:, -1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r["theta_h_max"], loop.max(axis=1), rtol=0, atol=1e-9)
    np.testing.assert_allclose(r["lol_h"], T.ageing_rate(loop).sum(axis=1) * 0.1 / 60, rtol=1e-12)


def test_steady_state_hand_check():
    """K = 1, theta_a = 20 degC, Table E.1 ONAN: 20 + 55 + 23 = 98 degC; one day at 98 degC = 24 h."""
    p = _onan()
    s0 = T.steady_state(p, 1.0, 1.0, 20.0)
    assert s0.theta_h == pytest.approx(98.0)
    r = T.simulate(p, np.ones(96), np.ones(96), np.full(96, 20.0), s0)
    np.testing.assert_allclose(r["theta_h_end"], 98.0, atol=1e-9)
    assert r["lol_h"].sum() == pytest.approx(24.0)


def test_load_factor_example():
    """Plan §3: dt_652_1 at the feeder peak, K = 0.977 / 1.068 / 0.842 -> K_oil 0.967, K_hs 1.068."""
    k = np.array([0.977, 1.068, 0.842])
    assert round(float(T.k_oil(k)), 3) == 0.967
    assert T.k_hs(k) == 1.068
    np.testing.assert_allclose(T.load_factor(np.array([[[213.34, 0.0, 106.67]]]), np.array([213.34])),
                               [[[1.0, 0.0, 0.5]]])


def test_ageing_rate_examples():
    """Plan §3: 104 degC -> 2, 110 degC -> 4, 120 degC -> 12.7."""
    assert T.ageing_rate(98.0) == 1.0 and T.ageing_rate(104.0) == 2.0 and T.ageing_rate(110.0) == 4.0
    assert round(float(T.ageing_rate(120.0)), 1) == 12.7


def test_dt_parameters():
    dts = T.read_dts(DTS)
    assert list(dts.index) == T.dt_names() and len(dts) == 13
    main, e1 = T.dt_params(dts, "main"), T.dt_params(dts, "table_e1")
    R = pd.Series(main.R, index=dts.index)
    assert round(R["dt_652_1"], 2) == 6.50 and round(R["dt_611_1"], 2) == 8.42
    assert set(R[dts.rating_kva == 630].round(2)) == {7.19}
    assert set(R[dts.rating_kva == 500].round(2)) == {7.64}
    d_or = pd.Series(main.dtheta_or, index=dts.index)
    assert set(d_or[d_or == 35.0].index) == {"dt_652_1", "dt_611_1"}
    assert (d_or[~d_or.index.isin(["dt_652_1", "dt_611_1"])] == 40.0).all()
    assert (e1.dtheta_or == 55.0).all() and (e1.dtheta_hr == 23.0).all() and (main.dtheta_hr == 23.0).all()
    np.testing.assert_array_equal(e1.R, main.R)
    assert (main.x, main.y, main.k11, main.k21, main.k22, main.tau_o, main.tau_w) == (
        0.8, 1.6, 1.0, 1.0, 2.0, 180.0, 4.0)


def test_rated_current():
    i = dict(zip(T.dt_names(), T.rated_current()))
    assert i["dt_652_1"] == pytest.approx(213.34)


def test_era5_file():
    ta = T.read_ambient()
    assert len(ta) == T.STEPS_2024 and not ta.isna().any()
    assert ta.index[0] == pd.Timestamp("2024-01-01 00:00") and ta.index[-1] == pd.Timestamp("2024-12-31 23:45")


def test_day_summary_crossings():
    """First crossing = step start + first sub-step above the threshold; NaN without a crossing."""
    p = _onan()
    k = np.full((2 * 96, 2), 0.5)
    k[40:48, 0] = 1.6                               # day 0, DT 0 overloaded 10:00-12:00
    s0 = T.steady_state(p, 0.5, 0.5, 20.0)
    r = T.simulate(p, k, k, np.full(k.shape, 20.0), s0, thresholds=(98.0,))
    d = T.day_summary(r)
    first = d["first_cross_min"][98.0]
    assert np.isnan(first[1]).all() and np.isnan(first[0, 1])
    j = int(np.flatnonzero(~np.isnan(r["cross_min"][98.0][:, 0]))[0])
    assert first[0, 0] == pytest.approx(j * 15 + r["cross_min"][98.0][j, 0])
    assert 600 < first[0, 0] < 720
    assert d["theta_h_max"][0, 0] > 98.0 and d["lol_h"].shape == (2, 2)


def test_forecast_run_with_actual_load_reproduces_reference():
    """A forecast equal to the actual load, started from the reference state at
    00:00, gives the reference trajectory (the gap is forecast error only)."""
    p = _onan(R=np.array([6.5, 7.6]), d_or=np.array([35.0, 40.0]), d_hr=np.array([23.0, 23.0]))
    rng = np.random.default_rng(1)
    k = rng.uniform(0.2, 1.3, (3 * 96, 2, 3))
    ta = rng.uniform(10, 40, 3 * 96)
    ref = T.reference_run(p, k, ta)
    days = np.array([1, 2])
    s0 = T.state_at(ref, days * 96 - 1)
    fc = T.forecast_run(p, k.reshape(3, 96, 2, 3)[days], ta.reshape(3, 96)[days], s0)
    np.testing.assert_allclose(fc["theta_h_end"], ref["theta_h_end"].reshape(3, 96, 2)[days], atol=1e-9)
    np.testing.assert_allclose(fc["lol_h"], ref["lol_h"].reshape(3, 96, 2)[days], rtol=1e-12)
