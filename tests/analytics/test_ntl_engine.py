"""Theft engine, balance part (Stage 2 run sheet §3, step B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import ntl_engine as N
from src.data_source import config

C = config.YEAR_CACHE_DIR


def _hh(values: dict, days) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.DatetimeIndex(days)).T


def test_day_type():
    d = pd.date_range("2024-01-01", periods=7)          # Monday .. Sunday
    assert list(N.day_type(d)) == [0, 0, 0, 0, 0, 1, 1]


def test_offline_estimate_same_month_day_type():
    days = pd.date_range("2024-01-01", "2024-01-31")
    x = np.where(N.day_type(days) == 0, 10.0, 4.0) + np.arange(len(days)) * 0.0
    x[[1, 6]] = np.nan                                   # Tue (weekday), Sun (weekend)
    x[2], x[3] = 12.0, 14.0                              # weekday values 10 x 19, 12, 14 -> median 10
    est, info = N.offline_estimate(_hh({"h": x}, days))
    assert est.loc["h"].iloc[1] == 10.0 and est.loc["h"].iloc[6] == 4.0
    assert est.loc["h"].drop(est.columns[[1, 6]]).eq(0).all()
    assert info["rule_counts"]["same_month_day_type"] == 2 and info["offline_household_days"] == 2


def test_offline_estimate_fallbacks():
    days = pd.date_range("2024-01-01", "2024-04-30")
    dt = N.day_type(days)
    m = days.month
    x = np.full(len(days), np.nan)
    # January weekend: 2 online days only (below 3), February weekend 3 days of 6.0,
    # April weekend 3 days of 9.0 -> a March weekend offline day goes to February
    # (distance 1 to both; the earlier month wins)
    jan_we = np.flatnonzero((m == 1) & (dt == 1))
    feb_we = np.flatnonzero((m == 2) & (dt == 1))
    apr_we = np.flatnonzero((m == 4) & (dt == 1))
    x[jan_we[:2]] = 1.0
    x[feb_we[:3]] = 6.0
    x[apr_we[:3]] = 9.0
    mar_we = np.flatnonzero((m == 3) & (dt == 1))[0]
    mar_wd = np.flatnonzero((m == 3) & (dt == 0))[0]
    est, info = N.offline_estimate(_hh({"h": x}, days))
    assert est.loc["h"].iloc[mar_we] == 6.0
    # January weekend offline day: 2 online January weekend days -> own month
    assert est.loc["h"].iloc[jan_we[2]] == 1.0
    # weekday: no weekday online at all -> all online days of the household
    assert est.loc["h"].iloc[mar_wd] == float(np.nanmedian(x))
    rc = info["rule_counts"]
    assert rc["nearest_month_day_type"] > 0 and rc["all_online_days"] > 0
    assert sum(rc.values()) == int(np.isnan(x).sum())
    # a household that is never online gets 0, counted
    est2, info2 = N.offline_estimate(_hh({"z": np.full(len(days), np.nan)}, days))
    assert est2.to_numpy().sum() == 0 and info2["rule_counts"]["no_online_days"] == len(days)


def test_balances_and_netting():
    days = pd.date_range("2024-07-01", periods=2)
    hh = _hh({"h1": [10.0, np.nan], "h2": [5.0, 6.0], "h3": [1.0, 2.0]}, days)
    idx = pd.Series({"h1": "dt_a", "h2": "dt_a", "h3": "dt_b"})
    head = pd.Series([30.0, 31.0], index=days)
    dt = pd.DataFrame({"dt_a": [20.0, 19.0], "dt_b": [5.0, 6.0]}, index=days)
    b = N.balances(head, dt, hh, idx)
    np.testing.assert_allclose(b["A"], [5.0, 6.0])
    np.testing.assert_allclose(b["B"]["dt_a"], [5.0, 13.0])
    np.testing.assert_allclose(b["B"]["dt_b"], [4.0, 4.0])
    np.testing.assert_allclose(b["C"], [14.0, 23.0])
    # identity: C = A + sum_DT B
    np.testing.assert_allclose(b["C"], b["A"] + b["B"].sum(axis=1))
    eoff = pd.Series([0.0, 10.0], index=days)
    ntl = N.netted_feeder(b["C"], pd.Series([4.0, 4.0], index=days), eoff)
    np.testing.assert_allclose(ntl, [10.0, 9.0])
    eoff_dt = pd.DataFrame({"dt_a": [0.0, 10.0], "dt_b": [0.0, 0.0]}, index=days)
    nb = N.netted_dt(b["B"], b["billed_dt"], eoff_dt, 0.02)
    np.testing.assert_allclose(nb["dt_a"], [5.0 - 0.02 * 15, 13.0 - 0.02 * 6 - 10.0])


@pytest.mark.skipif(not (C / "state.parquet").exists(), reason="year cache absent")
def test_ltech_reproduces_power_flow_losses_on_true_loads():
    """L^_tech's power flow, fed the TRUE DT loads of one day, reproduces the
    cached run's MV line and DT losses (the method check; the engine itself
    is fed the metered DT loads)."""
    day = ("2024-03-05 00:00", "2024-03-05 23:45")
    head, dts = N.physical_load_ids()
    true = pd.read_parquet(C / "meter_frame_true.parquet",
                           columns=["timestamp", "load_id", "phase", "active_power_kw", "reactive_power_kvar"],
                           filters=[("load_id", "in", dts), ("timestamp", ">=", pd.Timestamp(day[0])),
                                    ("timestamp", "<=", pd.Timestamp(day[1]))])
    lt = N.ltech_run(true)
    st = pd.read_parquet(C / "state.parquet", columns=["timestamp", "line_losses_mw", "dt_losses_mw"],
                         filters=[("timestamp", ">=", pd.Timestamp(day[0])),
                                  ("timestamp", "<=", pd.Timestamp(day[1]))]).set_index("timestamp")
    assert len(lt) == 96
    # per step, to the power-flow solver's convergence (observed max 3.5e-8 relative)
    ref = (st.line_losses_mw + st.dt_losses_mw) * 1e3
    np.testing.assert_allclose(lt.ltech_kw.to_numpy(), ref.to_numpy(), rtol=1e-6)


def test_bias_mae_and_dt_scoring():
    from src.analytics import evaluate as V
    days = pd.date_range("2024-06-29", periods=4)
    est = pd.Series([2.0, 4.0, 5.0, 1.0], index=days)
    truth = pd.Series([1.0, 5.0, 5.0, 2.0], index=days)
    r = V.bias_mae(est, truth)
    assert r["n"] == 4 and r["bias_kwh"] == pytest.approx(-0.25) and r["mae_kwh"] == pytest.approx(0.75)
    tr = {"dt_theft": pd.DataFrame({"d1": [1.0, 5.0, 5.0, 2.0]}, index=days),
          "dt_offline_meters": pd.DataFrame({"d1": [0, 0, 1, 0]}, index=days)}
    s = V.score_dt({"x": pd.DataFrame({"d1": est.to_numpy()}, index=days)}, tr)
    test = s["jul_dec_test"]["x"]
    assert test["all_dt_days"]["n"] == 2 and test["fully_online_dt_days"]["n"] == 1
    assert test["fully_online_dt_days"]["bias_kwh"] == pytest.approx(-1.0)
    ref = s["jan_jun_reference"]["x"]["all_dt_days"]
    assert ref["n"] == 2 and ref["bias_kwh"] == pytest.approx(0.0)


@pytest.mark.skipif(not (C / "household_daily_labels.parquet").exists(), reason="year cache absent")
def test_energy_identity_closes_on_test_window():
    from src.analytics import evaluate as V
    r = V.ntl_energy_identity(V.NTL_TEST)
    assert r["closes"] and r["steps"] == 184 * 96
