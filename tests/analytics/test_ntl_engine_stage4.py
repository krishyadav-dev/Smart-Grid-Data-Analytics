"""Theft engine, Stage 4 balance changes A1-A4 (Stage 4 run sheet §3, step B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import ntl_engine as N


def _hh(rows: dict, days) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.DatetimeIndex(days)).T


def test_gain_window_is_january_to_june():
    assert N.GAIN_WINDOW == ("2024-01-01", "2024-06-30")


def test_head_gain_estimate_recovers_a_head_gain():
    days = pd.date_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(0)
    dt = pd.DataFrame(rng.uniform(800, 1200, (len(days), 3)), index=days, columns=["d1", "d2", "d3"])
    lt = pd.Series(rng.uniform(40, 60, len(days)), index=days)
    g_true = 0.99653                                       # head reads low by this gain
    head = (dt.sum(axis=1) + lt) * g_true
    g = N.head_gain_estimate(head, dt, lt)
    assert g == pytest.approx(1 / g_true, rel=1e-12)
    # only January-June enters: a July-December change leaves g^ unchanged
    head2 = head.copy()
    head2[days >= "2024-07-01"] *= 1.05
    assert N.head_gain_estimate(head2, dt, lt) == pytest.approx(g, rel=1e-12)


def test_neighbour_ratio_estimate():
    days = pd.date_range("2024-01-01", "2024-01-31")
    nb = 5.0 + np.arange(len(days)) * 0.1                   # neighbours' level moves day to day
    x = {"a": 2.0 * nb, "b": nb.copy(), "c": nb.copy()}      # a uses twice its neighbours' mean
    x["a"][[10, 13]] = np.nan                                # Thu (weekday), Sun (weekend)
    hh = _hh(x, days)
    idx = pd.Series({"a": "dt_1", "b": "dt_1", "c": "dt_1"})
    est, info = N.offline_estimate_neighbour(hh, idx)
    assert est.loc["a"].iloc[10] == pytest.approx(2.0 * nb[10])
    assert est.loc["a"].iloc[13] == pytest.approx(2.0 * nb[13])
    assert est.loc[["b", "c"]].eq(0).all().all() and est.loc["a"].drop(est.columns[[10, 13]]).eq(0).all()
    assert info["rule_counts"]["ratio_same_month_day_type"] == 2 and info["offline_household_days"] == 2
    # the median rule would miss the day-to-day level: it gives a's median weekday value
    med, _ = N.offline_estimate(hh)
    assert abs(med.loc["a"].iloc[10] - 2.0 * nb[10]) > 0.1


def test_neighbour_ratio_uses_month_and_day_type():
    days = pd.date_range("2024-01-01", "2024-01-31")
    nb = np.full(len(days), 4.0)
    wkend = N.day_type(days) == 1
    a = np.where(wkend, 3.0, 1.0) * nb                      # ratio 1 on weekdays, 3 at weekends
    a[[12, 15]] = np.nan                                    # Sat, Tue
    hh = _hh({"a": a, "b": nb, "c": nb}, days)
    est, _ = N.offline_estimate_neighbour(hh, pd.Series({"a": "j", "b": "j", "c": "j"}))
    assert est.loc["a"].iloc[12] == pytest.approx(12.0) and est.loc["a"].iloc[15] == pytest.approx(4.0)


def test_neighbour_ratio_fallbacks_to_the_median_rule():
    days = pd.date_range("2024-01-01", "2024-01-31")
    a = np.full(len(days), 6.0)
    b = np.full(len(days), 3.0)
    a[[3, 20]] = np.nan
    b[20] = np.nan                                          # day 20: no online neighbour of a
    c = np.full(len(days), np.nan)                          # c: never online -> no ratio day
    hh = _hh({"a": a, "b": b, "c": c}, days)
    idx = pd.Series({"a": "j", "b": "j", "c": "j"})
    est, info = N.offline_estimate_neighbour(hh, idx)
    med, _ = N.offline_estimate(hh)
    assert est.loc["a"].iloc[3] == pytest.approx(6.0)          # ratio 2 x neighbour 3
    assert est.loc["a"].iloc[20] == med.loc["a"].iloc[20]      # no online neighbour
    assert est.loc["b"].iloc[20] == med.loc["b"].iloc[20]
    assert (est.loc["c"] == med.loc["c"]).all()                 # no ratio day: median rule (0)
    rc = info["rule_counts"]
    assert rc["median_rule_no_online_neighbour"] == 3           # a, b and c on day 20 (counted first)
    assert rc["median_rule_no_ratio_days"] == len(days) - 1     # c, every other day
    assert sum(rc.values()) == info["offline_household_days"]
    assert sum(info["median_rule_counts_on_fallback_days"].values()) == 2 + len(days)


def test_neighbour_ratio_households_on_other_dts_are_not_neighbours():
    days = pd.date_range("2024-01-01", "2024-01-31")
    a = np.full(len(days), 2.0)
    a[5] = np.nan
    hh = _hh({"a": a, "b": np.full(len(days), 1.0), "x": np.full(len(days), 100.0)}, days)
    hh.loc["x", hh.columns[5]] = 500.0                              # a different DT's spike on day 5
    est, _ = N.offline_estimate_neighbour(hh, pd.Series({"a": "j", "b": "j", "x": "k"}))
    assert est.loc["a"].iloc[5] == pytest.approx(2.0)


def test_balance_estimates_stage4_outputs():
    ts = pd.date_range("2024-01-01", "2024-12-31 23:45", freq="15min")
    rng = np.random.default_rng(1)
    head, dts = N.physical_load_ids()
    d1, d2 = dts[:2]
    rows = []
    p = {d: rng.uniform(10, 20, len(ts)) for d in dts}
    p[head] = (sum(p.values()) + 1.0) * 0.99
    for lid, v in p.items():
        rows.append(pd.DataFrame({"timestamp": ts, "load_id": lid, "phase": "a", "active_power_kw": v,
                                  "reactive_power_kvar": 0.0}))
    frame = pd.concat(rows, ignore_index=True)
    days = pd.date_range("2024-01-01", "2024-12-31")
    hh = _hh({"h1": np.full(len(days), 50.0), "h2": np.full(len(days), 60.0)}, days)
    hh.loc["h1", hh.columns[40]] = np.nan
    idx = pd.Series({"h1": d1, "h2": d2})
    lt = pd.DataFrame({"line_losses_kw": 1.0, "dt_losses_kw": 0.0, "p_head_kw": 0.0, "ltech_kw": 1.0},
                      index=pd.DatetimeIndex(ts, name="timestamp"))
    est = N.balance_estimates(frame, hh, idx, lt, lambdas_assumed=(0.0, 0.02, 0.04), offline="neighbour")
    assert est["offline_rule"] == "neighbour" and set(est["eoff_rules"]) == {"median", "neighbour"}
    pd.testing.assert_series_equal(est["bottom_up"], est["ntl_dt"][0.0].sum(axis=1))
    assert est["g_hat"] == pytest.approx(1 / 0.99, rel=1e-9)
    assert est["a_minus_ltech_corrected"].abs().max() < 1e-6
    np.testing.assert_allclose(est["ntl_feeder_corrected"], est["bottom_up"], atol=1e-6)
    with pytest.raises(ValueError):
        N.balance_estimates(frame, hh, idx, None, offline="interpolate")
