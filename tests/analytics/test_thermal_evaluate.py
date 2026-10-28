"""Thermal evaluation helpers in src/analytics/evaluate.py (Stage 3 run sheet §3, §5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics import evaluate as V
from src.analytics import thermal as T
from src.data_source.config import YEAR_CACHE_DIR


def test_true_current_check_is_zero_on_the_true_run():
    """Fed the true-current trajectory itself, the check reports no difference."""
    _, i_true = T.read_dt_currents(YEAR_CACHE_DIR / V.TRUE_FRAME.name)
    ta = T.read_ambient().to_numpy()
    p = T.dt_params(T.read_dts(YEAR_CACHE_DIR / "dts.parquet"), "main")
    r = T.reference_run(p, T.load_factor(i_true, T.rated_current()), ta)
    c = V.thermal_true_current_check(r["theta_h_end"], r["theta_h_max"], ta)
    assert c["max_abs_diff_theta_h_k"] == 0.0 and c["max_abs_diff_step_peak_k"] == 0.0
    assert c["true_peak_theta_h_c"] == float(r["theta_h_max"][96:].max())
    assert np.isfinite(c["true_peak_rise_k"]["value"])


def test_hot_spot_errors_horizons():
    ref = np.zeros((4, 96, 2))
    th = np.zeros((4, 96, 2))
    th[:, 3, 0] = 2.0                                   # 1 h, DT 0 only
    th[:, 95, :] = -1.0                                 # 24 h, both DTs
    e = V.hot_spot_errors(th, ref, ["a", "b"])
    assert e["pooled"]["1h"]["rmse"] == pytest.approx(np.sqrt(2.0)) and e["pooled"]["1h"]["mae"] == 1.0
    assert e["pooled"]["6h"]["rmse"] == 0.0 and e["pooled"]["24h"]["mae"] == 1.0
    assert e["per_dt"]["a"]["1h"]["rmse"] == 2.0 and e["per_dt"]["b"]["1h"]["rmse"] == 0.0
    assert e["pooled"]["all_96"]["n"] == 4 * 96 * 2


def test_event_metrics():
    nan = np.nan
    ref = np.full((12, 1), nan)
    ref[:10, 0] = 600.0                                  # 10 event DT-days, first crossing 10:00
    fc = np.full((12, 1), nan)
    fc[:6, 0] = 630.0                                    # 6 detected, 30 min late
    fc[11, 0] = 500.0                                    # one forecast-only day
    m = V.event_metrics(fc, ref)
    assert (m["event_dt_days"], m["detected"], m["forecast_only_dt_days"]) == (10, 6, 1)
    assert m["lead_time_h"]["mean"] == 10.0 and m["crossing_time_error_min"]["mean"] == 30.0
    assert m["recall"]["p"] == pytest.approx(0.6) and m["precision"]["p"] == pytest.approx(6 / 7)
    few = V.event_metrics(fc[:5], ref[:5])
    assert "precision" not in few and "precision_recall" in few
    none = V.event_metrics(np.full((3, 1), nan), np.full((3, 1), nan))
    assert none["event_dt_days"] == 0 and none["lead_time_h"] is None


def test_lol_errors():
    e = V.lol_errors(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert e["mean_h"] == 1.5 and e["median_h"] == 1.5 and e["n_dt_days"] == 4


def test_paired_bootstrap():
    rng = np.random.default_rng(3)
    ref = rng.normal(60, 5, (40, 96, 3))
    lol_ref = rng.uniform(0, 1, (40, 3))
    th = {"lstm": ref + rng.normal(0, 0.1, ref.shape), "persistence": ref + rng.normal(0, 1.0, ref.shape),
          "seasonal_naive": ref + rng.normal(0, 0.1, ref.shape)}
    lol = {"lstm": lol_ref + 0.01, "persistence": lol_ref + 0.2, "seasonal_naive": lol_ref + 0.01}
    b = V.paired_bootstrap(th, ref, lol, lol_ref, n=200)
    assert b["rmse_persistence_minus_lstm_k"]["lstm_better"]
    assert b["rmse_persistence_minus_lstm_k"]["ci95"][0] > 0.8
    assert not b["rmse_seasonal_naive_minus_lstm_k"]["lstm_better"]
    assert b["lol_error_persistence_minus_lstm_h"]["point"] == pytest.approx(0.19)
    same = V.paired_bootstrap({m: th["lstm"] for m in th}, ref, {m: lol["lstm"] for m in lol}, lol_ref, n=50)
    assert same["rmse_persistence_minus_lstm_k"]["ci95"] == [0.0, 0.0]
