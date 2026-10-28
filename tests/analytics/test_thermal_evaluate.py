"""Thermal evaluation helpers in src/analytics/evaluate.py (Stage 3 run sheet §3, §5)."""

from __future__ import annotations

import numpy as np

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
