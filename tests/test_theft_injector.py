"""
Unit tests for theft_injector — verified against hand-computable synthetic series.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_source.theft_injector import (
    constant_partial_reporting,
    random_reduction,
    reverse_reporting,
    time_of_day_dependent,
    zero_reporting,
)


@pytest.fixture
def constant_series():
    """A synthetic constant series: 100 kWh for 24 timesteps."""
    return pd.Series([100.0] * 24, name="test_consumption")


@pytest.fixture
def datetime_series():
    """Constant series with a DatetimeIndex spanning one day at 1-hour freq."""
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    return pd.Series([100.0] * 24, index=idx, name="test_consumption")


# -------------------------------------------------------------------- #
# constant_partial_reporting                                             #
# -------------------------------------------------------------------- #

class TestConstantPartialReporting:
    def test_output_shape(self, constant_series):
        manip, label, sev = constant_partial_reporting(constant_series, 0.5)
        assert len(manip) == len(constant_series)

    def test_label(self, constant_series):
        _, label, _ = constant_partial_reporting(constant_series, 0.5)
        assert label == "constant_partial_reporting"

    def test_values_half(self, constant_series):
        manip, _, _ = constant_partial_reporting(constant_series, 0.5)
        # 100 * 0.5 = 50 for every element
        assert (manip == 50.0).all()

    def test_severity(self, constant_series):
        _, _, sev = constant_partial_reporting(constant_series, 0.7)
        assert pytest.approx(sev, abs=1e-9) == 0.3

    def test_invalid_fraction(self, constant_series):
        with pytest.raises(ValueError):
            constant_partial_reporting(constant_series, 0.0)
        with pytest.raises(ValueError):
            constant_partial_reporting(constant_series, 1.0)
        with pytest.raises(ValueError):
            constant_partial_reporting(constant_series, -0.1)


# -------------------------------------------------------------------- #
# random_reduction                                                       #
# -------------------------------------------------------------------- #

class TestRandomReduction:
    def test_output_shape(self, constant_series):
        manip, _, _ = random_reduction(constant_series, seed=42)
        assert len(manip) == len(constant_series)

    def test_label(self, constant_series):
        _, label, _ = random_reduction(constant_series, seed=42)
        assert label == "random_reduction"

    def test_values_bounded(self, constant_series):
        manip, _, _ = random_reduction(
            constant_series, min_frac=0.3, max_frac=0.9, seed=42
        )
        # All values should be in [30, 90] for a constant-100 series
        assert (manip >= 30.0 - 1e-9).all()
        assert (manip <= 90.0 + 1e-9).all()

    def test_severity_range(self, constant_series):
        _, _, sev = random_reduction(
            constant_series, min_frac=0.3, max_frac=0.9, seed=42
        )
        # Mean reduction: 1 - mean(uniform(0.3, 0.9)) ≈ 0.4
        assert 0.0 < sev < 1.0

    def test_reproducible(self, constant_series):
        m1, _, _ = random_reduction(constant_series, seed=123)
        m2, _, _ = random_reduction(constant_series, seed=123)
        assert (m1 == m2).all()


# -------------------------------------------------------------------- #
# time_of_day_dependent                                                  #
# -------------------------------------------------------------------- #

class TestTimeOfDayDependent:
    def test_output_shape(self, datetime_series):
        manip, _, _ = time_of_day_dependent(datetime_series)
        assert len(manip) == len(datetime_series)

    def test_label(self, datetime_series):
        _, label, _ = time_of_day_dependent(datetime_series)
        assert label == "time_of_day_dependent"

    def test_peak_reduced_more(self, datetime_series):
        manip, _, _ = time_of_day_dependent(
            datetime_series,
            peak_fraction=0.3,
            offpeak_fraction=0.9,
            peak_hours=(9, 21),
        )
        # Hours 9–20 should be 30, hours 0–8 and 21–23 should be 90
        for i, ts in enumerate(datetime_series.index):
            if 9 <= ts.hour < 21:
                assert pytest.approx(manip.iloc[i]) == 30.0
            else:
                assert pytest.approx(manip.iloc[i]) == 90.0

    def test_requires_datetime_index(self, constant_series):
        with pytest.raises(TypeError):
            time_of_day_dependent(constant_series)


# -------------------------------------------------------------------- #
# reverse_reporting                                                      #
# -------------------------------------------------------------------- #

class TestReverseReporting:
    def test_all_zeros(self, constant_series):
        manip, _, _ = reverse_reporting(constant_series)
        assert (manip == 0.0).all()

    def test_label(self, constant_series):
        _, label, _ = reverse_reporting(constant_series)
        assert label == "reverse_reporting"

    def test_severity(self, constant_series):
        _, _, sev = reverse_reporting(constant_series)
        assert sev == 1.0

    def test_preserves_index(self, datetime_series):
        manip, _, _ = reverse_reporting(datetime_series)
        assert (manip.index == datetime_series.index).all()


# -------------------------------------------------------------------- #
# zero_reporting                                                         #
# -------------------------------------------------------------------- #

class TestZeroReporting:
    def test_output_shape(self, constant_series):
        manip, _, _ = zero_reporting(constant_series, seed=42)
        assert len(manip) == len(constant_series)

    def test_label(self, constant_series):
        _, label, _ = zero_reporting(constant_series, seed=42)
        assert label == "zero_reporting"

    def test_severity_equals_fraction(self, constant_series):
        _, _, sev = zero_reporting(constant_series, zero_fraction=0.4, seed=42)
        assert pytest.approx(sev) == 0.4

    def test_some_zeros(self, constant_series):
        manip, _, _ = zero_reporting(
            constant_series, zero_fraction=0.5, seed=42
        )
        # With 50% zero fraction, some values should be 0 and some 100
        assert (manip == 0.0).any()
        assert (manip == 100.0).any()

    def test_reproducible(self, constant_series):
        m1, _, _ = zero_reporting(constant_series, seed=99)
        m2, _, _ = zero_reporting(constant_series, seed=99)
        assert (m1 == m2).all()
