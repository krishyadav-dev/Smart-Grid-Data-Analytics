"""
Tests for dataset_loader — SGCC data loading and ground-truth table.

Acceptance criteria verified:
    B2a. No null feeder_true_kwh values.
    B2b. At least one row per consumer with is_theft_label = True.
    B3.  Seeded determinism.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_source import config


# Use the small fallback dataset for fast testing
SMALL_CSV = config.SGCC_SMALL_FALLBACK_PATH


@pytest.fixture
def sgcc_df():
    """Load the small SGCC dataset if available."""
    from src.data_source.dataset_loader import load_sgcc

    if not SMALL_CSV.exists():
        pytest.skip(f"Small SGCC dataset not found at {SMALL_CSV}")
    return load_sgcc(SMALL_CSV)


@pytest.fixture
def load_points():
    """Return feeder load points DataFrame."""
    from src.data_source.feeder_model import get_load_points
    return get_load_points()


# -------------------------------------------------------------------- #
# load_sgcc                                                              #
# -------------------------------------------------------------------- #

class TestLoadSGCC:
    def test_has_required_columns(self, sgcc_df):
        required = {"consumer_id", "date", "kwh", "is_theft"}
        assert required.issubset(set(sgcc_df.columns))

    def test_no_null_kwh(self, sgcc_df):
        assert sgcc_df["kwh"].notna().all()

    def test_positive_kwh(self, sgcc_df):
        assert (sgcc_df["kwh"] > 0).all()

    def test_has_both_classes(self, sgcc_df):
        assert sgcc_df["is_theft"].any(), "No theft-labeled consumers"
        assert (~sgcc_df["is_theft"]).any(), "No normal consumers"


# -------------------------------------------------------------------- #
# sample_load_profiles                                                   #
# -------------------------------------------------------------------- #

class TestSampleLoadProfiles:
    def test_profiles_not_empty(self, sgcc_df, load_points):
        from src.data_source.dataset_loader import sample_load_profiles

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        assert len(profiles) > 0

    def test_profiles_has_load_id_columns(self, sgcc_df, load_points):
        """Profiles DataFrame must have one column per load_id."""
        from src.data_source.dataset_loader import sample_load_profiles

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        for load_id in load_points["load_id"]:
            assert load_id in profiles.columns, (
                f"Missing column {load_id} in profiles DataFrame"
            )

    def test_profiles_has_datetime_index(self, sgcc_df, load_points):
        """Profiles DataFrame must have a DatetimeIndex."""
        from src.data_source.dataset_loader import sample_load_profiles

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        assert isinstance(profiles.index, pd.DatetimeIndex)

    def test_profiles_factors_positive(self, sgcc_df, load_points):
        """All scaling factors must be non-negative."""
        from src.data_source.dataset_loader import sample_load_profiles

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        assert (profiles >= 0).all().all()

    def test_profiles_seeded_determinism(self, sgcc_df, load_points):
        """Same seed must produce identical profiles."""
        from src.data_source.dataset_loader import sample_load_profiles

        p1 = sample_load_profiles(load_points, sgcc_df, seed=99)
        p2 = sample_load_profiles(load_points, sgcc_df, seed=99)
        pd.testing.assert_frame_equal(p1, p2)


# -------------------------------------------------------------------- #
# assign_theft_labels                                                    #
# -------------------------------------------------------------------- #

class TestAssignTheftLabels:
    def test_theft_labels_present(self, sgcc_df, load_points):
        """B2b — At least one consumer has is_theft_label = True."""
        from src.data_source.dataset_loader import (
            sample_load_profiles,
            assign_theft_labels,
        )

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        labelled = assign_theft_labels(
            profiles, sgcc_df, load_points, theft_fraction=0.3, seed=42
        )

        assert labelled["is_theft_label"].any(), (
            "No theft labels assigned — theft class did not survive sampling"
        )

    def test_no_null_feeder_true_kwh(self, sgcc_df, load_points):
        """B2a — Acceptance criterion: no null feeder_true_kwh."""
        from src.data_source.dataset_loader import (
            sample_load_profiles,
            assign_theft_labels,
        )

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        labelled = assign_theft_labels(profiles, sgcc_df, load_points, seed=42)

        assert labelled["feeder_true_kwh"].notna().all(), (
            "Found null feeder_true_kwh values"
        )

    def test_output_schema(self, sgcc_df, load_points):
        """Ground truth table has the required stable schema."""
        from src.data_source.dataset_loader import (
            sample_load_profiles,
            assign_theft_labels,
        )

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        labelled = assign_theft_labels(profiles, sgcc_df, load_points, seed=42)

        expected_cols = {
            "consumer_id", "timestamp", "reported_kwh",
            "feeder_true_kwh", "is_theft_label", "source",
        }
        assert expected_cols.issubset(set(labelled.columns))

    def test_is_theft_label_dtype(self, sgcc_df, load_points):
        """is_theft_label must be boolean."""
        from src.data_source.dataset_loader import (
            sample_load_profiles,
            assign_theft_labels,
        )

        profiles = sample_load_profiles(load_points, sgcc_df, seed=42)
        labelled = assign_theft_labels(profiles, sgcc_df, load_points, seed=42)

        assert labelled["is_theft_label"].dtype == bool

    def test_seeded_determinism(self, sgcc_df, load_points):
        """Same seed must produce identical ground-truth tables."""
        from src.data_source.dataset_loader import (
            sample_load_profiles,
            assign_theft_labels,
        )

        profiles = sample_load_profiles(load_points, sgcc_df, seed=7)
        t1 = assign_theft_labels(profiles, sgcc_df, load_points, seed=7)
        t2 = assign_theft_labels(profiles, sgcc_df, load_points, seed=7)

        pd.testing.assert_frame_equal(
            t1.reset_index(drop=True), t2.reset_index(drop=True)
        )
