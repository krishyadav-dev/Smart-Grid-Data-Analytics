"""config/layer3_ntl_params.yaml (Stage 2 run sheet §4, step D): frozen
detector settings on the fixed grid, tuned on January-June only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.analytics import ntl_engine as N

PARAMS = Path(__file__).resolve().parents[2] / "config" / "layer3_ntl_params.yaml"


def test_frozen_detector_settings():
    if not PARAMS.exists():
        pytest.skip("config/layer3_ntl_params.yaml is written by step D")
    d = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    assert list(d["folds"]) == list(N.FOLDS)
    for f, p in d["folds"].items():
        assert p["C"] in N.GRID_C and p["gamma"] in N.GRID_GAMMA, f
        assert 0.0 <= p["q"] <= 1.0 and 0.0 <= p["q_gated"] <= 1.0, f
    assert d["grid"]["C"] == list(N.GRID_C) and d["grid"]["gamma"] == list(N.GRID_GAMMA)
    assert d["seed"] == N.JOKAR_SEED and d["min_train_days"] == N.MIN_TRAIN_DAYS
    assert d["kmeans"]["k_range"] == list(N.K_RANGE)
    assert d["kmeans"]["min_cluster_share"] == N.MIN_CLUSTER_SHARE
    assert "2024-01-01..2024-06-30" in d["tuned_on"]["window"]
