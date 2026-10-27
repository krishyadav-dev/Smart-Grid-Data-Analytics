"""config/layer3_params.yaml (run sheet Step D): frozen values on the fixed grids,
tuned on the tuning half, readable by the engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.analytics import evaluate as V
from src.analytics import fault_engine as E

PARAMS = Path(__file__).resolve().parents[2] / "config" / "layer3_params.yaml"


def test_frozen_parameters():
    if not PARAMS.exists():
        pytest.skip("config/layer3_params.yaml is written by Step D")
    d = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    p = E.load_params(PARAMS)
    assert p.rho_star in V.GRID_RHO and p.coincidence_samples in V.GRID_WINDOW
    assert p.balance_tol in V.GRID_TOL and p.delta_pp in V.GRID_DELTA
    assert "faults_tune" in d["tuned_on"]["faults"] and "quiet_tune" in d["tuned_on"]["quiet"]
    assert d["order"] == ["rho_star", "coincidence_samples", "balance_tol", "delta_pp"]
