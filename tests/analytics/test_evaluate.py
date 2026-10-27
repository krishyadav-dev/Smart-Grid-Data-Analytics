"""Evaluator helpers (src/analytics/evaluate.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics import evaluate as V


def test_wilson_plan_illustration():
    w = V.wilson(45, 47)                     # plan §4: 0.957 [0.857, 0.988] (lower bound 0.8575, printed truncated)
    assert w["p"] == pytest.approx(0.957, abs=1e-3)
    assert w["lo"] == pytest.approx(0.857, abs=1e-3) and w["hi"] == pytest.approx(0.988, abs=1e-3)
    assert V.wilson(0, 0)["p"] is None
    assert V.wilson(0, 10)["lo"] == 0.0 and V.wilson(10, 10)["hi"] == 1.0


def test_best_plateau_middle():
    g = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    assert V.best_plateau_middle(g, [1, 3, 3, 3, 2, 3]) == (0.15, [0.10, 0.15, 0.20])
    assert V.best_plateau_middle(g, [3, 3, 1, 1, 3, 3])[0] == 0.05          # tie: lowest run, lower middle
    assert V.best_plateau_middle(g, [1, 1, 1, 1, 1, 1])[0] == 0.15          # 6 values: lower middle
    assert V.best_plateau_middle(g, [0, 0, 0, 5, 0, 0])[0] == 0.20


def test_match_onsets():
    lab = pd.DataFrame({"win_start_ms": [100, 1000], "win_end_ms": [200, 1500]}, index=[7, 9])
    assert V.match_onsets([99, 100, 150, 200, 201, 1000, 1600], lab) == [None, 7, 7, 7, None, 9, None]


@pytest.mark.parametrize("node,bus,rel", [
    ("652", "652", "same_node"), ("684", "652", "upstream"), ("632", "611", "upstream"),
    ("652", "684", "downstream"), ("675", "652", "wrong_branch"), (None, "652", "no_dt_detection"),
])
def test_error_direction(node, bus, rel):
    assert V._rel(node, bus) == rel


def test_grids_match_run_sheet():
    assert V.GRID_RHO[0] == 1.05 and V.GRID_RHO[-1] == 3.0 and len(V.GRID_RHO) == 40
    assert V.GRID_WINDOW == tuple(range(1, 11))
    assert V.GRID_TOL[0] == 0.05 and V.GRID_TOL[-1] == 0.5 and len(V.GRID_TOL) == 10
    assert V.GRID_DELTA[0] == 0.25 and V.GRID_DELTA[-1] == 3.0 and len(V.GRID_DELTA) == 12
