"""Layer 3 P2: voltage-unbalance scenarios (src/data_source/unbalance_scenarios.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_source import unbalance_scenarios as U

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unbalance_scenarios"
DTS = [f"dt_{k}" for k in range(13)]


def test_shift_keeps_total_and_moves_fraction():
    rng = np.random.default_rng(1)
    p, q = rng.uniform(10, 100, (13, 3)), rng.uniform(5, 50, (13, 3))
    rows, ph = np.array([0, 4, 7]), np.array([2, 0, 1])
    p2, q2 = U.shift_loads(p, q, rows, ph, 0.3)
    np.testing.assert_allclose(p2.sum(axis=1), p.sum(axis=1))
    np.testing.assert_allclose(q2.sum(axis=1), q.sum(axis=1))
    for r, j in zip(rows, ph):
        others = [k for k in range(3) if k != j]
        np.testing.assert_allclose(p2[r, others], 0.7 * p[r, others])
        assert p2[r, j] == pytest.approx(p[r, j] + 0.3 * p[r, others].sum())
    untouched = np.setdiff1d(np.arange(13), rows)
    np.testing.assert_array_equal(p2[untouched], p[untouched])
    np.testing.assert_array_equal(U.shift_loads(p, q, rows, ph, 0.0)[0], p)


def test_scenarios_seeded_and_complete():
    a = U.make_scenarios(DTS, np.random.default_rng(5))
    b = U.make_scenarios(DTS, np.random.default_rng(5))
    assert [(s.scenario_id, s.shift) for s in a] == [(s.scenario_id, s.shift) for s in b]
    assert a[0].s == 0.0 and a[0].kind == "baseline" and not a[0].shift
    assert len(a) == 1 + (len(U.S_SWEEP) - 1) * (U.N_COMMON + U.N_RANDOM)
    assert sorted({s.s for s in a}) == list(U.S_SWEEP)
    for s in a[1:]:
        assert s.shift and set(s.shift.values()) <= set("abc") and set(s.shift) <= set(DTS)


def test_generated_scenarios():
    """The generated run (skipped without it): checks passed, labels apart from inputs."""
    if not (OUT / "summary.json").exists():
        pytest.skip("results/unbalance_scenarios absent (scripts/run_unbalance_scenarios.py)")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "scenarios.json").read_text(encoding="utf-8"))
    assert summary["passed"] and summary["baseline_check"]["taps_equal"]
    assert "not claims about real feeders" in meta["note"]
    frames = sorted((OUT / "frames").glob("*.parquet"))
    assert [p.stem for p in frames] == sorted(s["scenario_id"] for s in meta["scenarios"])
    f = pd.read_parquet(frames[0])
    assert list(f.columns) == ["timestamp", "load_id", "bus", "phase", "voltage_v", "current_a",
                               "active_power_kw", "reactive_power_kvar", "voltage_angle_deg", "load_current_a"]
    assert not {"vu_true_pct", "exceeds_limit"} & set(f.columns)
    w = pd.read_parquet(OUT / U.WEEKLY_LABELS_FILE)
    assert len(w) == len(frames) * 14
