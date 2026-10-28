"""P1: IS 17036 unbalance from line-to-line magnitudes (src/analytics/unbalance.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.unbalance import vuf_pct_from_ll

ROOT = Path(__file__).resolve().parents[2]
TRUE_FRAME = ROOT / "results" / "year_2024" / "meter_frame_true.parquet"
TOL_PP = 1e-9                       # percentage points (prompt P1)
A = np.exp(2j * np.pi / 3)


def _vu_from_ll_phasors(vab, vbc, vca):
    """V_N / V_P x 100 % by symmetrical components of the line-to-line phasors."""
    v1 = (vab + A * vbc + A ** 2 * vca) / 3
    v2 = (vab + A ** 2 * vbc + A * vca) / 3
    return 100 * np.abs(v2) / np.abs(v1)


def test_balanced_gives_zero():
    assert vuf_pct_from_ll(415.0, 415.0, 415.0) == 0.0
    v = np.full(5, 11_000.0)
    assert np.all(vuf_pct_from_ll(v, v, v) == 0.0)


def test_matches_symmetrical_components_on_random_sets():
    rng = np.random.default_rng(20261027)
    n = 5000
    mag = 1 + rng.uniform(-0.3, 0.3, (n, 3))
    ang = np.deg2rad(np.array([0, -120, 120]) + rng.uniform(-20, 20, (n, 3)))
    va, vb, vc = (mag * np.exp(1j * ang)).T
    vab, vbc, vca = va - vb, vb - vc, vc - va
    ref = _vu_from_ll_phasors(vab, vbc, vca)
    got = vuf_pct_from_ll(np.abs(vab), np.abs(vbc), np.abs(vca))
    assert np.isfinite(got).all()
    assert ref.max() > 10 and ref.min() < 1             # a real spread of unbalance
    assert np.max(np.abs(got - ref)) <= TOL_PP


def test_matches_full_phasor_on_year_frame():
    """Feeder head and every DT meter over the whole 2024 true frame."""
    if not TRUE_FRAME.exists():
        pytest.skip("results/year_2024/meter_frame_true.parquet absent")
    from src.analytics import meters as M
    ids = [m.load_id for m in M.physical_meters()]
    f = pd.read_parquet(TRUE_FRAME, columns=["timestamp", "load_id", "phase", "voltage_v", "voltage_angle_deg"],
                        filters=[("load_id", "in", ids)])
    f["v"] = f.voltage_v * np.exp(1j * np.deg2rad(f.voltage_angle_deg))
    w = f.pivot(index=["load_id", "timestamp"], columns="phase", values="v")
    assert len(w) == 14 * 35_136
    va, vb, vc = (w[p].to_numpy() for p in "abc")
    vab, vbc, vca = va - vb, vb - vc, vc - va
    ref = _vu_from_ll_phasors(vab, vbc, vca)
    got = vuf_pct_from_ll(np.abs(vab), np.abs(vbc), np.abs(vca))
    assert np.max(np.abs(got - ref)) <= TOL_PP
