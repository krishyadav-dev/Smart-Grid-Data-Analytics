"""Theft-engine scenario inputs (Stage 2 run sheet §2, step A)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_source import ntl_inputs as N
from src.data_source import simulation as S


def test_lambda_construction_matches_lambda_sweep():
    rng = np.random.default_rng(0)
    p = pd.DataFrame(rng.uniform(0, 100, size=(96, 3)), columns=["a", "b", "c"])
    g = np.array([0.995, 1.0, 1.004])
    e0 = N.dt_meter_kwh(p, g, 0.0)
    np.testing.assert_allclose(e0.to_numpy(), p.to_numpy() * S.STEP_H * g)
    for lam in N.LAMBDA_TRUE:
        e = N.dt_meter_kwh(p, g, lam)
        # LV loss over the window = lam x the DT's energy, I^2R-shaped (P^2)
        np.testing.assert_allclose(e.sum() / e0.sum(), 1 + lam, rtol=1e-12)
        if lam:
            loss = e.to_numpy() - e0.to_numpy()
            k = loss / (p.to_numpy() ** 2 * g)
            np.testing.assert_allclose(k, np.broadcast_to(k[0], k.shape), rtol=1e-9)


def test_lambda_dirs():
    assert [N.lambda_dir(x).name for x in N.LAMBDA_TRUE] == ["lambda_true_0", "lambda_true_2", "lambda_true_4"]


def test_written_inputs():
    path = N.OUT_DIR / "inputs.json"
    if not path.exists():
        pytest.skip("written by scripts/build_ntl_inputs.py (step A)")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["ideal_meter"]["real_gains_reproduce_meter_frame_metered_exactly"] is True
    assert rec["ideal_meter"]["real_gains_reproduce_household_daily_exactly"] is True
    assert rec["lambda_true"]["zero_matches_metered_frame_max_abs_kwh"] < 1e-9
    for lam in N.LAMBDA_TRUE:
        assert rec["lambda_true"][f"{lam:.2f}"]["lv_loss_share_of_true_dt_energy"] == pytest.approx(lam, abs=1e-12)
    assert rec["consumer_index"]["households"] == 8020
    import pyarrow.parquet as pq
    d = N.OUT_DIR / "ideal_meter"
    assert pq.read_schema(d / "household_daily.parquet").names == ["household", "date", "metered_kwh"]
    f = pd.read_parquet(d / "meter_frame_metered.parquet", columns=["load_id"])
    ids = set(f.load_id.unique())
    assert N.HEAD in ids and len(ids) == 14 and all(i == N.HEAD or i.startswith("dt_") for i in ids)
