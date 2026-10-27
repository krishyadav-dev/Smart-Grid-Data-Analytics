"""Layers 1-2 without the broker (src/offline_pipeline.py): the batch documents
equal what the meter served, within the register quantum, and pass the listener."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics import meters as M
from src.data_source import meter_server as ms
from src.offline_pipeline import flatten_telemetry, run_stream

ROOT = Path(__file__).resolve().parents[1]
BATCH_BURSTS = ROOT / "results" / "layer3" / "fault" / "batch" / "bursts.jsonl.gz"
T = pd.Timestamp("2024-03-11 10:15:00")


def _record(vb, n=180, seed=0):
    rng = np.random.default_rng(seed)
    v = vb * rng.uniform(0.0, 1.2, (n, 3))
    i = rng.uniform(0.0, 900.0, (n, 3))
    return ms.EventRecord(event_id=7, t_start=T + pd.Timedelta(milliseconds=370), sample_interval_ms=10,
                          voltage_v=v, current_a=i, phases="abc", v_base_ln_v=vb)


def test_burst_round_trip():
    phys = M.physical_meters()
    head, dt = phys[0], phys[5]
    recs = {T: [(head.load_id, _record(head.v_base_ln_v, seed=1)), (dt.load_id, _record(dt.v_base_ln_v, seed=2))]}
    tel, bur = run_stream(None, recs, phys)
    assert tel == [] and len(bur) == 2
    for d, (_, rec) in zip(bur, recs[T]):
        assert d["n_samples"] == len(rec.voltage_v) and "ingest" in d
        assert d["t_start"].startswith("2024-03-11T10:15:00.370") and d["t_start"].endswith("+05:30")
        scale_a = max(1, int(np.ceil(rec.current_a.max() * 1000 / 65_000))) / 1000
        for k, p in enumerate("abc"):
            v = np.asarray(d["samples"]["voltage_v"][p])
            i = np.asarray(d["samples"]["current_a"][p])
            assert np.max(np.abs(v - rec.voltage_v[:, k])) <= 0.5e-4 * rec.v_base_ln_v + 1e-3
            assert np.max(np.abs(i - rec.current_a[:, k])) <= scale_a / 2 + 1e-3


def test_telemetry_round_trip():
    phys = M.physical_meters()[:3]
    rng = np.random.default_rng(3)
    rows = []
    for s in range(3):
        for m in phys:
            for k, p in enumerate("abc"):
                rows.append({"timestamp": T + pd.Timedelta(minutes=15 * s), "load_id": m.load_id, "phase": p,
                             "voltage_v": m.v_base_ln_v * rng.uniform(0.95, 1.05),
                             "voltage_angle_deg": -120.0 * k + rng.uniform(-2, 2),
                             "load_current_a": rng.uniform(10, 300), "current_a": 0.0,
                             "active_power_kw": rng.uniform(5, 100), "reactive_power_kvar": rng.uniform(1, 40)})
    frame = pd.DataFrame(rows)
    tel, bur = run_stream(frame, None, phys)
    assert bur == [] and len(tel) == 9
    flat = flatten_telemetry(tel)
    assert set(flat["flags"]) == {""} and (flat.gap_steps == 0).all()
    for d in tel:
        lid = next(m.load_id for m in phys if m.meter_id == d["meter_id"])
        ts = pd.Timestamp(d["t_sim"][:19])
        g = frame[(frame.load_id == lid) & (frame.timestamp == ts)].set_index("phase")
        ph = g.voltage_v * np.exp(1j * np.deg2rad(g.voltage_angle_deg))
        for p in "abc":
            assert abs(d["measurements"]["voltage_v"][p] - g.voltage_v[p]) <= 0.005 + 1e-9
            assert abs(d["measurements"]["current_a"][p] - g.load_current_a[p]) <= 0.0005 + 1e-9
        assert abs(d["measurements"]["voltage_ll_v"]["ab"] - abs(ph["a"] - ph["b"])) <= 0.005 + 1e-9


def test_batch_bursts_file():
    """The generated batch documents (skipped without them): 400 events x 14 physical meters."""
    if not BATCH_BURSTS.exists():
        pytest.skip("results/layer3/fault/batch/bursts.jsonl.gz absent (run_layer3_fault.py docs)")
    with gzip.open(BATCH_BURSTS, "rt", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    assert len(docs) == 400 * 14
    assert {d["meter_id"] for d in docs} == {m.meter_id for m in M.physical_meters()}
    assert all(len(d["samples"]["voltage_v"]["a"]) == d["n_samples"] for d in docs)
