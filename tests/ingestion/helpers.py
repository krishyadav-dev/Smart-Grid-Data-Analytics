"""Shared builders for Layer 2 tests."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timedelta, timezone

RUN_ID = "3f2b8c1e-5d4a-4e6b-9c7d-0a1b2c3d4e5f"
IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2024, 1, 1, 0, 15, tzinfo=IST)       # schema 2.0: t_sim is IST
NOW = datetime(2026, 10, 8, 10, 15, 0, 123456, tzinfo=timezone.utc)


def utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def valid_doc(meter_id="m05", unit_id=5, load_id="load_671_abc", seq=0,
              t_sim: datetime | None = T0, run_id=RUN_ID, t_poll=NOW, **over) -> dict:
    doc = {
        "schema_version": "2.0",
        "feeder_id": "ieee13_11kv",
        "meter_id": meter_id,
        "unit_id": unit_id,
        "load_id": load_id,
        "meter_role": "bus_virtual",
        "run_id": run_id,
        "seq": seq,
        "t_sim": t_sim.isoformat() if t_sim else None,
        "t_poll_utc": utc(t_poll),
        "t_pub_utc": utc(t_poll + timedelta(milliseconds=2)),
        "measurements": {
            "voltage_v": {"a": 6006.19, "b": 6056.69, "c": 6008.68},
            "voltage_ll_v": {"ab": 10432.1, "bc": 10455.7, "ca": 10398.2},
            "current_a": {"a": 69.5, "b": 34.2, "c": 64.8},
            "active_power_kw": {"a": 154.04, "b": 154.04, "c": 154.04, "total": 462.12},
            "reactive_power_kvar": {"a": 88.02, "b": 88.02, "c": 88.02, "total": 264.06},
            "apparent_power_kva": {"a": 177.41, "b": 177.41, "c": 177.41, "total": 532.23},
            "power_factor": {"a": 0.8683, "b": 0.8683, "c": 0.8683, "total": 0.8683},
            "energy_import_kwh": 1234.567,
        },
        "quality": {"flags": [], "gap_steps": 0},
    }
    for k, v in over.items():
        doc[k] = v
    return doc


def encode(doc: dict) -> bytes:
    return json.dumps(doc).encode()


def deep(doc: dict) -> dict:
    return copy.deepcopy(doc)


def topic_for(doc: dict) -> str:
    return f"sg/v2/{doc.get('feeder_id', 'ieee13_11kv')}/{doc.get('meter_id', 'm05')}/telemetry"


def new_run() -> str:
    return str(uuid.uuid4())
