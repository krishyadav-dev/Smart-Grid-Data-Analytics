"""C3 — golden messages vs the exported JSON schema and the pydantic model."""

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from src.ingestion.schema import (
    SCHEMA_PATH, Telemetry, ValidatedTelemetry, build_json_schema,
)
from tests.ingestion.helpers import valid_doc

GOLDEN = Path(__file__).parent / "golden"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator():
    V = jsonschema.Draft202012Validator
    return V(_schema(), format_checker=V.FORMAT_CHECKER)


def test_exported_schema_is_up_to_date():
    regenerated = json.loads(json.dumps(build_json_schema(), sort_keys=True))
    assert _schema() == regenerated, "run: python -m src.ingestion.schema"


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("valid_*.json")), ids=lambda p: p.stem)
def test_golden_valid(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    _validator().validate(doc)
    Telemetry.model_validate(doc)


def _mutations():
    def drop(k):
        def f(d):
            d.pop(k)
        return f

    def setv(k, v):
        def f(d):
            d[k] = v
        return f

    def meas(k, v):
        def f(d):
            d["measurements"][k] = v
        return f

    return {
        "missing_meter_id": drop("meter_id"),
        "missing_t_sim_key": drop("t_sim"),
        "bad_meter_id": setv("meter_id", "meter5"),
        "seq_negative": setv("seq", -1),
        "seq_string": setv("seq", "zero"),
        "run_id_not_uuid": setv("run_id", "abc"),
        "role_unknown": setv("meter_role", "prosumer"),
        "t_poll_naive": setv("t_poll_utc", "2026-10-08T10:15:00"),
        "extra_top_level": setv("colour", "blue"),
        "phase_d": meas("voltage_v", {"d": 1.0}),
        "voltage_string": meas("voltage_v", {"a": "high"}),
        "unknown_measurement": meas("thd_pct", 3.0),
        "unknown_flag": setv("quality", {"flags": ["WEIRD"], "gap_steps": 0}),
        "gap_negative": setv("quality", {"flags": [], "gap_steps": -1}),
        "version_2": setv("schema_version", "2.0"),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_invalid_samples_fail_both(name):
    doc = valid_doc()
    _mutations()[name](doc)
    with pytest.raises(jsonschema.ValidationError):
        _validator().validate(doc)
    with pytest.raises(ValidationError):
        Telemetry.model_validate(doc)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nan_inf_rejected_by_model(bad):
    raw = json.dumps(valid_doc()).replace("6006.19", bad)
    with pytest.raises(ValidationError):
        Telemetry.model_validate(json.loads(raw))


def test_wire_roundtrip_omits_absent_measurements():
    doc = valid_doc()
    del doc["measurements"]["reactive_power_kvar"]
    m = Telemetry.model_validate(doc)
    wire = json.loads(m.to_wire())
    assert "reactive_power_kvar" not in wire["measurements"]
    assert "frequency_hz" not in wire["measurements"]
    assert wire["t_sim"] == "2024-01-01T00:15:00"
    assert wire["measurements"]["voltage_v"] == doc["measurements"]["voltage_v"]


def test_validated_model_accepts_ingest_block():
    doc = valid_doc()
    doc["ingest"] = {"t_recv_utc": "2026-10-08T10:15:00.2Z", "ingest_seq": 1,
                     "latency_ms": {"poll_to_pub": 1.0, "pub_to_recv": 2.0, "total": 3.0}}
    ValidatedTelemetry.model_validate(doc)
