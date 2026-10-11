"""D1/D2 — every reject code and every flag; anomalies pass through unmodified."""

import json
from datetime import timedelta

import pytest

from src.ingestion.config import ListenerSettings, load_meters
from src.ingestion.listener.service import ListenerService
from src.ingestion.listener.validation import Accepted, Rejected
from src.ingestion.schema import QualityFlag, RejectReason
from tests.ingestion.helpers import NOW, T0, deep, encode, new_run, topic_for, valid_doc

T_RECV = NOW + timedelta(milliseconds=5)


@pytest.fixture
def svc():
    return ListenerService(ListenerSettings(mqtt_enabled=False), load_meters())


def process(svc, doc_or_bytes, topic=None, t_recv=T_RECV):
    if isinstance(doc_or_bytes, dict):
        topic = topic or topic_for(doc_or_bytes)
        payload = encode(doc_or_bytes)
    else:
        payload = doc_or_bytes
        topic = topic or "sg/v1/ieee13_11kv/m05/telemetry"
    return svc.validator.process(topic, payload, t_recv)


# ---------------------------------------------------------------- rejects
def test_reject_malformed_json(svc):
    r = process(svc, b'{"schema_version": "1.0", ')
    assert isinstance(r, Rejected) and r.reason == RejectReason.MALFORMED_JSON
    assert process(svc, b"\xff\xfe").reason == RejectReason.MALFORMED_JSON
    assert process(svc, b"[1,2]").reason == RejectReason.MALFORMED_JSON


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
def test_reject_nan_inf_as_schema_invalid(svc, bad):
    raw = json.dumps(valid_doc()).replace("6006.19", bad).encode()
    assert process(svc, raw).reason == RejectReason.SCHEMA_INVALID


def test_reject_schema_invalid(svc):
    d = valid_doc()
    d["seq"] = -3
    assert process(svc, d).reason == RejectReason.SCHEMA_INVALID


def test_reject_topic_payload_mismatch(svc):
    r = process(svc, valid_doc(), topic="sg/v1/ieee13_11kv/m06/telemetry")
    assert r.reason == RejectReason.SCHEMA_INVALID


def test_reject_unsupported_version(svc):
    assert process(svc, valid_doc(schema_version="1.7")).reason == \
        RejectReason.UNSUPPORTED_SCHEMA_VERSION
    assert process(svc, valid_doc(schema_version="2.0")).reason == \
        RejectReason.UNSUPPORTED_SCHEMA_VERSION


def test_reject_unknown_meter(svc):
    assert process(svc, valid_doc(meter_id="m99", unit_id=99)).reason == RejectReason.UNKNOWN_METER
    # known id but wrong load mapping
    assert process(svc, valid_doc(load_id="load_999_abc")).reason == RejectReason.UNKNOWN_METER


def test_d2_qos1_redelivery_counted_once(svc):
    d = valid_doc()
    first = process(svc, d)
    assert isinstance(first, Accepted)
    for _ in range(3):
        r = process(svc, deep(d))
        assert r.reason == RejectReason.DUPLICATE
    st = svc.validator
    assert st.step_counts["accepted"] == 1
    assert st.reject_counts["DUPLICATE"] == 3


def test_duplicate_by_t_sim_across_runs(svc):
    assert isinstance(process(svc, valid_doc()), Accepted)
    r = process(svc, valid_doc(run_id=new_run()))      # gateway restart re-reads same step
    assert r.reason == RejectReason.DUPLICATE


def test_out_of_order_reject_policy():
    s = ListenerService(ListenerSettings(mqtt_enabled=False, out_of_order_policy="reject"),
                        load_meters())
    assert isinstance(process(s, valid_doc(seq=1, t_sim=T0 + timedelta(minutes=15))), Accepted)
    r = process(s, valid_doc(seq=0, t_sim=T0))
    assert r.reason == RejectReason.OUT_OF_ORDER


# ---------------------------------------------------------------- flags
def _flags(r):
    assert isinstance(r, Accepted), r
    return r.doc["quality"]["flags"]


def test_flag_gap_listener_side(svc):
    _flags(process(svc, valid_doc(seq=0, t_sim=T0)))
    r = process(svc, valid_doc(seq=1, t_sim=T0 + timedelta(minutes=60)))
    assert "GAP" in _flags(r) and r.doc["quality"]["gap_steps"] == 3


def test_flag_gap_from_gateway_preserved(svc):
    d = valid_doc(quality={"flags": ["GAP"], "gap_steps": 2})
    r = process(svc, d)
    assert _flags(r) == ["GAP"] and r.doc["quality"]["gap_steps"] == 2


def test_flag_no_sim_time_passes(svc):
    r = process(svc, valid_doc(t_sim=None, quality={"flags": ["NO_SIM_TIME"], "gap_steps": 0}))
    assert _flags(r) == ["NO_SIM_TIME"]


def test_flag_out_of_order(svc):
    _flags(process(svc, valid_doc(seq=1, t_sim=T0 + timedelta(minutes=15))))
    assert "OUT_OF_ORDER" in _flags(process(svc, valid_doc(seq=0, t_sim=T0)))


def test_flag_stale(svc):
    r = process(svc, valid_doc(), t_recv=NOW + timedelta(seconds=31))
    assert "STALE" in _flags(r)


def test_flag_energy_decrease(svc):
    _flags(process(svc, valid_doc(seq=0, t_sim=T0)))
    d = valid_doc(seq=1, t_sim=T0 + timedelta(minutes=15))
    d["measurements"]["energy_import_kwh"] = 1000.0
    assert "ENERGY_DECREASE" in _flags(process(svc, d))


def test_flag_suspect_range_negative_voltage_current(svc):
    d = valid_doc()
    d["measurements"]["voltage_v"]["a"] = -5.0
    d["measurements"]["current_a"]["b"] = -1.0
    r = process(svc, d)
    assert "SUSPECT_RANGE" in _flags(r)
    assert r.doc["measurements"]["voltage_v"]["a"] == -5.0      # not clipped


def test_negative_active_power_not_suspect(svc):
    d = valid_doc()
    d["measurements"]["active_power_kw"] = {"a": -3.0, "b": 1.0, "c": 1.0, "total": -1.0}
    assert "SUSPECT_RANGE" not in _flags(process(svc, d))


@pytest.mark.parametrize("flag", ["PARTIAL_READ", "TORN_READ"])
def test_gateway_flags_forwarded(svc, flag):
    assert _flags(process(svc, valid_doc(quality={"flags": [flag], "gap_steps": 0}))) == [flag]


def test_every_flag_and_reject_covered():
    """Guard: this module exercises every enum member."""
    import inspect, sys
    src = inspect.getsource(sys.modules[__name__])
    for f in QualityFlag:
        assert f.value in src, f
    for r in RejectReason:
        assert r.name in src, r


# ---------------------------------------------------------------- signal passes through
def test_d1_sag_and_overcurrent_pass_unmodified(svc):
    v_base = 6350.85
    d = valid_doc()
    d["measurements"]["voltage_v"] = {"a": round(0.6 * v_base, 2), "b": 6056.69, "c": 6008.68}
    d["measurements"]["current_a"] = {"a": 3 * 69.5, "b": 34.2, "c": 64.8}
    original = deep(d)
    r = process(svc, d)
    assert isinstance(r, Accepted)
    assert r.doc["quality"]["flags"] == []
    assert r.doc["measurements"] == original["measurements"]
    out = svc._accept(r, T_RECV)
    wire = json.loads(out.payload)
    assert wire["measurements"] == original["measurements"]
    assert wire["measurements"]["voltage_v"]["a"] == 3810.51
    assert wire["measurements"]["current_a"]["a"] == 208.5


def test_handle_routes_rejects_to_dlq(svc):
    out = svc.handle("sg/v1/ieee13_11kv/m05/telemetry", b"not json", T_RECV)
    assert len(out) == 1 and out[0].topic == "sg/v1/ieee13_11kv/dlq" and out[0].qos == 0
    dlq = json.loads(out[0].payload)
    assert dlq["reason"] == "MALFORMED_JSON" and dlq["payload_head"] == "not json"


def test_enrichment_latency_and_ingest_seq(svc):
    out = svc.handle(topic_for(valid_doc()), encode(valid_doc()), T_RECV)
    doc = json.loads(out[0].payload)
    assert out[0].topic == "sg/v1/ieee13_11kv/validated/m05"
    assert doc["ingest"]["ingest_seq"] == 1
    assert doc["ingest"]["latency_ms"] == {"poll_to_pub": 2.0, "pub_to_recv": 3.0, "total": 5.0}
