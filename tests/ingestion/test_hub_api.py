"""D3 (hub) and D4 (REST)."""

import asyncio
import json
import random
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from src.ingestion.config import ListenerSettings
from src.ingestion.listener.app import create_app
from src.ingestion.listener.hub import TelemetryHub
from tests.ingestion.helpers import NOW, T0, encode, topic_for, valid_doc


# ------------------------------------------------------------------ hub
def test_d3_two_consumers_both_receive_and_slow_one_drops():
    async def go():
        hub = TelemetryHub(default_maxsize=10)
        fast_a = hub.subscribe(name="a")
        fast_b = hub.subscribe(name="b")
        slow = hub.subscribe(maxsize=5, name="slow")
        got_a, got_b = [], []

        async def consume(sub, sink, n):
            async for m in sub:
                sink.append(m["i"])
                if len(sink) == n:
                    return

        ta = asyncio.create_task(consume(fast_a, got_a, 100))
        tb = asyncio.create_task(consume(fast_b, got_b, 100))
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for i in range(100):
            hub.publish({"i": i})          # never awaits: cannot block
            await asyncio.sleep(0)
        publish_time = loop.time() - t0
        await asyncio.wait_for(asyncio.gather(ta, tb), 2)
        return got_a, got_b, slow, publish_time

    got_a, got_b, slow, publish_time = asyncio.run(go())
    assert got_a == list(range(100)) and got_b == list(range(100))
    assert slow.dropped == 95 and slow.qsize() == 5
    assert publish_time < 1.0


# ------------------------------------------------------------------ REST
@pytest.fixture
def client():
    app = create_app(ListenerSettings(mqtt_enabled=False))
    with TestClient(app) as c:
        yield c, app.state.service


def feed(svc, doc, t_recv=None):
    return svc.handle(topic_for(doc), encode(doc), t_recv or NOW + timedelta(milliseconds=5))


def test_rest_shapes_and_404(client):
    c, svc = client
    for i in range(3):
        feed(svc, valid_doc(seq=i, t_sim=T0 + timedelta(minutes=15 * i)))
    meters = c.get("/api/v1/meters").json()
    assert len(meters) == 8 and {m["meter_id"] for m in meters} >= {"m01", "m08"}
    m05 = next(m for m in meters if m["meter_id"] == "m05")
    assert m05["messages"] == 3 and m05["last_t_sim"] == "2024-01-01T00:45:00"
    latest = c.get("/api/v1/meters/m05/latest").json()
    assert latest["seq"] == 2 and "ingest" in latest
    hist = c.get("/api/v1/meters/m05/history", params={"limit": 2}).json()
    assert hist["count"] == 2 and [h["seq"] for h in hist["items"]] == [1, 2]
    assert c.get("/api/v1/meters/m99/latest").status_code == 404
    assert c.get("/api/v1/meters/m99/history").status_code == 404
    assert c.get("/api/v1/meters/m01/latest").status_code == 404   # known, no data yet
    st = c.get("/api/v1/stats").json()
    assert st["latency_ms"]["total"]["p50"] == 5.0
    assert st["latency_budget_ms"] is None


def test_health_false_when_broker_down(client):
    c, svc = client
    svc.broker_connected = False
    h = c.get("/health").json()
    assert h["ok"] is False and h["broker_connected"] is False
    svc.broker_connected = True
    svc.handle("sg/v1/ieee13_11kv/gateway/status",
               b'{"state":"online","run_id":null,"t_utc":"2026-10-08T10:00:00Z"}')
    h = c.get("/health").json()
    assert h["ok"] is True and h["gateway_state"] == "online"


def test_d4_stats_match_known_mix(client):
    """100 sent: 90 valid, 5 duplicates, 5 malformed."""
    c, svc = client
    rng = random.Random(1)
    valid = []
    for i in range(90):
        meter = rng.choice([("m05", 5, "load_671_abc"), ("m06", 6, "load_675_abc")])
        valid.append(valid_doc(meter_id=meter[0], unit_id=meter[1], load_id=meter[2],
                               seq=i, t_sim=T0 + timedelta(minutes=15 * i)))
    t_recv = NOW + timedelta(milliseconds=5)
    for d in valid:                                   # 90 valid
        svc.handle(topic_for(d), encode(d), t_recv)
    for d in valid[:5]:                               # 5 QoS-1 style redeliveries
        svc.handle(topic_for(d), encode(d), t_recv)
    for _ in range(5):                                # 5 malformed
        svc.handle("sg/v1/ieee13_11kv/m05/telemetry", b"{oops", t_recv)
    st = c.get("/api/v1/stats").json()
    assert st["received"] == 100
    assert st["accepted"] == 90
    assert st["rejects"]["DUPLICATE"] == 5
    assert st["rejects"]["MALFORMED_JSON"] == 5
    assert st["rejected"] == 10 and st["dlq_published"] == 10
    assert sum(m["accepted_total"] for m in st["per_meter"].values()) == 90
    assert st["steps"]["parse"] == 100 and st["steps"]["model"] == 95
