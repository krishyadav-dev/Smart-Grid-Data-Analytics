"""Gateway logic with a fake Modbus reader (C6, C7, C8, buffer, partial reads,
fault-burst events) — register map v2 / schema 2.0."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.ingestion.config import load_gateway_config, load_meters
from src.ingestion.gateway import (
    DropOldestBuffer, Gateway, MeterPoller, ModbusReadError, OutMessage,
)
from src.ingestion.register_map import RegisterMap
from src.ingestion.schema import QualityFlag

IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2024, 1, 1, 0, 0, tzinfo=IST)
REGMAP = RegisterMap.load()
ADDR = {r.name: r.address for r in REGMAP.registers}
T_ADDR = ADDR["sim_time"]


def _u32(v):
    v = int(round(v)) & 0xFFFFFFFF
    return [v >> 16, v & 0xFFFF]


def _u64(v):
    v = int(round(v))
    return [(v >> s) & 0xFFFF for s in (48, 32, 16, 0)]


class FakeMeter:
    """Words laid out like the v2 meter server (wire address = offset)."""

    def __init__(self):
        self.ir = [0] * 37020
        self.hr = [0]
        self.reads = []
        self.fail_fc = set()
        self.fail_addr = set()
        self.on_read = None          # hook(fc, address, count) before returning
        self.connected = True

    def set_frame(self, t_sim, v_a=6000.0, kw_a=100.0, wh=1000.0):
        self.ir[ADDR["voltage_a"]:ADDR["voltage_a"] + 2] = _u32(v_a * 100)
        self.ir[ADDR["power_a"]:ADDR["power_a"] + 2] = _u32(kw_a * 1000)
        self.ir[ADDR["frequency"]:ADDR["frequency"] + 2] = _u32(5000)
        self.ir[ADDR["energy_import"]:ADDR["energy_import"] + 4] = _u64(wh)
        self.set_time(t_sim)

    def set_time(self, t_sim):
        self.ir[T_ADDR:T_ADDR + 2] = _u32(t_sim.timestamp())

    def set_event(self, words, counter):
        from src.data_source.meter_server import EVENT_HEADER
        self.ir[EVENT_HEADER:EVENT_HEADER + len(words)] = words
        a = ADDR["event_counter"]
        self.ir[a:a + 2] = _u32(counter)

    async def connect(self):
        pass

    async def read(self, unit_id, fc, address, count):
        self.reads.append((fc, address, count))
        if self.on_read:
            self.on_read(fc, address, count)
        if fc in self.fail_fc or address in self.fail_addr:
            raise ModbusReadError(f"fc{fc}@{address} failed")
        src = self.ir if fc == 4 else self.hr
        return src[address:address + count]

    def close(self):
        pass


def _single_phase_meter():
    """Test-only single-phase meter config (phase 'a').  Since brief v2 X.3 every
    bus_virtual meter in meters.yaml is three-phase, so the single-phase gateway
    paths are exercised with this copy of m04 instead of the inventory entry."""
    return load_meters().by_id()["m04"].model_copy(update={"phases_present": ["a"]})


def _poller(meter_id=None, regmap=REGMAP):
    meter = _single_phase_meter() if meter_id is None else load_meters().by_id()[meter_id]
    fake = FakeMeter()
    p = MeterPoller(meter=meter, feeder_id="ieee13_11kv", run_id=uuid.uuid4(),
                    regmap=regmap, reader=fake, expected_step_s=900,
                    torn_read_retries=3)
    return p, fake


@pytest.fixture
def poller():
    return _poller()                     # test-only single-phase 'a' meter (m04 copy)


def run(coro):
    return asyncio.run(coro)


def test_c6_repeated_polls_publish_once(poller):
    p, fake = poller
    fake.set_frame(T0)
    outs = [run(p.poll_once()) for _ in range(5)]
    published = [o.published for o in outs if o.published]
    assert len(published) == 1
    assert p.skipped_same_t_sim == 4
    msg = published[0]
    assert msg.t_sim == T0 and msg.quality.flags == [] and msg.quality.gap_steps == 0
    assert msg.t_sim.isoformat() == "2024-01-01T00:00:00+05:30"
    assert msg.measurements.voltage_v == {"a": 6000.0}
    assert msg.measurements.voltage_ll_v is None            # single-phase meter
    assert msg.measurements.active_power_kw == {"a": 100.0, "total": 100.0}
    assert msg.measurements.energy_import_kwh == 1.0
    assert msg.measurements.frequency_hz == 50.0


def test_negative_power_decoded(poller):
    p, fake = poller
    fake.set_frame(T0, kw_a=-50.0)
    msg = run(p.poll_once()).published
    assert msg.measurements.active_power_kw == {"a": -50.0, "total": -50.0}


def test_c6_skipped_step_sets_gap(poller):
    p, fake = poller
    fake.set_frame(T0)
    run(p.poll_once())
    fake.set_frame(T0 + timedelta(minutes=15))
    assert run(p.poll_once()).published.quality.gap_steps == 0
    fake.set_frame(T0 + timedelta(minutes=60))          # missed 30 and 45
    msg = run(p.poll_once()).published
    assert QualityFlag.GAP in msg.quality.flags
    assert msg.quality.gap_steps == 2


def test_no_publish_before_first_frame(poller):
    p, fake = poller            # all registers 0 -> t_sim register 0
    out = run(p.poll_once())
    assert out.skipped and out.published is None and p.seq == 0


def test_c7_seq_monotonic_and_new_run_id_on_restart():
    meters = load_meters()
    cfg = load_gateway_config()
    seqs = []
    for restart in range(2):
        gw = Gateway(cfg)
        fake = FakeMeter()
        p = MeterPoller(meter=meters.by_id()["m05"], feeder_id=cfg.feeder_id,
                        run_id=gw.run_id, regmap=gw.regmap, reader=fake)
        run_seqs = []
        for i in range(4):
            fake.set_frame(T0 + timedelta(minutes=15 * i))
            m = run(p.poll_once()).published
            assert m.run_id == gw.run_id
            run_seqs.append(m.seq)
        seqs.append((gw.run_id, run_seqs))
    assert seqs[0][1] == [0, 1, 2, 3] and seqs[1][1] == [0, 1, 2, 3]
    assert seqs[0][0] != seqs[1][0]


def test_c8_torn_read_retried_then_flagged(poller):
    p, fake = poller
    fake.set_frame(T0)
    state = {"t": T0}

    def advance(fc, address, count):
        # Every read of the polled data block (starts at 0) moves time on.
        if fc == 4 and address == 0:
            state["t"] += timedelta(minutes=15)
            fake.set_time(state["t"])

    fake.on_read = advance
    msg = run(p.poll_once()).published
    assert QualityFlag.TORN_READ in msg.quality.flags
    assert p.torn_retries_total == 3                   # retried torn_read_retries times
    time_reads = [r for r in fake.reads if r == (4, T_ADDR, 2)]
    assert len(time_reads) == 2 * 4                    # start+end per attempt, 4 attempts


def test_torn_read_recovers_on_retry(poller):
    p, fake = poller
    fake.set_frame(T0)
    calls = {"n": 0}

    def once(fc, address, count):
        if fc == 4 and address == 0:
            calls["n"] += 1
            if calls["n"] == 1:
                fake.set_time(T0 + timedelta(minutes=15))

    fake.on_read = once
    msg = run(p.poll_once()).published
    assert QualityFlag.TORN_READ not in msg.quality.flags
    assert p.torn_retries_total == 1


def test_partial_read_flag_and_missing_energy():
    # v2 polls one fc4 block; split it (max 32 words per read) and fail the
    # second block, which holds the energy, time and event counter.
    small = RegisterMap.model_validate({**REGMAP.model_dump(), "max_registers_per_read": 32})
    p, fake = _poller(regmap=small)
    fake.set_frame(T0)
    assert [(b.start, b.count) for b in small.blocks()] == [(0, 32), (32, 26)]
    fake.fail_addr = {32}
    msg = run(p.poll_once()).published
    assert QualityFlag.PARTIAL_READ in msg.quality.flags
    assert QualityFlag.NO_SIM_TIME in msg.quality.flags     # time was in the failed block
    assert msg.measurements.energy_import_kwh is None
    assert msg.measurements.voltage_v == {"a": 6000.0}


def test_all_blocks_failing_raises(poller):
    p, fake = poller
    fake.set_frame(T0)
    fake.fail_fc = {3, 4}
    with pytest.raises(ModbusReadError):
        run(p.poll_once())


# ---------------------------------------------------------------------------
# Fault-burst events
# ---------------------------------------------------------------------------

def _event_words(event_id=1, n=300, phases="abc"):
    from src.data_source.meter_server import EventRecord, build_event_registers
    v = np.tile([5500.0, 6300.0, 6200.0], (n, 1))
    i = np.tile([1500.0, 90.0, 80.0], (n, 1))
    rec = EventRecord(event_id=event_id, t_start="2024-01-01 10:00:00.500",
                      sample_interval_ms=10, voltage_v=v, current_a=i,
                      phases=phases, v_base_ln_v=6350.85)
    return build_event_registers(rec)


def test_event_read_on_counter_change():
    p, fake = _poller("m05")                         # three-phase meter
    fake.set_frame(T0)
    assert run(p.poll_once()).event is None          # counter 0: no event
    fake.set_event(_event_words(n=300), counter=1)
    fake.set_time(T0 + timedelta(minutes=15))
    out = run(p.poll_once())
    ev = out.event
    assert ev is not None and ev.event_id == 1 and ev.n_samples == 300
    assert ev.t_start.isoformat() == "2024-01-01T10:00:00.500000+05:30"
    assert ev.sample_interval_ms == 10
    assert len(ev.samples.voltage_v["a"]) == 300
    assert ev.samples.voltage_v["a"][0] == pytest.approx(5500.0, abs=1.0)
    assert ev.samples.current_a["a"][0] == pytest.approx(1500.0, abs=0.1)
    sample_reads = [r for r in fake.reads if r[1] >= 120]
    assert all(c <= 125 for _, _, c in sample_reads)
    assert sum(c for _, _, c in sample_reads) == 300 * 6
    assert run(p.poll_once()).event is None          # same counter: not re-read


def test_event_counter_jump_counts_missed():
    p, fake = _poller("m05")
    fake.set_frame(T0)
    fake.set_event(_event_words(event_id=1, n=20), counter=1)
    run(p.poll_once())
    fake.set_event(_event_words(event_id=4, n=20), counter=4)
    ev = run(p.poll_once()).event
    assert ev.event_id == 4 and p.events_missed == 2 and p.events_fetched == 2


def test_event_overwritten_during_read_is_retried():
    p, fake = _poller("m05")
    fake.set_frame(T0)
    fake.set_event(_event_words(event_id=1, n=50), counter=1)
    state = {"bumped": False}

    def bump(fc, address, count):
        if address >= 120 and not state["bumped"]:
            state["bumped"] = True
            fake.set_event(_event_words(event_id=2, n=50), counter=2)

    fake.on_read = bump
    ev = run(p.poll_once()).event
    # The first pass saw counter 2 at the end: retried, and the stale
    # record is not published under the old counter.
    assert p.torn_retries_total >= 1
    assert ev is None or ev.event_id == 2


def test_buffer_drop_oldest():
    async def go():
        buf = DropOldestBuffer(3)
        for i in range(5):
            buf.put(OutMessage(f"t{i}", b"x", 1))
        assert buf.dropped == 2
        return [(await buf.get()).topic for _ in range(3)]
    assert run(go()) == ["t2", "t3", "t4"]


def test_c10_gateway_does_not_import_layer1():
    import subprocess
    import sys
    code = ("import sys; import src.ingestion.gateway, src.ingestion.listener.app; "
            "bad=[m for m in sys.modules if m.startswith(('pandapower','src.data_source'))]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


class FakeMqtt:
    def __init__(self, sink):
        self.sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def publish(self, topic, payload, qos=0, retain=False):
        self.sink.append((topic, payload, qos, retain))


def test_one_failing_unit_does_not_stop_others():
    import json

    cfg = load_gateway_config(poll_interval_s=0.01)
    cfg.backoff.initial_s = cfg.backoff.max_s = 0.01
    sink = []
    fakes = []

    def reader_factory():
        f = FakeMeter()
        f.set_frame(T0)
        if not fakes:                                  # first meter (m01) is broken
            async def boom(*a, **k):
                raise RuntimeError("decoder bug")
            f.read = boom
        fakes.append(f)
        return f

    gw = Gateway(cfg, reader_factory=reader_factory, mqtt_client_factory=lambda: FakeMqtt(sink))

    async def go():
        task = asyncio.create_task(gw.run())
        await asyncio.sleep(0.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    run(go())
    expected = {m.meter_id for m in load_meters().meters} - {"m01"}
    tel = {json.loads(p)["meter_id"] for t, p, q, r in sink if t.endswith("/telemetry")}
    assert tel == expected
    assert all(t.startswith("sg/v2/") for t, p, q, r in sink)
    m01 = [json.loads(p) for t, p, q, r in sink if t.endswith("/m01/status")]
    assert m01[-1]["state"] == "unreachable" and "decoder bug" in m01[-1]["last_error"]
    gw_status = [json.loads(p) for t, p, q, r in sink if t.endswith("/gateway/status")]
    assert gw_status[-1]["state"] == "offline"                 # graceful shutdown
    assert all(q == 1 and r for t, p, q, r in sink if t.endswith("/status"))
    assert all(q == 1 and not r for t, p, q, r in sink if t.endswith("/telemetry"))
