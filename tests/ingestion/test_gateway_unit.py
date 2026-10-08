"""Gateway logic with a fake Modbus reader (C6, C7, C8, buffer, partial reads)."""

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest

from src.ingestion.config import load_gateway_config, load_meters
from src.ingestion.gateway import (
    DropOldestBuffer, Gateway, MeterPoller, ModbusReadError, OutMessage,
)
from src.ingestion.register_map import RegisterMap
from src.ingestion.schema import QualityFlag

EPOCH = datetime(1970, 1, 1)
T0 = datetime(2024, 1, 1, 0, 0)


def _u32(v):
    v = int(round(v)) & 0xFFFFFFFF
    return [v >> 16, v & 0xFFFF]


class FakeMeter:
    """Words laid out like Layer 1 (wire address = offset, see register_map.yaml)."""

    def __init__(self):
        self.ir = [0] * 30
        self.hr = [0] * 2
        self.reads = []
        self.fail_fc = set()
        self.on_read = None          # hook(fc, address, count) before returning
        self.connected = True

    def set_frame(self, t_sim, v_a=6000.0, kw_a=100.0, wh=1000.0):
        self.ir[0:2] = _u32(v_a * 100)
        self.ir[12:14] = _u32(kw_a * 1000)
        self.ir[18:20] = _u32(5000)                 # 50.00 Hz
        self.ir[28:30] = _u32((t_sim - EPOCH).total_seconds())
        self.hr[0:2] = _u32(wh)

    def set_time(self, t_sim):
        self.ir[28:30] = _u32((t_sim - EPOCH).total_seconds())

    async def connect(self):
        pass

    async def read(self, unit_id, fc, address, count):
        self.reads.append((fc, address, count))
        if self.on_read:
            self.on_read(fc, address, count)
        if fc in self.fail_fc:
            raise ModbusReadError(f"fc{fc} failed")
        src = self.ir if fc == 4 else self.hr
        return src[address:address + count]

    def close(self):
        pass


@pytest.fixture
def poller():
    meter = load_meters().by_id()["m04"]           # single phase 'a'
    fake = FakeMeter()
    p = MeterPoller(meter=meter, feeder_id="ieee13_11kv", run_id=uuid.uuid4(),
                    regmap=RegisterMap.load(), reader=fake, expected_step_s=900,
                    torn_read_retries=3)
    return p, fake


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
    assert msg.measurements.voltage_v == {"a": 6000.0}
    assert msg.measurements.active_power_kw == {"a": 100.0, "total": 100.0}
    assert msg.measurements.energy_import_kwh == 1.0
    assert msg.measurements.frequency_hz == 50.0


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
    p = None
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
        # Every read of the fc4 data block (starts at 0) moves time on.
        if fc == 4 and address == 0:
            state["t"] += timedelta(minutes=15)
            fake.set_time(state["t"])

    fake.on_read = advance
    msg = run(p.poll_once()).published
    assert QualityFlag.TORN_READ in msg.quality.flags
    assert p.torn_retries_total == 3                   # retried torn_read_retries times
    time_reads = [r for r in fake.reads if r == (4, 28, 2)]
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


def test_partial_read_flag_and_missing_energy(poller):
    p, fake = poller
    fake.set_frame(T0)
    fake.fail_fc = {3}
    msg = run(p.poll_once()).published
    assert QualityFlag.PARTIAL_READ in msg.quality.flags
    assert msg.measurements.energy_import_kwh is None
    assert msg.measurements.voltage_v == {"a": 6000.0}


def test_all_blocks_failing_raises(poller):
    p, fake = poller
    fake.set_frame(T0)
    fake.fail_fc = {3, 4}
    with pytest.raises(ModbusReadError):
        run(p.poll_once())


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
    tel = {json.loads(p)["meter_id"] for t, p, q, r in sink if t.endswith("/telemetry")}
    assert tel == {f"m0{i}" for i in range(2, 9)}
    m01 = [json.loads(p) for t, p, q, r in sink if t.endswith("/m01/status")]
    assert m01[-1]["state"] == "unreachable" and "decoder bug" in m01[-1]["last_error"]
    gw_status = [json.loads(p) for t, p, q, r in sink if t.endswith("/gateway/status")]
    assert gw_status[-1]["state"] == "offline"                 # graceful shutdown
    assert all(q == 1 and r for t, p, q, r in sink if t.endswith("/status"))
    assert all(q == 1 and not r for t, p, q, r in sink if t.endswith("/telemetry"))
