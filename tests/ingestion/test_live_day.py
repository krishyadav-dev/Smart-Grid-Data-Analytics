"""Live, one simulated day through the whole chain (C1, C2, D3, D5).

Real meter server fed by real ``feeder_model.run_timeseries`` output,
real Mosquitto, gateway and listener as separate processes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict

import pandas as pd
import pytest

from devstack import Recorder, Stack, wait_for
from src.ingestion.config import load_meters
from src.ingestion.register_map import RegisterMap

pytestmark = pytest.mark.live

SPEED = 1800.0          # 0.5 s wall per 15-min step -> one day in 48 s
STEPS = 96              # 2024-01-01 00:00 .. 23:45


@pytest.fixture(scope="module")
def day():
    st = Stack()
    rec = None
    try:
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        assert rec.connected.wait(10)
        st.start_listener(persistent=True, client_id="sg-listener-day")
        st.start_gateway(poll_interval=0.05, playback_speed=SPEED)
        frame_path = st.workdir / "frame.csv"
        st.start_meter(SPEED, frame_out=frame_path, start_delay=3.0)
        meters = load_meters().meters

        def complete():
            tel = rec.snapshot(suffix="/telemetry")
            per = defaultdict(int)
            for m in tel:
                per[m.topic] += 1
            return len(per) == len(meters) and min(per.values()) >= STEPS

        wait_for(complete, timeout=STEPS * 0.5 + 90, interval=0.5, what="one simulated day")
        time.sleep(1.5)     # let listener drain
        from dump_registers import dump
        regs = asyncio.run(dump("127.0.0.1", st.modbus_port, list(range(0, 11)), 40))
        out = {
            "frame": pd.read_csv(frame_path, parse_dates=["timestamp"]),
            "telemetry": [m.json() for m in rec.snapshot(suffix="/telemetry")],
            "validated": [m.json() for m in rec.snapshot(contains="/validated/")],
            "dlq": [m.json() for m in rec.snapshot(suffix="/dlq")],
            "stats": st.http("/api/v1/stats"),
            "regs": regs,
            "stack": st,
        }
        yield out
    finally:
        if rec:
            rec.close()
        st.close()


# --------------------------------------------------------------------------
# C1 — decoded values equal the run_timeseries frame within quantisation
# --------------------------------------------------------------------------
TOL = {"voltage_v": 0.005 + 1e-9, "current_a": 0.0005 + 1e-9,
       "active_power_kw": 0.0005 + 1e-9, "reactive_power_kvar": 0.0005 + 1e-9}
COLS = {"voltage_v": "voltage_v", "current_a": "current_a",
        "active_power_kw": "active_power_kw", "reactive_power_kvar": "reactive_power_kvar"}


def test_c1_values_match_frame(day):
    frame = day["frame"]
    meters = {m.meter_id: m for m in load_meters().meters}
    tel = day["telemetry"]
    by_key = {(m["meter_id"], m["t_sim"]): m for m in tel}
    assert len(by_key) == len(tel), "gateway published a t_sim twice"
    worst = defaultdict(float)
    checked = 0
    for mid, mc in meters.items():
        sub = frame[frame.load_id == mc.load_id]
        energy_wh = 0.0
        for ts, rows in sub.groupby("timestamp", sort=True):
            msg = by_key[(mid, ts.isoformat())]
            meas = msg["measurements"]
            p_total = 0.0
            for _, row in rows.iterrows():
                ph = row.phase
                for key, col in COLS.items():
                    err = abs(meas[key][ph] - row[col])
                    worst[key] = max(worst[key], err)
                    assert err <= TOL[key], (mid, ts, key, ph, meas[key][ph], row[col])
                    checked += 1
                p_total += row.active_power_kw
            # absent phases are omitted, never zero-filled
            for key in COLS:
                assert set(meas[key]) - {"total"} == set(mc.phases_present)
            energy_wh += p_total * 1000.0 * 0.25
            assert abs(meas["energy_import_kwh"] - round(energy_wh) / 1000) <= 0.0005 + 1e-9
            assert meas["frequency_hz"] == 50.0
            q_sum = rows.reactive_power_kvar.sum()
            assert abs(meas["reactive_power_kvar"]["total"] - q_sum) <= 0.0005 + 1e-9
            assert msg["quality"]["flags"] == []
    print(f"\nC1: {checked} phase values checked; worst abs error {dict(worst)}")


def test_c1_every_step_published_once_in_order(day):
    per = defaultdict(list)
    for m in day["telemetry"]:
        per[m["meter_id"]].append((m["seq"], m["t_sim"]))
    expected = [t.isoformat() for t in pd.date_range("2024-01-01", periods=STEPS, freq="15min")]
    assert len(per) == 8
    for mid, items in per.items():
        assert [s for s, _ in items] == list(range(len(items))), mid
        assert [t for _, t in items][:STEPS] == expected, mid


# --------------------------------------------------------------------------
# C2 — register map vs live server; meters.yaml vs get_load_points()
# --------------------------------------------------------------------------
def test_c2_register_map_matches_live_server(day):
    rm = RegisterMap.load()
    mapped = rm.mapped_offsets()
    regs = day["regs"]
    units = {m.unit_id for m in load_meters().meters}
    for unit, fcs in regs.items():
        unit = int(unit)
        readable = any(v != "exc" and not str(v).startswith("err") for f in fcs.values() for v in f.values())
        assert readable == (unit in units), f"unit {unit} presence mismatch"
        if unit not in units:
            continue
        for fc_key, words in fcs.items():
            fc = int(fc_key[2:])
            for addr, v in words.items():
                if not isinstance(v, int) or v == 0:
                    continue
                offset = int(addr) - rm.address_base
                assert offset in mapped.get(fc, set()), \
                    f"unit {unit} fc{fc} addr {addr}=0x{v:04x} not in register_map.yaml"
        # every mapped word must be readable
        for fc, offs in mapped.items():
            for off in offs:
                v = fcs[f"fc{fc}"][rm.wire_address(off)]
                assert isinstance(v, int), (unit, fc, off, v)


def test_c2_meters_yaml_matches_live_phases(day):
    """Voltage registers are non-zero exactly on phases_present."""
    rm = RegisterMap.load()
    vregs = {r.phase: r for r in rm.registers if r.measurement == "voltage_v"}
    for m in load_meters().meters:
        words = day["regs"][m.unit_id]["fc4"]
        live = {ph for ph, r in vregs.items()
                if any(words[rm.wire_address(o)] for o in range(r.address, r.address + r.width_registers))}
        assert live == set(m.phases_present), m.meter_id


# --------------------------------------------------------------------------
# D3 / D5 — validated stream, accounting, ordering, zero unexplained loss
# --------------------------------------------------------------------------
def test_d3_validated_topic_receives_accepted(day):
    assert len(day["validated"]) == day["stats"]["accepted"] > 0
    assert all("ingest" in v for v in day["validated"])


def test_d5_end_to_end_accounting(day):
    s = day["stats"]
    published = len(day["telemetry"])
    known_rejects = sum(s["rejects"].values())
    assert s["received"] == published
    assert s["accepted"] == published - known_rejects
    assert len(day["dlq"]) == known_rejects
    # every (meter, t_sim) of the simulated day delivered exactly once
    frame = day["frame"]
    meters = load_meters().meters
    expected = {(m.meter_id, t.isoformat()) for m in meters
                for t in frame[frame.load_id == m.load_id].timestamp.unique()}
    got = [(v["meter_id"], v["t_sim"]) for v in day["validated"]]
    assert len(got) == len(set(got))
    assert set(got) == expected, f"missing {sorted(expected - set(got))[:5]}"


def test_d5_per_meter_order_preserved(day):
    per = defaultdict(list)
    for v in day["validated"]:
        per[v["meter_id"]].append(v)
    for mid, items in per.items():
        ts = [v["t_sim"] for v in items]
        assert ts == sorted(ts), mid
        seqs = [v["ingest"]["ingest_seq"] for v in items]
        assert seqs == sorted(seqs), mid


def test_live_latency_reported(day):
    lat = day["stats"]["latency_ms"]["total"]
    print(f"\nlive latency total ms: {json.dumps(lat)}")
    assert lat["n"] == day["stats"]["accepted"]
