"""Live fault burst end to end (brief v2 §2D acceptance; X.3.2 item 11/18).

A real fault from :mod:`src.data_source.faults` (sequence-network
superposition on the Stage B network, CP3 relay settings) is turned into 10 ms
bursts for all 23 meter points and served by the real meter server
(``--events``).  Real Mosquitto, gateway and listener run as separate
processes.  The fault is the CP4 event set's **longest burst** (its own
pre-fault state from the CP3 cache).  Every sample must arrive at the
listener; the gateway's reads and wall time for the burst are reported
(contiguous block, no paging).  The 60 s cap (6 050 samples) itself was run
end to end before the arc-resistance follow-up (event 12, progress file).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pandapower as pp
import pytest

from devstack import Recorder, Stack, wait_for
from src.ingestion.config import load_meters

pytestmark = pytest.mark.live

SPEED = 1800.0           # 0.5 s wall per 15-min step
STEPS = 9                # 00:00 .. 02:00
TRIGGER_STEP = 2         # bursts written right after the 00:30 step
EVENT_ID = 4242


def _fault(net, sn, ev):
    from src.data_source import faults as F
    if ev.rf_model == "arc":
        return F.simulate_arc_fault(net, sn, ev.bus, ev.ftype, ev.phases)
    return F.simulate_fault(net, sn, ev.bus, ev.ftype, ev.phases, float(ev.rf_ohm))


def _longest_fault():
    """The CP4 event set's longest burst (all 200 faults scanned, each with its
    own pre-fault state rebuilt from the CP3 cache: DT loads and OLTC tap of
    its step).  A fault the relay does not clear would run to the 60 s cap
    (6 050 samples); with arc resistance on 3PH/LL every fault clears and the
    longest burst is the slowest trip."""
    from src.data_source import faults as F
    from src.data_source.feeder_model import build_network
    net = build_network("B")
    relay = F.relay_settings(F.max_head_current_a())
    events = F.make_event_set(net)
    fl = events[events.kind == "fault"]
    cache = F.steps_from_cache(net, fl.step)
    best = None
    for ev in fl.itertuples():
        F.set_prefault(net, *cache[pd.Timestamp(ev.step)])
        sn = F.build_sequence_network(net)
        res = _fault(net, sn, ev)
        t_rel, element = F.relay_operate(relay, float(res.i_head_fault_a.max()), res.i_head_residual_a)
        t_clear = t_rel + relay.breaker_s
        n = sum(F.burst_lengths(t_clear))
        if best is None or n > best[0]:
            best = (n, ev, t_clear, element)
    n, ev, t_clear, element = best
    F.set_prefault(net, *cache[pd.Timestamp(ev.step)])        # leave net in that event's state
    sn = F.build_sequence_network(net)
    res = _fault(net, sn, ev)
    return net, sn, (n, ev.bus, ev.ftype, ev.phases, float(res.rf_ohm), res, t_clear, element,
                     int(ev.event_id), str(ev.step))


@pytest.fixture(scope="module")
def burst():
    from run_meter_server import build_frame
    from src.data_source import faults as F
    from src.data_source import meter_server as ms
    net, sn, (n, bus, ftype, ph, rf, res, t_clear, element, ev_id, step) = _longest_fault()
    meters = F.meter_faults(net, sn, res)
    frame = build_frame("2024-01-01 00:00", "2024-01-01 02:00")
    trigger = sorted(frame.timestamp.unique())[TRIGGER_STEP]
    t_start = pd.Timestamp(trigger) + pd.Timedelta(seconds=120) - pd.Timedelta(seconds=F.PRE_S)
    bursts, expected = [], {}
    for lid, m in meters.items():
        v, i = F.build_burst(m, t_clear)
        bursts.append({"trigger": trigger, "load_id": lid, "event_id": EVENT_ID, "t_start": t_start,
                       "v": v, "i": i, "v_base_ln_v": m.v_base_ln_v, "phases": "abc"})
        expected[lid] = (v, i, m.v_base_ln_v)
    return {"frame": frame, "bursts": bursts, "expected": expected, "n": n, "fault":
            {"event_id": ev_id, "step": step, "bus": bus, "ftype": ftype, "phases": ph,
             "rf_ohm": rf, "t_clear_s": t_clear, "element": element}}


@pytest.fixture(scope="module")
def live(burst):
    from src.data_source import meter_server as ms
    st = Stack()
    rec = None
    try:
        frame_path = st.workdir / "frame.csv"
        burst["frame"].to_csv(frame_path, index=False)
        ev_path = st.workdir / "events.npz"
        ms.save_event_file(ev_path, burst["bursts"])
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        assert rec.connected.wait(10)
        st.start_listener(persistent=True, client_id="sg-listener-event")
        st.start_gateway(poll_interval=0.05, playback_speed=SPEED)
        st.start_meter(SPEED, frame_in=frame_path, start_delay=3.0, events=ev_path)
        n_meters = len(load_meters().meters)

        def raw_events():
            return [m for m in rec.snapshot(suffix="/event") if "/validated/" not in m.topic]

        wait_for(lambda: len(raw_events()) >= n_meters, timeout=600, interval=1.0,
                 what="one burst per meter")
        wait_for(lambda: st.http("/api/v1/stats")["events"]["accepted"] >= n_meters,
                 timeout=60, interval=0.5, what="listener accepted every burst")
        yield {"events": [m.json() for m in raw_events()], "stats": st.http("/api/v1/stats"),
               "gateway_log": st.log_text("gateway"), "stack": st}
    finally:
        if rec:
            rec.close()
        st.close()


def test_every_sample_arrives(burst, live):
    meters = {m.meter_id: m for m in load_meters().meters}
    by_meter = {e["meter_id"]: e for e in live["events"]}
    assert set(by_meter) == set(meters), "one burst per meter point"
    total = 0
    for mid, ev in by_meter.items():
        lid = meters[mid].load_id
        v, i, vb = burst["expected"][lid]
        assert ev["event_id"] == EVENT_ID and ev["sample_interval_ms"] == 10
        assert ev["n_samples"] == len(v) == burst["n"]
        for k, p in enumerate("abc"):
            got_v = np.asarray(ev["samples"]["voltage_v"][p])
            got_i = np.asarray(ev["samples"]["current_a"][p])
            assert len(got_v) == len(got_i) == len(v), (mid, p)
            # quantisation: V in 1e-4 pu of v_base, v_base itself in 0.01 V in the
            # header, output rounded to 1 mV; I in the header's mA/count
            tol_v = 0.5e-4 * vb + np.abs(v[:, k]) * 0.005 / vb + 1e-3
            assert np.all(np.abs(got_v - v[:, k]) <= tol_v + 1e-9), (mid, p)
            scale = max(1, math.ceil(float(np.max(i)) * 1000 / 65_000)) / 1000
            assert np.max(np.abs(got_i - i[:, k])) <= 0.5 * scale + 1e-3, (mid, p)
        total += ev["n_samples"]
    st = live["stats"]["events"]
    assert st["accepted"] == len(meters) and st["samples"] == total
    print(f"\nburst: {burst['fault']}, {burst['n']} samples x {len(meters)} meters = {total} samples, "
          f"all arrived")


def test_gateway_reads_and_wall_time(burst, live):
    """Contiguous block (no paging): reads and wall time per burst, from the gateway log."""
    pat = re.compile(r"(m\d+): event (\d+) read: (\d+) samples, (\d+) reads, ([\d.]+) s")
    rows = [m.groups() for m in pat.finditer(live["gateway_log"])]
    rows = [r for r in rows if int(r[1]) == EVENT_ID]
    assert len(rows) == len(load_meters().meters)
    reads = [int(r[3]) for r in rows]
    secs = [float(r[4]) for r in rows]
    n = burst["n"]
    per_read = (125 // 6) * 6
    expected_reads = 1 + math.ceil(n * 6 / per_read) + 2       # header + samples + header + counter
    assert max(reads) >= expected_reads and min(reads) >= expected_reads
    print(f"\nlongest burst {n} samples ({n * 6} registers, {per_read} per read): "
          f"{expected_reads} reads per meter (no retry); measured reads {min(reads)}-{max(reads)}, "
          f"wall time per meter {min(secs):.3f}-{max(secs):.3f} s (median {np.median(secs):.3f} s), "
          f"sum over {len(rows)} meters {sum(secs):.2f} s")
