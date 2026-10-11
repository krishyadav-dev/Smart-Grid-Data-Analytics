"""Live resilience: C4 (meter server / broker restart), C5 (LWT), D7 (listener
restart with a persistent session).  Measured times are printed (run with -s)
and recorded in docs/build_report_*.md."""

from __future__ import annotations

import signal
import time
from collections import defaultdict

import pytest

from devstack import Recorder, Stack, kill_hard, retained, stop, wait_for

pytestmark = pytest.mark.live

FEEDER = "ieee13_11kv"


@pytest.fixture(scope="module")
def frame_csv(tmp_path_factory):
    """Real run_timeseries output for one day, written once."""
    from src.data_source.feeder_model import run_timeseries

    path = tmp_path_factory.mktemp("frame") / "frame.csv"
    run_timeseries(start="2024-01-01 00:00", end="2024-01-01 23:45").to_csv(path, index=False)
    return path


def _status(port, meter):
    return retained(port, f"sg/v1/{FEEDER}/{meter}/status") or {}


def _gw_status(port):
    return retained(port, f"sg/v1/{FEEDER}/gateway/status") or {}


def _count(rec, meter="m01"):
    return len(rec.snapshot(suffix=f"/{meter}/telemetry"))


# --------------------------------------------------------------------------
def test_c4_meter_server_restart(frame_csv):
    st = Stack()
    rec = None
    try:
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        assert rec.connected.wait(10)
        st.start_gateway(poll_interval=0.05, playback_speed=900)
        st.start_meter(900, frame_in=frame_csv, start_delay=1.0)
        wait_for(lambda: _count(rec) >= 3, 30, what="telemetry")
        assert _status(st.mqtt_port, "m01").get("state") == "reachable"

        t_kill = time.monotonic()
        stop(st.meter)
        wait_for(lambda: _status(st.mqtt_port, "m01").get("state") == "unreachable", 15,
                 interval=0.1, what="m01 unreachable")
        t_unreach = time.monotonic() - t_kill
        assert st.gateway.poll() is None, "gateway crashed"
        gw = _gw_status(st.mqtt_port)
        assert gw["state"] == "degraded" and "m01" in gw["unreachable_meters"]
        run_id = gw["run_id"]

        n_before = _count(rec)
        t_restart = time.monotonic()
        st.start_meter(900, frame_in=frame_csv)
        wait_for(lambda: _count(rec) > n_before, 30, what="publishing resumes")
        t_resume = time.monotonic() - t_restart
        wait_for(lambda: _status(st.mqtt_port, "m01").get("state") == "reachable", 10,
                 what="m01 reachable")
        assert st.gateway.poll() is None
        gw = _gw_status(st.mqtt_port)
        assert gw["state"] == "online" and gw["run_id"] == run_id    # no gateway restart
        print(f"\nC4 meter restart: unreachable after {t_unreach:.2f}s, "
              f"publishing resumed {t_resume:.2f}s after server restart")
    finally:
        if rec:
            rec.close()
        st.close()


def test_c4_broker_restart(frame_csv):
    st = Stack()
    rec = None
    try:
        st.start_broker()
        # persistent recorder session survives the broker restart (persistence on)
        rec = Recorder(st.mqtt_port, client_id="rec-c4-broker", persistent=True)
        assert rec.connected.wait(10)
        st.start_gateway(poll_interval=0.05, playback_speed=900)
        st.start_meter(900, frame_in=frame_csv, start_delay=1.0)
        wait_for(lambda: _count(rec) >= 3, 30, what="telemetry")
        run_id = _gw_status(st.mqtt_port)["run_id"]

        stop(st.broker)
        down_s = 4.0
        time.sleep(down_s)                       # ~4 simulated steps while broker is down
        assert st.gateway.poll() is None, "gateway crashed while broker down"
        n_before = _count(rec)
        t_up = time.monotonic()
        st.start_broker()
        wait_for(lambda: _count(rec) >= n_before + 4, 30, what="buffered + new telemetry")
        t_resume = time.monotonic() - t_up
        time.sleep(1.5)
        assert st.gateway.poll() is None
        gw = _gw_status(st.mqtt_port)
        assert gw["state"] == "online" and gw["run_id"] == run_id

        per = defaultdict(list)
        for m in rec.snapshot(suffix="/telemetry"):
            d = m.json()
            per[d["meter_id"]].append(d["seq"])
        for mid, seqs in per.items():
            uniq = sorted(set(seqs))
            assert uniq == list(range(len(uniq))), f"{mid} lost seq across broker restart: {uniq}"
        print(f"\nC4 broker restart: broker down {down_s:.0f}s, telemetry flowing "
              f"{t_resume:.2f}s after broker restart, no seq lost "
              f"({sum(len(set(v)) for v in per.values())} msgs)")
    finally:
        if rec:
            rec.close()
        st.close()


# --------------------------------------------------------------------------
def test_c5_lwt_on_abrupt_gateway_kill(frame_csv):
    st = Stack()
    try:
        st.start_broker()
        keepalive = 10
        st.start_gateway(poll_interval=0.1, playback_speed=900, keepalive=keepalive)
        st.start_meter(900, frame_in=frame_csv)
        wait_for(lambda: _gw_status(st.mqtt_port).get("state") == "online", 20, what="online")
        rec = Recorder(st.mqtt_port, f"sg/v1/{FEEDER}/gateway/status")
        assert rec.connected.wait(5)
        time.sleep(0.3)
        t0 = time.time()
        kill_hard(st.gateway)                     # SIGKILL: no graceful shutdown
        msgs = wait_for(lambda: [m for m in rec.snapshot() if m.json()["state"] == "offline"],
                        keepalive * 1.5 + 2, interval=0.01, what="LWT offline")
        dt = msgs[0].t_recv - t0
        rec.close()
        assert _gw_status(st.mqtt_port)["state"] == "offline"     # retained
        assert dt <= keepalive * 1.5
        print(f"\nC5: retained 'offline' (LWT) observed {dt * 1000:.0f} ms after SIGKILL "
              f"(keepalive {keepalive}s, window {keepalive * 1.5:.0f}s)")
    finally:
        st.close()


# --------------------------------------------------------------------------
@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "clean"])
def test_d7_listener_restart(frame_csv, persistent):
    st = Stack()
    rec = None
    try:
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        assert rec.connected.wait(10)
        cid = f"sg-listener-d7-{'p' if persistent else 'c'}"
        st.start_listener(persistent=persistent, client_id=cid)
        st.start_gateway(poll_interval=0.05, playback_speed=900)
        st.start_meter(900, frame_in=frame_csv, start_delay=1.0)
        wait_for(lambda: _count(rec) >= 3, 30, what="telemetry")

        stop(st.listener)                          # graceful (SIGTERM)
        n_tel_down = len(rec.snapshot(suffix="/telemetry"))
        time.sleep(5.0)                            # ~5 steps x 8 meters published meanwhile
        n_tel_up = len(rec.snapshot(suffix="/telemetry"))
        st.start_listener(persistent=persistent, client_id=cid)
        time.sleep(3.0)
        stop(st.gateway)
        time.sleep(1.0)

        tel = {(m.json()["meter_id"], m.json()["t_sim"])
               for m in rec.snapshot(suffix="/telemetry")[n_tel_down:n_tel_up]}
        val = {(m.json()["meter_id"], m.json()["t_sim"])
               for m in rec.snapshot(contains="/validated/")}
        delivered = len(tel & val)
        print(f"\nD7 {'persistent' if persistent else 'clean'} session: "
              f"{len(tel)} telemetry msgs published while listener down, "
              f"{delivered} delivered after restart")
        assert len(tel) > 0
        if persistent:
            assert delivered == len(tel)
        else:
            assert delivered == 0
    finally:
        if rec:
            rec.close()
        st.close()
