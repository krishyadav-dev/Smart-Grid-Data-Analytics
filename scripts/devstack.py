"""Process harness for live tests and the benchmark.

Starts the real chain as separate processes — Mosquitto, the Layer 1 meter
server (real ``run_timeseries`` output), the gateway and the listener — on
free localhost ports, plus an in-process MQTT recorder.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def find_mosquitto() -> Optional[str]:
    env = os.environ.get("SG_MOSQUITTO_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("mosquitto")
    if found:
        return found
    for cand in (Path.home() / "homebrew/sbin/mosquitto", Path("/opt/homebrew/sbin/mosquitto"),
                 Path("/usr/local/sbin/mosquitto"), Path("/usr/sbin/mosquitto"),
                 Path(r"C:\Program Files\mosquitto\mosquitto.exe")):
        if cand.exists():
            return str(cand)
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"port {port} not open after {timeout}s")


def wait_for(pred, timeout: float, interval: float = 0.05, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


def stop(proc: Optional[subprocess.Popen], sig: int = signal.SIGTERM, timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32" and sig != signal.SIGTERM:
            proc.kill()
        else:
            proc.send_signal(sig)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def kill_hard(proc: Optional[subprocess.Popen]) -> None:
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


# ===================================================================== #
#  Recorder                                                              #
# ===================================================================== #

@dataclass
class Recorded:
    topic: str
    payload: bytes
    retain: bool
    t_recv: float            # time.time()

    def json(self) -> dict:
        return json.loads(self.payload)


class Recorder:
    """paho-mqtt subscriber in a background thread, recording everything."""

    def __init__(self, port: int, topic: str = "sg/#", client_id: Optional[str] = None,
                 persistent: bool = False):
        import paho.mqtt.client as mqtt

        self.messages: list[Recorded] = []
        self._lock = threading.Lock()
        self.connected = threading.Event()
        self.topic = topic
        self._c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                              client_id=client_id or f"rec-{os.getpid()}-{port}-{time.time_ns()}",
                              clean_session=not persistent)
        self._c.on_connect = self._on_connect
        self._c.on_message = self._on_message
        self._c.on_disconnect = lambda *a, **k: self.connected.clear()
        self._c.reconnect_delay_set(0.1, 0.5)
        self._c.connect_async("127.0.0.1", port, keepalive=10)
        self._c.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe(self.topic, qos=1)
        self.connected.set()

    def _on_message(self, client, userdata, msg):
        with self._lock:
            self.messages.append(Recorded(msg.topic, bytes(msg.payload), bool(msg.retain), time.time()))

    def snapshot(self, suffix: Optional[str] = None, contains: Optional[str] = None) -> list[Recorded]:
        with self._lock:
            out = list(self.messages)
        if suffix:
            out = [m for m in out if m.topic.endswith(suffix)]
        if contains:
            out = [m for m in out if contains in m.topic]
        return out

    def close(self) -> None:
        self._c.loop_stop()
        self._c.disconnect()


def retained(port: int, topic: str, timeout: float = 2.0) -> Optional[dict]:
    """Fetch the retained message on ``topic`` with a fresh connection."""
    rec = Recorder(port, topic)
    try:
        if not rec.connected.wait(5):
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msgs = rec.snapshot()
            if msgs:
                return msgs[-1].json()
            time.sleep(0.02)
        return None
    finally:
        rec.close()


# ===================================================================== #
#  Stack                                                                 #
# ===================================================================== #

@dataclass
class Stack:
    workdir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="sg-stack-")))
    mqtt_port: int = field(default_factory=free_port)
    modbus_port: int = field(default_factory=free_port)
    http_port: int = field(default_factory=free_port)
    broker: Optional[subprocess.Popen] = None
    meter: Optional[subprocess.Popen] = None
    gateway: Optional[subprocess.Popen] = None
    listener: Optional[subprocess.Popen] = None

    def _log(self, name: str):
        return open(self.workdir / f"{name}.log", "ab")

    def log_text(self, name: str) -> str:
        p = self.workdir / f"{name}.log"
        return p.read_text(errors="replace") if p.exists() else ""

    # -- broker -----------------------------------------------------------
    def start_broker(self) -> None:
        exe = find_mosquitto()
        if exe is None:
            raise RuntimeError("mosquitto not found (set SG_MOSQUITTO_BIN)")
        data = self.workdir / "mosquitto-data"
        data.mkdir(exist_ok=True)
        conf = self.workdir / "mosquitto.conf"
        conf.write_text(
            f"listener {self.mqtt_port} 127.0.0.1\nallow_anonymous true\n"
            f"persistence true\npersistence_location {data.as_posix()}/\n"
            "autosave_interval 1\nmax_queued_messages 100000\nmax_inflight_messages 100\n"
            "log_type error\nlog_type warning\nlog_type notice\nconnection_messages true\n",
            encoding="utf-8")
        self.broker = subprocess.Popen([exe, "-c", str(conf)], stdout=self._log("broker"),
                                       stderr=subprocess.STDOUT)
        wait_port(self.mqtt_port, 10)

    # -- meter server -----------------------------------------------------
    def start_meter(self, speed: float, frame_in: Optional[Path] = None,
                    frame_out: Optional[Path] = None, start_delay: float = 0.0,
                    profiles: str = "synthetic") -> None:
        args = [PY, str(ROOT / "scripts" / "run_meter_server.py"), "--port", str(self.modbus_port),
                "--speed", str(speed), "--start-delay", str(start_delay),
                "--profiles", profiles]
        if frame_in:
            args += ["--frame-in", str(frame_in)]
        if frame_out:
            args += ["--frame-out", str(frame_out)]
        self.meter = subprocess.Popen(args, cwd=ROOT, stdout=self._log("meter"),
                                      stderr=subprocess.STDOUT)
        wait_port(self.modbus_port, 180)

    # -- gateway ------------------------------------------------------------
    def start_gateway(self, poll_interval: float = 0.05, playback_speed: float = 900.0,
                      keepalive: int = 10, backoff_max: float = 0.5) -> None:
        cfg = yaml.safe_load((ROOT / "config" / "gateway.yaml").read_text(encoding="utf-8"))
        cfg["modbus"]["port"] = self.modbus_port
        cfg["mqtt"]["port"] = self.mqtt_port
        cfg["mqtt"]["keepalive_s"] = keepalive
        cfg["poll_interval_s"] = poll_interval
        cfg["playback_speed"] = playback_speed
        cfg["backoff"] = {"initial_s": 0.1, "max_s": backoff_max, "jitter": 0.2}
        path = self.workdir / "gateway.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        self.gateway = subprocess.Popen([PY, "-m", "src.ingestion.gateway", "--config", str(path)],
                                        cwd=ROOT, stdout=self._log("gateway"),
                                        stderr=subprocess.STDOUT)

    # -- listener -------------------------------------------------------------
    def start_listener(self, persistent: bool = True, client_id: str = "sg-listener-test",
                       extra_env: Optional[dict] = None) -> None:
        env = dict(os.environ)
        env.update({"SG_MQTT_PORT": str(self.mqtt_port), "SG_MQTT_CLIENT_ID": client_id,
                    "SG_MQTT_PERSISTENT_SESSION": str(persistent).lower(),
                    "SG_BACKOFF_INITIAL_S": "0.1", "SG_BACKOFF_MAX_S": "0.5"})
        env.update(extra_env or {})
        self.listener = subprocess.Popen(
            [PY, "-m", "src.ingestion.listener", "--port", str(self.http_port),
             "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=self._log("listener"), stderr=subprocess.STDOUT)
        wait_port(self.http_port, 30)
        wait_for(lambda: self.http("/health").get("broker_connected"), 15, what="listener broker")

    def http(self, path: str) -> dict:
        import httpx
        return httpx.get(f"http://127.0.0.1:{self.http_port}{path}", timeout=5).json()

    def close(self) -> None:
        for p in (self.gateway, self.listener, self.meter, self.broker):
            stop(p)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
