"""
Modbus TCP Smart-Meter Server — register map v2 (schema 2.0).

Architecture choice
-------------------
A **single Modbus TCP server with multiple unit IDs** (one per simulated
meter) was chosen over N separate servers: one TCP port, one process, and
pymodbus serves many devices keyed by unit_id.  Each meter point of
``feeder_model.meter_points()`` maps to a distinct unit_id: 1-9 the load
points (bus_virtual), 10 the feeder head, 11+ the DT meters (Part 2C.3).
Consumer meters are not served; they go to the AMI export file.

DLMS/COSEM scope
-----------------
The register map is **data-model aligned** with DLMS/COSEM OBIS codes
(``config/register_map.yaml`` lists them).  **Not implemented:** the DLMS/COSEM
wire protocol, HDLC framing, the security association layer, A-XDR.

Register map v2 (brief v2 section 2E)
-------------------------------------
All measurements are **input registers (fc4)**, so a Modbus client cannot
write any measured value.  Two words per value, big-endian (high word first)
unless noted.  ``config/register_map.yaml`` is the source of truth; the
constants below mirror it.

+---------+---------------------------+-----------+------+
| Regs    | Quantity                  | Scaling   | Type |
+=========+===========================+===========+======+
|  0–5    | V a/b/c (line-to-neutral) | V × 100   | u32  |
|  6–11   | V ab/bc/ca (line-to-line) | V × 100   | u32  |
| 12–17   | I a/b/c (load current)    | mA        | u32  |
| 18–23   | P a/b/c                   | W         | s32  |
| 24–31   | Q a/b/c/total             | var       | s32  |
| 32–39   | S a/b/c/total             | VA        | u32  |
| 40–47   | PF a/b/c/total            | × 10 000  | s32  |
| 48–49   | Frequency                 | Hz × 100  | u32  |
| 50–53   | Energy import             | Wh        | u64  |
| 54–55   | Simulated time            | Unix s    | u32  |
| 56–57   | Event counter             | count     | u32  |
| 100–111 | Event record header       | see yaml  |      |
| 120–    | Event record samples      | see yaml  | u16  |
+---------+---------------------------+-----------+------+

- **Signed P and Q** (MB-1): negative values are encoded as two's complement,
  never clamped.
- **Energy** (MB-2, MB-3): u64 Wh in input registers.  It is **stamped at the
  end of the interval** (MB-7): the register written with ``t_sim = t`` holds
  the energy imported up to t; the interval's own energy is added after the
  write.  Only import (P_total > 0) accumulates.
- **Simulated time** (MB-10): the ``run_timeseries`` timestamps are IST wall
  time; the register holds the Unix seconds of that IST instant.
- **Apparent power and PF**: per phase S = sqrt(P² + Q²) and PF = P/S; the
  totals are vector values S_tot = sqrt(P_tot² + Q_tot²), PF_tot = P_tot/S_tot.
- **Line-to-line voltage** is formed from the phase phasors (magnitude and
  angle from the solved power flow); it is 0 when either phase is absent.
- One reserved holding register (HR 0, always 0) exists only because pymodbus
  3.15 requires a holding block.  Writes elsewhere get exception 2.
- **Event record**: :meth:`MeterState.write_event` writes the header and the
  samples and only then increments the event counter, so a reader that sees a
  new counter sees a complete record.

References
----------
- pymodbus v3.x documentation — asynchronous Modbus TCP server API.
- DLMS/COSEM Blue Book (IEC 62056-6-2) — OBIS object model.
"""

from __future__ import annotations

import asyncio
import cmath
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register addresses (input registers, fc4)
# ---------------------------------------------------------------------------
REG_VOLTAGE_A = 0          # V a/b/c        0, 2, 4
REG_VOLTAGE_LL = 6         # V ab/bc/ca     6, 8, 10
REG_CURRENT = 12           # I a/b/c        12, 14, 16
REG_POWER = 18             # P a/b/c        18, 20, 22   (s32)
REG_REACTIVE = 24          # Q a/b/c/total  24 … 30      (s32)
REG_APPARENT = 32          # S a/b/c/total  32 … 38
REG_PF = 40                # PF a/b/c/total 40 … 46      (s32)
REG_FREQUENCY = 48
REG_ENERGY = 50            # u64, 50–53
REG_SIM_TIME = 54
REG_EVENT_COUNTER = 56
POLL_REGISTERS = 58        # registers 0–57 are read every poll

# Event record (see config/register_map.yaml `event`)
EVENT_HEADER = 100
EVENT_HEADER_REGISTERS = 12
EVENT_SAMPLES = 120
EVENT_VALUES_PER_SAMPLE = 6
EVENT_MAX_SAMPLES = 6150   # 0.5 s + 60 s + 1.0 s at 10 ms
EVENT_VOLTAGE_SCALE_PU = 1e-4
IR_SIZE = EVENT_SAMPLES + EVENT_VALUES_PER_SAMPLE * EVENT_MAX_SAMPLES   # 37 020
# Contiguous event block, no paging (brief v2 X.3.2 item 11/18): it must fit the
# 16-bit address space and must not overlap the polled registers 0-57.
assert IR_SIZE <= 65_536, IR_SIZE
assert EVENT_HEADER >= POLL_REGISTERS and EVENT_SAMPLES >= EVENT_HEADER + EVENT_HEADER_REGISTERS

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_u32(value: float) -> tuple[int, int]:
    """Unsigned 32-bit (big-endian words).  Raises if out of range."""
    v = int(round(value))
    if not 0 <= v <= 0xFFFFFFFF:
        raise ValueError(f"u32 out of range: {value!r}")
    return (v >> 16) & 0xFFFF, v & 0xFFFF


def _encode_s32(value: float) -> tuple[int, int]:
    """Signed 32-bit two's complement (big-endian words).  Raises if out of range."""
    v = int(round(value))
    if not -(1 << 31) <= v < (1 << 31):
        raise ValueError(f"s32 out of range: {value!r}")
    v &= 0xFFFFFFFF
    return (v >> 16) & 0xFFFF, v & 0xFFFF


def _encode_u64(value: float) -> list[int]:
    """Unsigned 64-bit as four big-endian words."""
    v = int(round(value))
    if not 0 <= v < (1 << 64):
        raise ValueError(f"u64 out of range: {value!r}")
    return [(v >> s) & 0xFFFF for s in (48, 32, 16, 0)]


def _decode_u32(hi: int, lo: int) -> int:
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def _decode_s32(hi: int, lo: int) -> int:
    v = _decode_u32(hi, lo)
    return v - (1 << 32) if v >= (1 << 31) else v


def _decode_u64(words: Sequence[int]) -> int:
    v = 0
    for w in words:
        v = (v << 16) | (w & 0xFFFF)
    return v


def _sim_time_seconds(ts: Any) -> int:
    """Unix seconds of a simulated timestamp.

    Naive timestamps are IST wall time (schema 2.0); aware ones are converted.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t_utc = t - IST_OFFSET
    else:
        t_utc = t.tz_convert("UTC").tz_localize(None)
    return int((t_utc - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------

@dataclass
class PhaseReading:
    """One phase of a meter reading (absent phase: all zeros)."""
    voltage_v: float = 0.0
    angle_deg: float = 0.0
    current_a: float = 0.0
    p_kw: float = 0.0
    q_kvar: float = 0.0

    @property
    def phasor(self) -> complex:
        return cmath.rect(self.voltage_v, math.radians(self.angle_deg))


def _pf(p: float, s: float) -> float:
    return p / s if s > 0 else 0.0


def build_input_registers(phases: dict[str, PhaseReading], frequency_hz: float,
                          energy_wh: float, t_sim: Any, event_counter: int) -> list[int]:
    """The 58 polled input-register words for one reading (layout v2)."""
    ph = [phases.get(p, PhaseReading()) for p in "abc"]
    present = [p in phases for p in "abc"]
    words: list[int] = []
    for r in ph:
        words += _encode_u32(r.voltage_v * 100)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        vll = abs(ph[i].phasor - ph[j].phasor) if present[i] and present[j] else 0.0
        words += _encode_u32(vll * 100)
    for r in ph:
        words += _encode_u32(r.current_a * 1000)
    for r in ph:
        words += _encode_s32(r.p_kw * 1000)
    q_tot = sum(r.q_kvar for r in ph)
    for q in [r.q_kvar for r in ph] + [q_tot]:
        words += _encode_s32(q * 1000)
    p_tot = sum(r.p_kw for r in ph)
    s_ph = [math.hypot(r.p_kw, r.q_kvar) for r in ph]
    s_tot = math.hypot(p_tot, q_tot)
    for s in s_ph + [s_tot]:
        words += _encode_u32(s * 1000)
    for p, s in zip([r.p_kw for r in ph] + [p_tot], s_ph + [s_tot]):
        words += _encode_s32(_pf(p, s) * 10_000)
    words += _encode_u32(frequency_hz * 100)
    words += _encode_u64(energy_wh)
    words += _encode_u32(_sim_time_seconds(t_sim) if t_sim is not None else 0)
    words += _encode_u32(event_counter)
    assert len(words) == POLL_REGISTERS
    return words


@dataclass
class EventRecord:
    """A fault burst for one meter: 10 ms half-cycle RMS samples."""
    event_id: int
    t_start: Any                     # IST wall time (naive) or aware
    sample_interval_ms: int
    voltage_v: np.ndarray            # shape (n, 3), phases a/b/c (0 if absent)
    current_a: np.ndarray            # shape (n, 3)
    phases: str                      # e.g. "abc", "b"
    v_base_ln_v: float


def build_event_registers(rec: EventRecord) -> list[int]:
    """Header + sample words of an event record (layout in register_map.yaml)."""
    n = len(rec.voltage_v)
    if not 1 <= n <= EVENT_MAX_SAMPLES:
        raise ValueError(f"event has {n} samples; 1..{EVENT_MAX_SAMPLES} allowed")
    t = pd.Timestamp(rec.t_start)
    secs = _sim_time_seconds(t.floor("s"))
    ms = int(round((t - t.floor("s")) / pd.Timedelta(milliseconds=1)))
    i_max = float(np.max(rec.current_a)) if rec.current_a.size else 0.0
    # Current counts are u16: pick mA-per-count so the largest sample fits.
    scale_ma = max(1, math.ceil(i_max * 1000 / 65_000))
    mask = sum(bit for p, bit in (("a", 1), ("b", 2), ("c", 4)) if p in rec.phases)
    header = (list(_encode_u32(rec.event_id)) + list(_encode_u32(secs))
              + [ms, rec.sample_interval_ms, n, mask]
              + list(_encode_u32(rec.v_base_ln_v * 100)) + list(_encode_u32(scale_ma)))
    v_pu = np.clip(np.rint(rec.voltage_v / rec.v_base_ln_v / EVENT_VOLTAGE_SCALE_PU), 0, 65535)
    i_cnt = np.clip(np.rint(rec.current_a * 1000 / scale_ma), 0, 65535)
    samples = np.hstack([v_pu, i_cnt]).astype(int).ravel().tolist()
    pad = [0] * (EVENT_SAMPLES - EVENT_HEADER - len(header))   # samples start at 120
    return header + pad + samples


# ---------------------------------------------------------------------------
# Meter state
# ---------------------------------------------------------------------------

class MeterState:
    """Per-meter energy accumulator, event counter and register writer."""

    def __init__(self, unit_id: int, load_id: str):
        self.unit_id = unit_id
        self.load_id = load_id
        parts = load_id.split("_")
        self.bus_name = parts[-1] if load_id.startswith("feeder_head") else (
            parts[1] if len(parts) >= 2 else load_id)
        self.energy_wh: float = 0.0       # import up to the last written t_sim
        self.event_counter: int = 0
        self._last_poll: Optional[list[int]] = None

    async def update_registers(
        self,
        server_ctx: Any,
        phases: dict[str, PhaseReading],
        frequency_hz: float = 50.0,
        interval_hours: float = 0.25,
        t_sim: Any = None,
    ) -> None:
        """Write one reading; energy is stamped at the end of the interval.

        The registers written with ``t_sim`` carry the energy imported up to
        ``t_sim``; this interval's import (P_total × interval) is added after
        the write and appears with the next ``t_sim``.
        """
        words = build_input_registers(phases, frequency_hz, self.energy_wh,
                                      t_sim, self.event_counter)
        await server_ctx.async_setValues(self.unit_id, 4, 0, words)
        self._last_poll = words
        p_total = sum(r.p_kw for r in phases.values())
        self.energy_wh += max(p_total, 0.0) * 1000.0 * interval_hours

    async def write_event(self, server_ctx: Any, rec: EventRecord) -> None:
        """Write a burst record, then bump the event counter (atomic for readers)."""
        words = build_event_registers(rec)
        await server_ctx.async_setValues(self.unit_id, 4, EVENT_HEADER, words)
        self.event_counter += 1
        await server_ctx.async_setValues(self.unit_id, 4, REG_EVENT_COUNTER,
                                         list(_encode_u32(self.event_counter)))


# ---------------------------------------------------------------------------
# Server context builder
# ---------------------------------------------------------------------------

def build_server_context(unit_map: Optional[dict[str, int]] = None):
    """Create one ``SimDevice`` per meter.

    Returns ``(devices_list, devices_dict, meter_states)``.  ``unit_map``
    defaults to :func:`_load_id_unit_map`.
    """
    from pymodbus.simulator import DataType
    from pymodbus.simulator.simdevice import SimDevice
    from pymodbus.simulator.simdata import SimData

    devices_list: list[SimDevice] = []
    devices_dict: dict[int, Any] = {}
    meters: dict[int, MeterState] = {}

    for load_id, unit_id in (unit_map or _load_id_unit_map()).items():
        ir = SimData(0, values=[0] * IR_SIZE, datatype=DataType.REGISTERS)
        hr = SimData(0, values=[0], datatype=DataType.REGISTERS)   # reserved
        co = SimData(0, values=[False], datatype=DataType.BITS)
        di = SimData(0, values=[False], datatype=DataType.BITS)
        device = SimDevice(unit_id, simdata=([co], [di], [hr], [ir]))
        devices_list.append(device)
        devices_dict[unit_id] = device
        meters[unit_id] = MeterState(unit_id, load_id)

    return devices_list, devices_dict, meters


def _load_id_unit_map() -> dict[str, int]:
    """Map meter-point id → unit_id (``feeder_model.meter_points()`` order).

    Units 1-9 are the load points (bus_virtual), unit 10 the feeder head,
    units 11+ the DT meters.  Raises if the feeder model cannot be built: a
    silent fallback would serve meters with fake ids that never receive data.
    """
    from .feeder_model import meter_points
    mp = meter_points()
    return {r.load_id: int(r.unit_id) for r in mp.itertuples()}


# ---------------------------------------------------------------------------
# Background updater
# ---------------------------------------------------------------------------

def frame_readings(rows: pd.DataFrame) -> dict[str, PhaseReading]:
    """Per-phase readings of one meter at one step from ``run_timeseries`` rows.

    Uses ``load_current_a`` (the meter's own current) and
    ``voltage_angle_deg`` when the frame has them (schema 2.0 frames).
    """
    has_i = "load_current_a" in rows.columns
    has_ang = "voltage_angle_deg" in rows.columns
    out: dict[str, PhaseReading] = {}
    for _, r in rows.iterrows():
        out[r["phase"]] = PhaseReading(
            voltage_v=float(r["voltage_v"]),
            angle_deg=float(r["voltage_angle_deg"]) if has_ang else 0.0,
            current_a=float(r["load_current_a"] if has_i else r["current_a"]),
            p_kw=float(r["active_power_kw"]),
            q_kvar=float(r["reactive_power_kvar"]),
        )
    return out


def save_event_file(path, bursts: list[dict]) -> None:
    """Write fault bursts for playback (``.npz``).

    Each item: ``trigger`` (step timestamp after which the record is written),
    ``load_id``, ``event_id``, ``t_start`` (IST wall time of the first
    sample), ``v`` and ``i`` (n, 3) RMS samples, ``v_base_ln_v``,
    ``phases``.  Written by :mod:`faults` / the CP4-CP5 scripts.
    """
    import json
    arrays, index = {}, []
    for k, b in enumerate(bursts):
        arrays[f"v{k}"] = np.asarray(b["v"], float)
        arrays[f"i{k}"] = np.asarray(b["i"], float)
        index.append({"trigger": str(pd.Timestamp(b["trigger"])), "load_id": b["load_id"],
                      "event_id": int(b["event_id"]), "t_start": str(pd.Timestamp(b["t_start"])),
                      "v_base_ln_v": float(b["v_base_ln_v"]), "phases": b.get("phases", "abc")})
    np.savez_compressed(path, index=np.array(json.dumps(index)), **arrays)


def load_event_file(path) -> dict:
    """{trigger timestamp: [(load_id, EventRecord), ...]} from :func:`save_event_file`."""
    import json
    z = np.load(path, allow_pickle=False)
    out: dict = {}
    for k, e in enumerate(json.loads(str(z["index"]))):
        rec = EventRecord(event_id=e["event_id"], t_start=pd.Timestamp(e["t_start"]),
                          sample_interval_ms=10, voltage_v=z[f"v{k}"], current_a=z[f"i{k}"],
                          phases=e["phases"], v_base_ln_v=e["v_base_ln_v"])
        out.setdefault(pd.Timestamp(e["trigger"]), []).append((e["load_id"], rec))
    return out


async def _update_loop(
    server_ctx: Any,
    meters: dict[int, MeterState],
    timeseries_data: pd.DataFrame,
    playback_speed: float,
    interval_minutes: float = 15.0,
    events: Optional[dict] = None,
) -> None:
    """Background coroutine that updates meter registers from timeseries data.

    ``timeseries_data`` is a ``run_timeseries`` frame (columns ``timestamp,
    load_id, phase, voltage_v, current_a, active_power_kw,
    reactive_power_kvar`` and, for schema 2.0, ``voltage_angle_deg`` and
    ``load_current_a``).  ``events`` (:func:`load_event_file`): fault bursts
    written to their meters right after the step they are triggered by
    (record first, then the counter: :meth:`MeterState.write_event`).
    """
    interval_hours = interval_minutes / 60.0
    wall_sleep = (interval_minutes * 60.0) / playback_speed
    by_ts = {ts: g for ts, g in timeseries_data.groupby("timestamp", sort=True)}
    by_load_id = {m.load_id: m for m in meters.values()}

    for ts, ts_data in by_ts.items():
        by_load = {lid: g for lid, g in ts_data.groupby("load_id")}
        for meter in meters.values():
            rows = by_load.get(meter.load_id)
            if rows is None:
                continue
            await meter.update_registers(
                server_ctx, frame_readings(rows), frequency_hz=50.0,
                interval_hours=interval_hours, t_sim=ts)
        for load_id, rec in (events or {}).get(pd.Timestamp(ts), []):
            if load_id in by_load_id:
                await by_load_id[load_id].write_event(server_ctx, rec)
                logger.info("event %d written to %s (%d samples)", rec.event_id, load_id,
                            len(rec.voltage_v))
        logger.debug("Updated meters for timestamp %s", ts)
        await asyncio.sleep(wall_sleep)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def start_meter_server(
    host: str | None = None,
    port: int | None = None,
    timeseries_data: pd.DataFrame | None = None,
    playback_speed: float | None = None,
    events: Optional[dict] = None,
) -> None:
    """Start the Modbus TCP server and background updater."""
    from pymodbus.server import ModbusTcpServer

    if host is None:
        host = config.MODBUS_HOST
    if port is None:
        port = config.MODBUS_PORT
    if playback_speed is None:
        playback_speed = config.PLAYBACK_SPEED

    devices_list, _, meters = build_server_context()
    logger.info("Starting Modbus TCP server on %s:%d with %d meters", host, port, len(meters))
    server = ModbusTcpServer(context=devices_list, address=(host, port))
    if timeseries_data is not None and not timeseries_data.empty:
        asyncio.create_task(
            _update_loop(server.context, meters, timeseries_data, playback_speed, events=events))
    await server.serve_forever()


def run_server(
    host: str | None = None,
    port: int | None = None,
    timeseries_data: pd.DataFrame | None = None,
) -> None:
    """Synchronous wrapper to start the meter server."""
    asyncio.run(start_meter_server(host, port, timeseries_data))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Starting meter server on {config.MODBUS_HOST}:{config.MODBUS_PORT}")
    print(f"Unit IDs: {_load_id_unit_map()}")
    print("Press Ctrl+C to stop.\n")
    run_server()
