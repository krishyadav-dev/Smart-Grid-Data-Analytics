"""Layers 1 and 2 without the broker: validated telemetry and fault bursts for
the Layer 3 batch mode (Stage 1 run sheet rule 5).

The same code paths as the live system, composed in one process:

- Layer 1 meter state (:class:`meter_server.MeterState`: register encoding,
  energy accumulator, event record + counter) writes into in-memory input
  registers instead of a Modbus server;
- the Layer 2 gateway's :class:`gateway.MeterPoller` reads them through an
  in-memory :class:`RegisterReader` (torn-read brackets, event re-read check,
  GAP logic, quantised decoding) and builds the ``Telemetry`` / ``EventBurst``
  models;
- the Layer 2 listener's :class:`ListenerService` validates each wire message
  and returns the document it republishes on ``validated/{meter}`` or
  ``validated_event/{meter}``.

So a batch document differs from a live one only in the wall-clock fields
(``run_id``, ``t_poll_utc``, ``t_pub_utc``, ``ingest``).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.data_source import meter_server as ms
from src.ingestion import topics
from src.ingestion.config import ListenerSettings, MeterConfig, MetersFile, load_meters
from src.ingestion.gateway import MeterPoller
from src.ingestion.listener.service import ListenerService
from src.ingestion.register_map import RegisterMap

FEEDER_ID = "ieee13_11kv"
FRAME_VALUES = ("voltage_v", "voltage_angle_deg", "load_current_a", "active_power_kw", "reactive_power_kvar")


class _Registers:
    """Input registers of every unit (the meter server's IR tables)."""

    def __init__(self, units: Iterable[int]):
        self.ir = {u: [0] * ms.IR_SIZE for u in units}

    async def async_setValues(self, unit: int, fc: int, address: int, values: list[int]) -> None:  # noqa: N802
        assert fc == 4
        self.ir[unit][address:address + len(values)] = list(values)


class MemoryReader:
    """:class:`gateway.RegisterReader` over :class:`_Registers`."""

    def __init__(self, regs: _Registers):
        self.regs = regs

    @property
    def connected(self) -> bool:
        return True

    async def connect(self) -> None:
        return None

    async def read(self, unit_id: int, function_code: int, address: int, count: int) -> list[int]:
        return list(self.regs.ir[unit_id][address:address + count])

    def close(self) -> None:
        return None


@dataclass
class _Meter:
    cfg: MeterConfig
    state: ms.MeterState
    poller: MeterPoller


class OfflinePipeline:
    """Meter registers -> gateway poller -> listener, for the given meters."""

    def __init__(self, meters: Iterable[MeterConfig], run_name: str = "layer3-batch"):
        meters = list(meters)
        self.regs = _Registers(m.unit_id for m in meters)
        regmap = RegisterMap.load()
        run_id = uuid.uuid5(uuid.NAMESPACE_URL, run_name)
        reader = MemoryReader(self.regs)
        self.meters = {m.meter_id: _Meter(m, ms.MeterState(m.unit_id, m.load_id),
                                          MeterPoller(meter=m, feeder_id=FEEDER_ID, run_id=run_id,
                                                      regmap=regmap, reader=reader))
                       for m in meters}
        self.by_load = {m.cfg.load_id: m for m in self.meters.values()}
        allm = load_meters()
        self.listener = ListenerService(ListenerSettings(mqtt_enabled=False, _env_file=None), allm)

    def _validate(self, topic: str, model) -> Optional[dict]:
        out = self.listener.handle(topic, model.to_wire())
        assert len(out) == 1
        if out[0].topic.endswith("/dlq"):
            raise RuntimeError(f"listener rejected {topic}: {out[0].payload[:300]!r}")
        return json.loads(out[0].payload)

    async def _poll(self, m: _Meter) -> list[dict]:
        o = await m.poller.poll_once()
        docs = []
        if o.event is not None:                         # the gateway publishes the event first
            docs.append(self._validate(topics.event(FEEDER_ID, m.cfg.meter_id), o.event))
        if o.published is not None:
            docs.append(self._validate(topics.telemetry(FEEDER_ID, m.cfg.meter_id), o.published))
        return docs

    async def step(self, ts, readings: dict[str, dict], events: list) -> list[dict]:
        """One 15-min step: write every meter's reading, poll it; then write the
        step's event records (``[(load_id, EventRecord)]``) and poll those meters."""
        docs = []
        for lid, ph in readings.items():
            m = self.by_load[lid]
            await m.state.update_registers(self.regs, ph, frequency_hz=50.0, interval_hours=0.25, t_sim=ts)
            docs += await self._poll(m)
        for lid, rec in events:
            m = self.by_load.get(lid)
            if m is None:
                continue
            await m.state.write_event(self.regs, rec)
            docs += await self._poll(m)
        return docs


def frame_readings_fast(frame: pd.DataFrame, load_ids: list[str]):
    """{timestamp: {load_id: {phase: PhaseReading}}} for the given meters (same
    values as :func:`meter_server.frame_readings`)."""
    f = frame[frame.load_id.isin(load_ids)]
    out: dict = {}
    cols = [f[c].to_numpy(float) for c in FRAME_VALUES]
    for k, (ts, lid, ph) in enumerate(zip(f.timestamp.to_numpy(), f.load_id.to_numpy(), f.phase.to_numpy())):
        out.setdefault(pd.Timestamp(ts), {}).setdefault(lid, {})[ph] = ms.PhaseReading(
            voltage_v=cols[0][k], angle_deg=cols[1][k], current_a=cols[2][k], p_kw=cols[3][k], q_kvar=cols[4][k])
    return out


def run_stream(frame: Optional[pd.DataFrame], events: Optional[dict], meters: Iterable[MeterConfig],
               run_name: str = "layer3-batch", on_telemetry=None) -> tuple[list[dict], list[dict]]:
    """(validated telemetry docs, validated burst docs) for a frame and an event
    file (:func:`meter_server.load_event_file`).  ``frame=None`` serves the
    bursts only (each record written and read on its own).  With
    ``on_telemetry`` each telemetry document goes to that callback instead of
    the returned list (a year of documents does not fit in memory)."""
    meters = list(meters)
    pipe = OfflinePipeline(meters, run_name)
    events = events or {}
    tel, bur = [], []
    sink = on_telemetry or tel.append

    async def go():
        if frame is None:
            for trig in sorted(events):
                for d in await pipe.step(trig, {}, events[trig]):
                    bur.append(d)
            return
        readings = frame_readings_fast(frame, [m.load_id for m in meters])
        for ts in sorted(readings):
            for d in await pipe.step(ts, readings[ts], events.get(ts, [])):
                if "samples" in d:
                    bur.append(d)
                else:
                    sink(d)
    asyncio.run(go())
    return tel, bur


def flatten_doc(d: dict) -> dict:
    """One validated telemetry document as a flat row (the engine's 15-min view)."""
    m = d["measurements"]
    r = {"meter_id": d["meter_id"], "t_sim": d["t_sim"], "t_poll_utc": d["t_poll_utc"],
         "flags": ",".join(d["quality"]["flags"]), "gap_steps": d["quality"]["gap_steps"]}
    for key, sub in (("voltage_v", "abc"), ("voltage_ll_v", ("ab", "bc", "ca")), ("current_a", "abc")):
        for p in sub:
            r[f"{key}_{p}"] = m.get(key, {}).get(p, np.nan)
    return r


def flatten_telemetry(docs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([flatten_doc(d) for d in docs])
