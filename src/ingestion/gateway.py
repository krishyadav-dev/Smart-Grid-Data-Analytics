"""Modbus → MQTT gateway (Layer 2, Part C).

Polls every meter of the Layer 1 Modbus TCP server, decodes registers using
``config/register_map.yaml``, and publishes one JSON telemetry message per new
simulated time step to ``sg/v1/{feeder_id}/{meter_id}/telemetry`` (QoS 1).

Design
------
- One poll task and one Modbus connection per meter, so a failing unit never
  stalls the others.
- Change detection: a reading is published only when ``t_sim`` advanced;
  repeated polls of the same step are counted (``skipped_same_t_sim``).
  If ``t_sim`` jumps by more than ``expected_step_s`` the message carries
  ``GAP`` with ``gap_steps``.  Without a time register the gateway falls back
  to publishing on value change with ``t_sim = null`` + ``NO_SIM_TIME``.
- Torn reads: the time register is read before and after the unit's blocks;
  if the three time values disagree the sequence is retried
  (``torn_read_retries``) and then published with ``TORN_READ``.
- MQTT side: a single connection with Last Will (``offline``, retained),
  reconnect with exponential backoff + jitter, and a bounded drop-oldest
  buffer while the broker is unreachable.
- No import of ``feeder_model`` or ``pandapower``.

Windows note: aiomqtt (paho) needs ``add_reader``, which the default Proactor
event loop lacks, so :func:`main` runs on a ``SelectorEventLoop`` on Windows.

CLI (PowerShell)::

    python -m src.ingestion.gateway --config config/gateway.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol

from . import topics
from .backoff import Backoff
from .config import GatewayConfig, MeterConfig, load_gateway_config, load_meters
from .register_map import ReadBlock, RegisterMap
from .schema import (
    SCHEMA_VERSION,
    GatewayState,
    GatewayStatus,
    MeterLinkState,
    MeterStatus,
    QualityFlag,
    Telemetry,
    dumps_compact,
    utc_now,
)

logger = logging.getLogger("sg.gateway")


# ===================================================================== #
#  Modbus access                                                         #
# ===================================================================== #

class ModbusReadError(Exception):
    """A register read failed (exception response, timeout, disconnect)."""


class RegisterReader(Protocol):
    async def connect(self) -> None: ...
    async def read(self, unit_id: int, function_code: int,
                   address: int, count: int) -> list[int]: ...
    def close(self) -> None: ...
    @property
    def connected(self) -> bool: ...


class PymodbusReader:
    """Thin async wrapper over ``pymodbus.client.AsyncModbusTcpClient``."""

    def __init__(self, host: str, port: int, timeout_s: float):
        from pymodbus.client import AsyncModbusTcpClient

        # retries=0 / reconnect handled by the gateway's own backoff.
        self._client = AsyncModbusTcpClient(
            host, port=port, timeout=timeout_s, retries=0,
            reconnect_delay=0, reconnect_delay_max=0,
        )

    @property
    def connected(self) -> bool:
        return bool(self._client.connected)

    async def connect(self) -> None:
        ok = await self._client.connect()
        if not ok or not self._client.connected:
            raise ModbusReadError("connect failed")

    async def read(self, unit_id: int, function_code: int,
                   address: int, count: int) -> list[int]:
        fn = (self._client.read_holding_registers if function_code == 3
              else self._client.read_input_registers)
        try:
            rr = await fn(address, count=count, device_id=unit_id)
        except Exception as exc:  # ModbusException, timeouts, connection loss
            raise ModbusReadError(f"{type(exc).__name__}: {exc}") from exc
        if rr.isError():
            raise ModbusReadError(f"exception response fc{function_code} @{address}: {rr}")
        regs = list(rr.registers)
        if len(regs) < count:
            raise ModbusReadError(f"short read fc{function_code} @{address}: {len(regs)}/{count}")
        return regs[:count]

    def close(self) -> None:
        self._client.close()


# ===================================================================== #
#  Buffer                                                                #
# ===================================================================== #

@dataclass
class OutMessage:
    topic: str
    payload: Any            # Telemetry (stamped at publish) or bytes
    qos: int
    retain: bool = False

    def wire(self) -> bytes:
        if isinstance(self.payload, Telemetry):
            self.payload.t_pub_utc = utc_now()
            return self.payload.to_wire()
        return self.payload


class DropOldestBuffer:
    """Bounded FIFO; when full the oldest message is dropped and counted."""

    def __init__(self, maxlen: int):
        self._dq: deque[OutMessage] = deque()
        self.maxlen = maxlen
        self.dropped = 0
        self._event = asyncio.Event()

    def __len__(self) -> int:
        return len(self._dq)

    def put(self, msg: OutMessage) -> None:
        if len(self._dq) >= self.maxlen:
            self._dq.popleft()
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 100 == 0:
                logger.warning("publish buffer full: dropped %d message(s) so far", self.dropped)
        self._dq.append(msg)
        self._event.set()

    def put_front(self, msg: OutMessage) -> None:
        """Return an unsent message to the head (after a failed publish)."""
        if len(self._dq) >= self.maxlen:
            self.dropped += 1   # the returned message is the oldest: drop it
            return
        self._dq.appendleft(msg)
        self._event.set()

    async def get(self) -> OutMessage:
        while not self._dq:
            self._event.clear()
            await self._event.wait()
        return self._dq.popleft()


# ===================================================================== #
#  Per-meter polling                                                     #
# ===================================================================== #

@dataclass
class PollOutcome:
    published: Optional[Telemetry] = None
    skipped: bool = False
    error: Optional[str] = None


@dataclass
class MeterPoller:
    """Read, decode and change-detect one meter.  Transport-agnostic."""

    meter: MeterConfig
    feeder_id: str
    run_id: uuid.UUID
    regmap: RegisterMap
    reader: RegisterReader
    expected_step_s: float = 900.0
    torn_read_retries: int = 3

    seq: int = 0
    last_t_sim: Optional[datetime] = None
    last_values: Optional[dict] = None
    skipped_same_t_sim: int = 0
    torn_retries_total: int = 0
    link_state: Optional[MeterLinkState] = None
    link_since: Optional[datetime] = None
    last_error: Optional[str] = None
    _blocks: list[ReadBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._blocks = self.regmap.blocks()

    # -- reads ----------------------------------------------------------
    async def _read_time(self) -> Optional[datetime]:
        tr = self.regmap.time_register
        if tr is None:
            return None
        words = await self.reader.read(
            self.meter.unit_id, tr.function_code,
            self.regmap.wire_address(tr.address), tr.width_registers)
        return self.regmap.decode_time({tr.name: tr.decode(words)})

    async def _read_blocks(self) -> tuple[dict[int, dict[int, int]], list[str]]:
        raw: dict[int, dict[int, int]] = {}
        errors: list[str] = []
        for b in self._blocks:
            try:
                words = await self.reader.read(
                    self.meter.unit_id, b.function_code,
                    self.regmap.wire_address(b.start), b.count)
            except ModbusReadError as exc:
                errors.append(f"fc{b.function_code}@{b.start}+{b.count}: {exc}")
                continue
            dst = raw.setdefault(b.function_code, {})
            for i, w in enumerate(words):
                dst[b.start + i] = w
        return raw, errors

    async def read_consistent(self) -> tuple[dict[str, float], list[QualityFlag], list[str]]:
        """Read all blocks bracketed by time reads; retry on a torn read."""
        has_time = self.regmap.time_register is not None
        attempts = self.torn_read_retries + 1
        for attempt in range(attempts):
            t_start = await self._read_time() if has_time else None
            raw, errors = await self._read_blocks()
            if len(errors) == len(self._blocks):
                raise ModbusReadError("; ".join(errors))
            t_end = await self._read_time() if has_time else None
            values = self.regmap.decode_block_values(raw)
            t_block = self.regmap.decode_time(values) if has_time else None
            flags: list[QualityFlag] = []
            if errors:
                flags.append(QualityFlag.PARTIAL_READ)
            # A failed block may have held the time register: compare what we have.
            times = {t for t in (t_start, t_block, t_end) if t is not None}
            if len(times) <= 1:
                return values, flags, errors
            if attempt + 1 < attempts:
                self.torn_retries_total += 1
                logger.debug("%s torn read (attempt %d): %s", self.meter.meter_id,
                             attempt + 1, sorted(times))
                continue
            flags.append(QualityFlag.TORN_READ)
            return values, flags, errors
        raise AssertionError("unreachable")

    # -- one poll ---------------------------------------------------------
    async def poll_once(self) -> PollOutcome:
        values, flags, errors = await self.read_consistent()
        t_poll = utc_now()
        t_sim = self.regmap.decode_time(values)
        has_time_reg = self.regmap.time_register is not None
        gap_steps = 0

        if has_time_reg and self.regmap.time_register.name in values:
            if t_sim is None:
                # Server up but has not written its first frame yet.
                return PollOutcome(skipped=True)
            if self.last_t_sim is not None and t_sim == self.last_t_sim:
                self.skipped_same_t_sim += 1
                return PollOutcome(skipped=True)
            if self.last_t_sim is not None and t_sim > self.last_t_sim:
                steps = round((t_sim - self.last_t_sim).total_seconds() / self.expected_step_s)
                if steps > 1:
                    gap_steps = steps - 1
                    flags.append(QualityFlag.GAP)
        else:
            # No usable time register: publish on value change only.
            flags.append(QualityFlag.NO_SIM_TIME)
            if self.last_values is not None and values == self.last_values:
                self.skipped_same_t_sim += 1
                return PollOutcome(skipped=True)

        measurements = self.regmap.build_measurements(values, self.meter.phases_present)
        msg = Telemetry(
            schema_version=SCHEMA_VERSION,
            feeder_id=self.feeder_id,
            meter_id=self.meter.meter_id,
            unit_id=self.meter.unit_id,
            load_id=self.meter.load_id,
            meter_role=self.meter.role,
            run_id=self.run_id,
            seq=self.seq,
            t_sim=t_sim,
            t_poll_utc=t_poll,
            t_pub_utc=t_poll,          # re-stamped when actually published
            measurements=measurements,
            quality={"flags": _dedupe(flags), "gap_steps": gap_steps},
        )
        self.seq += 1
        if t_sim is not None:
            self.last_t_sim = t_sim
        self.last_values = values
        if errors:
            self.last_error = errors[0]
        return PollOutcome(published=msg)


def _dedupe(flags: list[QualityFlag]) -> list[QualityFlag]:
    out: list[QualityFlag] = []
    for f in flags:
        if f not in out:
            out.append(f)
    return out


# ===================================================================== #
#  Gateway                                                               #
# ===================================================================== #

class Gateway:
    def __init__(self, cfg: GatewayConfig, reader_factory=None, mqtt_client_factory=None):
        self.cfg = cfg
        self.run_id = uuid.uuid4()
        self.meters_file = load_meters(cfg.meters_file)
        self.regmap = RegisterMap.load(cfg.register_map_file)
        if self.meters_file.feeder_id != cfg.feeder_id:
            raise ValueError(f"meters.yaml feeder_id {self.meters_file.feeder_id!r} "
                             f"!= gateway feeder_id {cfg.feeder_id!r}")
        self.buffer = DropOldestBuffer(cfg.buffer_max_messages)
        self._reader_factory = reader_factory or (
            lambda: PymodbusReader(cfg.modbus.host, cfg.modbus.port, cfg.modbus.timeout_s))
        self._mqtt_factory = mqtt_client_factory
        self.pollers: dict[str, MeterPoller] = {}
        self.published = 0
        self.mqtt_connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self._last_gw_state: Optional[tuple] = None

    # -- status helpers -----------------------------------------------------
    def _gateway_status(self, state: GatewayState) -> bytes:
        unreachable = sorted(m for m, p in self.pollers.items()
                             if p.link_state == MeterLinkState.UNREACHABLE)
        return GatewayStatus(
            state=state, run_id=self.run_id, t_utc=utc_now(),
            buffer_dropped=self.buffer.dropped, unreachable_meters=unreachable,
        ).model_dump_json().encode()

    def _current_gateway_state(self) -> GatewayState:
        bad = any(p.link_state == MeterLinkState.UNREACHABLE for p in self.pollers.values())
        return GatewayState.DEGRADED if bad or self.buffer.dropped else GatewayState.ONLINE

    def _queue_gateway_status(self, force: bool = False) -> None:
        state = self._current_gateway_state()
        unreachable = tuple(sorted(m for m, p in self.pollers.items()
                                   if p.link_state == MeterLinkState.UNREACHABLE))
        key = (state, unreachable)
        if force or key != self._last_gw_state:
            self._last_gw_state = key
            self.buffer.put(OutMessage(topics.gateway_status(self.cfg.feeder_id),
                                       self._gateway_status(state), topics.QOS_STATUS, True))

    def _set_link(self, poller: MeterPoller, state: MeterLinkState, error: Optional[str]) -> None:
        if poller.link_state == state:
            return
        poller.link_state = state
        poller.link_since = utc_now()
        if error:
            poller.last_error = error
        level = logging.WARNING if state == MeterLinkState.UNREACHABLE else logging.INFO
        logger.log(level, "%s -> %s%s", poller.meter.meter_id, state.value,
                   f" ({error})" if error else "")
        payload = MeterStatus(state=state, since_utc=poller.link_since,
                              last_error=poller.last_error).model_dump_json().encode()
        self.buffer.put(OutMessage(topics.meter_status(self.cfg.feeder_id, poller.meter.meter_id),
                                   payload, topics.QOS_STATUS, True))
        self._queue_gateway_status()

    # -- tasks --------------------------------------------------------------
    def _sampling_warning(self) -> None:
        speed = self.cfg.playback_speed
        if not speed:
            return
        wall_step = self.cfg.expected_step_s / speed
        if self.cfg.poll_interval_s >= wall_step / 2:
            logger.warning(
                "poll_interval_s=%.3f is too large for playback_speed=%.1f "
                "(one simulated step every %.3f s wall); expect GAP flags. "
                "Use poll_interval_s < %.3f.",
                self.cfg.poll_interval_s, speed, wall_step, wall_step / 2)

    async def _poll_meter(self, meter: MeterConfig) -> None:
        poller = MeterPoller(
            meter=meter, feeder_id=self.cfg.feeder_id, run_id=self.run_id,
            regmap=self.regmap, reader=self._reader_factory(),
            expected_step_s=self.cfg.expected_step_s,
            torn_read_retries=self.cfg.torn_read_retries,
        )
        self.pollers[meter.meter_id] = poller
        b = self.cfg.backoff
        backoff = Backoff(b.initial_s, b.max_s, b.jitter)
        topic = topics.telemetry(self.cfg.feeder_id, meter.meter_id)
        loop = asyncio.get_running_loop()
        while not self._stopping.is_set():
            started = loop.time()
            try:
                if not poller.reader.connected:
                    await poller.reader.connect()
                outcome = await poller.poll_once()
            except (ModbusReadError, OSError, asyncio.TimeoutError) as exc:
                if self._stopping.is_set():
                    break           # pymodbus turns our cancellation into an IO error
                self._set_link(poller, MeterLinkState.UNREACHABLE, str(exc) or type(exc).__name__)
                poller.reader.close()
                await self._sleep(backoff.next_delay())
                continue
            except Exception as exc:  # noqa: BLE001 — one bad unit must not stop the others
                if self._stopping.is_set():
                    break
                logger.exception("%s: unexpected error while polling", meter.meter_id)
                self._set_link(poller, MeterLinkState.UNREACHABLE, f"{type(exc).__name__}: {exc}")
                poller.reader.close()
                await self._sleep(backoff.next_delay())
                continue
            backoff.reset()
            self._set_link(poller, MeterLinkState.REACHABLE, None)
            if outcome.published is not None:
                self.buffer.put(OutMessage(topic, outcome.published, topics.QOS_TELEMETRY, False))
            elapsed = loop.time() - started
            await self._sleep(max(0.0, self.cfg.poll_interval_s - elapsed))
        poller.reader.close()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    def _make_mqtt_client(self):
        import aiomqtt

        m = self.cfg.mqtt
        will = aiomqtt.Will(
            topic=topics.gateway_status(self.cfg.feeder_id),
            payload=self._gateway_status(GatewayState.OFFLINE),
            qos=topics.QOS_STATUS, retain=True,
        )
        return aiomqtt.Client(
            hostname=m.host, port=m.port, keepalive=m.keepalive_s, will=will,
            identifier=f"{m.client_id_prefix}-{self.run_id.hex[:8]}",
            clean_session=True, timeout=10,
        )

    async def _mqtt_loop(self) -> None:
        import aiomqtt

        b = self.cfg.backoff
        backoff = Backoff(b.initial_s, b.max_s, b.jitter)
        factory = self._mqtt_factory or self._make_mqtt_client
        while not self._stopping.is_set():
            try:
                async with factory() as client:
                    self._client = client
                    backoff.reset()
                    self.mqtt_connected.set()
                    logger.info("MQTT connected to %s:%d", self.cfg.mqtt.host, self.cfg.mqtt.port)
                    # Re-assert retained state after every (re)connect.
                    await client.publish(topics.gateway_status(self.cfg.feeder_id),
                                         self._gateway_status(self._current_gateway_state()),
                                         qos=topics.QOS_STATUS, retain=True)
                    for p in self.pollers.values():
                        if p.link_state is not None:
                            await client.publish(
                                topics.meter_status(self.cfg.feeder_id, p.meter.meter_id),
                                MeterStatus(state=p.link_state, since_utc=p.link_since,
                                            last_error=p.last_error).model_dump_json().encode(),
                                qos=topics.QOS_STATUS, retain=True)
                    while True:
                        msg = await self.buffer.get()
                        try:
                            await client.publish(msg.topic, msg.wire(), qos=msg.qos, retain=msg.retain)
                        except BaseException:
                            self.buffer.put_front(msg)
                            raise
                        if isinstance(msg.payload, Telemetry):
                            self.published += 1
            except aiomqtt.MqttError as exc:
                self.mqtt_connected.clear()
                delay = backoff.next_delay()
                logger.warning("MQTT unavailable (%s); retry in %.1fs, buffered=%d dropped=%d",
                               exc, delay, len(self.buffer), self.buffer.dropped)
                await self._sleep(delay)

    async def _shutdown_mqtt(self) -> None:
        """Best-effort: let the MQTT task drain the buffer (≤2 s), then
        publish the retained 'offline' status ourselves."""
        client = getattr(self, "_client", None)
        if client is None or not self.mqtt_connected.is_set():
            return
        try:
            async with asyncio.timeout(2.0):
                while len(self.buffer):
                    await asyncio.sleep(0.02)
                await client.publish(topics.gateway_status(self.cfg.feeder_id),
                                     self._gateway_status(GatewayState.OFFLINE),
                                     qos=topics.QOS_STATUS, retain=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("graceful shutdown incomplete: %s", exc)

    async def run(self) -> None:
        logger.info("gateway run_id=%s meters=%d modbus=%s:%d mqtt=%s:%d",
                    self.run_id, len(self.meters_file.meters), self.cfg.modbus.host,
                    self.cfg.modbus.port, self.cfg.mqtt.host, self.cfg.mqtt.port)
        self._sampling_warning()
        tasks = [asyncio.create_task(self._poll_meter(m), name=f"poll-{m.meter_id}")
                 for m in self.meters_file.meters]
        mqtt_task = asyncio.create_task(self._mqtt_loop(), name="mqtt")
        try:
            # asyncio.wait (unlike gather) does not cancel the children when
            # this coroutine is cancelled, so MQTT stays up for the shutdown.
            done, _ = await asyncio.wait([mqtt_task, *tasks],
                                         return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                t.result()      # surface an unexpected crash
        except asyncio.CancelledError:
            logger.info("shutting down")
            self._stopping.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._shutdown_mqtt()
            mqtt_task.cancel()
            await asyncio.gather(mqtt_task, return_exceptions=True)
            raise

    def stats(self) -> dict:
        return {
            "run_id": str(self.run_id),
            "published": self.published,
            "buffered": len(self.buffer),
            "buffer_dropped": self.buffer.dropped,
            "skipped_same_t_sim": {k: p.skipped_same_t_sim for k, p in self.pollers.items()},
            "torn_retries": {k: p.torn_retries_total for k, p in self.pollers.items()},
        }


# ===================================================================== #
#  CLI                                                                   #
# ===================================================================== #

def run_async(coro) -> Any:
    """``asyncio.run`` on a selector loop on Windows (aiomqtt requirement)."""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(coro)
    return asyncio.run(coro)


def install_term_handler() -> None:
    """Treat SIGTERM like Ctrl+C (cancel the main task) on POSIX."""
    if sys.platform == "win32":
        return
    import signal

    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    if task is not None:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ingestion.gateway",
                                 description="Modbus -> MQTT gateway")
    ap.add_argument("--config", default="config/gateway.yaml")
    ap.add_argument("--modbus-port", type=int)
    ap.add_argument("--mqtt-port", type=int)
    ap.add_argument("--poll-interval", type=float)
    ap.add_argument("--playback-speed", type=float)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_gateway_config(
        args.config, modbus__port=args.modbus_port, mqtt__port=args.mqtt_port,
        poll_interval_s=args.poll_interval, playback_speed=args.playback_speed)
    gw = Gateway(cfg)

    async def _main() -> None:
        install_term_handler()
        await gw.run()

    try:
        run_async(_main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    logger.info("stopped: %s", gw.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
