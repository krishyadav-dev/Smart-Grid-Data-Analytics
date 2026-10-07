"""
Modbus TCP Smart-Meter Server — DLMS/COSEM Data-Model Aligned.

Architecture choice
-------------------
A **single Modbus TCP server with multiple unit IDs** (one per simulated load
point from the IEEE 13-bus feeder) was chosen over N separate servers because:
  - It requires only one TCP port and one process.
  - pymodbus's ``ModbusServerContext`` natively supports multiple device
    contexts keyed by unit_id (slave address).
  - Orchestration is simpler: one coroutine drives all meter updates.

Each load point from ``get_load_points()`` maps to a distinct unit_id.

DLMS/COSEM scope
-----------------
The register map is **data-model aligned** with DLMS/COSEM OBIS codes — each
register's comment notes the corresponding OBIS object it conceptually maps
to (e.g. ``1.8.0`` for total active energy import, per IEC 62056-6-2 Blue
Book).

**Explicitly NOT implemented:** DLMS/COSEM wire protocol, HDLC framing, COSEM
security association layer, or A-XDR encoding.  This is a deliberate scoping
decision: Layer 1 stops at Modbus TCP registers.  A future DLMS/COSEM gateway
(Layer 2) would translate on top of this server.

Register Map
------------
All values are stored as **32-bit unsigned integers** encoded across two
consecutive 16-bit Modbus registers in **big-endian word order** (high word
first).  This gives a 32-bit unsigned range of 0–4,294,967,295 without
negative-number complications.

Energy register overflow: the Wh accumulator uses 32-bit unsigned integers.
At the maximum representable value (4,294,967,295 Wh ≈ 4.3 GWh per meter),
the counter will saturate (clamp to max, not wrap).  In practice a single
load-bus meter will not reach this in the simulation window.

+----------+---------------------+----------+--------------+----------+-------+
| Regs     | Description         | Scaling  | OBIS Code    | Fn Code  | Type  |
+==========+=====================+==========+==============+==========+=======+
|   0–1    | Voltage Phase A     | V × 100  | 1-1:32.7.0   | IR (4)   | u32   |
|   2–3    | Voltage Phase B     | V × 100  | 1-1:52.7.0   | IR (4)   | u32   |
|   4–5    | Voltage Phase C     | V × 100  | 1-1:72.7.0   | IR (4)   | u32   |
|   6–7    | Current Phase A     | mA (×1k) | 1-1:31.7.0   | IR (4)   | u32   |
|   8–9    | Current Phase B     | mA (×1k) | 1-1:51.7.0   | IR (4)   | u32   |
|  10–11   | Current Phase C     | mA (×1k) | 1-1:71.7.0   | IR (4)   | u32   |
|  12–13   | Active Power Ph A   | W        | 1-1:21.7.0   | IR (4)   | u32   |
|  14–15   | Active Power Ph B   | W        | 1-1:41.7.0   | IR (4)   | u32   |
|  16–17   | Active Power Ph C   | W        | 1-1:61.7.0   | IR (4)   | u32   |
|  18–19   | Total Energy Import | Wh       | 1-1:1.8.0    | HR (3)   | u32   |
+----------+---------------------+----------+--------------+----------+-------+

Notes on 32-bit word order: high word at base address, low word at base+1.
Client reads 2 registers; decode as ``value = (regs[0] << 16) | regs[1]``.

Open question (not built — see final report)
---------------------------------------------
The brief specifies that meter registers reflect the *true* feeder values.
However, theft evaluation likely needs meters to report the *manipulated*
(reported) values for theft consumers.  The current implementation reports
true feeder values.  A future extension would inject
``theft_injector``-transformed series for theft-labeled consumers.

References
----------
- pymodbus v3.x documentation — asynchronous Modbus TCP server API.
- DLMS/COSEM Blue Book (IEC 62056-6-2) — OBIS object model.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, Dict, Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register addresses (base register index for each 32-bit measurement)
# ---------------------------------------------------------------------------
REG_VOLTAGE_A   = 0    # OBIS 1-1:32.7.0  — Phase A voltage
REG_VOLTAGE_B   = 2    # OBIS 1-1:52.7.0  — Phase B voltage
REG_VOLTAGE_C   = 4    # OBIS 1-1:72.7.0  — Phase C voltage
REG_CURRENT_A   = 6    # OBIS 1-1:31.7.0  — Phase A current
REG_CURRENT_B   = 8    # OBIS 1-1:51.7.0  — Phase B current
REG_CURRENT_C   = 10   # OBIS 1-1:71.7.0  — Phase C current
REG_POWER_A     = 12   # OBIS 1-1:21.7.0  — Phase A active power
REG_POWER_B     = 14   # OBIS 1-1:41.7.0  — Phase B active power
REG_POWER_C     = 16   # OBIS 1-1:61.7.0  — Phase C active power
REG_ENERGY      = 18   # OBIS 1-1:1.8.0   — Total energy import (holding)

TOTAL_REGISTERS = 20   # 10 measurements × 2 registers each

# IR (input register) count and HR (holding register) start/count
IR_COUNT        = 18   # registers 0–17  (voltage, current, power)
HR_START        = 0    # holding registers start from address 0
HR_COUNT        = 2    # registers 18–19 (energy accumulator stored in HR 0–1)

# Load point → unit_id mapping (stable — Part B contract)
LOAD_BUS_UNIT_MAP = {
    "634": 1,
    "645": 2,
    "646": 3,
    "652": 4,
    "671": 5,
    "675": 6,
    "692": 7,
    "611": 8,
}


# ---------------------------------------------------------------------------
# 32-bit encoding helpers
# ---------------------------------------------------------------------------

def _encode_u32(value: float) -> tuple[int, int]:
    """Encode a float as two 16-bit unsigned registers (big-endian u32).

    Saturates at 0 (below) and 0xFFFFFFFF (above).
    """
    clamped = max(0, int(round(value)))
    clamped = min(clamped, 0xFFFFFFFF)
    hi = (clamped >> 16) & 0xFFFF
    lo = clamped & 0xFFFF
    return hi, lo


def _decode_u32(hi: int, lo: int) -> int:
    """Decode two 16-bit registers back to a 32-bit unsigned integer."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


# ---------------------------------------------------------------------------
# Meter state
# ---------------------------------------------------------------------------

class MeterState:
    """Tracks per-meter energy accumulator and writes register values.

    The meter owns a pair of raw register-value lists extracted from the
    ``ModbusDeviceContext``'s internal ``simdata`` blocks.  In pymodbus 3.15
    the legacy ``setValues``/``getValues`` API on ``ModbusDeviceContext`` was
    removed.  Data is now written directly to:

    - **Input registers** (fc=4): ``device.simdevice.simdata[3][0].values``
    - **Holding registers** (fc=3): ``device.simdevice.simdata[2][0].values``

    The ``ModbusSequentialDataBlock`` was created at *address=1*, so the
    Modbus client reads at address=1 to access offset 0 of the values list.
    The values list is zero-indexed from that base address.

    Energy accumulator (REG_ENERGY) is stored in holding registers (fc=3).
    Instantaneous measurements (voltage, current, power) are stored in input
    registers (fc=4).
    """

    def __init__(self, unit_id: int, load_id: str, ir_values: list, hr_values: list):
        self.unit_id = unit_id
        self.load_id = load_id
        # Derive bus_name from load_id (format: "load_<bus>_<phase>")
        parts = load_id.split("_")
        self.bus_name = parts[1] if len(parts) >= 2 else load_id
        self.energy_wh: float = 0.0  # monotonically increasing
        # Direct references to the underlying register value lists
        self._ir_values = ir_values   # input registers (fc=4)
        self._hr_values = hr_values   # holding registers (fc=3)

    def update_registers(
        self,
        slave_ctx: Any = None,          # kept for backward-compat; ignored
        voltage_a: float = 0.0,
        voltage_b: float = 0.0,
        voltage_c: float = 0.0,
        current_a: float = 0.0,
        current_b: float = 0.0,
        current_c: float = 0.0,
        power_a_kw: float = 0.0,
        power_b_kw: float = 0.0,
        power_c_kw: float = 0.0,
        interval_hours: float = 0.25,
    ) -> None:
        """Write measurement values directly into the raw register value lists.

        Parameters
        ----------
        slave_ctx : any, optional
            Kept for backward-compatibility with callers that pass the device
            context.  Not used in pymodbus 3.15+; data is written via the
            ``_ir_values`` / ``_hr_values`` lists stored at construction.
        voltage_a/b/c : float
            Phase voltages in volts (V).
        current_a/b/c : float
            Phase currents in amps (A).
        power_a/b/c_kw : float
            Phase active powers in kilowatts (kW).
        interval_hours : float
            Time since last update (hours), used for energy accumulation.
        """
        # Accumulate energy (monotonically increasing — never decreases)
        total_power_kw = power_a_kw + power_b_kw + power_c_kw
        self.energy_wh += total_power_kw * 1000.0 * interval_hours  # kW → Wh

        # Build the 18 input-register values (9 measurements × 2 regs each)
        ir_flat: list[int] = []
        for raw_val in [
            voltage_a * 100,        # V → V×100      (OBIS 32/52/72.7.0)
            voltage_b * 100,
            voltage_c * 100,
            current_a * 1000,       # A → mA×1000    (OBIS 31/51/71.7.0)
            current_b * 1000,
            current_c * 1000,
            power_a_kw * 1000,      # kW → W         (OBIS 21/41/61.7.0)
            power_b_kw * 1000,
            power_c_kw * 1000,
        ]:
            hi, lo = _encode_u32(raw_val)
            ir_flat.extend([hi, lo])

        # Write input registers.
        # In pymodbus 3.15 with a block created at address=1, a Modbus client
        # reading at address N receives values[N].  values[0] is unreachable.
        # We therefore write data starting at index 1 so that:
        #   Modbus address 1 → values[1] = hi word of register pair 0
        #   Modbus address 2 → values[2] = lo word of register pair 0  ... etc.
        for i, v in enumerate(ir_flat):
            idx = i + 1   # skip values[0] (unreachable)
            if idx < len(self._ir_values):
                self._ir_values[idx] = v

        # Write energy accumulator to holding registers at index 1 & 2
        hi_e, lo_e = _encode_u32(self.energy_wh)
        if len(self._hr_values) >= 3:
            self._hr_values[1] = hi_e
            self._hr_values[2] = lo_e


# ---------------------------------------------------------------------------
# Server context builder
# ---------------------------------------------------------------------------

def build_server_context():
    """Create a Modbus server context with one device per load point.

    In pymodbus 3.15 the ``ModbusDeviceContext`` wraps a ``SimDevice`` whose
    ``simdata`` tuple is ordered ``(coils, discrete_inputs, holding_regs,
    input_regs)`` (indices 0–3).  We grab direct references to the
    ``SimData.values`` lists at construction so that ``MeterState`` can write
    without calling the removed ``setValues`` method.

    Returns
    -------
    (server_context, devices, meter_states)
        - ``server_context`` — ``ModbusServerContext`` for passing to
          ``StartAsyncTcpServer``.
        - ``devices`` — ``dict[unit_id, ModbusDeviceContext]`` — the per-device
          datastores keyed by unit_id.
        - ``meter_states`` — ``dict[unit_id, MeterState]``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from pymodbus.datastore import (
            ModbusSequentialDataBlock,
            ModbusServerContext,
            ModbusDeviceContext,
        )

        devices: dict[int, Any] = {}
        meters: dict[int, MeterState] = {}

        for load_id, unit_id in _load_id_unit_map().items():
            # Blocks start at address=1 (pymodbus 3.15 requires address >= 1).
            # In this layout Modbus address N maps to values[N], so values[0]
            # is unreachable.  We allocate TOTAL_REGISTERS+1 entries so that
            # useful data at values[1..TOTAL_REGISTERS] maps to Modbus
            # addresses 1..TOTAL_REGISTERS.  The +1 extra entry is consumed
            # by the holding-register energy pair which also starts at index 1.
            n = TOTAL_REGISTERS + 1
            ir = ModbusSequentialDataBlock(1, [0] * n)
            hr = ModbusSequentialDataBlock(1, [0] * n)
            co = ModbusSequentialDataBlock(1, [0])
            di = ModbusSequentialDataBlock(1, [0])
            device = ModbusDeviceContext(ir=ir, hr=hr, co=co, di=di)
            devices[unit_id] = device

            # Extract raw value lists from SimDevice.simdata:
            #   simdata[0] = coils, [1] = discrete inputs,
            #   [2] = holding regs (fc=3), [3] = input regs (fc=4)
            sd = device.simdevice
            ir_values: list = sd.simdata[3][0].values   # input regs
            hr_values: list = sd.simdata[2][0].values   # holding regs

            meters[unit_id] = MeterState(unit_id, load_id, ir_values, hr_values)

        ctx = ModbusServerContext(devices=devices, single=False)

    return ctx, devices, meters


def _load_id_unit_map() -> dict[str, int]:
    """Map load_id → unit_id.  Tries feeder model; falls back to static map."""
    try:
        from .feeder_model import get_load_points
        lp = get_load_points()
        return {row["load_id"]: i + 1
                for i, (_, row) in enumerate(lp.iterrows())}
    except Exception:
        # Fallback: derive from LOAD_BUS_UNIT_MAP
        return {f"load_{bus}_?": uid for bus, uid in LOAD_BUS_UNIT_MAP.items()}


# ---------------------------------------------------------------------------
# Background updater
# ---------------------------------------------------------------------------

async def _update_loop(
    devices: dict[int, Any],
    meters: dict[int, MeterState],
    timeseries_data: pd.DataFrame,
    playback_speed: float,
    interval_minutes: float = 15.0,
) -> None:
    """Background coroutine that updates meter registers from timeseries data.

    Parameters
    ----------
    devices : dict[unit_id, ModbusDeviceContext]
        Per-device datastores (returned by ``build_server_context``).
    meters : dict[unit_id, MeterState]
        Meter state objects.
    timeseries_data : DataFrame
        Output of ``run_timeseries`` — must have columns
        ``timestamp, load_id, phase, voltage_v, current_a,
        active_power_kw, reactive_power_kvar``.
    playback_speed : float
        Acceleration factor (10 = 10× real-time).
    interval_minutes : float
        Simulation time step in minutes.
    """
    interval_hours = interval_minutes / 60.0
    wall_sleep = (interval_minutes * 60.0) / playback_speed

    timestamps = sorted(timeseries_data["timestamp"].unique())

    for ts in timestamps:
        ts_data = timeseries_data[timeseries_data["timestamp"] == ts]

        for unit_id, meter in meters.items():
            slave_ctx = devices.get(unit_id)
            if slave_ctx is None:
                continue

            # Match by load_id
            rows = ts_data[ts_data["load_id"] == meter.load_id]
            if rows.empty:
                continue

            # Per-phase lookup
            def _phase_val(col: str, ph: str) -> float:
                row = rows[rows["phase"] == ph]
                return float(row[col].iloc[0]) if not row.empty else 0.0

            meter.update_registers(
                slave_ctx,
                voltage_a=_phase_val("voltage_v", "a"),
                voltage_b=_phase_val("voltage_v", "b"),
                voltage_c=_phase_val("voltage_v", "c"),
                current_a=_phase_val("current_a", "a"),
                current_b=_phase_val("current_a", "b"),
                current_c=_phase_val("current_a", "c"),
                power_a_kw=_phase_val("active_power_kw", "a"),
                power_b_kw=_phase_val("active_power_kw", "b"),
                power_c_kw=_phase_val("active_power_kw", "c"),
                interval_hours=interval_hours,
            )

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
) -> None:
    """Start the Modbus TCP server and background updater."""
    from pymodbus.server import ModbusTcpServer

    if host is None:
        host = config.MODBUS_HOST
    if port is None:
        port = config.MODBUS_PORT
    if playback_speed is None:
        playback_speed = config.PLAYBACK_SPEED

    ctx, devices, meters = build_server_context()

    if timeseries_data is not None and not timeseries_data.empty:
        asyncio.create_task(
            _update_loop(devices, meters, timeseries_data, playback_speed)
        )

    logger.info(
        "Starting Modbus TCP server on %s:%d with %d meters",
        host, port, len(meters),
    )
    server = ModbusTcpServer(context=ctx, address=(host, port))
    await server.serve_forever()


def run_server(
    host: str | None = None,
    port: int | None = None,
    timeseries_data: pd.DataFrame | None = None,
) -> None:
    """Synchronous wrapper to start the meter server."""
    asyncio.run(start_meter_server(host, port, timeseries_data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Starting meter server on {config.MODBUS_HOST}:{config.MODBUS_PORT}")
    print(f"Unit IDs: {LOAD_BUS_UNIT_MAP}")
    print("Press Ctrl+C to stop.\n")
    run_server()
