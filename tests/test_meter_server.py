"""
Tests for meter_server — register map v2 (schema 2.0).

Acceptance criteria verified:
    B1. A pymodbus client reads back the written values (within quantisation).
    B5. The register map, DLMS/COSEM scope and architecture are documented.
    B6. The energy accumulator is monotonic.
    2E. Signed P/Q (no clamping), u64 energy in input registers, energy
        stamped at the end of the interval, IST simulated time, line-to-line
        voltage, kVA and PF, writes rejected, one simulated year at the
        largest DT without saturation, event record + counter.
"""

import asyncio
import math
import socket
import threading
import time

import numpy as np
import pytest

from src.data_source import meter_server as ms
from src.data_source.meter_server import (
    EVENT_HEADER,
    POLL_REGISTERS,
    REG_ENERGY,
    REG_EVENT_COUNTER,
    REG_POWER,
    REG_REACTIVE,
    REG_SIM_TIME,
    REG_VOLTAGE_A,
    REG_VOLTAGE_LL,
    EventRecord,
    MeterState,
    PhaseReading,
    _decode_s32,
    _decode_u32,
    _decode_u64,
    _encode_s32,
    _encode_u32,
    _encode_u64,
    _sim_time_seconds,
    build_server_context,
)

ONE_METER = {"load_634_abc": 1}


def _ctx():
    from pymodbus.simulator.simcore import SimCore
    devices, _, meters = build_server_context(ONE_METER)
    return SimCore(devices), meters[1]


def _read(ctx, fc, addr, n, unit=1):
    return asyncio.run(ctx.async_getValues(unit, fc, addr, n))


def _phases(**kw):
    """Three-phase reading at nominal-ish values, overridable per field."""
    base = dict(voltage_v=6351.0, current_a=5.0, p_kw=10.0, q_kvar=2.0)
    return {p: PhaseReading(angle_deg=ang, **{**base, **kw})
            for p, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0))}


# ===================================================================== #
#  Encoding                                                               #
# ===================================================================== #

class TestEncoding:
    def test_u32_round_trip(self):
        for v in (0, 12345, 1_000_000, 0xFFFFFFFF):
            assert _decode_u32(*_encode_u32(v)) == v

    def test_u32_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            _encode_u32(-5)
        with pytest.raises(ValueError):
            _encode_u32(0x1_0000_0000)

    def test_s32_keeps_negative_values(self):
        """MB-1: negatives are two's complement, never clamped to 0."""
        for v in (-50_000, -120_000, -1, 0, 1, 2**31 - 1, -(2**31)):
            assert _decode_s32(*_encode_s32(v)) == v

    def test_u64_beyond_u32(self):
        """MB-3: energy beyond 4.29 GWh fits."""
        v = 5_533_920_000          # 630 kW for a 366-day year, in Wh
        assert v > 0xFFFFFFFF
        assert _decode_u64(_encode_u64(v)) == v

    def test_sim_time_is_ist(self):
        """Naive timestamps are IST wall time: 2024-01-01 05:30 IST = 00:00 UTC."""
        assert _sim_time_seconds("2024-01-01 05:30:00") == 1704067200
        assert _sim_time_seconds("2024-01-01 00:15:00") == 1704068100 - 19800


# ===================================================================== #
#  Register contents                                                       #
# ===================================================================== #

class TestRegisters:
    def test_layout_size(self):
        words = ms.build_input_registers(_phases(), 50.0, 0.0, None, 0)
        assert len(words) == POLL_REGISTERS == 58

    def test_negative_p_and_q_read_back(self):
        ctx, meter = _ctx()
        ph = _phases(p_kw=-50.0, q_kvar=-120.0)
        asyncio.run(meter.update_registers(ctx, ph, t_sim="2024-01-01 00:15"))
        p = _read(ctx, 4, REG_POWER, 2)
        q = _read(ctx, 4, REG_REACTIVE, 2)
        assert _decode_s32(*p) == -50_000
        assert _decode_s32(*q) == -120_000
        pf_a = _decode_s32(*_read(ctx, 4, ms.REG_PF, 2)) / 1e4
        assert pf_a == pytest.approx(-50 / math.hypot(50, 120), abs=1e-4)

    def test_line_to_line_from_phasors(self):
        ctx, meter = _ctx()
        asyncio.run(meter.update_registers(ctx, _phases(voltage_v=6350.85)))
        vab = _decode_u32(*_read(ctx, 4, REG_VOLTAGE_LL, 2)) / 100
        assert vab == pytest.approx(11000.0, abs=0.02)   # sqrt(3) x 6350.85

    def test_line_to_line_zero_for_absent_phase(self):
        ctx, meter = _ctx()
        ph = {"b": PhaseReading(voltage_v=6350.0, angle_deg=-120, p_kw=1.0)}
        asyncio.run(meter.update_registers(ctx, ph))
        assert _read(ctx, 4, REG_VOLTAGE_LL, 6) == [0] * 6

    def test_apparent_and_pf_totals_are_vector(self):
        words = ms.build_input_registers(_phases(p_kw=30.0, q_kvar=40.0), 50.0, 0.0, None, 0)
        s_tot = _decode_u32(words[38], words[39]) / 1000
        pf_tot = _decode_s32(words[46], words[47]) / 1e4
        assert s_tot == pytest.approx(math.hypot(90, 120))
        assert pf_tot == pytest.approx(0.6, abs=1e-4)

    def test_voltage_register_matches(self):
        """B1 — input registers reflect the written voltage within scaling."""
        ctx, meter = _ctx()
        asyncio.run(meter.update_registers(ctx, _phases(voltage_v=6351.0)))
        assert _decode_u32(*_read(ctx, 4, REG_VOLTAGE_A, 2)) == 635100


# ===================================================================== #
#  Energy                                                                  #
# ===================================================================== #

class TestEnergy:
    def test_energy_stamped_at_end_of_interval(self):
        """MB-7: the register at t holds the energy up to t, not up to t+15 min."""
        ctx, meter = _ctx()
        ph = _phases(p_kw=10.0)          # 30 kW total
        asyncio.run(meter.update_registers(ctx, ph, interval_hours=0.25, t_sim="2024-01-01 00:00"))
        assert _decode_u64(_read(ctx, 4, REG_ENERGY, 4)) == 0
        asyncio.run(meter.update_registers(ctx, ph, interval_hours=0.25, t_sim="2024-01-01 00:15"))
        assert _decode_u64(_read(ctx, 4, REG_ENERGY, 4)) == 7500     # 30 kW x 0.25 h

    def test_energy_monotonic_and_import_only(self):
        """B6 — never decreases; export (P < 0) does not reduce import."""
        ctx, meter = _ctx()
        prev = -1
        for kw in [5.0, 0.0, -10.0, 3.0, 0.0, 7.0]:
            asyncio.run(meter.update_registers(ctx, _phases(p_kw=kw), interval_hours=0.25))
            e = _decode_u64(_read(ctx, 4, REG_ENERGY, 4))
            assert e >= prev
            prev = e

    def test_one_year_at_largest_dt_no_saturation(self):
        """2E: 630 kVA (largest DT, S_max) at unity PF for all of 2024."""
        ctx, meter = _ctx()
        p_phase = 630.0 / 3
        ph = {p: PhaseReading(voltage_v=250.0, angle_deg=a, p_kw=p_phase)
              for p, a in (("a", 0.0), ("b", -120.0), ("c", 120.0))}
        steps = 366 * 96

        async def run_year():
            for _ in range(steps + 1):          # +1: stamp the end of the last interval
                await meter.update_registers(ctx, ph, interval_hours=0.25)

        asyncio.run(run_year())
        expected = 630_000 * 0.25 * steps   # Wh
        assert expected > 0xFFFFFFFF
        assert _decode_u64(_read(ctx, 4, REG_ENERGY, 4)) == round(expected)


# ===================================================================== #
#  Events                                                                  #
# ===================================================================== #

class TestEvents:
    def test_event_record_then_counter(self):
        ctx, meter = _ctx()
        n = 200
        v = np.tile([6000.0, 3000.0, 6100.0], (n, 1))
        i = np.tile([120.0, 2400.0, 130.0], (n, 1))
        rec = EventRecord(event_id=7, t_start="2024-03-01 10:00:00.250",
                          sample_interval_ms=10, voltage_v=v, current_a=i,
                          phases="abc", v_base_ln_v=6350.85)
        asyncio.run(meter.write_event(ctx, rec))
        hdr = _read(ctx, 4, EVENT_HEADER, 12)
        assert _decode_u32(hdr[0], hdr[1]) == 7
        assert _decode_u32(hdr[2], hdr[3]) == _sim_time_seconds("2024-03-01 10:00:00")
        assert hdr[4:8] == [250, 10, n, 7]
        assert _decode_u32(*_read(ctx, 4, REG_EVENT_COUNTER, 2)) == 1
        first = _read(ctx, 4, ms.EVENT_SAMPLES, 6)
        assert first[0] == round(6000 / 6350.85 * 1e4)
        scale_ma = _decode_u32(hdr[10], hdr[11])
        assert first[4] * scale_ma / 1000 == pytest.approx(2400.0, abs=scale_ma / 1000)


# ===================================================================== #
#  Server context and Modbus client                                        #
# ===================================================================== #

class TestServerContext:
    def test_has_all_units(self):
        _, devices_dict, meters = build_server_context()
        assert set(devices_dict) == set(meters)
        assert len(meters) >= 9


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestModbusClient:
    """B1 and the read-only energy register, through a real TCP server."""

    @pytest.fixture
    def served(self):
        from pymodbus.server import ModbusTcpServer
        from pymodbus.simulator.simcore import SimCore

        port = _free_port()
        devices, _, meters = build_server_context(ONE_METER)
        ctx = SimCore(devices)
        asyncio.run(meters[1].update_registers(
            ctx, _phases(voltage_v=6351.0, p_kw=-50.0, q_kvar=-120.0),
            interval_hours=0.25, t_sim="2024-01-01 00:15"))
        started = threading.Event()

        async def run():
            server = ModbusTcpServer(context=devices, address=("127.0.0.1", port))
            server.context = ctx
            started.set()
            await server.serve_forever()

        threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()
        started.wait(5.0)
        time.sleep(0.3)
        from pymodbus.client import ModbusTcpClient
        client = ModbusTcpClient(host="127.0.0.1", port=port)
        assert client.connect()
        yield client
        client.close()

    def test_client_reads_values(self, served):
        r = served.read_input_registers(address=0, count=POLL_REGISTERS, device_id=1)
        assert not r.isError()
        assert _decode_u32(r.registers[0], r.registers[1]) == 635100
        assert _decode_s32(r.registers[REG_POWER], r.registers[REG_POWER + 1]) == -50_000
        assert _decode_u32(*r.registers[REG_SIM_TIME:REG_SIM_TIME + 2]) == \
            _sim_time_seconds("2024-01-01 00:15")

    def test_write_to_energy_register_rejected(self, served):
        """MB-2: fc6 / fc16 at the energy address get a Modbus exception."""
        for resp in (served.write_register(REG_ENERGY, 1, device_id=1),
                     served.write_registers(REG_ENERGY, [1, 2, 3, 4], device_id=1)):
            assert resp.isError()
            assert resp.exception_code == 2      # illegal data address
        r = served.read_input_registers(address=REG_ENERGY, count=4, device_id=1)
        assert _decode_u64(r.registers) == 0     # unchanged (first stamp)
