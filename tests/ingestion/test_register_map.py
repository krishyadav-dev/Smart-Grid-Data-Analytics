"""Table-driven decoding (no Modbus needed)."""

import pytest

from src.ingestion.register_map import RegisterDef, RegisterMap


@pytest.fixture(scope="module")
def rm():
    return RegisterMap.load()


def test_blocks_respect_limit_and_cover_map(rm):
    blocks = rm.blocks()
    assert all(b.count <= 125 for b in blocks)
    covered = {(b.function_code, b.start + i) for b in blocks for i in range(b.count)}
    mapped = {(fc, o) for fc, offs in rm.mapped_offsets().items() for o in offs}
    assert mapped <= covered
    # v2 layout: everything polled is one fc4 block (58 words); the event
    # record (group "event") is not polled
    assert sorted((b.function_code, b.start, b.count) for b in blocks) == [(4, 0, 58)]


def test_block_split_at_125():
    regs = [dict(name=f"r{i}", measurement="x", function_code=4, address=i * 2,
                 width_registers=2, unit="u") for i in range(100)]
    rm = RegisterMap(address_base=0, registers=regs)
    blocks = rm.blocks()
    assert [b.count for b in blocks] == [124, 76]


def test_overlap_rejected():
    with pytest.raises(ValueError):
        RegisterMap(registers=[
            dict(name="a", measurement="x", function_code=4, address=0, width_registers=2, unit="u"),
            dict(name="b", measurement="y", function_code=4, address=1, width_registers=2, unit="u"),
        ])


@pytest.mark.parametrize("words,order,signed,expected", [
    ([0x0001, 0x0002], "big", False, 0x00010002),
    ([0x0002, 0x0001], "little", False, 0x00010002),
    ([0xFFFF, 0xF63C], "big", True, -2500),
    ([0xFFFF, 0xF63C], "big", False, 0xFFFFF63C),
    ([0x8000, 0x0000], "big", True, -(1 << 31)),
])
def test_word_order_and_sign(words, order, signed, expected):
    r = RegisterDef(name="r", measurement="x", function_code=4, address=0,
                    width_registers=2, word_order=order, signed=signed, unit="u")
    assert r.decode_raw(words) == expected


def test_scale_and_quantum(rm):
    v = next(r for r in rm.registers if r.name == "voltage_a")
    assert v.decode([9, 10795]) == 6006.19          # 600619 * 0.01
    e = next(r for r in rm.registers if r.name == "energy_import")
    assert e.decode([0, 0, 0, 1234]) == 1.234       # u64 Wh -> kWh
    big = 5_533_920_000                              # > u32: 630 kW for 2024
    assert e.decode([(big >> s) & 0xFFFF for s in (48, 32, 16, 0)]) == 5_533_920.0


def test_build_measurements_omits_absent_phases_and_totals(rm):
    values = {"voltage_b": 6020.4, "current_b": 23.0, "power_b": 110.5,
              "voltage_a": 0.0, "power_a": 0.0, "reactive_b": 50.05, "energy_import": 9.0}
    m = rm.build_measurements(values, ["b"])
    assert m["voltage_v"] == {"b": 6020.4}
    assert m["active_power_kw"] == {"b": 110.5, "total": 110.5}
    assert m["reactive_power_kvar"] == {"b": 50.05, "total": 50.05}
    assert m["energy_import_kwh"] == 9.0
    assert "t_sim" not in m


def test_total_only_when_all_present_phases_read(rm):
    m = rm.build_measurements({"power_a": 1.0, "power_b": 2.0}, ["a", "b", "c"])
    assert m["active_power_kw"] == {"a": 1.0, "b": 2.0}


def test_decode_time(rm):
    tr = rm.time_register
    secs = 1704048300   # 2024-01-01T00:15:00 IST = 2023-12-31T18:45:00Z
    t = rm.decode_time({tr.name: tr.decode([secs >> 16, secs & 0xFFFF])})
    assert t.isoformat() == "2024-01-01T00:15:00+05:30"
    assert rm.decode_time({tr.name: 0.0}) is None


def test_served_total_is_used_not_recomputed(rm):
    m = rm.build_measurements({"reactive_b": 50.05, "reactive_total": 50.05, "frequency": 50.0}, ["b"])
    assert m["reactive_power_kvar"] == {"b": 50.05, "total": 50.05}
    assert m["frequency_hz"] == 50.0


def test_line_to_line_only_with_both_phases(rm):
    values = {"voltage_ab": 10990.0, "voltage_bc": 11010.0, "voltage_ca": 11000.0,
              "apparent_a": 10.0, "apparent_total": 10.0, "pf_a": 0.9, "pf_total": 0.9}
    m = rm.build_measurements(values, ["a", "b"])
    assert m["voltage_ll_v"] == {"ab": 10990.0}
    assert m["apparent_power_kva"] == {"a": 10.0, "total": 10.0}
    assert m["power_factor"] == {"a": 0.9, "total": 0.9}
    assert "voltage_ll_v" not in rm.build_measurements(values, ["a"])


def test_meta_registers_not_measurements(rm):
    m = rm.build_measurements({"sim_time": 1704048300, "event_counter": 3, "voltage_a": 1.0}, ["a"])
    assert set(m) == {"voltage_v"}
    assert rm.decode_event_counter({"event_counter": 3}) == 3


def test_event_layout_and_sample_blocks(rm):
    ev = rm.event
    assert (ev.header_address, ev.sample_address, ev.values_per_sample) == (100, 120, 6)
    assert ev.max_samples == 6150
    blocks = rm.event_sample_blocks(300)
    assert all(b.count <= 125 and b.count % 6 == 0 for b in blocks)
    assert sum(b.count for b in blocks) == 1800 and blocks[0].start == 120
