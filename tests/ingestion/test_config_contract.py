"""C2 (static part) — meters.yaml vs feeder_model.meter_points()."""

import math

import pytest

from src.ingestion.config import load_meters
from src.ingestion.topics import meter_id_for


@pytest.fixture(scope="module")
def net():
    from src.data_source.feeder_model import build_ieee13_network
    return build_ieee13_network()


@pytest.fixture(scope="module")
def points(net):
    from src.data_source.feeder_model import meter_points
    return meter_points(net)


def test_meters_yaml_matches_meter_points(points):
    meters = load_meters()
    assert meters.feeder_id == "ieee13_11kv"
    yaml_rows = {m.load_id: m for m in meters.meters}
    assert set(yaml_rows) == set(points.load_id)
    for r in points.itertuples():
        m = yaml_rows[r.load_id]
        # unit ids follow meter_server._load_id_unit_map (meter_points order)
        assert m.unit_id == r.unit_id, r.load_id
        assert m.meter_id == meter_id_for(m.unit_id)
        assert m.role == r.role, r.load_id
        assert "".join(m.phases_present) == r.phase, r.load_id


def test_roles_and_counts(points):
    by_role = load_meters().meters
    roles = [m.role for m in by_role]
    assert roles.count("bus_virtual") == 9          # Layer 2 compatibility
    assert roles.count("feeder_head") == 1
    assert roles.count("dt") == 13                  # Part 2C.1 with brief v2 X.3: 13 DTs


def test_unit_map_agrees_with_meter_server():
    from src.data_source.meter_server import _load_id_unit_map
    assert {m.load_id: m.unit_id for m in load_meters().meters} == _load_id_unit_map()


def test_base_voltages_match_network(net):
    d = net["dts"]
    vn = dict(zip(net.bus.index, net.bus.vn_kv))
    for m in load_meters().meters:
        if m.role == "feeder_head":
            kv = float(net.bus[net.bus.name == "650"].vn_kv.iloc[0])
        elif m.role == "dt":
            kv = vn[int(d[d.name == m.load_id].lv_bus.iloc[0])]
        else:   # bus_virtual: the LV side of its DTs
            kv = vn[int(d[d.load_id == m.load_id].lv_bus.iloc[0])]
        assert m.v_base_ln_v == pytest.approx(kv * 1000 / 3 ** 0.5, abs=0.01), m.meter_id


def test_dt_rated_current(net):
    d = net["dts"].set_index("name")
    for m in load_meters().meters:
        if m.role == "dt":
            i = d.at[m.load_id, "rating_kva"] / (math.sqrt(3) * 0.433)
            assert m.i_rated_a == pytest.approx(i, abs=0.01), m.meter_id
