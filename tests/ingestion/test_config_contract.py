"""C2 (static part) — meters.yaml vs feeder_model.get_load_points()."""

import pytest

from src.ingestion.config import load_meters
from src.ingestion.topics import meter_id_for


@pytest.fixture(scope="module")
def load_points():
    from src.data_source.feeder_model import get_load_points
    return get_load_points()


def test_meters_yaml_matches_get_load_points(load_points):
    meters = load_meters()
    assert meters.feeder_id == "ieee13_11kv"
    yaml_rows = {m.load_id: m for m in meters.meters}
    assert set(yaml_rows) == set(load_points.load_id)
    for i, row in load_points.reset_index(drop=True).iterrows():
        m = yaml_rows[row.load_id]
        # unit ids follow meter_server._load_id_unit_map (row order, 1-based)
        assert m.unit_id == i + 1, row.load_id
        assert m.meter_id == meter_id_for(m.unit_id)
        assert "".join(m.phases_present) == row.phase, row.load_id


def test_unit_map_agrees_with_meter_server():
    from src.data_source.meter_server import _load_id_unit_map
    assert {m.load_id: m.unit_id for m in load_meters().meters} == _load_id_unit_map()


def test_base_voltages_match_network():
    from src.data_source.feeder_model import build_ieee13_network
    net = build_ieee13_network()
    for m in load_meters().meters:
        bus = m.load_id.split("_")[1]
        vn_kv = float(net.bus[net.bus.name == bus].vn_kv.iloc[0])
        assert m.v_base_ln_v == pytest.approx(vn_kv * 1000 / 3 ** 0.5, abs=0.01)
