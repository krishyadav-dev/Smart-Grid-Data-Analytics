"""config/geo_layout.geojson stays consistent with the network data and config (plan.md §5.1)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "config" / "geo_layout.geojson"
spec = importlib.util.spec_from_file_location("build_geo_layout", ROOT / "scripts" / "build_geo_layout.py")
bgl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bgl)


@pytest.fixture(scope="module")
def layout() -> dict:
    return json.loads(LAYOUT.read_text(encoding="utf-8"))


def _kind(layout, kind):
    return [f["properties"] | {"_geom": f["geometry"]} for f in layout["features"] if f["properties"]["kind"] == kind]


def test_committed_layout_is_current(layout):
    """The committed file equals a fresh schematic build (re-run the script after config changes)."""
    if layout["metadata"]["mode"] != "schematic":
        pytest.skip("committed layout was routed with Mapbox; geometry is not reproducible offline")
    assert bgl.build("schematic") == json.loads(json.dumps(bgl.build("schematic")))   # deterministic
    assert layout == json.loads(json.dumps(bgl.build("schematic")))


def test_every_bus_and_dt_once(layout):
    zdoc = yaml.safe_load((ROOT / "config" / "zones.yaml").read_text(encoding="utf-8"))
    buses = set(zdoc["feeder_tree"]) | set(zdoc["feeder_tree"].values())
    got = [b["bus"] for b in _kind(layout, "bus")]
    assert sorted(got) == sorted(str(b) for b in buses)
    meters = yaml.safe_load((ROOT / "config" / "meters.yaml").read_text(encoding="utf-8"))["meters"]
    dts = sorted(m["load_id"] for m in meters if m["role"] == "dt")
    assert sorted(d["load_id"] for d in _kind(layout, "dt")) == dts
    by_id = {m["load_id"]: m["meter_id"] for m in meters}
    assert all(d["meter_id"] == by_id[d["load_id"]] for d in _kind(layout, "dt"))


def test_segments_match_feeder_tree_and_zones(layout):
    zdoc = yaml.safe_load((ROOT / "config" / "zones.yaml").read_text(encoding="utf-8"))
    bus_zone = {str(b): z["id"] for z in zdoc["zones"] for b in z["buses"]}
    tree_edges = {f"{p}-{c}" for c, p in zdoc["feeder_tree"].items()}
    segs = _kind(layout, "segment")
    assert {s["segment"] for s in segs} == tree_edges
    for s in segs:
        assert s["zone"] == bus_zone[s["to"]]
    for b in _kind(layout, "bus"):
        assert b["zone"] == bus_zone.get(b["bus"])


def test_drawn_lengths_within_10pct_of_model(layout):
    for s in _kind(layout, "segment"):
        if s["model_length_m"] == 0:
            continue
        drawn = bgl.path_length_m(s["_geom"]["coordinates"])
        assert abs(drawn / s["model_length_m"] - 1) <= 0.10, s["segment"]


def test_model_lengths_follow_network_data():
    edges = bgl.model_edges()
    # 650-632 is 2000 ft at 4.16 kV; Stage B scales by (11/4.16)^2.
    assert edges["650-632"]["model_length_m"] == pytest.approx(2000 * 0.3048 * (11 / 4.16) ** 2, abs=0.1)
    assert edges["632-670"]["length_ft_ieee"] + edges["670-671"]["length_ft_ieee"] == 2000
    assert edges["671-692"]["element"] == "switch" and edges["633-634"]["element"] == "transformer"


def test_length_factor_matches_stage_b():
    pytest.importorskip("pandapower")
    from src.data_source import stage_b
    assert bgl.LENGTH_FACTOR == pytest.approx(stage_b.LENGTH_FACTOR)


def test_line_lengths_match_pandapower_model():
    pytest.importorskip("pandapower")
    from src.data_source import feeder_model
    net = feeder_model.build_network("B")
    model = {n.removeprefix("Line_"): km * 1000 for n, km in zip(net.line.name, net.line.length_km)}
    edges = bgl.model_edges()
    checked = 0
    for key, e in edges.items():
        if e["element"] != "line":
            continue
        name = (key.replace("650-", "RG60-") if key.startswith("650-") else key).replace("-", "_")
        assert name in model, f"no pandapower line for {key}"
        assert e["model_length_m"] == pytest.approx(model[name], rel=1e-3), key
        checked += 1
    assert checked == len(model) == 11


def test_dt_rating_from_rated_current(layout):
    for d in _kind(layout, "dt"):
        assert d["rating_kva"] == round(d["i_rated_a"] * math.sqrt(3) * 433 / 1000)
    assert {d["load_id"]: d["rating_kva"] for d in _kind(layout, "dt")}["dt_675_1"] == 630
