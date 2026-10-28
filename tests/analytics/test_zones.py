"""P4: zones and feeder tree (config/zones.yaml, src/analytics/zones.py)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.analytics import meters as M
from src.analytics import zones as Z

ROOT = Path(__file__).resolve().parents[2]
FAULT_EVENTS = ROOT / "results" / "faults_2024" / "fault_events.parquet"
MV_BUSES = ("650", "632", "633", "645", "646", "670", "671", "680", "684", "611", "652", "692", "675")


def _network_tree() -> dict:
    """Parent of every IEEE bus from the Stage B network: lines, closed bus-bus
    switches and transformers between IEEE-named buses, walked from 650.  RG60
    (the regulator position, joined to 650 by a closed switch) is contracted."""
    from src.data_source.feeder_model import build_network
    logging.getLogger("src.data_source").setLevel(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = build_network("B")
    name = net.bus.name.astype(str)
    keep = set(MV_BUSES) | {"634", "RG60"}
    edges = [(name[r.from_bus], name[r.to_bus]) for r in net.line.itertuples() if r.in_service]
    edges += [(name[r.bus], name[r.element]) for r in net.switch.itertuples() if r.et == "b" and r.closed]
    edges += [(name[r.hv_bus], name[r.lv_bus]) for r in net.trafo.itertuples() if r.in_service]
    adj: dict[str, set] = {}
    for a, b in edges:
        if a in keep and b in keep:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    parent, seen, stack = {}, {"650"}, ["650"]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                stack.append(v)
    assert seen == keep, f"unreached buses: {keep - seen}"
    rg = parent.pop("RG60")
    assert rg == "650"
    return {b: ("650" if p == "RG60" else p) for b, p in parent.items()}


def test_feeder_tree_matches_network():
    assert Z.load().tree == _network_tree()


def test_every_bus_in_exactly_one_zone():
    t = Z.load()
    assert len(t.zones) == 7 and t.zone_ids() == tuple("ABCDEFG")
    for b in MV_BUSES + ("634",):
        assert sum(b in z.buses for z in t.zones) == 1, b
    all_buses = [b for z in t.zones for b in z.buses]
    assert len(all_buses) == len(set(all_buses))
    assert set(all_buses) == set(MV_BUSES) | {"634"}


def test_every_dt_meter_maps_to_one_zone_and_bus_virtual_rejected():
    dts = M.dt_meters()
    assert [m.meter_id for m in dts] == [f"m{k}" for k in range(11, 24)]
    zones = {m.meter_id: Z.zone_of_meter(m.meter_id) for m in dts}
    assert set(zones.values()) <= set("ABCDEFG")
    assert zones["m11"] == "B" and zones["m14"] == "G" and zones["m23"] == "D"
    assert Z.zone_of_meter("m10") == "A"
    for k in range(1, 10):
        with pytest.raises(M.NotPhysicalMeter):
            Z.zone_of_meter(f"m{k:02d}")


def test_physical_meters_by_role():
    ph = M.physical_meters()
    assert [m.meter_id for m in ph] == [f"m{k}" for k in range(10, 24)]
    assert {str(m.role) for m in ph} == {"feeder_head", "dt"}


def test_fault_buses_map_to_zones():
    if not FAULT_EVENTS.exists():
        pytest.skip("results/faults_2024/fault_events.parquet absent")
    ev = pd.read_parquet(FAULT_EVENTS, columns=["kind", "bus"])
    buses = sorted(ev[ev.kind == "fault"].bus.unique())
    assert len(buses) == 12
    for b in buses:
        assert Z.zone_of_bus(b) in "ABCDEFG"


def test_common_upstream_node():
    assert Z.common_upstream_node(["645", "671"]) == "632"
    assert Z.common_upstream_node(["611", "675"]) == "671"
    assert Z.common_upstream_node(["611", "652"]) == "684"
    assert Z.common_upstream_node(["652"]) == "652"
    assert Z.path_to_root("652") == ["652", "684", "671", "670", "632", "650"]
