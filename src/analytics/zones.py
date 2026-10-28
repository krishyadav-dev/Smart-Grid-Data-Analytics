"""Fault-location zones and the feeder tree (Layer 3 P4; docs/Layer3_plan.md §1
Step 7), loaded from ``config/zones.yaml``.

Configuration, not labels: the fault engine reads it for zone location and
``evaluate.py`` maps each fault's true bus through the same table.  A DT's zone
comes from its bus via the network's DT table (``feeder_model.build_network``),
never from DT names in the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from src.data_source.config import PROJECT_ROOT

from . import meters as M

ZONES_YAML = PROJECT_ROOT / "config" / "zones.yaml"
ROOT_BUS = "650"


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    buses: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class ZoneTable:
    tree: dict                       # bus -> parent bus
    zones: tuple[Zone, ...]

    def zone_ids(self) -> tuple[str, ...]:
        return tuple(z.id for z in self.zones)


@lru_cache(maxsize=4)
def load(path: str | Path = ZONES_YAML) -> ZoneTable:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tree = {str(k): str(v) for k, v in doc["feeder_tree"].items()}
    zones = tuple(Zone(str(z["id"]), str(z["name"]), tuple(str(b) for b in z["buses"]), str(z["note"]))
                  for z in doc["zones"])
    return ZoneTable(tree, zones)


def parent(bus: str, table: ZoneTable | None = None) -> str | None:
    """Parent of ``bus`` on the feeder tree; ``None`` for the root (650)."""
    t = table or load()
    bus = str(bus)
    if bus == ROOT_BUS:
        return None
    if bus not in t.tree:
        raise KeyError(f"bus {bus} is not on the feeder tree")
    return t.tree[bus]


def path_to_root(bus: str, table: ZoneTable | None = None) -> list[str]:
    """[bus, parent, ..., 650]."""
    out = [str(bus)]
    while (p := parent(out[-1], table)) is not None:
        out.append(p)
    return out


def common_upstream_node(buses: Iterable[str], table: ZoneTable | None = None) -> str:
    """Most upstream common node of a set of buses: the deepest bus that lies on
    every bus's path to the root (Step 7, item 3)."""
    paths = [path_to_root(b, table) for b in buses]
    if not paths:
        raise ValueError("empty bus set")
    common = set(paths[0]).intersection(*map(set, paths[1:]))
    return next(b for b in paths[0] if b in common)        # deepest first along the path


def zone_of_bus(bus: str, table: ZoneTable | None = None) -> str:
    t = table or load()
    hits = [z.id for z in t.zones if str(bus) in z.buses]
    if len(hits) != 1:
        raise KeyError(f"bus {bus} is in {len(hits)} zones")
    return hits[0]


@lru_cache(maxsize=1)
def dt_bus() -> dict[str, str]:
    """DT load_id -> bus, from the network's DT table (Stage B)."""
    import logging
    import warnings

    from src.data_source.feeder_model import build_network
    lvl = logging.getLogger("src.data_source").level
    logging.getLogger("src.data_source").setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = build_network("B")
    finally:
        logging.getLogger("src.data_source").setLevel(lvl)
    return {str(d.name): str(d.bus) for d in net["dts"].itertuples()}


def bus_of_meter(meter_id: str) -> str:
    """Bus a physical meter reads: 650 for the feeder head, the DT's bus for a DT
    meter.  Any other meter id raises :class:`meters.NotPhysicalMeter`."""
    m = M.meter(meter_id)
    if str(m.role) == "feeder_head":
        return ROOT_BUS
    return dt_bus()[m.load_id]


def zone_of_meter(meter_id: str, table: ZoneTable | None = None) -> str:
    return zone_of_bus(bus_of_meter(meter_id), table)
