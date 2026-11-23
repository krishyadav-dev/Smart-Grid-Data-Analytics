"""Map layout (``config/geo_layout.geojson``, written by ``scripts/build_geo_layout.py``)
and the static network metadata the map and pages need."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import yaml

LV_V = 433.0


@lru_cache(maxsize=4)
def layout(config_dir: Path) -> dict:
    path = config_dir / "geo_layout.geojson"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python scripts/build_geo_layout.py")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def meta(config_dir: Path) -> dict:
    meters = yaml.safe_load((config_dir / "meters.yaml").read_text(encoding="utf-8"))
    zdoc = yaml.safe_load((config_dir / "zones.yaml").read_text(encoding="utf-8"))
    bus_zone = {str(b): z["id"] for z in zdoc["zones"] for b in z["buses"]}
    out_meters = []
    for m in meters["meters"]:
        bus = m["load_id"].split("_")[1] if m["role"] in ("dt", "bus_virtual") else "650"
        kva = round(m["i_rated_a"] * math.sqrt(3) * LV_V / 1000.0) if m.get("i_rated_a") else None
        out_meters.append({**m, "bus": bus, "zone": bus_zone.get(bus), "rating_kva": kva,
                           "physical": m["role"] != "bus_virtual"})
    return {
        "feeder_id": meters["feeder_id"],
        "meters": out_meters,
        "zones": [{"id": z["id"], "name": z["name"], "buses": [str(b) for b in z["buses"]], "note": z["note"]}
                  for z in zdoc["zones"]],
        "feeder_tree": {str(k): str(v) for k, v in zdoc["feeder_tree"].items()},
    }
