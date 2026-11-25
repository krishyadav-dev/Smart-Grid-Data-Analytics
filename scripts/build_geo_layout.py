"""Build config/geo_layout.geojson: the Stage B feeder placed on a map of Delhi (plan.md §5.1).

The IEEE 13-bus test feeder has line lengths but no coordinates.  This script
gives it an *illustrative* placement so the Layer 4 dashboard can draw it on a
Mapbox base map.  Lengths come from the network data, never from this script:
``ieee13_data.LINE_SEGMENTS`` (feet) x the Stage B length factor (11/4.16)^2,
with line 632-671 split at 670 one third along, exactly as
``feeder_model.build_network`` does.

Modes
-----
``--mode schematic`` (default, no network): starting at the anchor in
``config/geo_anchor.yaml``, walk the feeder tree (``config/zones.yaml``) and
draw every edge as a straight line on its configured bearing at its model
length.

``--mode mapbox`` (needs ``MAPBOX_TOKEN``): the same walk, but each line
segment is routed along real roads with the Mapbox Directions API, starting at
the (already routed) parent bus.  The route is cut at the model length, so the
drawn length equals the model length whenever the road route is long enough;
otherwise the full route is kept and its ratio is reported (the test allows
+/- 10 %).

PowerShell::

    python scripts/build_geo_layout.py                       # schematic
    $env:MAPBOX_TOKEN = "pk...."; python scripts/build_geo_layout.py --mode mapbox
    python scripts/build_geo_layout.py --mode mapbox        # or MAPBOX_TOKEN=pk.... in .env
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_source import ieee13_data as ieee  # noqa: E402  (pure data, no pandapower)

ANCHOR_YAML = ROOT / "config" / "geo_anchor.yaml"
ZONES_YAML = ROOT / "config" / "zones.yaml"
METERS_YAML = ROOT / "config" / "meters.yaml"
NTL_INPUTS = ROOT / "results" / "layer3" / "ntl" / "inputs.json"
OUT = ROOT / "config" / "geo_layout.geojson"

FT_TO_M = 0.3048
V_STAGE_B_KV = 11.0
# Same expression as stage_b.LENGTH_FACTOR (that module imports pandapower; this one
# must run without it).  tests/dashboard/test_geo_layout.py checks the two agree.
LENGTH_FACTOR = (V_STAGE_B_KV / ieee.V_PRIMARY_KV) ** 2
LV_V = 433.0                       # DT LV line-to-line voltage (IS 1180), meters.yaml
EARTH_R = 6_371_008.8


# --------------------------------------------------------------------------- geometry

def offset(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Point dist_m from (lat, lon) on the given bearing (great-circle, spherical earth)."""
    br, la1, lo1, d = math.radians(bearing_deg), math.radians(lat), math.radians(lon), dist_m / EARTH_R
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(br))
    lo2 = lo1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance between two (lon, lat) points, metres."""
    lo1, la1, lo2, la2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def path_length_m(coords: list) -> float:
    return sum(haversine_m(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def cut_path(coords: list, length_m: float) -> list:
    """The first length_m metres of a (lon, lat) path (the whole path if shorter)."""
    out, acc = [coords[0]], 0.0
    for a, b in zip(coords, coords[1:]):
        seg = haversine_m(a, b)
        if acc + seg >= length_m:
            f = (length_m - acc) / seg if seg else 0.0
            out.append([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])])
            return out
        out.append(b)
        acc += seg
    return out


def circle(lat: float, lon: float, radius_m: float, n: int = 48) -> list:
    ring = [[round(x, 7) for x in reversed(offset(lat, lon, 360.0 * k / n, radius_m))] for k in range(n)]
    return ring + [ring[0]]


# --------------------------------------------------------------------------- network data

def model_edges() -> dict[str, dict]:
    """Feeder-tree edge "parent-child" -> element and Stage B length (m)."""
    edges = {}
    for a, b, ft, cfg in ieee.LINE_SEGMENTS:
        if (a, b) == ("632", "671"):          # split at the lumped distributed load (feeder_model)
            parts = [("632", "670", 667), ("670", "671", 1333)]
        else:
            parts = [(a, b, ft)]
        for fa, fb, f in parts:
            element = {"XFM-1": "transformer", "Switch": "switch"}.get(cfg, "line")
            edges[f"{fa}-{fb}"] = {"from": fa, "to": fb, "element": element,
                                   "config": None if element != "line" else int(cfg),
                                   "length_ft_ieee": f,
                                   "model_length_m": round(f * FT_TO_M * LENGTH_FACTOR, 1)}
    return edges


def load_inputs() -> tuple[dict, dict, dict, dict, dict]:
    anchor = yaml.safe_load(ANCHOR_YAML.read_text(encoding="utf-8"))
    zdoc = yaml.safe_load(ZONES_YAML.read_text(encoding="utf-8"))
    tree = {str(k): str(v) for k, v in zdoc["feeder_tree"].items()}
    bus_zone = {str(b): z["id"] for z in zdoc["zones"] for b in z["buses"]}
    meters = yaml.safe_load(METERS_YAML.read_text(encoding="utf-8"))["meters"]
    households = {}
    if NTL_INPUTS.exists():
        households = json.loads(NTL_INPUTS.read_text(encoding="utf-8"))["consumer_index"]["per_dt"]
    return anchor, tree, bus_zone, meters, households


# --------------------------------------------------------------------------- Mapbox routing

def mapbox_route(token: str, profile: str, a: tuple[float, float], b: tuple[float, float]) -> list:
    """(lon, lat) road path from a to b (Mapbox Directions API v5, GeoJSON geometry)."""
    coords = f"{a[0]:.6f},{a[1]:.6f};{b[0]:.6f},{b[1]:.6f}"
    q = urllib.parse.urlencode({"geometries": "geojson", "overview": "full", "access_token": token})
    url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}?{q}"
    with urllib.request.urlopen(url, timeout=30) as r:          # honours HTTPS_PROXY
        doc = json.loads(r.read().decode("utf-8"))
    if doc.get("code") != "Ok" or not doc.get("routes"):
        raise RuntimeError(f"Mapbox Directions failed for {coords}: {doc.get('code')} {doc.get('message')}")
    return doc["routes"][0]["geometry"]["coordinates"]


# --------------------------------------------------------------------------- build

def build(mode: str = "schematic", token: str | None = None) -> dict:
    anchor, tree, bus_zone, meters, households = load_inputs()
    edges = model_edges()
    bearings = anchor["bearings_deg"]
    disp = float(anchor["zero_length_display_m"])
    root = str(anchor["anchor"]["bus"])

    children: dict[str, list[str]] = {}
    for child, par in tree.items():
        children.setdefault(par, []).append(child)
    missing = {f"{p}-{c}" for c, p in tree.items()} - set(edges)
    if missing:
        raise SystemExit(f"feeder_tree edges without network data: {sorted(missing)}")

    pos = {root: (float(anchor["anchor"]["lat"]), float(anchor["anchor"]["lon"]))}
    features, report = [], []
    order = [root]
    for bus in order:                                  # breadth-first walk from the root
        for child in sorted(children.get(bus, [])):
            key = f"{bus}-{child}"
            e = edges[key]
            la, lo = pos[bus]
            target_m = e["model_length_m"]
            draw_m = target_m if target_m > 0 else disp
            end = offset(la, lo, float(bearings[key]), draw_m)
            path = [[lo, la], [end[1], end[0]]]
            if mode == "mapbox" and e["element"] == "line":
                route = mapbox_route(token, anchor.get("mapbox_profile", "walking"), (lo, la), (end[1], end[0]))
                path = cut_path([[lo, la]] + route[1:], target_m)
            drawn = path_length_m(path)
            pos[child] = (path[-1][1], path[-1][0])
            order.append(child)
            features.append({"type": "Feature",
                             "geometry": {"type": "LineString", "coordinates": [[round(x, 7) for x in p] for p in path]},
                             "properties": {"kind": "segment", "id": f"seg:{key}", "segment": key,
                                            "from": bus, "to": child, "zone": bus_zone.get(child),
                                            "element": e["element"], "config": e["config"],
                                            "length_ft_ieee": e["length_ft_ieee"],
                                            "model_length_m": target_m, "drawn_length_m": round(drawn, 1)}})
            if target_m > 0:
                report.append((key, target_m, drawn))

    for bus, (la, lo) in pos.items():
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lo, 7), round(la, 7)]},
                         "properties": {"kind": "bus", "id": f"bus:{bus}", "bus": bus, "zone": bus_zone.get(bus),
                                        "role": "substation" if bus == root else "bus",
                                        "label": anchor["anchor"]["label"] if bus == root else f"Bus {bus}"}})

    ring = anchor["dt_ring"]
    by_bus: dict[str, list[dict]] = {}
    for m in meters:
        if m["role"] == "dt":
            by_bus.setdefault(m["load_id"].split("_")[1], []).append(m)
    for bus, dts in by_bus.items():
        la, lo = pos[bus]
        for k, m in enumerate(sorted(dts, key=lambda d: d["load_id"])):
            dla, dlo = offset(la, lo, float(ring["angles_deg"][k]), float(ring["radius_m"]))
            kva = round(m["i_rated_a"] * math.sqrt(3) * LV_V / 1000.0)
            n_hh = households.get(m["load_id"])
            props = {"kind": "dt", "id": f"dt:{m['load_id']}", "load_id": m["load_id"], "meter_id": m["meter_id"],
                     "bus": bus, "zone": bus_zone.get(bus), "rating_kva": kva, "i_rated_a": m["i_rated_a"],
                     "households": n_hh}
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(dlo, 7), round(dla, 7)]},
                             "properties": props})
            if n_hh:
                features.append({"type": "Feature",
                                 "geometry": {"type": "Polygon", "coordinates": [circle(dla, dlo, anchor["service_area_k_m"] * math.sqrt(n_hh))]},
                                 "properties": {"kind": "service_area", "id": f"area:{m['load_id']}",
                                                "load_id": m["load_id"], "zone": bus_zone.get(bus), "households": n_hh}})

    head = next(m for m in meters if m["role"] == "feeder_head")
    la, lo = pos[root]
    features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lo, 7), round(la, 7)]},
                     "properties": {"kind": "feeder_head_meter", "id": f"meter:{head['meter_id']}",
                                    "meter_id": head["meter_id"], "bus": root}})

    return {"type": "FeatureCollection",
            "metadata": {"generator": "scripts/build_geo_layout.py", "mode": mode,
                         "illustrative": True,
                         "note": "Illustrative placement of the IEEE 13-bus test feeder (Stage B, 11 kV) in Delhi; "
                                 "not a real DISCOM network.",
                         "anchor": anchor["anchor"], "length_factor": LENGTH_FACTOR,
                         "length_ratio": {k: round(d / t, 4) for k, t, d in report}},
            "features": features}


def dotenv_value(path: Path, key: str) -> str | None:
    """KEY=value from a .env file (the dashboard reads the same file), or None."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == key and not line.lstrip().startswith("#"):
            return v.strip().strip("'\"") or None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("schematic", "mapbox"), default="schematic")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    token = os.environ.get("MAPBOX_TOKEN") or dotenv_value(ROOT / ".env", "MAPBOX_TOKEN")
    if args.mode == "mapbox" and not token:
        raise SystemExit("--mode mapbox needs MAPBOX_TOKEN in the environment")
    doc = build(args.mode, token)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    ratios = doc["metadata"]["length_ratio"]
    worst = max(ratios.items(), key=lambda kv: abs(kv[1] - 1))
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}: "
          f"{len(doc['features'])} features, mode={args.mode}, worst length ratio {worst[0]} = {worst[1]}")
    bad = {k: r for k, r in ratios.items() if abs(r - 1) > 0.10}
    if bad:
        print(f"WARNING: drawn length off by more than 10 %: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
