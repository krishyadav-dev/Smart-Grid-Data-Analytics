"""
Download and prepare ERA5 2 m air temperature for the thermal engine (Layer 3, decision T2).

Source: Copernicus Climate Data Store, dataset ``reanalysis-era5-single-levels``
(Hersbach et al., 2020, Q. J. R. Meteorol. Soc. 146(730):1999-2049,
doi:10.1002/qj.3803). Licence: Copernicus licence; attribute as
"Contains modified Copernicus Climate Change Service information 2024".

Decision T2 (Layer 3 design, 27 Oct 2026):
- hourly 2 m temperature, 2023-12-31 00:00 to 2024-12-31 23:00 UTC;
- the ERA5 0.25 deg grid point nearest 28.6 N, 77.2 E (Delhi);
- converted K -> deg C and UTC -> IST (+5:30), then linearly interpolated from
  hourly values to the 15-minute steps of the simulated year (stated assumption).
The extra day (2023-12-31) is needed because 2024-01-01 00:00 IST is
2023-12-31 18:30 UTC.

Usage (from the project root, with the venv active):

    python scripts/download_era5.py            # download (if missing) + convert
    python scripts/download_era5.py --convert-only

Needs a CDS account with the ERA5 licence accepted and the API key in
~/.cdsapirc (or the CDSAPI_URL / CDSAPI_KEY environment variables).

Outputs (data/ is git-ignored):
- data/era5/raw/era5_t2m_{2023-12-31,2024}.nc   the files as delivered by the CDS
- data/era5/request.json                        the exact requests and when they ran
- data/era5/era5_t2m_delhi_2024_ist_15min.parquet
      columns: timestamp (naive IST, same convention as results/year_2024),
               t2m_c (deg C)
- data/era5/era5_meta.json                      grid point used, checks, summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERA5_DIR = PROJECT_ROOT / "data" / "era5"
RAW_DIR = ERA5_DIR / "raw"
OUT_PARQUET = ERA5_DIR / "era5_t2m_delhi_2024_ist_15min.parquet"
REQUEST_JSON = ERA5_DIR / "request.json"
META_JSON = ERA5_DIR / "era5_meta.json"

DATASET = "reanalysis-era5-single-levels"
TARGET_LAT, TARGET_LON = 28.6, 77.2
# North, West, South, East: a 2 x 2 box of 0.25 deg grid points around the target,
# so the nearest point is chosen here and recorded, not left to the server.
AREA = [28.75, 77.0, 28.5, 77.25]
HOURS = [f"{h:02d}:00" for h in range(24)]
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
YEAR = 2024
STEPS_PER_YEAR = 366 * 96  # 2024 is a leap year: 35,136 fifteen-minute steps

REQUESTS = {
    "2023-12-31": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2023"],
        "month": ["12"],
        "day": ["31"],
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    },
    "2024": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2024"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    },
}


def raw_path(key: str) -> Path:
    return RAW_DIR / f"era5_t2m_{key}.nc"


def download() -> None:
    """Retrieve every request whose file is not already present."""
    import cdsapi  # imported here so --convert-only works without it

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = json.loads(REQUEST_JSON.read_text()) if REQUEST_JSON.exists() else {}
    client = cdsapi.Client()
    for key, request in REQUESTS.items():
        target = raw_path(key)
        if target.exists():
            print(f"[skip] {target.name} already present")
            continue
        print(f"[cds] {DATASET} {key} -> {target}")
        client.retrieve(DATASET, request).download(str(target))
        log[key] = {
            "dataset": DATASET,
            "request": request,
            "file": str(target.relative_to(PROJECT_ROOT)),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        REQUEST_JSON.write_text(json.dumps(log, indent=2))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_hourly() -> tuple[pd.Series, dict]:
    """Hourly t2m (K) at the grid point nearest the target, indexed by naive UTC."""
    import xarray as xr

    parts, grid = [], None
    for key in REQUESTS:
        path = raw_path(key)
        if not path.exists():
            sys.exit(f"missing {path}; run without --convert-only first")
        with xr.open_dataset(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            # nearest grid point by great-circle distance
            best = min(
                ((float(la), float(lo)) for la in lats for lo in lons),
                key=lambda p: _haversine_km(TARGET_LAT, TARGET_LON, *p),
            )
            if grid is None:
                grid = best
            elif grid != best:
                sys.exit(f"grid point differs between files: {grid} vs {best}")
            da = ds["t2m"].sel(latitude=best[0], longitude=best[1])
            extra = [d for d in da.dims if d != time_name]
            for d in extra:  # e.g. a length-1 'number' or 'expver' dimension
                if da.sizes[d] != 1:
                    sys.exit(f"unexpected dimension {d} of size {da.sizes[d]} in {path.name}")
                da = da.isel({d: 0})
            s = da.to_series()
            s.index = pd.to_datetime(s.index)
            parts.append(s)
    hourly = pd.concat(parts).sort_index()
    if hourly.index.has_duplicates:
        sys.exit("duplicate hours across the downloaded files")
    info = {
        "grid_lat": grid[0],
        "grid_lon": grid[1],
        "distance_km": round(_haversine_km(TARGET_LAT, TARGET_LON, *grid), 2),
    }
    return hourly, info


def convert(hourly_k: pd.Series, grid: dict) -> pd.DataFrame:
    """K -> deg C, UTC -> IST, linear interpolation to the 2024 15-minute IST steps."""
    expected_utc = pd.date_range("2023-12-31 00:00", "2024-12-31 23:00", freq="h")
    missing = expected_utc.difference(hourly_k.index)
    if len(missing):
        sys.exit(f"{len(missing)} expected UTC hours missing, first {missing[0]}")
    hourly_k = hourly_k.loc[expected_utc]
    if hourly_k.isna().any():
        sys.exit(f"{int(hourly_k.isna().sum())} NaN hours in ERA5 data")

    hourly_c = hourly_k - 273.15
    hourly_c.index = hourly_c.index + IST_OFFSET  # naive IST, like results/year_2024

    steps = pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:45", freq="15min")
    assert len(steps) == STEPS_PER_YEAR
    union = hourly_c.index.union(steps)
    t2m = hourly_c.reindex(union).interpolate(method="time").loc[steps]
    if t2m.isna().any():
        sys.exit(f"{int(t2m.isna().sum())} 15-minute steps without a value")

    out = pd.DataFrame({"timestamp": steps, "t2m_c": t2m.to_numpy(dtype=np.float64)})
    meta = {
        "dataset": DATASET,
        "citation": "Hersbach et al. 2020, QJRMS 146(730):1999-2049, doi:10.1002/qj.3803",
        "target": {"lat": TARGET_LAT, "lon": TARGET_LON},
        "grid_point": grid,
        "area_requested_NWSE": AREA,
        "hours_utc": [str(expected_utc[0]), str(expected_utc[-1])],
        "conversion": "K - 273.15; UTC + 5:30 = IST (naive); linear in time to 15-min steps",
        "steps": len(out),
        "first_step": str(out.timestamp.iloc[0]),
        "last_step": str(out.timestamp.iloc[-1]),
        "t2m_c": {
            "min": round(float(out.t2m_c.min()), 2),
            "max": round(float(out.t2m_c.max()), 2),
            "mean": round(float(out.t2m_c.mean()), 2),
            "max_at": str(out.timestamp.iloc[int(out.t2m_c.idxmax())]),
        },
        "monthly_mean_c": {
            str(m): round(float(v), 2)
            for m, v in out.groupby(out.timestamp.dt.month).t2m_c.mean().items()
        },
    }
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--convert-only", action="store_true", help="skip the CDS download")
    args = ap.parse_args()
    if not args.convert_only:
        download()
    hourly, grid = load_hourly()
    out, meta = convert(hourly, grid)
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {OUT_PARQUET.relative_to(PROJECT_ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
