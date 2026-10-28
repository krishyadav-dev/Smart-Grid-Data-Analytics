"""Physical meters for Layer 3 (docs/Layer3_plan.md §0 rule 2).

Engines read only the meters whose role in ``config/meters.yaml`` is
``feeder_head`` or ``dt`` (m10-m23, 14 meters), selected by role, never by unit
number.  The ``bus_virtual`` meter points are sums of DT meters with no
information of their own and are rejected here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.ingestion.config import DEFAULT_METERS_FILE, MeterConfig, load_meters

PHYSICAL_ROLES = ("feeder_head", "dt")


class NotPhysicalMeter(ValueError):
    """A meter id that is not a physical meter (bus_virtual or unknown)."""


@lru_cache(maxsize=4)
def physical_meters(path: str | Path = DEFAULT_METERS_FILE) -> tuple[MeterConfig, ...]:
    """The physical meters in ``meters.yaml`` order (feeder head first)."""
    return tuple(m for m in load_meters(path).meters if str(m.role) in PHYSICAL_ROLES)


def physical_by_id(path: str | Path = DEFAULT_METERS_FILE) -> dict[str, MeterConfig]:
    return {m.meter_id: m for m in physical_meters(path)}


def meter(meter_id: str, path: str | Path = DEFAULT_METERS_FILE) -> MeterConfig:
    """The physical meter ``meter_id``; raises :class:`NotPhysicalMeter` otherwise."""
    m = physical_by_id(path).get(meter_id)
    if m is None:
        raise NotPhysicalMeter(f"{meter_id} is not a physical meter (roles {PHYSICAL_ROLES})")
    return m


def feeder_head(path: str | Path = DEFAULT_METERS_FILE) -> MeterConfig:
    (m,) = [m for m in physical_meters(path) if str(m.role) == "feeder_head"]
    return m


def dt_meters(path: str | Path = DEFAULT_METERS_FILE) -> tuple[MeterConfig, ...]:
    return tuple(m for m in physical_meters(path) if str(m.role) == "dt")
