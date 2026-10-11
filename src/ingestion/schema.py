"""Telemetry contract for Layer 2 (schema version 1.0).

Pydantic v2 models shared by the gateway (producer) and the listener
(validator).  ``schemas/telemetry_v1.json`` is exported from
:class:`Telemetry` by :func:`export_json_schema`; a test regenerates it and
compares, so the file and the models cannot drift.

Conventions
-----------
- Canonical engineering units (V, A, kW, kvar, kWh, Hz), already scaled.
- Per-phase quantities are objects keyed by phase; phases a meter does not
  have are **omitted**, never zero-filled.
- ``t_sim`` is timezone-naive ISO 8601 (Layer 1 convention) or ``null`` when
  the meter has no simulated-time register (flag ``NO_SIM_TIME``).
- ``t_*_utc`` are timezone-aware wall-clock UTC.
- NaN / Infinity are rejected (``allow_inf_nan=False``).
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NaiveDatetime,
)

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "telemetry_v1.json"


class MeterRole(StrEnum):
    CONSUMER = "consumer"
    DT = "dt"
    FEEDER_HEAD = "feeder_head"


class QualityFlag(StrEnum):
    GAP = "GAP"
    NO_SIM_TIME = "NO_SIM_TIME"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    STALE = "STALE"
    ENERGY_DECREASE = "ENERGY_DECREASE"
    SUSPECT_RANGE = "SUSPECT_RANGE"
    PARTIAL_READ = "PARTIAL_READ"
    TORN_READ = "TORN_READ"


class RejectReason(StrEnum):
    MALFORMED_JSON = "MALFORMED_JSON"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    UNKNOWN_METER = "UNKNOWN_METER"
    DUPLICATE = "DUPLICATE"
    # Only used when out_of_order_policy = "reject".
    OUT_OF_ORDER = "OUT_OF_ORDER"


class GatewayState(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


class MeterLinkState(StrEnum):
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"


# JSON Schema cannot express naive vs aware datetimes; patterns make the
# exported schema enforce it (pydantic enforces it via the types below).
NAIVE_ISO = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$"
AWARE_ISO = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"

Phase = Literal["a", "b", "c"]
PhaseOrTotal = Literal["a", "b", "c", "total"]

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)


class Measurements(BaseModel):
    """Measured values.  A key is present only if the register map serves it
    and the read succeeded."""

    model_config = _STRICT

    voltage_v: Optional[dict[Phase, float]] = Field(
        None, description="Line-to-neutral voltage per phase, V")
    current_a: Optional[dict[Phase, float]] = Field(
        None, description="Current per phase, A (see docs: Layer 1 serves the "
                          "largest connected line current, not the load's own)")
    active_power_kw: Optional[dict[PhaseOrTotal, float]] = Field(
        None, description="Active power per phase and total, kW")
    reactive_power_kvar: Optional[dict[PhaseOrTotal, float]] = Field(
        None, description="Reactive power per phase and total, kvar (signed)")
    energy_import_kwh: Optional[float] = Field(
        None, description="Cumulative active energy import, kWh")
    frequency_hz: Optional[float] = Field(
        None, description="Frequency, Hz (not served by Layer 1)")


class Quality(BaseModel):
    model_config = _STRICT

    flags: list[QualityFlag] = Field(default_factory=list)
    gap_steps: int = Field(0, ge=0, description="Missed simulated steps before this one")


class Telemetry(BaseModel):
    """One meter reading, as published by the gateway."""

    model_config = _STRICT

    schema_version: str = Field(pattern=r"^1\.\d+$")
    feeder_id: str = Field(min_length=1)
    meter_id: str = Field(pattern=r"^m\d{2,}$")
    unit_id: int = Field(ge=1, le=247)
    load_id: str = Field(min_length=1)
    meter_role: MeterRole
    run_id: UUID
    seq: int = Field(ge=0)
    t_sim: Optional[NaiveDatetime] = Field(json_schema_extra={"pattern": NAIVE_ISO})
    t_poll_utc: AwareDatetime = Field(json_schema_extra={"pattern": AWARE_ISO})
    t_pub_utc: AwareDatetime = Field(json_schema_extra={"pattern": AWARE_ISO})
    measurements: Measurements
    quality: Quality = Field(default_factory=Quality)

    def to_wire(self) -> bytes:
        """Compact UTF-8 JSON; absent measurements and phases are omitted."""
        return dumps_compact(self.to_doc())

    def to_doc(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["measurements"] = {k: v for k, v in doc["measurements"].items()
                               if v is not None}
        return doc


class LatencyMs(BaseModel):
    model_config = _STRICT

    poll_to_pub: float
    pub_to_recv: float
    total: float


class IngestInfo(BaseModel):
    model_config = _STRICT

    t_recv_utc: AwareDatetime
    ingest_seq: int = Field(ge=0)
    latency_ms: LatencyMs


class ValidatedTelemetry(Telemetry):
    """Telemetry plus the listener's ``ingest`` block (validated topic)."""

    ingest: IngestInfo


class GatewayStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: GatewayState
    run_id: Optional[UUID] = None
    t_utc: AwareDatetime
    # Additive 1.x fields (diagnostics)
    buffer_dropped: Optional[int] = None
    unreachable_meters: Optional[list[str]] = None


class MeterStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: MeterLinkState
    since_utc: AwareDatetime
    last_error: Optional[str] = None


class DlqMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: RejectReason
    detail: str
    topic: str
    t_recv_utc: AwareDatetime
    payload_head: str


def dumps_compact(doc: dict) -> bytes:
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def utc_now() -> datetime:
    from datetime import timezone
    return datetime.now(timezone.utc)


def build_json_schema() -> dict:
    schema = Telemetry.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://smart-grid-data-analytics/schemas/telemetry_v1.json"
    schema["title"] = f"Telemetry v{SCHEMA_VERSION}"
    return schema


def export_json_schema(path: Path = SCHEMA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_json_schema(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


if __name__ == "__main__":
    print(export_json_schema())
