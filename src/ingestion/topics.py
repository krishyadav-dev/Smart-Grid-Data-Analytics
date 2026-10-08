"""MQTT topic names (MQTT 3.1.1, UTF-8, lowercase).  See brief §4.2."""

from __future__ import annotations

from dataclasses import dataclass

PREFIX = "sg/v1"

QOS_TELEMETRY = 1
QOS_STATUS = 1
QOS_VALIDATED = 1
QOS_DLQ = 0


def meter_id_for(unit_id: int) -> str:
    return f"m{unit_id:02d}"


def telemetry(feeder_id: str, meter_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/{meter_id}/telemetry"


def gateway_status(feeder_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/gateway/status"


def meter_status(feeder_id: str, meter_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/{meter_id}/status"


def validated(feeder_id: str, meter_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/validated/{meter_id}"


def dlq(feeder_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/dlq"


# Subscription filters
def telemetry_filter(feeder_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/+/telemetry"


def status_filter(feeder_id: str) -> str:
    # matches gateway/status and every {meter_id}/status
    return f"{PREFIX}/{feeder_id}/+/status"


def validated_filter(feeder_id: str) -> str:
    return f"{PREFIX}/{feeder_id}/validated/+"


@dataclass(frozen=True)
class ParsedTopic:
    feeder_id: str
    kind: str            # telemetry | status | gateway_status | validated | dlq
    meter_id: str | None


def parse(topic: str) -> ParsedTopic | None:
    parts = topic.split("/")
    if len(parts) < 4 or "/".join(parts[:2]) != PREFIX:
        return None
    feeder = parts[2]
    if len(parts) == 4 and parts[3] == "dlq":
        return ParsedTopic(feeder, "dlq", None)
    if len(parts) != 5:
        return None
    a, b = parts[3], parts[4]
    if a == "validated":
        return ParsedTopic(feeder, "validated", b)
    if a == "gateway" and b == "status":
        return ParsedTopic(feeder, "gateway_status", None)
    if b == "telemetry":
        return ParsedTopic(feeder, "telemetry", a)
    if b == "status":
        return ParsedTopic(feeder, "status", a)
    return None
