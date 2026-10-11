"""Listener core: validation → enrichment → hand-off.  Transport-agnostic, so
tests can drive it without a broker; ``mqtt_client`` feeds it from MQTT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .. import topics
from ..config import ListenerSettings, MetersFile
from ..schema import (
    DlqMessage,
    GatewayStatus,
    MeterStatus,
    dumps_compact,
    utc_now,
)
from .hub import TelemetryHub
from .stats import LatencyStats, RateTracker
from .validation import Accepted, Rejected, Validator


@dataclass
class Outbound:
    topic: str
    payload: bytes
    qos: int
    retain: bool = False


def _ms(a: datetime, b: datetime) -> float:
    return round((b - a).total_seconds() * 1000.0, 3)


class ListenerService:
    def __init__(self, settings: ListenerSettings, meters: MetersFile):
        self.s = settings
        self.meters = meters.by_id()
        self.validator = Validator(settings, self.meters)
        self.hub = TelemetryHub(settings.hub_queue_size)
        self.history: dict[str, deque[dict]] = {
            m: deque(maxlen=settings.history_len) for m in self.meters}
        self.latency = LatencyStats(settings.latency_sample_size)
        self.rates = RateTracker(settings.rate_window_s)
        self.ingest_seq = 0
        self.received = 0
        self.dlq_count = 0
        self.gateway_status: Optional[dict] = None
        self.meter_status: dict[str, dict] = {}
        self.broker_connected = False
        self.last_message_utc: Optional[datetime] = None
        self.status_errors = 0

    # ------------------------------------------------------------------ #
    def handle(self, topic: str, payload: bytes,
               t_recv: Optional[datetime] = None) -> list[Outbound]:
        """Process one incoming MQTT message; return messages to publish."""
        t_recv = t_recv or utc_now()
        pt = topics.parse(topic)
        if pt is not None and pt.kind in ("gateway_status", "status"):
            self._handle_status(pt, payload)
            return []
        self.received += 1
        self.last_message_utc = t_recv
        result = self.validator.process(topic, payload, t_recv)
        if isinstance(result, Rejected):
            self.dlq_count += 1
            dlq = DlqMessage(
                reason=result.reason, detail=result.detail[:500], topic=topic,
                t_recv_utc=t_recv,
                payload_head=payload[:2048].decode("utf-8", errors="replace"),
            )
            return [Outbound(topics.dlq(self.s.feeder_id),
                             dlq.model_dump_json().encode(), topics.QOS_DLQ)]
        return [self._accept(result, t_recv)]

    def _accept(self, acc: Accepted, t_recv: datetime) -> Outbound:
        m = acc.model
        latency = {
            "poll_to_pub": _ms(m.t_poll_utc, m.t_pub_utc),
            "pub_to_recv": _ms(m.t_pub_utc, t_recv),
            "total": _ms(m.t_poll_utc, t_recv),
        }
        self.ingest_seq += 1
        doc = acc.doc
        doc["ingest"] = {
            "t_recv_utc": t_recv.isoformat().replace("+00:00", "Z"),
            "ingest_seq": self.ingest_seq,
            "latency_ms": latency,
        }
        self.latency.add(latency)
        self.rates.add(m.meter_id)
        self.history[m.meter_id].append(doc)
        self.hub.publish(doc)
        return Outbound(topics.validated(self.s.feeder_id, m.meter_id),
                        dumps_compact(doc), topics.QOS_VALIDATED)

    def _handle_status(self, pt: topics.ParsedTopic, payload: bytes) -> None:
        try:
            if pt.kind == "gateway_status":
                self.gateway_status = GatewayStatus.model_validate_json(payload).model_dump(mode="json")
            elif pt.meter_id in self.meters:
                self.meter_status[pt.meter_id] = MeterStatus.model_validate_json(
                    payload).model_dump(mode="json")
        except Exception:  # noqa: BLE001 — status is advisory; never crash ingestion
            self.status_errors += 1

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        v = self.validator
        return {
            "received": self.received,
            "accepted": v.step_counts["accepted"],
            "rejected": sum(v.reject_counts.values()),
            "dlq_published": self.dlq_count,
            "steps": dict(v.step_counts),
            "rejects": dict(v.reject_counts),
            "flags": dict(v.flag_counts),
            "per_meter": {
                m: {"accepted_total": self.rates.totals.get(m, 0),
                    "rate_per_s": rate}
                for m, rate in {**{k: 0.0 for k in self.meters}, **self.rates.rates()}.items()
            },
            "latency_ms": self.latency.summary(),
            "latency_budget_ms": self.s.LATENCY_BUDGET_MS,
            "hub": self.hub.stats(),
            "status_parse_errors": self.status_errors,
        }
