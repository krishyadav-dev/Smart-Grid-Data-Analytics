"""REST API (response models in pydantic)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ..schema import utc_now
from .service import ListenerService

router = APIRouter()


class Health(BaseModel):
    ok: bool
    broker_connected: bool
    last_message_age_s: Optional[float]
    gateway_state: Optional[str]


class MeterInfo(BaseModel):
    meter_id: str
    unit_id: int
    load_id: str
    role: str
    phases_present: list[str]
    v_base_ln_v: Optional[float]
    i_rated_a: Optional[float]
    link_state: Optional[str]
    messages: int
    last_t_sim: Optional[str]


class HistoryResponse(BaseModel):
    meter_id: str
    count: int
    items: list[dict[str, Any]]


class LatencyHop(BaseModel):
    n: int
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    max: Optional[float]


class MeterRate(BaseModel):
    accepted_total: int
    rate_per_s: float


class StatsResponse(BaseModel):
    received: int
    accepted: int
    rejected: int
    dlq_published: int
    steps: dict[str, int]
    rejects: dict[str, int]
    flags: dict[str, int]
    per_meter: dict[str, MeterRate]
    latency_ms: dict[str, LatencyHop]
    latency_budget_ms: Optional[float]
    hub: dict[str, Any]
    status_parse_errors: int


def _svc(request: Request) -> ListenerService:
    return request.app.state.service


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    svc = _svc(request)
    age = None
    if svc.last_message_utc is not None:
        age = round((utc_now() - svc.last_message_utc).total_seconds(), 3)
    gw = svc.gateway_status["state"] if svc.gateway_status else None
    return Health(ok=svc.broker_connected, broker_connected=svc.broker_connected,
                  last_message_age_s=age, gateway_state=gw)


@router.get("/api/v1/meters", response_model=list[MeterInfo])
def meters(request: Request) -> list[MeterInfo]:
    svc = _svc(request)
    out = []
    for mid, mc in svc.meters.items():
        hist = svc.history[mid]
        out.append(MeterInfo(
            meter_id=mid, unit_id=mc.unit_id, load_id=mc.load_id, role=mc.role.value,
            phases_present=list(mc.phases_present), v_base_ln_v=mc.v_base_ln_v,
            i_rated_a=mc.i_rated_a,
            link_state=(svc.meter_status.get(mid) or {}).get("state"),
            messages=svc.rates.totals.get(mid, 0),
            last_t_sim=hist[-1].get("t_sim") if hist else None,
        ))
    return out


def _known(svc: ListenerService, meter_id: str) -> None:
    if meter_id not in svc.meters:
        raise HTTPException(status_code=404, detail=f"unknown meter {meter_id}")


@router.get("/api/v1/meters/{meter_id}/latest", response_model=dict[str, Any])
def latest(meter_id: str, request: Request) -> dict:
    svc = _svc(request)
    _known(svc, meter_id)
    hist = svc.history[meter_id]
    if not hist:
        raise HTTPException(status_code=404, detail=f"no data yet for {meter_id}")
    return hist[-1]


@router.get("/api/v1/meters/{meter_id}/history", response_model=HistoryResponse)
def history(meter_id: str, request: Request,
            limit: int = Query(96, ge=1, le=100_000)) -> HistoryResponse:
    svc = _svc(request)
    _known(svc, meter_id)
    items = list(svc.history[meter_id])[-limit:]
    return HistoryResponse(meter_id=meter_id, count=len(items), items=items)


@router.get("/api/v1/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    return StatsResponse.model_validate(_svc(request).stats())
