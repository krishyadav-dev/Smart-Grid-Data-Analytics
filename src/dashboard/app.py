"""FastAPI app: report-mode API, map layout, and the built web app (plan.md §4)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import detail, geo, results
from .settings import DashboardSettings

api = APIRouter(prefix="/api")
DAY = r"^\d{4}-\d{2}-\d{2}$"


def _s(request: Request) -> DashboardSettings:
    return request.app.state.settings


@api.get("/config")
def config(request: Request) -> dict:
    s = _s(request)
    return {"feeder_id": s.feeder_id, "mapbox_token": s.mapbox_token,
            "mode": {"report": True, "live": False},
            "map_note": "Illustrative placement of the IEEE 13-bus test feeder (Stage B, 11 kV) in Delhi; "
                        "not a real DISCOM network."}


@api.get("/geo/layout")
def geo_layout(request: Request) -> JSONResponse:
    try:
        return JSONResponse(geo.layout(_s(request).config_dir))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@api.get("/meta")
def meta(request: Request) -> dict:
    return geo.meta(_s(request).config_dir)


@api.get("/results/faults")
def results_faults(request: Request) -> dict:
    return results.faults(_s(request).results_dir)


@api.get("/results/faults/events")
def results_fault_events(request: Request) -> dict:
    s = _s(request)
    return results.fault_events(s.results_dir, s.config_dir)


@api.get("/results/theft")
def results_theft(request: Request) -> dict:
    return results.theft(_s(request).results_dir)


@api.get("/results/theft/dt-daily")
def results_theft_dt_daily(request: Request) -> dict:
    return detail.theft_dt_daily(_s(request).results_dir)


@api.get("/results/theft/inspection")
def results_theft_inspection(request: Request, dt: Optional[str] = None,
                             limit: int = Query(25, ge=1, le=500)) -> dict:
    return detail.theft_inspection(_s(request).results_dir, dt, limit)


@api.get("/results/faults/bursts/{event_id}")
def results_fault_burst(request: Request, event_id: int) -> dict:
    return detail.fault_burst(_s(request).results_dir, event_id)


@api.get("/results/grid/daily")
def results_grid_daily(request: Request) -> dict:
    s = _s(request)
    return detail.grid_daily(s.results_dir, s.config_dir, s.data_dir)


@api.get("/results/grid/day/{day}")
def results_grid_day(request: Request, day: str = Path(pattern=DAY)) -> dict:
    s = _s(request)
    return detail.grid_day(s.results_dir, s.config_dir, s.data_dir, day)


@api.get("/results/thermal/daily")
def results_thermal_daily(request: Request) -> dict:
    s = _s(request)
    return detail.thermal_daily(s.results_dir, s.config_dir, s.data_dir)


@api.get("/results/thermal/day/{day}")
def results_thermal_day(request: Request, day: str = Path(pattern=DAY)) -> dict:
    s = _s(request)
    return detail.thermal_day(s.results_dir, s.config_dir, s.data_dir, day)


@api.get("/results/unbalance/daily")
def results_unbalance_daily(request: Request) -> dict:
    s = _s(request)
    return detail.vuf_daily(s.results_dir, s.config_dir)


@api.get("/results/thermal")
def results_thermal(request: Request) -> dict:
    return results.thermal(_s(request).results_dir)


@api.get("/results/unbalance")
def results_unbalance(request: Request) -> dict:
    return results.unbalance(_s(request).results_dir)


@api.get("/results/ingestion")
def results_ingestion(request: Request) -> dict:
    return results.ingestion(_s(request).results_dir)


def create_app(settings: Optional[DashboardSettings] = None) -> FastAPI:
    s = settings or DashboardSettings()
    app = FastAPI(title="Smart Grid Dashboard", version="0.1.0",
                  description="Layer 4: report mode over the frozen Layer 3 results, and the feeder map.")
    app.state.settings = s
    if s.dev_origin:
        app.add_middleware(CORSMiddleware, allow_origins=[s.dev_origin], allow_methods=["GET"], allow_headers=["*"])
    app.include_router(api)

    index = s.web_dist / "index.html"
    if index.exists():
        app.mount("/assets", StaticFiles(directory=s.web_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):                             # client-side routes fall back to index.html
            if path.startswith("api/"):
                raise HTTPException(status_code=404)
            f = (s.web_dist / path).resolve()
            if path and f.is_file() and f.is_relative_to(s.web_dist.resolve()):
                return FileResponse(f)
            return FileResponse(index)
    else:
        @app.get("/", include_in_schema=False)
        def no_web() -> dict:
            return {"detail": "web app not built: cd web && npm install && npm run build", "api": "/docs"}
    return app
