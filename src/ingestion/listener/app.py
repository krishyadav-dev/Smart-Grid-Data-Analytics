"""FastAPI app factory with a lifespan-managed MQTT subscriber.

PowerShell::

    python -m uvicorn src.ingestion.listener.app:create_app --factory --host 127.0.0.1 --port 8000

On Windows, uvicorn must run on a selector event loop for aiomqtt; use
``python -m src.ingestion.listener`` which handles this.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..config import ListenerSettings, load_meters
from .api import router
from .mqtt_client import MqttIngest
from .service import ListenerService

logger = logging.getLogger("sg.listener")


def create_app(settings: Optional[ListenerSettings] = None) -> FastAPI:
    settings = settings or ListenerSettings()
    meters = load_meters(settings.meters_file)
    if meters.feeder_id != settings.feeder_id:
        raise ValueError(f"meters file feeder {meters.feeder_id!r} != {settings.feeder_id!r}")
    service = ListenerService(settings, meters)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ingest = None
        if settings.mqtt_enabled:
            ingest = MqttIngest(settings, service)
            app.state.ingest = ingest
            ingest.start()
        try:
            yield
        finally:
            if ingest is not None:
                await ingest.stop()
            service.hub.close()

    app = FastAPI(title="Smart Grid Ingestion Listener", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.state.ingest = None
    app.include_router(router)
    return app
