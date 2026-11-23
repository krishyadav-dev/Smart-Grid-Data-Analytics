"""Dashboard settings (environment / ``.env``; all optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SG_DASH_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8100
    feeder_id: str = Field("ieee13_11kv", validation_alias=AliasChoices("SG_DASH_FEEDER_ID", "SG_FEEDER_ID"))
    results_dir: Path = PROJECT_ROOT / "results"
    config_dir: Path = PROJECT_ROOT / "config"
    web_dist: Path = PROJECT_ROOT / "web" / "dist"
    # Mapbox *public* token (pk.*), URL-restricted in the Mapbox account.  Served to
    # the browser by /api/config; never a secret (sk.*) token.
    mapbox_token: Optional[str] = Field(None, validation_alias=AliasChoices("MAPBOX_TOKEN", "SG_DASH_MAPBOX_TOKEN"))
    # Development: the Vite dev server origin allowed by CORS.
    dev_origin: Optional[str] = "http://127.0.0.1:5173"
