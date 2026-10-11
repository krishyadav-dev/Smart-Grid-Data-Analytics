"""Layer 2 configuration: YAML files (gateway, meters, register map) and the
listener's environment settings.  All paths resolve from the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schema import MeterRole

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_GATEWAY_CONFIG = CONFIG_DIR / "gateway.yaml"
DEFAULT_METERS_FILE = CONFIG_DIR / "meters.yaml"
DEFAULT_REGISTER_MAP_FILE = CONFIG_DIR / "register_map.yaml"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with resolve(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------
# meters.yaml
# --------------------------------------------------------------------------

class MeterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meter_id: str
    unit_id: int
    load_id: str
    role: MeterRole
    phases_present: list[Literal["a", "b", "c"]]
    v_base_ln_v: Optional[float] = None
    i_rated_a: Optional[float] = None


class MetersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feeder_id: str
    meters: list[MeterConfig]

    def by_id(self) -> dict[str, MeterConfig]:
        return {m.meter_id: m for m in self.meters}


def load_meters(path: str | Path = DEFAULT_METERS_FILE) -> MetersFile:
    return MetersFile.model_validate(load_yaml(path))


# --------------------------------------------------------------------------
# gateway.yaml
# --------------------------------------------------------------------------

class ModbusSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5020
    timeout_s: float = 1.0


class MqttSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 1883
    keepalive_s: int = 10
    client_id_prefix: str = "sg-gateway"


class BackoffSettings(BaseModel):
    initial_s: float = 0.5
    max_s: float = 15.0
    jitter: float = Field(0.3, ge=0.0, le=1.0)


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feeder_id: str = "ieee13_11kv"
    meters_file: str = str(DEFAULT_METERS_FILE)
    register_map_file: str = str(DEFAULT_REGISTER_MAP_FILE)
    modbus: ModbusSettings = Field(default_factory=ModbusSettings)
    mqtt: MqttSettings = Field(default_factory=MqttSettings)
    poll_interval_s: float = Field(0.25, gt=0)
    expected_step_s: float = Field(900, gt=0)
    playback_speed: Optional[float] = None
    torn_read_retries: int = Field(3, ge=0)
    backoff: BackoffSettings = Field(default_factory=BackoffSettings)
    buffer_max_messages: int = Field(10000, ge=1)


def load_gateway_config(path: str | Path = DEFAULT_GATEWAY_CONFIG,
                        **overrides) -> GatewayConfig:
    data = load_yaml(path)
    for key, value in overrides.items():
        if value is None:
            continue
        node = data
        parts = key.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return GatewayConfig.model_validate(data)


# --------------------------------------------------------------------------
# Listener settings (Part D) — environment variables, prefix SG_
# --------------------------------------------------------------------------

from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402


class ListenerSettings(BaseSettings):
    """FastAPI listener settings.  Every field can be set as ``SG_<NAME>``
    (e.g. ``SG_MQTT_PORT=1883``) or in a ``.env`` file."""

    model_config = SettingsConfigDict(env_prefix="SG_", env_file=".env",
                                      env_file_encoding="utf-8", extra="ignore")

    feeder_id: str = "ieee13_11kv"
    meters_file: str = str(DEFAULT_METERS_FILE)

    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_client_id: str = "sg-listener"
    mqtt_keepalive_s: int = 10
    # Persistent session (clean_session=False): the broker queues QoS 1
    # messages while the listener is down.
    mqtt_persistent_session: bool = True
    mqtt_enabled: bool = True

    backoff_initial_s: float = 0.5
    backoff_max_s: float = 15.0
    backoff_jitter: float = 0.3

    # Validation policies
    out_of_order_policy: Literal["flag", "reject"] = "flag"
    stale_after_s: float = 30.0
    expected_step_s: float = 900.0
    dedup_window: int = Field(1024, ge=1)        # remembered keys per meter

    # Hand-off
    history_len: int = Field(96, ge=1)           # one simulated day at 15 min
    hub_queue_size: int = Field(1000, ge=1)
    latency_sample_size: int = Field(10000, ge=100)
    rate_window_s: float = 60.0

    # Deliberately unset: the project owner sets it from benchmark numbers.
    LATENCY_BUDGET_MS: Optional[float] = None
