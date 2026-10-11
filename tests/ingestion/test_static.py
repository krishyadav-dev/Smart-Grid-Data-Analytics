"""C9 (broker config) and D8 (no DB / analytics / Kafka in Layer 2)."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_c9_compose_binds_localhost_only():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    ports = compose["services"]["mosquitto"]["ports"]
    assert ports and all(str(p).startswith("127.0.0.1:") for p in ports)
    image = compose["services"]["mosquitto"]["image"]
    assert re.fullmatch(r"eclipse-mosquitto:2\.\d+\.\d+", image), image


def _directives(name):
    out = []
    for line in (ROOT / "config" / "mosquitto" / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.split())
    return out


def test_c9_native_conf_no_anonymous_remote_access():
    d = _directives("mosquitto.native.conf")
    listeners = [x for x in d if x[0] == "listener"]
    assert listeners and all(len(x) >= 3 and x[2] in ("127.0.0.1", "::1") for x in listeners)
    assert ["persistence", "true"] in d


def test_c9_docker_conf_persistence_and_explicit_listener():
    d = _directives("mosquitto.conf")
    assert any(x[0] == "listener" for x in d)       # Mosquitto 2.x needs it explicitly
    assert ["persistence", "true"] in d


def test_d8_no_database_analytics_or_kafka_code():
    banned = re.compile(r"^\s*(import|from)\s+(psycopg|psycopg2|asyncpg|sqlalchemy|sqlite3|"
                        r"kafka|aiokafka|confluent_kafka|sklearn|torch|tensorflow|statsmodels|"
                        r"timescale)", re.M)
    hits = []
    for path in (ROOT / "src" / "ingestion").rglob("*.py"):
        if banned.search(path.read_text(encoding="utf-8")):
            hits.append(str(path))
    assert hits == []
