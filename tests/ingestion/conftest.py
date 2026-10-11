import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs mosquitto + the real Layer 1 meter server")


def pytest_collection_modifyitems(config, items):
    from devstack import find_mosquitto

    if find_mosquitto() is not None:
        return
    skip = pytest.mark.skip(reason="mosquitto binary not found (set SG_MOSQUITTO_BIN)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
