"""Layer 3 leakage guards (plan §4; Stage 1 run sheet rules 1-2; Stage 2 run sheet §0).

1. No engine module references a label file or label column
   (:mod:`src.analytics.leakage`) or imports the evaluator.
2. No engine module uses the event record's ``event_id`` (rule 1).
3. No engine module reads a non-physical meter: meters are selected by role in
   :mod:`src.analytics.meters` only, never by unit number or meter id.
4. No engine module imports a Layer 1 module that reads the true frame or
   rebuilds the labelled population (:data:`leakage.LABEL_MODULES`).
5. The consumer index the theft engine reads holds household, DT and bus only.

Engine modules = every module under ``src/analytics`` except the evaluator
(the only module allowed to read labels), the label list itself and the
package ``__init__``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.analytics import leakage as L
from src.data_source import ntl_inputs

ROOT = Path(__file__).resolve().parents[2]
ANALYTICS = ROOT / "src" / "analytics"
NOT_ENGINE = {"evaluate.py", "leakage.py", "__init__.py"}
ROLE_FILTER = "meters.py"          # the one place that names the excluded role


def engine_modules() -> list[Path]:
    return sorted(p for p in ANALYTICS.glob("*.py") if p.name not in NOT_ENGINE)


def test_engine_modules_exist():
    names = {p.name for p in engine_modules()}
    assert {"zones.py", "unbalance.py", "meters.py"} <= names


@pytest.mark.parametrize("path", engine_modules(), ids=lambda p: p.name)
def test_no_label_files_or_columns(path):
    text = path.read_text(encoding="utf-8")
    hits = [n for n in L.LABEL_FILES + L.LABEL_COLUMNS if n in text]
    assert hits == [], f"{path.name} references labels {hits}"
    assert not re.search(r"^\s*(from|import)\s+[\w.]*evaluate\b|from\s+\.\s+import\s+[^\n]*\bevaluate\b",
                         text, re.M), f"{path.name} imports the evaluator"


@pytest.mark.parametrize("path", engine_modules(), ids=lambda p: p.name)
def test_no_event_id(path):
    assert L.EVENT_ID_FIELD not in path.read_text(encoding="utf-8"), \
        f"{path.name} uses the event record's {L.EVENT_ID_FIELD} (run sheet rule 1)"


@pytest.mark.parametrize("path", engine_modules(), ids=lambda p: p.name)
def test_no_virtual_meter_reads(path):
    text = path.read_text(encoding="utf-8")
    assert "unit_id" not in text, f"{path.name} selects meters by unit number"
    assert not re.search(r"\bm0[1-9]\b", text), f"{path.name} names a bus-level meter id"
    assert not re.search(r"\bload_\d{3}_", text), f"{path.name} names a bus-level meter point"
    if path.name != ROLE_FILTER:
        assert "bus_virtual" not in text, f"{path.name} names the bus-level meter role"


def test_role_filter_returns_physical_meters_only():
    from src.analytics import meters as M
    from src.ingestion.config import load_meters
    allm = load_meters().meters
    virtual = {m.meter_id for m in allm if str(m.role) == "bus_virtual"}
    assert len(virtual) == 9
    ph = {m.meter_id for m in M.physical_meters()}
    assert ph == {f"m{k}" for k in range(10, 24)} and not ph & virtual
    for mid in virtual:
        with pytest.raises(M.NotPhysicalMeter):
            M.meter(mid)



def _imports_module(text: str, mod: str) -> bool:
    return bool(re.search(rf"^\s*(from|import)\s+[\w.]*\b{mod}\b|^\s*from\s+[\w.]+\s+import\s+[^\n]*\b{mod}\b",
                          text, re.M))


@pytest.mark.parametrize("path", engine_modules(), ids=lambda p: p.name)
def test_no_label_module_imports(path):
    text = path.read_text(encoding="utf-8")
    hits = [m for m in L.LABEL_MODULES if _imports_module(text, m)]
    assert hits == [], f"{path.name} imports the Layer 1 module(s) {hits}"


def test_label_module_regex_catches_imports():
    bad = ("from src.data_source import simulation as S\n", "import src.data_source.metering\n",
           "from src.data_source.ntl_inputs import build\n", "from src.data_source import faults, households\n")
    for src in bad:
        assert any(_imports_module(src, m) for m in L.LABEL_MODULES), src
    assert not any(_imports_module("from src.data_source import theft_injector\n", m) for m in L.LABEL_MODULES)


def test_consumer_index_columns():
    assert tuple(ntl_inputs.CONSUMER_INDEX_COLUMNS) == L.CONSUMER_INDEX_COLUMNS
    path = ntl_inputs.OUT_DIR / "consumer_index.parquet"
    if not path.exists():
        pytest.skip("consumer_index.parquet is written by scripts/build_ntl_inputs.py (step A)")
    import pyarrow.parquet as pq
    assert tuple(pq.read_schema(path).names) == L.CONSUMER_INDEX_COLUMNS
