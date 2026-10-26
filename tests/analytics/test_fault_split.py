"""F4: the fixed fault split (config/fault_split.yaml, plan Appendix A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "config" / "fault_split.yaml"
FAULT_EVENTS = ROOT / "results" / "faults_2024" / "fault_events.parquet"
KEYS = ("faults_tune", "faults_test", "quiet_tune", "quiet_test")


@pytest.fixture(scope="module")
def split():
    return yaml.safe_load(SPLIT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def events():
    if not FAULT_EVENTS.exists():
        pytest.skip("results/faults_2024/fault_events.parquet absent")
    return pd.read_parquet(FAULT_EVENTS).sort_values("event_id")


def _classes(ev: pd.DataFrame) -> pd.Series:
    fl = ev[ev.kind == "fault"]
    rc = np.where(fl.rf_model == "arc", "arc", fl.rf_ohm.map(lambda r: f"{r:g}"))
    return pd.Series(fl.ftype.to_numpy() + "_" + rc, index=fl.event_id.to_numpy())


def test_lists_disjoint_complete_and_sized(split):
    assert split["seed"] == 20261025 and isinstance(split["rule"], str)
    lists = [split[k] for k in KEYS]
    assert all(len(x) == 100 == len(set(x)) for x in lists)
    allv = [e for x in lists for e in x]
    assert len(allv) == len(set(allv)) == 400
    assert set(allv) == set(range(1, 401))
    assert set(split["faults_tune"]) | set(split["faults_test"]) == set(range(1, 201))
    assert set(split["quiet_tune"]) | set(split["quiet_test"]) == set(range(201, 401))


def test_every_class_in_both_fault_halves(split, events):
    cls = _classes(events)
    tune, test = cls.loc[split["faults_tune"]], cls.loc[split["faults_test"]]
    assert set(tune) == set(test) == set(cls)
    assert len(set(cls)) == 10


def test_split_rederives_from_rule(split, events):
    """Plan §1 "Split": the lists must follow from the rule (the lists win; a
    mismatch is a stop-and-report condition)."""
    cls = _classes(events)
    rng = np.random.default_rng(split["seed"])
    tune, test, odd = [], [], 0
    for c in sorted(set(cls)):
        ids = rng.permutation(cls.index[cls == c].to_numpy())
        h = len(ids) // 2
        if len(ids) % 2:
            h += 1 if odd % 2 == 0 else 0
            odd += 1
        tune += ids[:h].tolist()
        test += ids[h:].tolist()
    quiet = rng.permutation(events[events.kind == "quiet"].event_id.to_numpy())
    assert sorted(tune) == sorted(split["faults_tune"])
    assert sorted(test) == sorted(split["faults_test"])
    assert sorted(quiet[:100].tolist()) == sorted(split["quiet_tune"])
    assert sorted(quiet[100:].tolist()) == sorted(split["quiet_test"])
