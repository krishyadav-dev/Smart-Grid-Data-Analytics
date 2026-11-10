"""Second theft injection (Layer 3 Stage 4 run sheet §2, step A; D1)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_source import config
from src.data_source import theft_reinjection as T

INJ2 = config.PROJECT_ROOT / "results" / "year_2024_inj2"
NTL_INJ2 = config.PROJECT_ROOT / "results" / "layer3" / "ntl_inj2"


def test_stream_names():
    assert T.stream_names("") == ("theft.pick", "theft.attack")
    assert T.stream_names("inj2") == ("theft.pick/inj2", "theft.attack/inj2")


def test_tagged_streams_are_new_draws():
    a, b = config.rng("theft.pick"), config.rng("theft.pick/inj2")
    assert not np.array_equal(a.integers(0, 2 ** 31, 8), b.integers(0, 2 ** 31, 8))
    np.testing.assert_array_equal(config.rng("theft.pick").integers(0, 2 ** 31, 8),
                                  config.rng("theft.pick").integers(0, 2 ** 31, 8))


def test_refuses_v1_outputs(tmp_path):
    for p in (T.V1_CACHE, T.V1_CACHE / "ami", T.V1_NTL, T.V1_NTL / "ideal_meter"):
        with pytest.raises(ValueError):
            T._check_out(p)
    T._check_out(tmp_path / "year_2024_inj2")


def test_reproduction_check_record():
    path = NTL_INJ2 / "reproduction_check.json"
    if not path.exists():
        pytest.skip("written by scripts/reinject_theft.py --tag \"\" (step A)")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["all_equal"] is True and rec["streams"] == ["theft.pick", "theft.attack"]
    names = {r["file"] for r in rec["files"]}
    assert {*T.THEFT_FILES, "ami_2024-01.parquet", "ami_2024-12.parquet", "lambda_sweep/metrics.json"} <= names


def test_injection2_inputs():
    path = NTL_INJ2 / "inputs.json"
    if not path.exists() or not (INJ2 / "households.parquet").exists():
        pytest.skip("written by scripts/reinject_theft.py --tag inj2 (step A)")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["theft_independence_checks"]["passed"] is True
    h1 = pd.read_parquet(T.V1_CACHE / "households.parquet", columns=["household", "thief", "dt", "meter_gain"])
    h2 = pd.read_parquet(INJ2 / "households.parquet", columns=["household", "thief", "dt", "meter_gain"])
    assert list(h1.household) == list(h2.household) and list(h1.dt) == list(h2.dt)
    np.testing.assert_array_equal(h1.meter_gain, h2.meter_gain)
    assert h1.thief.sum() == h2.thief.sum() == 684                 # prevalence unchanged
    assert (h1.thief != h2.thief).any()                            # a new draw
