"""Phase files of the Stage 4 detector runs: save and reload, the one-off date
conversion and the saved-chunk loader (step E recovery, 30 Oct)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_layer3_ntl_inj2", ROOT / "scripts" / "run_layer3_ntl_inj2.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _arr(n_hh: int, n_days: int) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for v in ("static", "update"):
        for f in ("h1", "circular"):
            score = rng.normal(size=(n_hh, n_days)).astype(np.float32)
            flag = (score > 0).astype(np.int8)
            score[0, 1], flag[0, 1] = np.nan, -1                       # an offline day
            out[v, f] = {"flag": flag, "score": score, "added": rng.uniform(size=(n_hh, n_days)) < 0.3}
    return out


def test_save_and_reload(tmp_path):
    ids = ["hh_00001", "hh_00002", "hh_00003"]
    days = pd.date_range("2024-07-01", periods=5, freq="D")
    arr = _arr(len(ids), len(days))
    R._save_phase(tmp_path / "p.npz", ids, days, arr)
    z = np.load(tmp_path / "p.npz")                                    # no pickle needed
    assert z["days"].dtype == np.dtype("U10")
    ids2, days2, arr2 = R._load_phase(tmp_path / "p.npz")
    assert ids2 == ids and days2.equals(days) and set(arr2) == set(arr)
    for key, a in arr.items():
        for k in ("flag", "score", "added"):
            assert arr2[key][k].dtype == a[k].dtype
            np.testing.assert_array_equal(arr2[key][k], a[k])


def test_convert_dates_once(tmp_path, monkeypatch):
    days = pd.date_range("2024-07-01", periods=4, freq="D")
    arr = _arr(2, len(days))
    names = ("a.npz", "b.npz")
    for name in names:                                                 # the old layout: days pickled
        np.savez_compressed(tmp_path / name, ids=np.array(["hh_1", "hh_2"]),
                            days=np.array(days.strftime("%Y-%m-%d")),
                            **{f"{k}__{v}__{f}": a[k] for (v, f), a in arr.items() for k in ("flag", "score", "added")})
    with pytest.raises(ValueError):
        R._load_phase(tmp_path / "a.npz")
    rec = {"files": {n: {"sha256": hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()} for n in names}}
    (tmp_path / "rec.json").write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr(R, "RECORD", tmp_path / "rec.json")
    monkeypatch.setattr(R, "PHASE_FILES", names)
    assert R.cmd_jokar_convert_dates(None) == 0
    rec2 = json.loads((tmp_path / "rec.json").read_text(encoding="utf-8"))
    for n in names:
        assert rec2["converted"][n]["old_sha256"] == rec["files"][n]["sha256"]
        assert rec2["converted"][n]["new_sha256"] == hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
        _, days2, arr2 = R._load_phase(tmp_path / n)
        assert days2.equals(days)
        np.testing.assert_array_equal(arr2["update", "h1"]["score"], arr["update", "h1"]["score"])
    R._check_record(names)                                             # new hashes are the record now
    with pytest.raises(SystemExit):
        R.cmd_jokar_convert_dates(None)                                # once


def test_check_record_refuses_a_changed_file(tmp_path, monkeypatch):
    (tmp_path / "x.pkl").write_bytes(b"1")
    rec = {"files": {"x.pkl": {"sha256": hashlib.sha256(b"1").hexdigest()}}}
    (tmp_path / "rec.json").write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr(R, "RECORD", tmp_path / "rec.json")
    R._check_record(["x.pkl"])
    (tmp_path / "x.pkl").write_bytes(b"2")
    with pytest.raises(SystemExit):
        R._check_record(["x.pkl"])


def test_load_parts_refuses_missing_chunks(tmp_path):
    for k in (0, 1):
        (tmp_path / f"{k:05d}.pkl").write_bytes(pickle.dumps([{"k": k}, {"k": k}]))
    assert len(R._load_parts(tmp_path, 4)) == 4
    with pytest.raises(SystemExit):
        R._load_parts(tmp_path, 5)                                     # a household short
    (tmp_path / "00003.pkl").write_bytes(pickle.dumps([{"k": 3}]))
    with pytest.raises(SystemExit):
        R._load_parts(tmp_path, 5)                                     # chunk 2 missing
