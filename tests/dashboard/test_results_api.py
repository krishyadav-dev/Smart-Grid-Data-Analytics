"""Dashboard report API returns the frozen Layer 3 numbers unchanged (plan.md §9)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import create_app
from src.dashboard.results import VU_LIMIT_PCT, transpose
from src.dashboard.settings import DashboardSettings

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(DashboardSettings(web_dist=ROOT / "web" / "__no_build__")))


def test_config_and_meta(client):
    c = client.get("/api/config").json()
    assert c["feeder_id"] == "ieee13_11kv" and c["mode"]["report"] is True
    m = client.get("/api/meta").json()
    assert len(m["meters"]) == 23 and sum(x["physical"] for x in m["meters"]) == 14
    assert [z["id"] for z in m["zones"]] == list("ABCDEFG")


def test_geo_layout(client):
    g = client.get("/api/geo/layout").json()
    assert g["type"] == "FeatureCollection" and g["metadata"]["illustrative"] is True


def test_faults_numbers_unchanged(client):
    src = _json(RES / "layer3" / "fault" / "summary.json")
    f = client.get("/api/results/faults").json()
    assert f["detection"]["recall"] == src["detection"]["recall"]
    assert f["zone"]["accuracy"]["p"] == pytest.approx(0.85)
    assert f["zone"]["baseline"]["p"] == pytest.approx(0.70)
    # confusion rows are TRUE classes: row sums equal the true-zone counts
    rows = f["zone"]["confusion_true_rows"]
    for z, v in src["zone"]["per_true_zone"].items():
        assert sum(rows[z].values()) == v["n"]
        assert rows[z][z] == v["k"]
    trow = f["type"]["confusion_true_rows"]
    for t, v in src["type"]["per_type_accuracy"].items():
        assert sum(trow[t].values()) == v["n"]
    assert f["live"]["pooled"]["900x"]["live_batch_identical"] is True


def test_transpose():
    assert transpose({"A": {"A": 2, "B": 1}, "B": {"A": 0, "B": 3}}) == {"A": {"A": 2, "B": 0}, "B": {"A": 1, "B": 3}}


def test_fault_events(client):
    pytest.importorskip("pyarrow")
    e = client.get("/api/results/faults/events").json()
    assert e["available"] is True and len(e["events"]) == 100
    assert {x["zone"] for x in e["events"]} <= set("ABCDEFG")


def test_fault_events_truth_join(tmp_path):
    """With a fault set present, events gain true bus / zone and hit flags."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    (tmp_path / "layer3" / "fault").mkdir(parents=True)
    shutil.copy(RES / "layer3" / "fault" / "summary_events.parquet", tmp_path / "layer3" / "fault")
    ev = pd.read_parquet(RES / "layer3" / "fault" / "summary_events.parquet")
    (tmp_path / "faults_2024").mkdir()
    pd.DataFrame({"event_id": ev.matched, "bus": ev.zone_node, "ftype": ev.ftype, "phases": "abc",
                  "rf_ohm": 5.0, "rf_model": "fixed", "kind": "fault"}).to_parquet(tmp_path / "faults_2024" / "fault_events.parquet")
    c = TestClient(create_app(DashboardSettings(results_dir=tmp_path, web_dist=ROOT / "web" / "__no_build__")))
    out = c.get("/api/results/faults/events").json()
    assert out["truth_available"] is True
    first = out["events"][0]
    assert first["true_bus"] == first["zone_node"] and first["type_hit"] is True and first["rf_class"] == "5"


def test_theft_numbers_unchanged(client):
    inj2 = (RES / "layer3" / "ntl_inj2" / "summary.json").exists()
    src = _json(RES / "layer3" / ("ntl_inj2" if inj2 else "ntl") / "summary.json")
    t = client.get("/api/results/theft").json()
    assert t["injection"] == ("ntl_inj2" if inj2 else "ntl")
    assert t["energy_identity"]["closes"] is True
    assert t["energy_identity"]["residual_kwh"] == src["balance"]["energy_identity_test_window"]["residual_kwh"]
    assert t["jokar"]["macro_energy_theft_h1_h4"]["precision"] < 0.05
    assert sum(t["households_per_dt"].values()) == 8020
    zfull = RES / "layer3" / "sgcc_full" / "summary.json"
    zsrc = _json(zfull if zfull.exists() else RES / "layer3" / "sgcc" / "summary.json")
    assert t["zheng"]["mean_and_range"]["auc"]["mean"] == zsrc["mean_and_range"]["auc"]["mean"]
    if inj2:
        assert t["jokar"]["variant"] == "update" and t["jokar"]["gate"] == src["jokar"]["primary_gate"]
        assert t["annual_mwh"] == src["balance"]["annual_mwh"]


def test_missing_detail_files_degrade(tmp_path):
    """With no git-ignored detail files, every detail view says how to make them."""
    for d in ("layer3/fault", "layer3/ntl", "layer3/thermal", "layer3/sgcc", "unbalance_scenarios"):
        (tmp_path / d).mkdir(parents=True)
    for f in ("layer3/fault/summary.json", "layer3/ntl/summary.json", "layer3/ntl/inputs.json",
              "layer3/thermal/summary.json", "layer3/sgcc/summary.json", "unbalance_scenarios/summary.json"):
        shutil.copy(RES / f, tmp_path / f)
    c = TestClient(create_app(DashboardSettings(results_dir=tmp_path, data_dir=tmp_path / "data",
                                                web_dist=ROOT / "web" / "__no_build__")))
    assert c.get("/api/results/theft").json()["injection"] == "ntl"
    for u in ("/api/results/theft/dt-daily", "/api/results/theft/inspection", "/api/results/faults/bursts/1",
              "/api/results/grid/daily", "/api/results/grid/day/2024-06-01", "/api/results/thermal/daily",
              "/api/results/thermal/day/2024-06-01", "/api/results/unbalance/daily", "/api/results/faults/events"):
        d = c.get(u).json()
        assert d["available"] is False and d["how"], u


def test_thermal_and_unbalance(client):
    th = client.get("/api/results/thermal").json()
    assert th["acceptance"]["passes"] is True
    assert set(th["forecast"]["models"]) == {"persistence", "seasonal_naive", "lstm"}
    assert th["forecast"]["bootstrap"]["rmse_persistence_minus_lstm_k"]["lstm_better"] is False
    u = client.get("/api/results/unbalance").json()
    assert u["limit_pct"] == 2.0 and u["n_scenarios"] == 31


def test_vu_limit_matches_analytics():
    from src.analytics import unbalance
    assert VU_LIMIT_PCT == unbalance.VU_LIMIT_PCT


def test_ingestion(client):
    i = client.get("/api/results/ingestion").json()
    assert i["available"] is True and i["runs"]
    assert any(r["results"] and "latency_ms" in r["results"][0] for r in i["runs"])
