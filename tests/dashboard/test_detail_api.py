"""Detail views over the git-ignored Layer 1-3 files reproduce the committed
summaries (plan.md §9).  Skipped where the detail files are not present."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import create_app
from src.dashboard.settings import DashboardSettings

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
INJ2 = RES / "layer3" / "ntl_inj2"

pytest.importorskip("pyarrow")


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def need(*paths: Path):
    missing = [str(p) for p in paths if not p.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"detail files missing: {missing}")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(DashboardSettings(web_dist=ROOT / "web" / "__no_build__")))


@need(RES / "faults_2024" / "fault_events.parquet")
def test_zone_hits_match_zone_accuracy(client):
    e = client.get("/api/results/faults/events").json()
    acc = _json(RES / "layer3" / "fault" / "summary.json")["zone"]["accuracy"]
    acc = acc["p"] if isinstance(acc, dict) else acc
    assert e["truth_available"] is True
    assert sum(x["zone_hit"] for x in e["events"]) / len(e["events"]) == pytest.approx(acc)


@need(RES / "faults_2024" / "event_records.npz")
def test_fault_burst(client):
    b = client.get("/api/results/faults/bursts/38").json()
    assert b["available"] is True and len(b["meters"]) > 0
    m = b["meters"][0]
    assert len(m["v_pu"]["a"]) == len(m["i_a"]["c"]) <= len(b["t_ms"])
    assert client.get("/api/results/faults/bursts/999999").json()["available"] is False


@need(INJ2 / "summary.json", INJ2 / "dt_day.parquet", RES / "year_2024_inj2" / "household_daily_labels.parquet")
def test_theft_dt_daily_truth_matches_scored_mean(client):
    np = pytest.importorskip("numpy")
    d = client.get("/api/results/theft/dt-daily").json()
    src = _json(INJ2 / "summary.json")["balance"]["dt"]["jul_dec_test"]["netted_lambda_0"]["all_dt_days"]
    i = d["days"].index("2024-07-01")
    truth = np.array([d["true_theft_kwh"][k][i:] for k in d["dts"]], dtype=float)
    est = np.array([d["estimate_kwh"][k][i:] for k in d["dts"]], dtype=float)
    assert truth.mean() == pytest.approx(src["mean_truth_kwh"], abs=1e-3)
    assert np.nanmean(est) == pytest.approx(src["mean_estimate_kwh"], abs=1e-3)


@need(INJ2 / "summary.json", INJ2 / "jokar_test.npz", RES / "year_2024_inj2" / "households.parquet")
def test_inspection_rule_reproduces_fold_counts():
    """The ranking's per-fold rule (share of online days flagged >= q) gives the
    scored tp / fp on each fold's evaluation set (honest + that attack's thieves)."""
    pd = pytest.importorskip("pandas")
    from src.dashboard.detail import ENERGY_THEFT_FOLDS, _jokar_rank
    pf = _json(INJ2 / "summary.json")["jokar"]["update"]["ungated"]["per_fold"]
    qs = tuple((f, pf[f]["q"]) for f in ENERGY_THEFT_FOLDS)
    r = _jokar_rank(INJ2 / "jokar_test.npz", 0.0, "update", qs)
    h = pd.read_parquet(RES / "year_2024_inj2" / "households.parquet", columns=["household", "attack", "thief"])
    r = r.merge(h, on="household")
    for f, q in qs:
        ev = r[(~r.thief | (r.attack == f)) & r[f"score_{f}"].notna()]
        flagged = ev[f"score_{f}"] >= q
        assert len(ev) == pf[f]["households_scored"]
        assert int((flagged & (ev.attack == f)).sum()) == pf[f]["tp"]
        assert int((flagged & ~ev.thief).sum()) == pf[f]["fp"]


@need(INJ2 / "jokar_test.npz")
def test_inspection_endpoint(client):
    j = client.get("/api/results/theft/inspection?dt=dt_652_1&limit=5").json()
    assert j["available"] is True and len(j["rows"]) == 5 and {r["dt"] for r in j["rows"]} == {"dt_652_1"}
    assert j["folds"] == ["h1", "h2", "h3", "h4"] and set(j["q"]) == set(j["folds"])
    assert all(set(r["scores"]) == set(j["folds"]) for r in j["rows"])
    folds = [r["folds_flagging"] for r in j["rows"]]
    assert folds == sorted(folds, reverse=True)


@need(RES / "year_2024" / "meter_frame_metered.parquet")
def test_vuf_daily_matches_fault_summary(client):
    u = client.get("/api/results/unbalance/daily").json()
    per = _json(RES / "layer3" / "fault" / "summary.json")["unbalance_per_meter_2024"]["per_meter"]
    for key, v in per.items():
        lid = key.split()[1]
        got = max(x for x in u["max_pct"][lid] if x is not None)
        assert got == pytest.approx(v["vu_max_pct_engine"], abs=2e-3), key


@need(RES / "year_2024" / "state.parquet")
def test_grid_daily_and_day(client):
    g = client.get("/api/results/grid/daily").json()
    assert len(g["days"]) == 366 and len(g["dt_load_pct_max"]) == 13
    d = client.get("/api/results/grid/day/2024-06-01").json()
    assert len(d["times"]) == 96 and max(d["p_head_mw"]) == pytest.approx(g["head_peak_mw"][152], abs=1e-3)
    assert client.get("/api/results/grid/day/2025-01-01").json()["available"] is False


@need(RES / "layer3" / "thermal" / "reference_main.npz")
def test_thermal_daily_peaks_match_summary(client):
    t = client.get("/api/results/thermal/daily").json()
    ref = _json(RES / "layer3" / "thermal" / "summary.json")["reference"]["main"]
    for dt, v in t["theta_h_max_c"].items():
        peak = ref["per_dt"][dt].get("peak_theta_h_c", ref["per_dt"][dt].get("theta_h_max_c"))
        if peak is not None:
            assert max(v) == pytest.approx(peak, abs=0.06), dt
    d = client.get("/api/results/thermal/day/2024-06-01").json()
    assert d["available"] is True and len(d["times"]) == 96 and d["forecast"] is not None
