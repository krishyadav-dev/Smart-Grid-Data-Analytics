"""Jokar detector, Stage 4 changes B1, B9a, B9b (Stage 4 run sheet §4, step D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import ntl_detector as D
from src.analytics import ntl_engine as N
from src.data_source.theft_injector import ATTACKS, apply_attack


def test_grid_b9a():
    assert D.GRID_C == tuple(2.0 ** k for k in (-6, -4, -2, 0, 2, 4, 6, 8, 10, 12))
    assert D.GRID_GAMMA == tuple(2.0 ** k for k in (-17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3))
    assert len(D.GRID) == 110 and set(N.GRID_C) <= set(D.GRID_C) and set(N.GRID_GAMMA) <= set(D.GRID_GAMMA)
    assert D.edge(2.0 ** 12, 2.0 ** -9) == ["C_high"] and D.edge(1.0, 2.0 ** -17) == ["gamma_low"]
    assert D.edge(1.0, 2.0 ** -9) == []


@pytest.mark.parametrize("kind", ATTACKS)
def test_apply_attack_96_steps_is_jokar_hourly(kind):
    """B9b check (Jokar §VI, PDF p. 7): at 96 steps the parameters stay hourly
    (alpha per day; beta_t, gamma_t per hour; off windows of whole hours, at
    least minOFFtime = 4 h), so with the same draws the 96-step attack summed to
    hours equals the hourly attack of the hourly sums."""
    rng = np.random.default_rng(5)
    x96 = rng.uniform(0.05, 0.6, (40, 96))
    x24 = x96.reshape(40, 24, 4).sum(axis=2)
    a96, p96 = apply_attack(kind, x96, np.random.default_rng(11))
    a24, p24 = apply_attack(kind, x24, np.random.default_rng(11))
    np.testing.assert_allclose(a96.reshape(40, 24, 4).sum(axis=2), a24, rtol=1e-12, atol=1e-12)
    if kind in ("h2", "h3", "h4"):
        ratio = np.divide(a96, x96 if kind != "h4" else x96.mean(axis=1, keepdims=True))
        r = ratio.reshape(40, 24, 4)
        np.testing.assert_allclose(r, r[:, :, :1].repeat(4, axis=2), rtol=1e-12)     # constant within the hour
    if kind == "h2":
        off = (a96 == 0).reshape(40, 24, 4)
        assert (off.all(axis=2) | ~off.any(axis=2)).all()                            # whole hours only
        hours = off.all(axis=2).sum(axis=1)
        np.testing.assert_array_equal(hours, p96["t_end_h"] - p96["t_start_h"])
        assert (hours >= 4).all()                                                    # minOFFtime


def test_ami_vectors_96(tmp_path):
    ts = pd.date_range("2024-03-01", periods=2 * 96, freq="15min")
    a = np.arange(len(ts), dtype=float)
    b = np.ones(len(ts))
    b[100] = np.nan
    pd.DataFrame({"timestamp": ts, "hh_1": pd.array(a, dtype="UInt16"),
                  "hh_2": pd.array(np.where(np.isnan(b), None, b), dtype="UInt16")}).to_parquet(
        tmp_path / "ami_2024-03.parquet", index=False)
    ids, days, x = D.ami_vectors(tmp_path, 96)
    assert ids == ["hh_1", "hh_2"] and x.shape == (2, 2, 96)
    np.testing.assert_allclose(x[0, 1, :3], np.array([96, 97, 98]) / 1e3)
    assert np.isnan(x[1, 1]).sum() == 1
    ids24, _, x24 = D.ami_vectors(tmp_path, 24)
    np.testing.assert_allclose(np.nansum(x24[0]), np.nansum(x[0]), rtol=1e-6)
    with pytest.raises(ValueError):
        D.ami_vectors(tmp_path, 48)


def _household(samples=24, seed=3):
    days = pd.date_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(seed)
    shape = np.sin(np.linspace(0, 2 * np.pi, samples)) + 1.5
    x = (shape[None, :] * rng.uniform(0.8, 1.2, (len(days), 1))).astype(np.float32)
    x[10] = np.nan
    x[200] = np.nan
    return x, days


def test_static_reproduces_stage2_scores():
    x, days = _household()
    p2 = N.prepare_household(x, days, 0, "validation")
    s2, f2 = N.score_household(p2, N.FOLDS["h1"], 4.0, 2 ** -3)
    p4 = D.prepare(x, days, 0, "validation")
    r = D.run_fold(p4, N.FOLDS["h1"], 4.0, 2 ** -3, "static")
    sc = p4["scored"][p4["out"]]
    np.testing.assert_array_equal(r["flag"][sc], f2)
    np.testing.assert_allclose(r["score"][sc], s2, rtol=1e-6, equal_nan=True)
    assert not r["added"].any()
    assert (r["flag"][~sc & p4["online"][p4["out"]]] >= 0).all()          # training days predicted too


def test_update_adds_benign_days_monthly():
    x, days = _household()
    p = D.prepare(x, days, 0, "test")
    r = D.run_fold(p, N.FOLDS["h2"], 4.0, 2 ** -3, "update")
    out_days = days[p["out"]]
    benign = r["flag"] == 0
    dec = out_days.month == 12
    # every benign online day of July-November joins the benign set; December never does
    np.testing.assert_array_equal(r["added"], benign & ~dec)
    assert set(r["k_prime"]) == {7, 8, 9, 10, 11, 12}
    st = D.run_fold(p, N.FOLDS["h2"], 4.0, 2 ** -3, "static")
    jul = out_days.month == 7
    np.testing.assert_array_equal(st["flag"][jul], r["flag"][jul])           # same model before the first update
    assert not st["added"].any()
    # deterministic, and the attack patterns of a month do not depend on the fold
    r2 = D.run_fold(D.prepare(x, days, 0, "test"), N.FOLDS["h2"], 4.0, 2 ** -3, "update")
    np.testing.assert_array_equal(r["flag"], r2["flag"])
    d1, a1 = D.month_attacks(p, 8)
    d2, a2 = D.month_attacks(D.prepare(x, days, 0, "test"), 8)
    np.testing.assert_array_equal(d1, d2) and np.testing.assert_array_equal(a1["h3"], a2["h3"])


def test_grid_auc_variants():
    x, days = _household()
    p = D.prepare(x, days, 0, "validation")
    grid = [(4.0, 2 ** -3), (1.0, 2 ** -7)]
    g = D.grid_auc(p, "h1", grid)
    assert set(g) == {"static", "update"} and all(len(v) == 2 for v in g.values())
    assert all(0.0 <= v <= 1.0 for v in g["static"] + g["update"])
    # the static objective is the Stage 2 objective
    np.testing.assert_allclose(g["static"], N.grid_auc(N.prepare_household(x, days, 0, "validation"), 0, "h1", grid))
    assert set(D.grid_auc(p, "h1", grid, variants=("static",))) == {"static"}


def test_resolution_choice():
    b = {24: {f: 0.80 for f in D.LOO_FOLDS}, 96: {f: 0.80 for f in D.LOO_FOLDS}}
    assert D.resolution_choice(b)["samples_per_day"] == 24                   # tie keeps 24
    b[96]["h1"] = 0.81
    assert D.resolution_choice(b)["samples_per_day"] == 96
    b[96]["h2"] = 0.79
    assert D.resolution_choice(b)["samples_per_day"] == 24
    b[24]["circular"] = 0.1                                                  # the circular fold is a reference
    assert D.resolution_choice(b)["folds"] == list(D.LOO_FOLDS)


def test_96_samples_household_runs():
    x, days = _household(96)
    p = D.prepare(x, days, 1, "validation")
    r = D.run_fold(p, N.FOLDS["h5"], 4.0, 2 ** -5, "update")
    assert r["score"].shape == (p["out"].sum(),) and (r["flag"][p["online"][p["out"]]] >= 0).all()


# --------------------------------------------------------------------------
# Scoring (evaluate.py, Stage 4 run sheet §4 metrics)
# --------------------------------------------------------------------------
from src.analytics import evaluate as V  # noqa: E402


def _labels():
    return pd.DataFrame({"thief": [False, False, False, True, True, True],
                         "attack": ["", "", "", "h1", "h1", "h2"],
                         "onset": pd.to_datetime([None, None, None, "2024-07-01", "2024-08-15", "2024-07-01"])},
                        index=[f"hh_{i}" for i in range(6)])


def test_budget_k():
    assert V.budget_k(0.01, 7350) == 74 and V.budget_k(0.01, 20) == 1 and V.budget_k(0.05, 100) == 5


def test_precision_at_budgets_and_recall_by_month():
    days = pd.date_range("2024-07-01", "2024-08-31")
    ids = list(_labels().index)
    fl = np.zeros((6, len(days)), np.int8)
    sc = np.zeros((6, len(days)), np.float32)
    jul = np.asarray(days.month == 7)
    fl[3] = 1                                      # h1 thief from July: always flagged
    fl[1, jul] = 1                                 # an honest household flagged in July
    sc[1] = 5.0                                    # ... with a higher mean score than the thief
    fl[2, :3] = -1                                 # offline days are not counted
    r = V.precision_at_budgets(fl, sc, days, ids, _labels(), ("h1",), budgets=(0.01, 0.5))
    m7, m8 = r["months"][7], r["months"][8]
    # July: hh_4 (onset 15 Aug) excluded, hh_5 (h2) excluded -> 4 scored, 1 positive
    assert m7["scored"] == 4 and m7["positives"] == 1 and m7["prevalence"] == 0.25
    # tie on the flag share (1.0): the honest household's higher mean score ranks first
    assert m7["precision_at_0.01"]["k"] == 0 and m7["precision_at_0.5"]["k"] == 1
    assert m8["scored"] == 5 and m8["positives"] == 2 and m8["precision_at_0.01"]["k"] == 1
    assert r["pooled"]["precision_at_0.01"]["n"] == 2
    rm = V.recall_by_month(fl, sc, days, ids, _labels(), ("h1",), q=0.5)
    assert rm[7]["k"] == 1 and rm[7]["n"] == 1 and rm[8]["k"] == 1 and rm[8]["n"] == 2


def test_honest_share_and_absorption():
    days = pd.date_range("2024-07-01", "2024-08-31")
    ids = list(_labels().index)
    fl = np.zeros((6, len(days)), np.int8)
    fl[0] = 1
    fl[2, :5] = -1
    sh = V.honest_flag_share_by_month({"h1": fl}, days, ids, _labels())
    assert sh[7]["n"] == 3 * 31 - 5 and sh[7]["k"] == 31
    added = np.zeros_like(fl, bool)
    jul = np.asarray(days.month == 7)
    added[3, jul] = True                           # all of hh_3's July days absorbed
    added[0, jul] = True                           # honest days do not count
    ab = V.absorption({"h1": fl}, {"h1": added}, days, ids, _labels())
    assert ab["months_with_updates"] == [7]
    # July: hh_3 31 attacked days all absorbed, hh_5 31 attacked days none; hh_4 not yet active
    assert ab["pooled"]["k"] == 31 and ab["pooled"]["n"] == 62
    assert ab["by_attack"]["h1"]["p"] == 1.0 and ab["by_attack"]["h2"]["p"] == 0.0
    assert ab["held_out_fold_only"]["n"] == 31 and ab["per_thief_fold_share"]["median"] == 0.5


def test_frozen_inj2_detector_settings():
    import yaml
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "config" / "layer3_ntl_params_inj2.yaml"
    if not path.exists():
        pytest.skip("config/layer3_ntl_params_inj2.yaml is written by step D")
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert d["samples_per_day"] in D.RESOLUTIONS and set(d["folds"]) == set(D.VARIANTS)
    for v in D.VARIANTS:
        assert list(d["folds"][v]) == list(D.FOLDS)
        for f, p in d["folds"][v].items():
            assert p["C"] in D.GRID_C and p["gamma"] in D.GRID_GAMMA, (v, f)
            assert 0.0 <= p["q"] <= 1.0 and 0.0 <= p["q_gated"] <= 1.0, (v, f)
    assert d["grid"]["C"] == list(D.GRID_C) and d["grid"]["gamma"] == list(D.GRID_GAMMA)
    assert d["seed"] == N.JOKAR_SEED and "January-June" in d["tuned_on"]["data"]
