"""Feeder energy identity on the cached year (Layer 3 P3, N5).

head energy - billed = MV and DT losses + theft + offline (would-have-reported)
+ meter error, over 2024, from the cache files:

- head energy: feeder-head meter (``meter_frame_metered``; true value from
  ``meter_frame_true``); billed: household metered kWh (``household_daily``);
- losses: 11 kV lines and DTs (``state.parquet``; the head meter is at bus 650,
  downstream of the substation transformer);
- theft and offline: ``household_daily_labels.parquet``;
- meter error: (head metered - head true) - (billed - reported).

Tolerance, from the cache itself: the power flow's largest per-step feeder
balance error (``acceptance.json``) over every step, plus float32 storage
rounding of the household daily values (2^-23 of their total).
Skipped when the cache (or the P3 label file) is absent.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data_source import config
from src.data_source import simulation as S

C = config.YEAR_CACHE_DIR


@pytest.fixture(scope="module")
def cache():
    if not (C / S.LABELS_FILE).exists():
        pytest.skip("year cache with the P3 label file absent")
    f = [("load_id", "==", "feeder_head_650")]
    head_true = pd.read_parquet(C / "meter_frame_true.parquet", columns=["load_id", "active_power_kw"],
                                filters=f).active_power_kw.sum() * S.STEP_H
    head_met = pd.read_parquet(C / "meter_frame_metered.parquet", columns=["load_id", "active_power_kw"],
                               filters=f).active_power_kw.sum() * S.STEP_H
    st = pd.read_parquet(C / "state.parquet", columns=["line_losses_mw", "dt_losses_mw"])
    hd = pd.read_parquet(C / "household_daily.parquet",
                         columns=["household", "date", "true_kwh", "reported_kwh", "metered_kwh"])
    lab = pd.read_parquet(C / S.LABELS_FILE)
    acc = json.loads((C / "acceptance.json").read_text(encoding="utf-8"))
    assert np.array_equal(hd.household.to_numpy(), lab.household.to_numpy())
    assert np.array_equal(hd.date.to_numpy(), lab.date.to_numpy())
    true = hd.true_kwh.to_numpy(float)
    tol = (acc["feeder_balance_max_abs_mw"] * 1e3 * S.STEP_H * acc["steps"]
           + 2.0 ** -23 * np.abs(true).sum())
    return {
        "head_true": float(head_true), "head_metered": float(head_met),
        "losses": float((st.line_losses_mw + st.dt_losses_mw).sum() * 1e3 * S.STEP_H),
        "true": float(true.sum()),
        "reported": float(np.nansum(hd.reported_kwh.to_numpy(float))),
        "billed": float(np.nansum(hd.metered_kwh.to_numpy(float))),
        "theft": float(lab.theft_kwh.sum()), "offline": float(lab.offline_kwh.sum()),
        "tol": float(tol)}


def test_energy_identity_closes(cache):
    c = cache
    meter_error = (c["head_metered"] - c["head_true"]) - (c["billed"] - c["reported"])
    lhs = c["head_metered"] - c["billed"]
    rhs = c["losses"] + c["theft"] + c["offline"] + meter_error
    print("\n" + "; ".join(f"{k} {v / 1e3:,.3f} MWh" for k, v in c.items() if k != "tol")
          + f"; meter error {meter_error / 1e3:,.3f} MWh; residual {lhs - rhs:.4f} kWh "
          f"(tolerance {c['tol']:.3f} kWh)")
    assert abs(lhs - rhs) <= c["tol"]


def test_true_energy_splits_without_double_counting(cache):
    """Sum of true household energy = reported (online) + offline + theft."""
    c = cache
    assert abs(c["true"] - (c["reported"] + c["offline"] + c["theft"])) <= c["tol"]


def test_head_energy_is_households_plus_losses(cache):
    """Power-flow balance: head (true) = true household energy + line and DT losses."""
    c = cache
    assert abs(c["head_true"] - c["true"] - c["losses"]) <= c["tol"]
