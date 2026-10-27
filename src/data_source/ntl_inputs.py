"""
Theft-engine scenario inputs (Layer 3 Stage 2 run sheet §2, step A).

Layer 1 code: it reads the true frame and rebuilds the seeded population, and
writes **metered-format** inputs that the theft engine reads exactly like the
normal metered data (the engine never sees the true frame).

1. **lambda_true grid (N3):** DT-meter 15-min energies with an I^2R-shaped LV
   loss added at lambda_true in {0, 2, 4} %, built exactly as
   :func:`simulation.lambda_sweep` builds them (``lv_loss_kw`` on the true DT
   power, times each DT meter's gain).  ``lambda_true_{0,2,4}/dt_meter_kwh.parquet``
   (rows = timestamps, columns = DT names, kWh per 15 min).
2. **Ideal meters (N6):** feeder-head and DT meter rows and household daily
   metered kWh with every meter gain = 1.0, through the existing metering code
   (:func:`simulation.apply_meter_errors`, :func:`metering.daily_table`).
   ``ideal_meter/``.  The same code with the real gains must reproduce the
   cached metered values exactly (checked, :func:`build`).
3. **Consumer index:** household, DT and bus only, the utility's
   consumer-to-DT indexing (user decision 27 Oct: no LV phase).
   ``consumer_index.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import metering as M
from . import simulation as S

OUT_DIR = config.PROJECT_ROOT / "results" / "layer3" / "ntl"
LAMBDA_TRUE = (0.0, 0.02, 0.04)              # N3 true-vs-assumed grid (plan §2)
CONSUMER_INDEX_COLUMNS = ("household", "dt", "bus")
HEAD = "feeder_head_650"


def lambda_dir(lam: float, out: Path = OUT_DIR) -> Path:
    return Path(out) / f"lambda_true_{round(lam * 100):d}"


def dt_true_kw(frame_true: pd.DataFrame, dts: pd.DataFrame) -> pd.DataFrame:
    """True DT LV power (steps x DTs, kW), summed over phases, as lambda_sweep."""
    rows = frame_true[frame_true.load_id.str.startswith("dt_")]
    return rows.pivot_table(index="timestamp", columns="load_id", values="active_power_kw",
                            aggfunc="sum")[dts.name]


def dt_meter_kwh(p_true: pd.DataFrame, gains: np.ndarray, lam: float) -> pd.DataFrame:
    """DT-meter 15-min energy with LV loss ``lam`` (lambda_sweep's construction)."""
    p = p_true.to_numpy()
    e = (p + S.lv_loss_kw(p, lam)) * S.STEP_H * gains
    return pd.DataFrame(e, index=p_true.index, columns=p_true.columns)


def physical_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Feeder-head and DT meter rows (the 14 physical meters)."""
    keep = (frame.load_id == HEAD) | frame.load_id.str.startswith("dt_")
    return frame[keep].reset_index(drop=True)


def build(cache: Path = config.YEAR_CACHE_DIR, out: Path = OUT_DIR) -> dict:
    """Write every step-A input and return the check record."""
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {}
    dts = pd.read_parquet(cache / "dts.parquet")
    frame_true = pd.read_parquet(cache / "meter_frame_true.parquet")
    metered = pd.read_parquet(cache / "meter_frame_metered.parquet")
    pop = S.build_population()

    # the rebuilt population is the cached one
    if not (np.array_equal(pop.gain_dt, dts.meter_gain.to_numpy())
            and list(pop.net["dts"].name) == list(dts.name)):
        raise ValueError("rebuilt DT meter gains differ from dts.parquet")
    g_fh = json.loads((cache / "run_info.json").read_text(encoding="utf-8"))["feeder_head_gain"]
    if pop.gain_feeder_head != g_fh:
        raise ValueError("rebuilt feeder-head gain differs from run_info.json")

    # 1. lambda_true grid; lambda_true = 0 must be the metered DT energy
    p_true = dt_true_kw(frame_true, dts)
    m_dt = metered[metered.load_id.str.startswith("dt_")]
    e_metered = (m_dt.pivot_table(index="timestamp", columns="load_id", values="active_power_kw",
                                  aggfunc="sum")[dts.name] * S.STEP_H)
    rec["lambda_true"] = {}
    for lam in LAMBDA_TRUE:
        e = dt_meter_kwh(p_true, dts.meter_gain.to_numpy(), lam)
        d = lambda_dir(lam, out)
        d.mkdir(parents=True, exist_ok=True)
        e.to_parquet(d / "dt_meter_kwh.parquet")
        rec["lambda_true"][f"{lam:.2f}"] = {
            "dt_energy_mwh": float(e.to_numpy().sum() / 1e3),
            "lv_loss_share_of_true_dt_energy": float(e.to_numpy().sum() / (p_true.to_numpy()
                                                     * S.STEP_H * dts.meter_gain.to_numpy()).sum() - 1)}
        if lam == 0.0:
            dev = float(np.abs(e.to_numpy() - e_metered.to_numpy()).max())
            rec["lambda_true"]["zero_matches_metered_frame_max_abs_kwh"] = dev
            if not np.allclose(e.to_numpy(), e_metered.to_numpy(), rtol=1e-12, atol=1e-9):
                raise ValueError(f"lambda_true = 0 does not reproduce the metered DT energy ({dev})")

    # 2. ideal meters: the same code with the real gains reproduces the cache exactly
    gains_dt = dict(zip(dts.name, dts.meter_gain))
    again = S.apply_meter_errors(frame_true, gains_dt, g_fh)
    pd.testing.assert_frame_equal(again, metered, check_exact=True)
    daily_cached = pd.read_parquet(cache / "household_daily.parquet")
    daily_again = M.daily_table(pop.hh, pop.theft, pop.gain_household)
    pd.testing.assert_frame_equal(daily_again, daily_cached, check_exact=True)
    rec["ideal_meter"] = {"real_gains_reproduce_meter_frame_metered_exactly": True,
                          "real_gains_reproduce_household_daily_exactly": True}
    del again, daily_again, daily_cached
    d = out / "ideal_meter"
    d.mkdir(parents=True, exist_ok=True)
    ideal = S.apply_meter_errors(frame_true, {n: 1.0 for n in dts.name}, 1.0)
    physical_rows(ideal).to_parquet(d / "meter_frame_metered.parquet", index=False)
    ones = np.ones(len(pop.hh.hh_id))
    M.daily_table(pop.hh, pop.theft, ones)[["household", "date", "metered_kwh"]].to_parquet(
        d / "household_daily.parquet", index=False)
    rec["ideal_meter"]["head_energy_mwh"] = float(
        ideal.loc[ideal.load_id == HEAD, "active_power_kw"].sum() * S.STEP_H / 1e3)
    rec["ideal_meter"]["head_energy_mwh_real_gain"] = float(
        metered.loc[metered.load_id == HEAD, "active_power_kw"].sum() * S.STEP_H / 1e3)
    rec["ideal_meter"]["feeder_head_gain"] = g_fh

    # 3. consumer index
    idx = pd.DataFrame({"household": pop.hh.hh_id, "dt": pop.hh.dt, "bus": pop.hh.bus})
    idx = idx[list(CONSUMER_INDEX_COLUMNS)]
    idx.to_parquet(out / "consumer_index.parquet", index=False)
    rec["consumer_index"] = {"households": int(len(idx)), "columns": list(idx.columns),
                             "per_dt": {k: int(v) for k, v in idx.dt.value_counts().sort_index().items()}}
    (out / "inputs.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec
