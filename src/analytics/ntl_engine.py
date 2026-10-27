"""Theft (non-technical loss) engine (docs/Layer3_plan.md §2; Stage 2 run sheet §3-4).

Inputs (engine side only, plan §0): the metered meter frame of the 14 physical
meters (feeder head m10, DT meters m11-m23, selected by role), household daily
metered kWh (``household_daily.metered_kwh``; NaN = meter offline), the
consumer index (household -> DT; step A), and the network model.  It never
reads true loads, labels or meter gains.

Balances (plan §2 "Balances"; DT meters are on the LV side, so each DT's own
losses fall inside A), per day and for the year:

    A    = E_head - sum_DT E_DT
    B_DT = E_DT - sum_{i in DT} E_i
    C    = E_head - sum_i E_i

Netted estimates (run sheet §3):

    feeder  NTL^   = C - L^_tech - E^_offline
    DT      NTL^_B = B_DT - lambda_assumed sum_{i in DT} E_i - E^_offline,DT

with the un-netted baselines C and B_DT.

- **L^_tech** [our choice of loads, interpreting the plan]: a ``runpp_3ph`` run
  for every 15-min step with each DT's load set to its own metered P and Q per
  phase (the DT meter readings: the load the utility can see, theft included,
  since stolen energy does flow through the MV network and the DT).  The plan's
  wording ("metered, under-reported household energy") would understate losses
  by the loss on stolen energy.  L^_tech = MV line losses + DT losses of that
  run (the head meter is at bus 650, downstream of the substation
  transformer).  Never true loads.
- **E^_offline** (N5): per household offline day, the median of the
  household's own online days in the same calendar month and day type
  (weekday Monday-Friday / weekend).  Fallbacks, in order [our choice, run
  sheet §3]: same day type in the nearest month with at least 3 online days of
  that type (equal distance: the earlier month [our choice]); then all of the
  household's online days.  Nothing is tuned.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_source.feeder_model import build_network, run_timeseries

from . import meters as M

logger = logging.getLogger(__name__)

STEP_H = 0.25                       # 15-min metering interval (h)
PHASES = ("a", "b", "c")
FALLBACK_MIN_DAYS = 3               # run sheet §3: nearest month with >= 3 online days of the type
OFFLINE_RULES = ("same_month_day_type", "nearest_month_day_type", "all_online_days", "no_online_days")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def physical_load_ids() -> tuple[str, list[str]]:
    """(feeder-head load id, DT-meter load ids), selected by role."""
    return M.feeder_head().load_id, [m.load_id for m in M.dt_meters()]


def read_meter_frame(path: str | Path) -> pd.DataFrame:
    """The 14 physical meters' rows of a metered frame (P and Q per phase)."""
    head, dts = physical_load_ids()
    return pd.read_parquet(path, columns=["timestamp", "load_id", "phase", "active_power_kw",
                                          "reactive_power_kvar"],
                           filters=[("load_id", "in", [head, *dts])])


def meter_energy_15min(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Feeder-head and DT-meter energy per 15-min step (kWh), summed over phases
    (the frame has no energy column: P x 0.25 h, as ``lambda_sweep``)."""
    head, dts = physical_load_ids()
    e = frame.pivot_table(index="timestamp", columns="load_id", values="active_power_kw",
                          aggfunc="sum") * STEP_H
    return e[head], e[dts]


def daily(e15: pd.Series | pd.DataFrame):
    """Sum 15-min energies to calendar days."""
    return e15.groupby(pd.DatetimeIndex(e15.index).normalize()).sum()


def household_matrix(daily_long: pd.DataFrame) -> pd.DataFrame:
    """households x days metered kWh (NaN = offline) from ``household, date, metered_kwh``."""
    m = daily_long.pivot(index="household", columns="date", values="metered_kwh").astype(float)
    m.columns = pd.DatetimeIndex(m.columns)
    return m


def read_household_daily(path: str | Path) -> pd.DataFrame:
    return household_matrix(pd.read_parquet(path, columns=["household", "date", "metered_kwh"]))


def read_consumer_index(path: str | Path) -> pd.Series:
    """household -> DT name."""
    idx = pd.read_parquet(path, columns=["household", "dt"])
    return idx.set_index("household").dt


# --------------------------------------------------------------------------
# Offline estimate (N5)
# --------------------------------------------------------------------------

def day_type(days: pd.DatetimeIndex) -> np.ndarray:
    """0 = weekday (Monday-Friday), 1 = weekend."""
    return (pd.DatetimeIndex(days).weekday >= 5).astype(int)


def offline_estimate(hh: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """E^_offline per household and day (0 on online days) and the rule counts.

    ``hh``: households x days metered kWh, NaN on offline days.
    """
    days = pd.DatetimeIndex(hh.columns)
    month = days.month.to_numpy()
    dtype = day_type(days)
    x = hh.to_numpy(float)
    off = np.isnan(x)
    est = np.zeros_like(x)
    counts = dict.fromkeys(OFFLINE_RULES, 0)
    months = np.unique(month)
    for i in np.flatnonzero(off.any(axis=1)):
        row = x[i]
        online = ~off[i]
        med, n = {}, {}
        for m in months:
            for t in (0, 1):
                sel = online & (month == m) & (dtype == t)
                n[m, t] = int(sel.sum())
                med[m, t] = float(np.median(row[sel])) if n[m, t] else np.nan
        all_med = float(np.median(row[online])) if online.any() else np.nan
        for d in np.flatnonzero(off[i]):
            m, t = month[d], dtype[d]
            if n[m, t] >= 1:
                est[i, d], rule = med[m, t], OFFLINE_RULES[0]
            else:
                cand = sorted((abs(int(k) - int(m)), int(k)) for k in months
                              if n[k, t] >= FALLBACK_MIN_DAYS)
                if cand:
                    est[i, d], rule = med[cand[0][1], t], OFFLINE_RULES[1]
                elif online.any():
                    est[i, d], rule = all_med, OFFLINE_RULES[2]
                else:
                    est[i, d], rule = 0.0, OFFLINE_RULES[3]
            counts[rule] += 1
    out = pd.DataFrame(est, index=hh.index, columns=hh.columns)
    return out, {"offline_household_days": int(off.sum()), "rule_counts": counts}


# --------------------------------------------------------------------------
# Balances and netted estimates
# --------------------------------------------------------------------------

def per_dt(hh_values: pd.DataFrame, consumer_dt: pd.Series, dt_names) -> pd.DataFrame:
    """days x DT sums of a households x days table (NaN counted as 0)."""
    s = hh_values.fillna(0.0).groupby(consumer_dt.reindex(hh_values.index).to_numpy()).sum()
    return s.reindex(list(dt_names)).T


def balances(head_day: pd.Series, dt_day: pd.DataFrame, hh: pd.DataFrame,
             consumer_dt: pd.Series) -> dict:
    """A, B_DT and C per day (kWh) plus the billed sums they use."""
    billed_dt = per_dt(hh, consumer_dt, dt_day.columns).reindex(dt_day.index)
    billed = hh.fillna(0.0).sum(axis=0).reindex(head_day.index)
    return {"A": head_day - dt_day.sum(axis=1), "B": dt_day - billed_dt,
            "C": head_day - billed, "billed": billed, "billed_dt": billed_dt}


def netted_feeder(c_day: pd.Series, ltech_day: pd.Series, eoff_day: pd.Series) -> pd.Series:
    """NTL^ = C - L^_tech - E^_offline per day."""
    return c_day - ltech_day.reindex(c_day.index) - eoff_day.reindex(c_day.index)


def netted_dt(b_day: pd.DataFrame, billed_dt: pd.DataFrame, eoff_dt: pd.DataFrame,
              lam_assumed: float) -> pd.DataFrame:
    """NTL^_B = B_DT - lambda_assumed sum_i E_i - E^_offline,DT per DT-day."""
    return b_day - lam_assumed * billed_dt - eoff_dt.reindex(index=b_day.index, columns=b_day.columns)


# --------------------------------------------------------------------------
# L^_tech: power flow on the DT-meter loads
# --------------------------------------------------------------------------

def dt_meter_loads(frame: pd.DataFrame, dt_names) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DT-meter P (kW) and Q (kvar) per step, columns (DT, phase) in ``dt_names`` x a, b, c."""
    rows = frame[frame.load_id.isin(list(dt_names))]
    cols = pd.MultiIndex.from_product([list(dt_names), PHASES])
    p = rows.pivot_table(index="timestamp", columns=["load_id", "phase"], values="active_power_kw",
                         aggfunc="sum").reindex(columns=cols)
    q = rows.pivot_table(index="timestamp", columns=["load_id", "phase"], values="reactive_power_kvar",
                         aggfunc="sum").reindex(columns=cols)
    if p.isna().any().any() or q.isna().any().any():
        raise ValueError("DT-meter frame is missing a DT phase or step")
    return p, q


def ltech_run(frame: pd.DataFrame, start=None, end=None, progress_every: int = 0) -> pd.DataFrame:
    """L^_tech per 15-min step (kW): ``runpp_3ph`` with every DT's load = its
    own metered P and Q per phase.  Columns ``line_losses_kw, dt_losses_kw,
    ltech_kw`` (MV lines + DTs; the substation transformer is upstream of the
    head meter) and ``p_head_kw`` (the run's feeder-head power)."""
    net = build_network("B")
    names = list(net["dts"].name)
    p, q = dt_meter_loads(frame, names)
    start = p.index[0] if start is None else pd.Timestamp(start)
    end = p.index[-1] if end is None else pd.Timestamp(end)
    pv = p.loc[start:end].to_numpy().reshape(-1, len(names), 3)
    qv = q.loc[start:end].to_numpy().reshape(-1, len(names), 3)
    pos = {ts: k for k, ts in enumerate(p.loc[start:end].index)}

    def loads(i, ts):
        k = pos[pd.Timestamp(ts)]
        return pv[k], qv[k]

    t0 = time.perf_counter()
    _, state = run_timeseries(start, end, net=net, return_state=True, dt_loads=loads,
                              progress_every=progress_every)
    logger.info("L_tech power flow: %d steps in %.0f s", len(state), time.perf_counter() - t0)
    out = pd.DataFrame({"line_losses_kw": state.line_losses_mw.to_numpy() * 1e3,
                        "dt_losses_kw": state.dt_losses_mw.to_numpy() * 1e3,
                        "p_head_kw": state.p_head_mw.to_numpy() * 1e3},
                       index=pd.DatetimeIndex(state.timestamp, name="timestamp"))
    out["ltech_kw"] = out.line_losses_kw + out.dt_losses_kw
    return out


def ltech_daily(ltech: pd.DataFrame) -> pd.DataFrame:
    """Daily L^_tech energies (kWh) from :func:`ltech_run`."""
    return daily(ltech[["line_losses_kw", "dt_losses_kw", "ltech_kw"]] * STEP_H)


# --------------------------------------------------------------------------
# One run of the balance engine
# --------------------------------------------------------------------------

def balance_estimates(frame: pd.DataFrame, hh: pd.DataFrame, consumer_dt: pd.Series,
                      ltech: Optional[pd.DataFrame], lambdas_assumed=(0.0,),
                      dt_energy_15min: Optional[pd.DataFrame] = None) -> dict:
    """Every balance-engine output for one set of inputs.

    ``dt_energy_15min`` replaces the DT-meter energies of ``frame`` (the
    lambda_true scenario inputs, read like the normal DT-meter energies).
    ``ltech`` = :func:`ltech_run` output (None: feeder netting skipped).
    """
    head15, dt15 = meter_energy_15min(frame)
    if dt_energy_15min is not None:
        dt15 = dt_energy_15min[dt15.columns]
    head_day, dt_day = daily(head15), daily(dt15)
    hh = hh.reindex(columns=head_day.index)
    bal = balances(head_day, dt_day, hh, consumer_dt)
    eoff, eoff_info = offline_estimate(hh)
    eoff_day = eoff.sum(axis=0)
    eoff_dt = per_dt(eoff, consumer_dt, dt_day.columns).reindex(dt_day.index)
    out = {"head_day": head_day, "dt_day": dt_day, **bal, "eoff_day": eoff_day, "eoff_dt": eoff_dt,
           "eoff_info": eoff_info,
           "ntl_dt": {lam: netted_dt(bal["B"], bal["billed_dt"], eoff_dt, lam) for lam in lambdas_assumed}}
    if ltech is not None:
        ld = ltech_daily(ltech).reindex(head_day.index)
        out["ltech_day"] = ld
        out["ntl_feeder"] = netted_feeder(bal["C"], ld.ltech_kw, eoff_day)
        out["a_minus_ltech"] = bal["A"] - ld.ltech_kw
    return out
