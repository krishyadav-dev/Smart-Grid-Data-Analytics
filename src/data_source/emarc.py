"""
Household intraday load shapes from Prayas eMARC (brief v2 X.3.3 item 1).

Source: Prayas (Energy Group), *eMARC* (Electricity Monitoring and Reporting
for Consumers) dataset, Harvard Dataverse, doi:10.7910/DVN/YJ5SP1, licence
CC0 1.0.  Files in ``config.EMARC_DIR`` (git-ignored, never committed):
``eMARC load blocks.csv`` (15-min average kW per deployment, block 0 =
00:00-00:15) and ``eMAR daily consumption.csv`` (deployment type, household,
region per deployment).  ``Household-Deployment basic info.xlsx`` holds the
same deployment-household-region mapping and is not needed.

Library rules (brief):

- **Mainline** deployments only (whole-household meter; "appliance"
  deployments are left out); **complete days only** (all 96 blocks).
- **Data-validity rule (brief v2 X.3.3, user decision 2026-10-14):** a
  complete day with daily kWh = 0 carries no intraday shape (vacant home or
  dead meter) and is dropped before averaging, so the >= 15-day rule counts
  non-zero complete days.  Without it, near-zero household-months normalised
  to mean 1 gave spikes up to 77 x mean.  The days and household-months it
  drops are reported per region.
- Jan 2018 - Feb 2020; the COVID lockdown 2020-03-01 .. 2020-06-30 is
  excluded.  Days are grouped by **calendar month** (2018 and 2019 pooled).
- One profile per eMARC household x calendar month: the mean 96-block
  profile over its complete days, normalised to mean 1.  Household-months
  with fewer than 15 complete days (or zero mean) are dropped.
- A missing month is filled from the household's **nearest available
  month** (circular month distance; on a tie the earlier month).

Limitations (stated): eMARC covers households in Maharashtra (Pune,
Aurangabad) and Uttar Pradesh (Gonda, Kanpur rural), not Delhi; 116
mainline households; every shape is reused by many simulated households.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

STEPS = 96
WINDOW = (pd.Timestamp("2018-01-01"), pd.Timestamp("2020-02-29"))
LOCKDOWN = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))
MIN_DAYS = 15
BLOCKS_FILE = "eMARC load blocks.csv"
DAILY_FILE = "eMAR daily consumption.csv"


@dataclass
class ShapeLibrary:
    households: np.ndarray        # eMARC household ids (n_e,)
    region: np.ndarray            # region per household (n_e,)
    profiles: np.ndarray          # (n_profiles, 96), each mean 1
    profile_month: np.ndarray     # (n_profiles,) calendar month 1..12 of the profile
    profile_household: np.ndarray  # (n_profiles,) household row
    profile_days: np.ndarray      # (n_profiles,) complete days averaged
    month_profile: np.ndarray     # (n_e, 12) profile id per household and month (nearest filled)
    has_month: np.ndarray         # (n_e, 12) bool: own data for that month
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    lockdown_days_used: int
    daily_kwh_by_region: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def n_households(self) -> int:
        return len(self.households)


def _nearest_fill(has: np.ndarray) -> np.ndarray:
    """(n, 12) bool -> (n, 12) month index (0..11) of the nearest available month."""
    out = np.full(has.shape, -1, int)
    for m in range(12):
        # candidates by circular distance; on a tie the earlier (preceding) month first
        def key(k, m=m):
            ahead = (k - m) % 12
            return min(ahead, 12 - ahead), 0 < ahead <= 6
        for k in sorted(range(12), key=key):
            sel = (out[:, m] < 0) & has[:, k]
            out[sel, m] = k
    return out


@functools.lru_cache(maxsize=2)
def load_library(emarc_dir: Optional[str] = None) -> ShapeLibrary:
    """Build the eMARC shape library (cached per directory)."""
    import pyarrow.csv as pc

    d = Path(emarc_dir) if emarc_dir else config.EMARC_DIR
    meta = pd.read_csv(d / DAILY_FILE, usecols=["Region", "household_id", "Deployment type",
                                                "deployment_id"]).drop_duplicates("deployment_id")
    main = meta[meta["Deployment type"].str.lower() == "mainline"]
    dep_hh = dict(zip(main.deployment_id, main.household_id))
    hh_region = dict(zip(main.household_id, main.Region))
    tb = pc.read_csv(d / BLOCKS_FILE).to_pandas()
    tb = tb[tb.deployment_id.isin(dep_hh)]
    tb["date"] = pd.to_datetime(tb["date"], format="%m/%d/%Y")
    tb = tb[(tb.date >= WINDOW[0]) & (tb.date <= WINDOW[1])]
    tb = tb[~((tb.date >= LOCKDOWN[0]) & (tb.date <= LOCKDOWN[1]))]
    n_blocks = tb.groupby(["deployment_id", "date"]).block.transform("nunique")
    tb = tb[n_blocks == STEPS].sort_values(["deployment_id", "date", "block"])
    days = tb[["deployment_id", "date"]].iloc[::STEPS].reset_index(drop=True)
    kw = tb["Load (kW)"].to_numpy(float).reshape(-1, STEPS)
    days["household"] = days.deployment_id.map(dep_hh)
    days["month"] = days.date.dt.month
    days["region"] = days.household.map(hh_region)
    days["kwh"] = kw.sum(axis=1) * 0.25
    lock = int(((days.date >= LOCKDOWN[0]) & (days.date <= LOCKDOWN[1])).sum())
    # Data-validity rule: zero-kWh days carry no intraday shape (vacant home / dead meter)
    zero = (days.kwh <= 0).to_numpy()
    n_all = days.groupby(["household", "month"]).size()
    n_nonzero = days[~zero].groupby(["household", "month"]).size().reindex(n_all.index, fill_value=0)
    lost = n_all[(n_all >= MIN_DAYS) & (n_nonzero < MIN_DAYS)]          # months dropped by the rule
    lost_region = pd.Series([hh_region[h] for h, _ in lost.index], dtype=object)
    dropped = {
        "household_days_by_region": {str(k): int(v) for k, v in
                                     days[zero].groupby("region").size().items()},
        "household_days_total": int(zero.sum()),
        "household_months_by_region": {str(k): int(v) for k, v in lost_region.value_counts().items()},
        "household_months_total": int(len(lost)),
    }
    days = days[~zero].reset_index(drop=True)
    kw = kw[~zero]

    hh_ids = np.array(sorted(days.household.unique()))
    row = {h: i for i, h in enumerate(hh_ids)}
    profiles, p_month, p_hh, p_days = [], [], [], []
    for (h, m), idx in days.groupby(["household", "month"]).indices.items():
        if len(idx) < MIN_DAYS:
            continue
        prof = kw[idx].mean(axis=0)
        if prof.mean() <= 0:
            continue
        profiles.append(prof / prof.mean())
        p_month.append(m)
        p_hh.append(row[h])
        p_days.append(len(idx))
    profiles = np.array(profiles)
    p_month, p_hh, p_days = np.array(p_month), np.array(p_hh), np.array(p_days)
    has = np.zeros((len(hh_ids), 12), bool)
    pid = np.full((len(hh_ids), 12), -1, int)
    has[p_hh, p_month - 1] = True
    pid[p_hh, p_month - 1] = np.arange(len(profiles))
    keep = has.any(axis=1)                      # households with at least one valid month
    near = _nearest_fill(has[keep])
    month_profile = np.take_along_axis(pid[keep], near, axis=1)
    remap = np.full(len(hh_ids), -1, int)
    remap[keep] = np.arange(int(keep.sum()))
    p_hh = remap[p_hh]
    by_region = (days.groupby("region").kwh.median().round(4).to_dict())
    lib = ShapeLibrary(
        households=hh_ids[keep], region=np.array([hh_region[h] for h in hh_ids[keep]]),
        profiles=profiles, profile_month=p_month, profile_household=p_hh, profile_days=p_days,
        month_profile=month_profile, has_month=has[keep],
        first_date=days.date.min(), last_date=days.date.max(), lockdown_days_used=lock,
        daily_kwh_by_region=by_region,
    )
    lib.info = {
        "source": "Prayas (Energy Group) eMARC, Harvard Dataverse doi:10.7910/DVN/YJ5SP1, CC0 1.0",
        "mainline_households": int(len(dep_hh)), "households_in_library": lib.n_households,
        "complete_days": int(len(days)), "profiles": int(len(profiles)),
        "household_months_filled_from_nearest": int((~lib.has_month).sum()),
        "first_date": str(lib.first_date.date()), "last_date": str(lib.last_date.date()),
        "lockdown_days_used": lock,
        "zero_kwh_rule_dropped": dropped,
        "median_daily_kwh_by_region": by_region,
        "median_daily_kwh_all": float(days.kwh.median()),
        "profile_peak_to_mean": {"median": float(np.median(profiles.max(axis=1))),
                                 "p95": float(np.percentile(profiles.max(axis=1), 95)),
                                 "p99": float(np.percentile(profiles.max(axis=1), 99)),
                                 "max": float(profiles.max())},
    }
    logger.info("eMARC library: %s", lib.info)
    return lib


def assign(n: int, lib: ShapeLibrary, rng: np.random.Generator) -> np.ndarray:
    """One eMARC household (library row) per consumer, uniformly at random."""
    return rng.integers(0, lib.n_households, size=n)


def profile_ids(lib: ShapeLibrary, emarc_row: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    """(n, n_days) profile id of each consumer on each simulated day (by calendar month)."""
    months = pd.DatetimeIndex(days).month.to_numpy() - 1
    return lib.month_profile[np.asarray(emarc_row)][:, months]
