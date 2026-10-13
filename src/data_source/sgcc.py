"""
SGCC dataset (Zheng et al. 2018) as a consumer x day matrix: preprocessing,
calendar mapping and the disjoint consumer pools of Part 2C.4.

Source: Zheng, Yang, Niu, Dai, Zhou, "Wide and Deep Convolutional Neural
Networks for Electricity-Theft Detection to Secure Smart Grids", IEEE TII 2018
(author manuscript ``ALL_TII-17-2030.pdf``):

- §V-A, PDF p. 6: 42 372 customers, 1 035 days (2014-01-01 .. 2016-10-31),
  3 615 thieves (FLAG = 1).
- §IV-A Eq. (1), PDF p. 3: a missing day with both neighbours present is the
  mean of its neighbours; a missing day next to another missing day is 0.
- §IV-A Eq. (2), PDF p. 4: x > avg(x) + 2 std(x) is set to avg + 2 std
  (positive side only), x = the consumer's day-by-day vector.
- §IV-A Eq. (3), PDF p. 4: max-min scaling.
- §V-C, PDF p. 7: training ratios 50/60/70/80 % (train/test, no validation
  set).  The detector split uses 80 : 20.

Detector pools get Eq. (1)-(3) exactly as published.  The feeder pool gets
the project rule instead (user decision 2026-10-12): isolated single-day gaps
interpolated as in Eq. (1), Eq. (2) applied, genuine zeros kept, longer gap
runs left missing (never zero-filled); Eq. (3) is not applied because the
feeder needs kWh.

Calendar (user decision 2026-10-12): simulated 2024 day d maps to SGCC day
2015-01-05 + (d - 2024-01-01).  Both anchors are Mondays, so weekdays line
up; holidays and seasons shift by 4 days (stated limitation).  The last 2024
day maps to 2016-01-05, inside the SGCC range.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

SIM_ANCHOR = pd.Timestamp("2024-01-01")       # Monday
SGCC_ANCHOR = pd.Timestamp("2015-01-05")      # Monday
DETECTOR_TEST_FRACTION = 0.20                 # Zheng §V-C, highest training ratio 80 %


@dataclass
class SGCCMatrix:
    """Daily kWh per consumer.  ``kwh[i, j]`` = consumer i on ``dates[j]``; NaN = missing."""
    consumer_id: np.ndarray        # str, shape (n,)
    flag: np.ndarray               # int8, shape (n,)
    dates: pd.DatetimeIndex        # shape (d,), sorted
    kwh: np.ndarray                # float32, shape (n, d)

    @property
    def theft_prevalence(self) -> float:
        """p = share of FLAG = 1 consumers in the whole file (Part 2C.4)."""
        return float(self.flag.mean())

    def columns_for(self, sim_days: pd.DatetimeIndex) -> np.ndarray:
        """Column index of the SGCC day each simulated day maps to."""
        mapped = map_to_sgcc(sim_days)
        pos = self.dates.get_indexer(mapped)
        if (pos < 0).any():
            bad = mapped[pos < 0][0]
            raise KeyError(f"simulated day maps to {bad.date()}, outside the SGCC range")
        return pos


def map_to_sgcc(sim_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """SGCC day = 2015-01-05 + (d - 2024-01-01) (weekday-preserving offset)."""
    days = pd.DatetimeIndex(sim_days).normalize()
    return SGCC_ANCHOR + (days - SIM_ANCHOR)


def load_matrix(path: Optional[Path | str] = None) -> SGCCMatrix:
    """Read the SGCC CSV straight into a float32 consumer x day matrix.

    No long-format melt (the full file would need ~8 GB).  Date columns are
    sorted chronologically; non-numeric cells become NaN.
    """
    if path is None:
        path = config.SGCC_FALLBACK_PATH
    path = Path(path)
    df = pd.read_csv(path, dtype={"CONS_NO": str, "FLAG": "int8"}, low_memory=False)
    meta = {"CONS_NO", "FLAG"}
    date_cols = [c for c in df.columns if c not in meta]
    dates = pd.to_datetime(pd.Index(date_cols), format="%m/%d/%Y")
    order = np.argsort(dates.values, kind="stable")
    vals = df[date_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    m = SGCCMatrix(
        consumer_id=df["CONS_NO"].to_numpy(str),
        flag=df["FLAG"].to_numpy(np.int8),
        dates=pd.DatetimeIndex(dates[order]),
        kwh=np.ascontiguousarray(vals[:, order]),
    )
    logger.info("SGCC matrix %s: %d consumers x %d days (%s .. %s), %d FLAG=1, p = %.4f",
                path.name, len(m.flag), len(m.dates), m.dates[0].date(),
                m.dates[-1].date(), int(m.flag.sum()), m.theft_prevalence)
    return m


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def _neighbours(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left and right neighbour of every day (NaN beyond the record)."""
    left = np.full_like(x, np.nan)
    right = np.full_like(x, np.nan)
    left[:, 1:] = x[:, :-1]
    right[:, :-1] = x[:, 1:]
    return left, right


def isolated_gaps(x: np.ndarray) -> np.ndarray:
    """Mask of missing days whose two neighbours are both present."""
    left, right = _neighbours(x)
    return np.isnan(x) & ~np.isnan(left) & ~np.isnan(right)


def zheng_eq1(x: np.ndarray) -> np.ndarray:
    """Zheng Eq. (1): isolated gap -> neighbour mean; other gaps -> 0."""
    left, right = _neighbours(x)
    nan = np.isnan(x)
    iso = nan & ~np.isnan(left) & ~np.isnan(right)
    out = x.copy()
    out[iso] = 0.5 * (left[iso] + right[iso])
    out[nan & ~iso] = 0.0
    return out


def zheng_eq2(x: np.ndarray) -> np.ndarray:
    """Zheng Eq. (2): clip each consumer at avg + 2 std of its own vector.

    avg and std ignore missing days (only relevant to the feeder rule, which
    keeps long gaps missing); population std (ddof = 0).
    """
    with np.errstate(invalid="ignore"):
        thr = np.nanmean(x, axis=1, keepdims=True) + 2.0 * np.nanstd(x, axis=1, keepdims=True)
        return np.where(x > thr, thr, x).astype(x.dtype)


def zheng_eq3(x: np.ndarray) -> np.ndarray:
    """Zheng Eq. (3): max-min scaling per consumer (constant vector -> 0)."""
    lo = x.min(axis=1, keepdims=True)
    span = x.max(axis=1, keepdims=True) - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(span > 0, (x - lo) / span, 0.0).astype(x.dtype)


def zheng_preprocess(x: np.ndarray) -> np.ndarray:
    """Detector pools: Zheng Eq. (1), (2), (3) in the published order."""
    return zheng_eq3(zheng_eq2(zheng_eq1(x)))


def zheng_eq2_loo(x: np.ndarray) -> np.ndarray:
    """Leave-one-out Eq. (2): clip each day at avg + 2 std of the consumer's
    **other** days (missing days ignored, ddof = 0).

    Stated deviation from Zheng for the feeder pool only (brief v2 X.3.3,
    user decision 2026-10-14): in the published form a single extreme value
    inflates its own std and survives (masking); e.g. a 13,442 kWh day was
    capped only to 977 kWh.  Still one pass with Zheng's 2 sigma.
    """
    v = x.astype(np.float64)
    ok = np.isfinite(v)
    z = np.where(ok, v, 0.0)
    n = ok.sum(axis=1, keepdims=True)
    s1 = z.sum(axis=1, keepdims=True)
    s2 = (z * z).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (s1 - z) / (n - 1)
        var = np.maximum((s2 - z * z) / (n - 1) - m * m, 0.0)
        thr = m + 2.0 * np.sqrt(var)
    clip = ok & (n > 1) & (v > thr)
    return np.where(clip, thr, v).astype(x.dtype)


def feeder_preprocess(x: np.ndarray) -> np.ndarray:
    """Feeder pool: interpolate isolated gaps, then leave-one-out Eq. (2)
    (:func:`zheng_eq2_loo`); longer gaps stay NaN."""
    left, right = _neighbours(x)
    iso = np.isnan(x) & ~np.isnan(left) & ~np.isnan(right)
    out = x.copy()
    out[iso] = 0.5 * (left[iso] + right[iso])
    return zheng_eq2_loo(out)


# --------------------------------------------------------------------------
# Eligibility and pools
# --------------------------------------------------------------------------

COMPLETENESS_MIN = 0.95          # user decision 2026-10-12
TOP_EXCLUDE_FRACTION = 0.05      # top 5 % by median daily kWh: likely non-residential
MOSTLY_ZERO_FRACTION = 0.5       # "mostly zero" = more than half of the window days are 0


def quantiles(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    q = np.percentile(v, [25, 50, 75, 95, 99]) if v.size else [np.nan] * 5
    return {"n": int(v.size), "median": float(q[1]), "p25": float(q[0]),
            "p75": float(q[2]), "iqr": float(q[2] - q[0]), "p95": float(q[3]),
            "p99": float(q[4])}


@dataclass
class Eligibility:
    window_cols: np.ndarray        # SGCC columns of the simulated days
    completeness: np.ndarray       # share of present (raw) days in the window
    median_kwh: np.ndarray         # per-consumer median daily kWh in the window (raw)
    top_threshold_kwh: float       # 95th percentile of median_kwh among FLAG 0, complete
    eligible: np.ndarray           # bool mask
    report: dict


def _zero_share(w: np.ndarray) -> np.ndarray:
    """Share of present days that are exactly 0 kWh, per consumer."""
    return np.where(np.isfinite(w), w == 0, False).sum(axis=1) / np.maximum(
        np.isfinite(w).sum(axis=1), 1)


def mostly_zero(m: SGCCMatrix, el: "Eligibility") -> np.ndarray:
    """Bool per consumer: more than half of the present window days are 0 kWh.

    Not an exclusion criterion (reported only)."""
    return _zero_share(m.kwh[:, el.window_cols]) > MOSTLY_ZERO_FRACTION


def eligibility(m: SGCCMatrix, sim_days: pd.DatetimeIndex) -> Eligibility:
    """Feeder-pool filter, in the user's order: FLAG 0, >= 95 % complete in the
    mapped window, not in the top 5 % by median daily kWh."""
    cols = m.columns_for(sim_days)
    w = m.kwh[:, cols]
    complete = np.isfinite(w).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN
        med = np.nanmedian(w, axis=1)
    flag0 = m.flag == 0
    ok = flag0 & (complete >= COMPLETENESS_MIN)
    thr = float(np.percentile(med[ok], 100 * (1 - TOP_EXCLUDE_FRACTION)))
    eligible = ok & (med <= thr)
    zero_share = _zero_share(w)
    report = {
        "consumers": int(len(m.flag)),
        "flag0": int(flag0.sum()),
        "flag0_complete": int(ok.sum()),
        "excluded_incomplete": int((flag0 & ~(complete >= COMPLETENESS_MIN)).sum()),
        "excluded_top5": int((ok & ~(med <= thr)).sum()),
        "eligible": int(eligible.sum()),
        "top5_threshold_kwh_per_day": thr,
        "median_kwh_all": quantiles(med),
        "median_kwh_flag0_complete": quantiles(med[ok]),
        "median_kwh_eligible": quantiles(med[eligible]),
        "eligible_mostly_zero": int((eligible & (zero_share > MOSTLY_ZERO_FRACTION)).sum()),
    }
    logger.info("SGCC eligibility: %s", report)
    return Eligibility(cols, complete, med, thr, eligible, report)


def mean_household_kw(m: SGCCMatrix, el: Eligibility, mask: Optional[np.ndarray] = None) -> float:
    """h-bar: mean daily kWh / 24 over the window (feeder rule applied), all eligible."""
    mask = el.eligible if mask is None else mask
    x = feeder_preprocess(m.kwh[mask])[:, el.window_cols]
    return float(np.nanmean(np.nanmean(x, axis=1)) / 24.0)


@dataclass
class Pools:
    feeder: np.ndarray             # row indices into the matrix
    detector_train: np.ndarray
    detector_test: np.ndarray

    def assert_valid(self, m: SGCCMatrix) -> None:
        a, b, c = (set(map(int, p)) for p in (self.feeder, self.detector_train, self.detector_test))
        assert not (a & b) and not (a & c) and not (b & c), "pools overlap"
        assert (m.flag[self.feeder] == 0).all(), "feeder pool contains FLAG = 1"
        assert len(a) + len(b) + len(c) == len(m.flag), "pools do not cover the file"


def build_pools(m: SGCCMatrix, el: Eligibility, n_feeder: int,
                rng_feeder: np.random.Generator, rng_split: np.random.Generator,
                test_fraction: float = DETECTOR_TEST_FRACTION) -> Pools:
    """Feeder pool sampled from the eligible set; everything else split by
    consumer, stratified by FLAG, into detector-train / detector-test."""
    eligible = np.flatnonzero(el.eligible)
    if n_feeder > len(eligible):
        raise ValueError(f"need {n_feeder} feeder households, only {len(eligible)} eligible")
    feeder = np.sort(rng_feeder.choice(eligible, size=n_feeder, replace=False))
    rest = np.setdiff1d(np.arange(len(m.flag)), feeder)
    train, test = [], []
    for f in (0, 1):
        idx = rest[m.flag[rest] == f]
        idx = rng_split.permutation(idx)
        n_test = int(round(test_fraction * len(idx)))
        test.append(idx[:n_test])
        train.append(idx[n_test:])
    pools = Pools(feeder, np.sort(np.concatenate(train)), np.sort(np.concatenate(test)))
    pools.assert_valid(m)
    logger.info("SGCC pools: feeder %d, detector-train %d (%d FLAG=1), detector-test %d (%d FLAG=1)",
                len(pools.feeder), len(pools.detector_train),
                int(m.flag[pools.detector_train].sum()), len(pools.detector_test),
                int(m.flag[pools.detector_test].sum()))
    return pools
