"""
Synthetic electricity-theft attack functions.

Implements the attack types described in:
    Jokar, P., Arianpoo, N., Leung, V.C.M., "Electricity Theft Detection in
    AMI Using Customers' Consumption Patterns," IEEE Trans. Smart Grid, vol. 7,
    no. 1, pp. 216–226, 2016.

Each function takes a true consumption Series and returns a tuple of:
    (manipulated_series, label_string, severity_value)

This module is **independent** of ``dataset_loader`` — it operates on any
``pandas.Series`` (real SGCC data, raw pandapower output, or a hand-crafted
test vector).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def constant_partial_reporting(
    series: pd.Series,
    fraction: float = 0.5,
) -> Tuple[pd.Series, str, float]:
    """Report a fixed fraction of true consumption at every timestep.

    Parameters
    ----------
    series : pd.Series
        True consumption values (kWh or similar).
    fraction : float, optional
        Fraction of true consumption reported (0 < fraction < 1).
        Default is 0.5 (50 %).

    Returns
    -------
    manipulated : pd.Series
        ``fraction * series``.
    label : str
        ``"constant_partial_reporting"``.
    severity : float
        ``1 - fraction`` (higher ⇒ more energy stolen).
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    manipulated = series * fraction
    severity = 1.0 - fraction
    return manipulated, "constant_partial_reporting", severity


def random_reduction(
    series: pd.Series,
    min_frac: float = 0.3,
    max_frac: float = 0.9,
    seed: int | None = None,
) -> Tuple[pd.Series, str, float]:
    """Scale each timestep by a uniformly-random factor in [min_frac, max_frac].

    Parameters
    ----------
    series : pd.Series
        True consumption values.
    min_frac, max_frac : float
        Lower/upper bounds of the per-timestep scaling factor.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    manipulated : pd.Series
        Element-wise ``series * uniform(min_frac, max_frac)``.
    label : str
        ``"random_reduction"``.
    severity : float
        Mean reduction factor: ``1 - mean(scaling_factors)``.
    """
    if not (0 < min_frac <= max_frac < 1):
        raise ValueError(
            f"Need 0 < min_frac <= max_frac < 1, got ({min_frac}, {max_frac})"
        )
    rng = np.random.default_rng(seed)
    factors = rng.uniform(min_frac, max_frac, size=len(series))
    manipulated = series * factors
    severity = float(1.0 - np.mean(factors))
    return manipulated, "random_reduction", severity


def time_of_day_dependent(
    series: pd.Series,
    peak_fraction: float = 0.3,
    offpeak_fraction: float = 0.9,
    peak_hours: Tuple[int, int] = (9, 21),
) -> Tuple[pd.Series, str, float]:
    """Heavy reduction during peak demand hours, mild during off-peak.

    The series **must** have a ``DatetimeIndex`` so that ``.hour`` is available.

    Parameters
    ----------
    series : pd.Series
        True consumption with a DatetimeIndex.
    peak_fraction : float
        Fraction reported during peak hours.
    offpeak_fraction : float
        Fraction reported during off-peak hours.
    peak_hours : (int, int)
        Start and end hour (24-h clock) defining the peak window.
        Default is 09:00–21:00.

    Returns
    -------
    manipulated : pd.Series
    label : str
    severity : float
        Weighted-average reduction across all timesteps.
    """
    if not hasattr(series.index, "hour"):
        raise TypeError(
            "Series must have a DatetimeIndex for time-of-day attacks"
        )
    start_h, end_h = peak_hours
    is_peak = series.index.hour.to_series(index=series.index).between(
        start_h, end_h - 1
    )
    factors = pd.Series(
        np.where(is_peak, peak_fraction, offpeak_fraction),
        index=series.index,
    )
    manipulated = series * factors
    severity = float(1.0 - factors.mean())
    return manipulated, "time_of_day_dependent", severity


def reverse_reporting(
    series: pd.Series,
) -> Tuple[pd.Series, str, float]:
    """Meter bypassed or running backward — reports zero at every timestep.

    Returns
    -------
    manipulated : pd.Series
        All zeros, same shape/index as input.
    label : str
        ``"reverse_reporting"``.
    severity : float
        ``1.0`` (total theft).
    """
    manipulated = pd.Series(0.0, index=series.index, name=series.name)
    return manipulated, "reverse_reporting", 1.0


def zero_reporting(
    series: pd.Series,
    zero_fraction: float = 0.3,
    seed: int | None = None,
) -> Tuple[pd.Series, str, float]:
    """Randomly zero-out a fraction of readings.

    Parameters
    ----------
    series : pd.Series
        True consumption values.
    zero_fraction : float
        Proportion of timesteps set to zero.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    manipulated : pd.Series
    label : str
        ``"zero_reporting"``.
    severity : float
        ``zero_fraction``.
    """
    if not 0 < zero_fraction < 1:
        raise ValueError(
            f"zero_fraction must be in (0, 1), got {zero_fraction}"
        )
    rng = np.random.default_rng(seed)
    mask = rng.random(len(series)) < zero_fraction
    manipulated = series.copy()
    manipulated[mask] = 0.0
    severity = float(zero_fraction)
    return manipulated, "zero_reporting", severity
