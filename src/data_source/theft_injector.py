"""
Electricity-theft attacks h1-h6 of Jokar, Arianpoo & Leung (Part 2C.4).

Source: P. Jokar, N. Arianpoo, V.C.M. Leung, "Electricity Theft Detection in
AMI Using Customers' Consumption Patterns", IEEE Trans. Smart Grid 7(1), 2016,
early-access version in References, §VI items 1-6, PDF p. 7.  The paper works
on a customer's daily vector of hourly readings x = [x_1 .. x_24]:

    h1(x_t) = alpha x_t,           alpha = random(0.1, 0.8)
    h2(x_t) = beta_t x_t,          beta_t = 0 for t_s <= t < t_s + duration, else 1
                                   t_s = random(0, 23 - minOFFtime),
                                   duration = random(minOFFtime, 24), minOFFtime = 4 h
    h3(x_t) = gamma_t x_t,         gamma_t = random(0.1, 0.8)
    h4(x_t) = gamma_t mean(x),     gamma_t = random(0.1, 0.8)
    h5(x_t) = mean(x)
    h6(x_t) = x_(24 - t)

At 15-minute resolution (96 samples a day) the parameters stay hourly:
gamma_t and beta_t are drawn per hour and held for that hour's four samples;
mean(x) is the daily mean; h6 reverses the day's samples (the paper's index
24 - t is off by one at t = 24, so it is implemented as a pure reversal).
t_s and duration are whole hours (the paper's vectors are hourly) and the h2
window is clipped at the end of the day.  Every parameter is redrawn per day,
as the paper generates one attack sample per daily vector.

h5 and h6 keep the daily energy (load-shifting attacks), so their severity is
0; h1-h4 report at most 0.8 of the true daily energy.

All functions take ``x`` with shape (days, samples_per_day) (a 1-D vector is
one day) and a numpy Generator, and return the reported array of the same
shape plus the drawn parameters.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

ATTACKS = ("h1", "h2", "h3", "h4", "h5", "h6")
# Brief v2 X.3.1 item 4 / X.3.2 item 20: h5 and h6 leave the daily energy
# unchanged ("load-shifting"); h1-h4 reduce it ("energy-reducing").  Every
# NTL / lambda result is reported with all attacks, without h6, and without h5 + h6.
LOAD_SHIFTING = ("h5", "h6")


def attack_class(kind: str) -> str:
    """"load-shifting" for h5/h6, "energy-reducing" for h1-h4."""
    if kind not in ATTACKS:
        raise ValueError(f"unknown attack {kind!r}")
    return "load-shifting" if kind in LOAD_SHIFTING else "energy-reducing"
RANGE_LO, RANGE_HI = 0.1, 0.8          # alpha and gamma_t, Jokar §VI items 1, 3, 4
MIN_OFF_TIME_H = 4                     # Jokar §VI item 2
HOURS = 24


def _as_days(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] % HOURS:
        raise ValueError(f"samples per day ({x.shape[1]}) is not a multiple of 24")
    return x


def _per_sample(hourly: np.ndarray, samples_per_day: int) -> np.ndarray:
    """Hold each hourly value for that hour's samples."""
    return np.repeat(hourly, samples_per_day // HOURS, axis=1)


def h1(x, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    x = _as_days(x)
    alpha = rng.uniform(RANGE_LO, RANGE_HI, size=(len(x), 1))
    return alpha * x, {"alpha": alpha[:, 0]}


def h2(x, rng: np.random.Generator,
       min_off_h: int = MIN_OFF_TIME_H) -> Tuple[np.ndarray, dict]:
    x = _as_days(x)
    n = len(x)
    start = rng.integers(0, HOURS - 1 - min_off_h, size=n, endpoint=True)
    duration = rng.integers(min_off_h, HOURS, size=n, endpoint=True)
    end = np.minimum(start + duration, HOURS)          # clipped at 24 h (stated)
    hour = np.arange(HOURS)[None, :]
    beta = ~((hour >= start[:, None]) & (hour < end[:, None]))
    return x * _per_sample(beta.astype(float), x.shape[1]), {
        "t_start_h": start, "duration_h": duration, "t_end_h": end}


def h3(x, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    x = _as_days(x)
    gamma = rng.uniform(RANGE_LO, RANGE_HI, size=(len(x), HOURS))
    return x * _per_sample(gamma, x.shape[1]), {"gamma": gamma}


def h4(x, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    x = _as_days(x)
    gamma = rng.uniform(RANGE_LO, RANGE_HI, size=(len(x), HOURS))
    return x.mean(axis=1, keepdims=True) * _per_sample(gamma, x.shape[1]), {"gamma": gamma}


def h5(x, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, dict]:
    x = _as_days(x)
    return np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1), {}


def h6(x, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, dict]:
    x = _as_days(x)
    return x[:, ::-1].copy(), {}


ATTACK_FUNCS: Dict[str, Callable] = {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "h6": h6}


def apply_attack(kind: str, x, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Reported daily vectors under attack ``kind`` (one draw per day)."""
    try:
        f = ATTACK_FUNCS[kind]
    except KeyError:
        raise ValueError(f"attack must be one of {ATTACKS}, got {kind!r}") from None
    return f(x, rng)


def severity(true_energy, reported_energy) -> float:
    """1 - sum(E_reported) / sum(E_true) (Part 2C.4); 0 if no true energy."""
    t = float(np.nansum(true_energy))
    return 0.0 if t <= 0 else 1.0 - float(np.nansum(reported_energy)) / t
