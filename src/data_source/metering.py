"""
Meters and theft on the household set (Parts 2C.3 and 2C.4).

Meters
------
| Meter          | Location                      | Measures                   | Class |
|----------------|-------------------------------|----------------------------|-------|
| feeder head    | 11 kV bus 650 (Sub_XFMR LV)    | true, from the power flow  | 0.5S  |
| DT meter       | each DT's LV terminals        | true, from the power flow  | 0.5S  |
| consumer meter | each household (AMI export)   | reported (theft applied)   | 1.0   |

Classes: CEA (Installation and Operation of Meters) Regulations 2006 as
amended, 2019 amendment, Schedule Part I para (2), PDF pp. 18-19 (household
LT direct-connected 1.0; distribution energy-accounting meters and the 11 kV
feeder meter 0.5S).  Error model (stated simplification): one seeded gain
error per meter, uniform within +/- the class index, constant over the year;
it scales P, Q, I and energy, not voltage.  The exact error limits are in
IS 13779 / IS 16444 / IS 14697 (``verify``, not in References).

The ``bus_virtual`` meters m01-m09 are the sum of their bus's DT meters
(with the DT meters' errors) and have no error of their own.

Theft
-----
Thieves = round(p x households), p = the FLAG = 1 share of the SGCC file,
drawn only from households that are **not** SGCC "mostly zero" consumers
(brief v2 X.3.1 item 3).  Attack type uniform over Jokar h1-h6, onset day
uniform over the simulated days, the attack persists after onset (stated);
everything seeded.  Attacks act on the household's own 15-min profile
(SGCC daily kWh / 24 x its eMARC shape for the day).  h5 and h6 are
"load-shifting" (daily energy unchanged; X.3.1 item 4, X.3.2 item 20).

Three quantities per household and day:
- ``true``      the household's real consumption (drives the power flow);
- ``reported``  what the (tampered) meter registers: true before onset,
                the Jokar attack after it; NaN while the meter is offline;
- ``metered``   reported x the meter's gain error (what the head-end sees).
Severity and "reported <= true" are about ``reported`` (theft only); the
class error is separate, so an honest meter with a positive gain error can
read above the true value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .households import HouseholdSet
from .theft_injector import ATTACKS, apply_attack, attack_class, severity

logger = logging.getLogger(__name__)

CLASS_HOUSEHOLD = 1.0       # LT direct-connected consumer meter
CLASS_DT = 0.5              # 0.5S, distribution energy accounting
CLASS_FEEDER_HEAD = 0.5     # 0.5S, 11 kV (650 V - 33 kV)


def meter_gains(n: int, class_index: float, rng: np.random.Generator) -> np.ndarray:
    """Per-meter gain 1 + e, e ~ U(-class, +class) %."""
    return 1.0 + rng.uniform(-class_index, class_index, size=n) / 100.0


@dataclass
class Theft:
    thief: np.ndarray            # bool (n_hh,)
    attack: np.ndarray           # "h1".."h6" or "" (n_hh,)
    onset_day: np.ndarray        # int, -1 for honest households
    thief_rows: np.ndarray       # household rows of the thieves
    reported_kw: np.ndarray      # float32 (n_thief, n_days, steps): thieves' reported kW
    p: float


def assign_theft(hh: HouseholdSet, p: float, rng_pick: np.random.Generator,
                 rng_attack: np.random.Generator) -> Theft:
    n = len(hh.hh_id)
    n_days = len(hh.days)
    n_thief = int(round(p * n))
    candidates = np.flatnonzero(~np.asarray(hh.mostly_zero, bool))
    if n_thief > len(candidates):
        raise ValueError(f"{n_thief} thieves needed, {len(candidates)} households not mostly-zero")
    rows = np.sort(rng_pick.choice(candidates, size=n_thief, replace=False))
    kinds = rng_pick.choice(np.array(ATTACKS), size=n_thief)
    onset = rng_pick.integers(0, n_days, size=n_thief)
    thief = np.zeros(n, bool)
    thief[rows] = True
    attack = np.full(n, "", dtype=object)
    attack[rows] = kinds
    onset_all = np.full(n, -1)
    onset_all[rows] = onset
    rep = np.empty((n_thief, n_days, hh.steps_per_day), np.float32)
    for j, (i, kind, a) in enumerate(zip(rows, kinds, onset)):
        x = (hh.true_kwh[i].astype(float) / 24.0)[:, None] * hh.profiles[hh.profile_id[i]]
        r = x.copy()
        r[a:] = apply_attack(kind, x[a:], rng_attack)[0]
        r[hh.offline[i]] = np.nan
        rep[j] = r
    logger.info("Theft: p = %.4f, %d thieves of %d households; attacks %s",
                p, n_thief, n, dict(zip(*np.unique(kinds, return_counts=True))))
    return Theft(thief, attack.astype(str), onset_all, rows, rep, p)


def reported_daily_kwh(hh: HouseholdSet, theft: Theft) -> np.ndarray:
    """Reported daily kWh (n_hh, n_days); NaN while offline."""
    rep = np.where(hh.offline, np.nan, hh.true_kwh.astype(float))
    step_h = 24.0 / hh.steps_per_day
    # offline days are NaN in reported_kw, so their sums stay NaN
    rep[theft.thief_rows] = theft.reported_kw.astype(float).sum(axis=2) * step_h
    return rep


def reported_kw_day(hh: HouseholdSet, theft: Theft, day: int) -> np.ndarray:
    """Reported 15-min kW of every household on one day (NaN offline)."""
    out = hh.true_kw_day(day)
    out[hh.offline[:, day]] = np.nan
    out[theft.thief_rows] = theft.reported_kw[:, day, :]
    return out


def ground_truth(hh: HouseholdSet, theft: Theft, gains: np.ndarray) -> pd.DataFrame:
    """One row per household: household, DT, bus, thief flag, attack type,
    onset, severity, true / reported / metered kWh over the year."""
    rep = reported_daily_kwh(hh, theft)
    true = hh.true_kwh.astype(float)
    sev = np.zeros(len(hh.hh_id))
    for i in theft.thief_rows:
        a = theft.onset_day[i]
        online = ~hh.offline[i, a:]
        sev[i] = severity(true[i, a:][online], rep[i, a:][online])
    onset = np.where(theft.onset_day >= 0,
                     hh.days[np.maximum(theft.onset_day, 0)].strftime("%Y-%m-%d"), "")
    return pd.DataFrame({
        "household": hh.hh_id, "sgcc_consumer": hh.consumer_id, "dt": hh.dt,
        "bus": hh.bus, "lv_phase": np.array(list("abc"))[hh.phase],
        "thief": theft.thief, "attack": theft.attack,
        "attack_class": np.array([attack_class(a) if a else "" for a in theft.attack]),
        "onset": onset,
        "severity": sev, "true_kwh": true.sum(axis=1),
        "reported_kwh": np.nansum(rep, axis=1),
        "metered_kwh": np.nansum(rep, axis=1) * gains,
        "offline_days": hh.offline.sum(axis=1), "meter_gain": gains,
    })


def daily_table(hh: HouseholdSet, theft: Theft, gains: np.ndarray) -> pd.DataFrame:
    """Long table household x day: true, reported and metered kWh."""
    rep = reported_daily_kwh(hh, theft)
    n, d = rep.shape
    return pd.DataFrame({
        "household": np.repeat(hh.hh_id, d),
        "date": np.tile(hh.days.values, n),
        "true_kwh": hh.true_kwh.astype(np.float32).ravel(),
        "reported_kwh": rep.astype(np.float32).ravel(),
        "metered_kwh": (rep * gains[:, None]).astype(np.float32).ravel(),
        "attack_active": (theft.thief[:, None]
                          & (np.arange(d)[None, :] >= theft.onset_day[:, None])).ravel(),
    })
