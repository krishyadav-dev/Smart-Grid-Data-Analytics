"""
Households behind the distribution transformers (Part 2C.2).

- **Count (brief v2 X.3 item 1):** N_bus = round(P_bus,nominal / (h_bar x
  s_design)).  The IEEE spot loads are **design peak** loads; h_bar is the
  mean household demand (kW) of all eligible SGCC candidates (:mod:`sgcc`)
  and s_design = (yearly maximum of the pool-average 15-min demand over the
  mapped 2024 calendar) / h_bar (:func:`design_peak`).  Since Layer 3 P3 the
  counts are pinned in ``config/household_counts.yaml`` (the pre-fix values);
  the rule is still computed and every pinned count must be within 1 of it
  (:func:`check_pinned_counts`).
- **Split:** across a bus's DTs in proportion to rating, and across LV phases
  in proportion to :func:`lv_phase_loads` (largest-remainder rounding);
  which household lands in which slot is a seeded permutation.
- **LV phasing (X.3 item 2, stated deviation from IEEE phasing):** buses that
  are three-phase in IEEE (634, 671, 675, 670) keep the IEEE per-phase
  proportions; single- and two-phase IEEE buses (645, 646, 652, 692, 611)
  spread their households equally over the three LV phases, because their
  laterals are three-phase in Stage B.  For delta MV loads the IEEE pair
  index (AB, BC, CA) is the LV phase index (a, b, c): behind a Dyn11 DT an
  LV phase-a load draws 11 kV current in lines A and B (checked by
  ``tests/test_metering.py::test_delta_loads_mv_currents``).
- **Intraday shape (brief v2 X.3.3 item 1):** each household has its own
  Prayas eMARC household (doi:10.7910/DVN/YJ5SP1, CC0 1.0; :mod:`emarc`),
  drawn with the seeded stream ``households.shape``; on each day it uses
  that household's mean-1 profile for the day's calendar month.  True 15-min
  kW = SGCC daily kWh / 24 x that profile.
- **Storage:** each household's daily kWh (float32) plus a profile id per
  household and day into the shared profile library; 15-minute kW are
  generated on demand.  The DT LV load is the sum of **true** household load
  (theft never touches the power flow), computed per day by grouping the
  households of each DT phase by profile id (weights @ profiles).
- **Q:** each household takes its bus phase's IEEE Q/P ratio (stated).
- **Gaps:** the meter reading is missing (offline) on days the feeder rule
  leaves missing; the true load across those days is linearly interpolated
  because the power flow needs a value, so no reported data is fabricated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ieee13_data as ieee

logger = logging.getLogger(__name__)

PHASES = "abc"


# --------------------------------------------------------------------------
# Bus loads in LV phase terms
# --------------------------------------------------------------------------

EQUAL_SPLIT_BUSES = ("645", "646", "652", "692", "611")   # single-/two-phase in IEEE


def lv_phasing(bus: str, p, q) -> tuple[np.ndarray, np.ndarray]:
    """LV per-phase (P, Q) of a bus from its IEEE per-phase values (X.3 item 2).

    Single- and two-phase IEEE buses: the bus total split equally over a/b/c.
    Three-phase IEEE buses: unchanged.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    if bus in EQUAL_SPLIT_BUSES:
        return np.full(3, p.sum() / 3), np.full(3, q.sum() / 3)
    return p.copy(), q.copy()


def ieee_phase_loads() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """IEEE nominal (P kW, Q kvar) per bus, indexed by LV phase a/b/c, IEEE phasing.

    Wye loads keep their phase; delta pairs AB/BC/CA map to LV a/b/c (Dyn11).
    The distributed load is lumped at 670 (Phase 1).
    """
    out = {bus: (np.array(p, float), np.array(q, float))
           for bus, (_, _, p, q) in ieee.SPOT_LOADS.items()}
    _, _, _, _, p, q = ieee.DISTRIBUTED_LOAD
    out["670"] = (np.array(p, float), np.array(q, float))
    return out


def lv_phase_loads() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Design-peak (P kW, Q kvar) per bus and LV phase with the X.3 LV phasing."""
    return {bus: lv_phasing(bus, p, q) for bus, (p, q) in ieee_phase_loads().items()}


def household_counts(bus_p_kw: dict[str, float], h_bar_kw: float, s_design: float) -> dict[str, int]:
    """N_bus = round(P_bus,nominal / (h_bar x s_design))."""
    return {bus: int(round(p / (h_bar_kw * s_design))) for bus, p in bus_p_kw.items()}


PINNED_COUNT_MAX_DIFF = 1      # user decision 2026-10-27: pinned N_bus within 1 of the rule


def pinned_household_counts(path=None) -> dict[str, int]:
    """Households per bus pinned in ``config/household_counts.yaml`` (Layer 3 P3)."""
    import yaml
    from . import config
    data = yaml.safe_load(open(path or config.HOUSEHOLD_COUNTS_YAML, encoding="utf-8"))
    counts = {str(b): int(n) for b, n in data["counts"].items()}
    if sum(counts.values()) != int(data["total"]):
        raise ValueError(f"pinned counts sum to {sum(counts.values())}, file says {data['total']}")
    return counts


def check_pinned_counts(pinned: dict[str, int], computed: dict[str, int]) -> dict[str, int]:
    """Pinned minus computed per bus; raises if a bus is missing or differs by more than 1."""
    if set(pinned) != set(computed):
        raise ValueError(f"pinned buses {sorted(pinned)} != computed {sorted(computed)}")
    diff = {b: pinned[b] - computed[b] for b in computed}
    bad = {b: d for b, d in diff.items() if abs(d) > PINNED_COUNT_MAX_DIFF}
    if bad:
        raise ValueError(f"pinned household counts differ from the rule by more than "
                         f"{PINNED_COUNT_MAX_DIFF}: {bad}")
    return diff


def largest_remainder(total: int, weights) -> np.ndarray:
    """Integer split of ``total`` in proportion to ``weights`` (sums exactly)."""
    w = np.asarray(weights, float)
    if total == 0 or w.sum() <= 0:
        return np.zeros(len(w), int)
    exact = total * w / w.sum()
    n = np.floor(exact).astype(int)
    rem = total - n.sum()
    order = np.argsort(-(exact - n), kind="stable")
    n[order[:rem]] += 1
    return n


# --------------------------------------------------------------------------
# Household set
# --------------------------------------------------------------------------

@dataclass
class HouseholdSet:
    hh_id: np.ndarray              # "hh_00001", ...
    sgcc_row: np.ndarray           # row in the SGCC matrix
    consumer_id: np.ndarray        # SGCC CONS_NO
    bus: np.ndarray                # IEEE bus
    dt: np.ndarray                 # DT name (dt_<bus>_<k>)
    dt_index: np.ndarray           # row in the DT table
    phase: np.ndarray              # LV phase index 0/1/2
    qp_ratio: np.ndarray           # Q/P of the household's bus phase
    true_kwh: np.ndarray           # float32 (n_hh, n_days): true daily kWh
    offline: np.ndarray            # bool (n_hh, n_days): meter reading missing
    days: pd.DatetimeIndex         # simulated days
    profiles: np.ndarray           # (n_profiles, steps_per_day), each mean 1 (eMARC library)
    profile_id: np.ndarray         # int (n_hh, n_days): profile of each household on each day
    emarc_row: np.ndarray          # (n_hh,) assigned eMARC household (library row)
    n_dt: int
    mostly_zero: np.ndarray = None  # bool (n_hh,): SGCC "mostly zero" consumer (sgcc.mostly_zero)
    info: dict = field(default_factory=dict)

    @property
    def steps_per_day(self) -> int:
        return self.profiles.shape[1]

    def day_shapes(self, day: int) -> np.ndarray:
        """(n_hh, steps) mean-1 shape of every household on one day."""
        return self.profiles[self.profile_id[:, day]]

    def _group_weights(self, day: int, w: np.ndarray) -> np.ndarray:
        """(n_dt * 3, n_profiles): sum of ``w`` over the households of each DT
        phase that use each profile on ``day``."""
        m = np.zeros((self.n_dt * 3, len(self.profiles)))
        np.add.at(m, (self.dt_index * 3 + self.phase, self.profile_id[:, day]), w)
        return m

    def dt_phase_kw_day(self, day: int) -> tuple[np.ndarray, np.ndarray]:
        """True (P kW, Q kvar) per (DT, LV phase, step) on one day, each (n_dt, 3, steps).

        Households grouped by profile id: sum_k (sum of daily kW of the group) x profile_k.
        """
        kw = self.true_kwh[:, day].astype(float) / 24.0
        p = self._group_weights(day, kw) @ self.profiles
        q = self._group_weights(day, kw * self.qp_ratio) @ self.profiles
        s = (self.n_dt, 3, self.steps_per_day)
        return p.reshape(s), q.reshape(s)

    def dt_phase_daily_kw(self) -> np.ndarray:
        """Mean kW per (day, DT, LV phase): sum of household daily kWh / 24."""
        out = np.zeros((len(self.days), self.n_dt, 3))
        np.add.at(out.transpose(1, 2, 0), (self.dt_index, self.phase),
                  self.true_kwh.astype(float) / 24.0)
        return out

    def dt_phase_daily_kvar(self) -> np.ndarray:
        out = np.zeros((len(self.days), self.n_dt, 3))
        np.add.at(out.transpose(1, 2, 0), (self.dt_index, self.phase),
                  self.true_kwh.astype(float) / 24.0 * self.qp_ratio[:, None])
        return out

    def true_kw_day(self, day: int) -> np.ndarray:
        """True 15-min kW of every household on one day, shape (n_hh, steps)."""
        return (self.true_kwh[:, day, None].astype(float) / 24.0) * self.day_shapes(day)

    def memory_bytes(self) -> int:
        return int(self.true_kwh.nbytes + self.offline.nbytes + self.profiles.nbytes
                   + self.profile_id.nbytes)


def interpolate_days(x: np.ndarray) -> np.ndarray:
    """Linear interpolation across missing days per row (edges: nearest value)."""
    out = x.astype(float).copy()
    idx = np.arange(x.shape[1])
    for i in np.flatnonzero(np.isnan(out).any(axis=1)):
        ok = ~np.isnan(out[i])
        if not ok.any():
            raise ValueError(f"household row {i} has no reading in the window")
        out[i, ~ok] = np.interp(idx[~ok], idx[ok], out[i, ok])
    return out


def build_households(daily_kwh: np.ndarray, sgcc_rows: np.ndarray, consumer_ids: np.ndarray,
                     counts: dict[str, int], dts: pd.DataFrame, days: pd.DatetimeIndex,
                     rng: np.random.Generator, profiles: np.ndarray, profile_id: np.ndarray,
                     emarc_row: np.ndarray, day_slice: slice = slice(None),
                     mostly_zero: np.ndarray | None = None) -> HouseholdSet:
    """Place the sampled feeder-pool households on DTs and LV phases.

    ``daily_kwh``: feeder-rule daily kWh of the sampled households (NaN =
    long gap), one row per household, in pool order.  Gaps are interpolated
    over all its columns, then ``day_slice`` selects the simulated ``days``
    (so a short window inside the year gets the year's values).
    ``profiles`` / ``profile_id`` (n_hh, n_days of ``days``) / ``emarc_row``:
    each household's eMARC shape (in pool order, like ``daily_kwh``).
    ``dts``: DT table with columns ``name, bus, rating_kva`` (row order = DT index).
    """
    n_total = sum(counts.values())
    if len(daily_kwh) != n_total:
        raise ValueError(f"{len(daily_kwh)} households given, counts need {n_total}")
    loads = lv_phase_loads()
    perm = rng.permutation(n_total)
    slot_bus, slot_dt, slot_phase, slot_qp = [], [], [], []
    for bus in sorted(counts):
        bus_dts = dts.index[dts.bus == bus].to_numpy()
        per_dt = largest_remainder(counts[bus], dts.loc[bus_dts, "rating_kva"])
        p, q = loads[bus]
        qp = np.divide(q, p, out=np.zeros(3), where=p > 0)
        for dt_i, n_dt in zip(bus_dts, per_dt):
            per_ph = largest_remainder(int(n_dt), p)
            for ph, n_ph in enumerate(per_ph):
                slot_bus += [bus] * n_ph
                slot_dt += [dt_i] * n_ph
                slot_phase += [ph] * n_ph
                slot_qp += [qp[ph]] * n_ph
    slot_dt = np.asarray(slot_dt)
    order = perm                      # household perm[k] fills slot k
    hh = np.empty(n_total, int)
    hh[order] = np.arange(n_total)    # slot of each household
    offline = np.isnan(daily_kwh)[:, day_slice]
    true = interpolate_days(daily_kwh)[:, day_slice].astype(np.float32)
    if true.shape[1] != len(days):
        raise ValueError(f"{true.shape[1]} day columns for {len(days)} simulated days")
    s = HouseholdSet(
        hh_id=np.array([f"hh_{i + 1:05d}" for i in range(n_total)]),
        sgcc_row=np.asarray(sgcc_rows), consumer_id=np.asarray(consumer_ids),
        bus=np.asarray(slot_bus)[hh], dt=dts["name"].to_numpy()[slot_dt[hh]],
        dt_index=slot_dt[hh], phase=np.asarray(slot_phase)[hh],
        qp_ratio=np.asarray(slot_qp, float)[hh], true_kwh=true, offline=offline,
        days=pd.DatetimeIndex(days), profiles=np.asarray(profiles, float),
        profile_id=np.asarray(profile_id), emarc_row=np.asarray(emarc_row), n_dt=len(dts),
        mostly_zero=(np.zeros(n_total, bool) if mostly_zero is None else np.asarray(mostly_zero)),
    )
    if s.profile_id.shape != (n_total, len(days)):
        raise ValueError(f"profile_id shape {s.profile_id.shape}, expected {(n_total, len(days))}")
    s.info = {"households": n_total, "offline_household_days": int(offline.sum()),
              "households_with_offline_days": int(offline.any(axis=1).sum()),
              "memory_mb": round(s.memory_bytes() / 1e6, 2)}
    logger.info("Households: %s", s.info)
    return s
