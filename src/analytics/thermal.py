"""Thermal engine: IS 2026 (Part 7):2009 / IEC 60076-7:2005 hot-spot and ageing
model for the 13 distribution transformers (docs/Layer3_plan.md §3, T1-T6;
Stage 3 run sheet §2-3).

Inputs (engine side only, plan §0; run sheet §6): the DT meters' metered
per-phase current (``current_a`` of the metered frame, meters m11-m23 selected
by role), the rated LV current ``i_rated_a`` from ``config/meters.yaml``, the
DT ratings and IS 1180 losses from ``dts.parquet`` (``name, rating_kva, p0_w,
pk_w`` only) and the ERA5 ambient series.

Model (IS 2026-7 Cl. 8.2.3, Fig. 10, PDF p. 30), per DT:

    top oil    [(1 + R K_oil^2)/(1 + R)]^x  Dtheta_or = k11 tau_o dtheta_o/dt + (theta_o - theta_a)
    gradient   k21 K_hs^y Dtheta_hr        = k22 tau_w dDtheta_h1/dt + Dtheta_h1
               (k21 - 1) K_hs^y Dtheta_hr  = (tau_o/k22) dDtheta_h2/dt + Dtheta_h2
    theta_h = theta_o + Dtheta_h1 - Dtheta_h2

solved with the Cl. 8.2.3 difference equations (explicit Euler; the load and
ambient of step n with the state of step n-1).  Time step Dt = 0.1 min, each
15-min load and ambient value held constant over its 150 sub-steps (user
decision 28 Oct, Stage 3 step A: the run sheet's Dt = 1 min fails the Annex E
acceptance test, theta_h(30) = 114.391 -> 114.4 against 114.2; 0.1 min gives
114.240, the setting of the References README check of 25 Oct; both satisfy
Cl. 8.2.3's Dt below half the smallest time constant).  With the inputs
constant over a step, the sub-steps of each first-order equation are summed in
closed form, x_m = u + (x_0 - u)(1 - a)^m; this is the same recurrence, not
another method (checked against the sub-step loop in the tests).

Load factor for an unbalanced DT (plan §3, stated adaptation of the standard):
K_p = I_p / I_r per phase, K_oil = sqrt(mean_p K_p^2) (top oil follows total
losses), K_hs = max_p K_p (gradient of the most loaded winding).

Parameter sets (T1; run sheet §0):
- ``main``: Table 5 ONAN constants; R = P_k / P_0 per DT from its IS 1180
  losses; Dtheta_or = IS 1180 top-oil rise limit, 35 K up to 200 kVA and 40 K
  above; Dtheta_hr = H g_r = 23 K (Table E.1, ``verify``).
- ``table_e1`` (sensitivity): as ``main`` but Dtheta_or = 55 K and
  Dtheta_hr = 23 K for every DT (Table E.1).  R stays per DT (run sheet §0
  names only Dtheta_or and H g_r).

Ageing (T6): non-upgraded paper, V = 2^((theta_h - 98)/6) per sub-step
(Cl. 6.2 eq. (2), PDF p. 15); daily loss of life L = sum V Dt over the
calendar day, in hours of normal life per day (a day at 98 degC gives 24 h).
A step's sub-steps belong to the step's calendar day.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import meters as M

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERA5_FILE = PROJECT_ROOT / "data" / "era5" / "era5_t2m_delhi_2024_ist_15min.parquet"
STEPS_2024 = 35_136                 # 366 days x 96 (run sheet §0; stop condition §6)
STEPS_PER_DAY = 96
STEP_MIN = 15
PHASES = ("a", "b", "c")

# IS 2026-7 Table 5, distribution transformers ONAN (PDF p. 29, printed p. 24)
TABLE5_ONAN = {"x": 0.8, "y": 1.6, "k11": 1.0, "k21": 1.0, "k22": 2.0, "tau_o": 180.0, "tau_w": 4.0}
# T1: IS 1180 (Part 1):2014 top-oil temperature-rise test limits (bound on Dtheta_or)
DTHETA_OR_UP_TO_200_KVA = 35.0      # Cl. 6.10.2, PDF p. 5 (three-phase DTs up to 200 kVA)
DTHETA_OR_ABOVE_200_KVA = 40.0      # Cl. 7.10.2, PDF pp. 6-7 (above 200 kVA up to 2 500 kVA)
SMALL_DT_MAX_KVA = 200.0            # Cl. 6.10.2 scope
H_GR = 23.0                         # verify: IS 2026-7 Annex E Table E.1 (PDF p. 52), informative example
TABLE_E1_DTHETA_OR = 55.0           # Annex E Table E.1, distribution ONAN (PDF p. 52)
PARAM_SETS = ("main", "table_e1")

# Cl. 8.2.3 difference equations; Dt = 0.1 min (user decision 28 Oct after the
# Annex E test failed at the run sheet's 1 min), 150 sub-steps per 15-min step
DT_MIN = 0.1
SUBSTEPS = 150

# Ageing (T6): IS 2026-7 Cl. 6.2 eq. (2), PDF p. 15
THETA_REF_C = 98.0
AGEING_DOUBLING_K = 6.0
# T4: 98 degC scored alarm (ageing reference); Table 4 (PDF p. 17) 120 / 140 degC
THRESHOLDS_C = (98.0, 120.0, 140.0)

# Acceptance test: IS 2026-7 Annex E worked example, Table E.1 OF column (PDF pp. 52-53)
ANNEX_E_OF = {"x": 1.0, "y": 1.3, "k11": 1.0, "k21": 1.3, "k22": 1.0, "tau_o": 90.0, "tau_w": 7.0,
              "R": 6.0, "dtheta_or": 56.0, "dtheta_hr": 22.0}
ANNEX_E_THETA_A = 20.0
ANNEX_E_K = (0.8, 1.4, 0.8)         # steady, for 30 min, then for 1 410 min
ANNEX_E_STEP_MIN = 30
ANNEX_E_END_MIN = 1440
ANNEX_E_EXPECTED = {30: 114.2, 1440: 75.2}     # theta_h, rounded to the one decimal Annex E prints


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalParams:
    x: float
    y: float
    k11: float
    k21: float
    k22: float
    tau_o: float
    tau_w: float
    R: np.ndarray               # per DT (or scalar)
    dtheta_or: np.ndarray       # rated top-oil rise (K)
    dtheta_hr: np.ndarray       # rated hot-spot-to-top-oil gradient H g_r (K)


def dt_names() -> list[str]:
    """The 13 DT meters' load ids, in ``meters.yaml`` order (selected by role)."""
    return [m.load_id for m in M.dt_meters()]


def rated_current() -> np.ndarray:
    """Rated LV current per DT (A), ``i_rated_a`` from ``config/meters.yaml``."""
    return np.array([float(m.i_rated_a) for m in M.dt_meters()])


def read_dts(path: str | Path) -> pd.DataFrame:
    """Rating and IS 1180 losses per DT (``dts.parquet``), in DT-meter order."""
    d = pd.read_parquet(path, columns=["name", "rating_kva", "p0_w", "pk_w"]).set_index("name")
    return d.loc[dt_names()]


def dt_params(dts: pd.DataFrame, param_set: str = "main") -> ThermalParams:
    """Parameter set ``main`` or ``table_e1`` for the DTs in ``dts`` (T1)."""
    if param_set not in PARAM_SETS:
        raise ValueError(param_set)
    R = (dts.pk_w / dts.p0_w).to_numpy(float)
    if param_set == "main":
        d_or = np.where(dts.rating_kva.to_numpy(float) <= SMALL_DT_MAX_KVA,
                        DTHETA_OR_UP_TO_200_KVA, DTHETA_OR_ABOVE_200_KVA)
    else:
        d_or = np.full(len(dts), TABLE_E1_DTHETA_OR)
    return ThermalParams(**TABLE5_ONAN, R=R, dtheta_or=d_or, dtheta_hr=np.full(len(dts), H_GR))


def annex_e_params() -> ThermalParams:
    c = dict(ANNEX_E_OF)
    return ThermalParams(**{k: c[k] for k in ("x", "y", "k11", "k21", "k22", "tau_o", "tau_w")},
                         R=np.float64(c["R"]), dtheta_or=np.float64(c["dtheta_or"]),
                         dtheta_hr=np.float64(c["dtheta_hr"]))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def read_dt_currents(frame_path: str | Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Metered per-phase current of the 13 DT meters: (timestamps, I[T, DT, phase]) in A."""
    names = dt_names()
    f = pd.read_parquet(frame_path, columns=["timestamp", "load_id", "phase", "current_a"],
                        filters=[("load_id", "in", names)])
    p = f.pivot(index="timestamp", columns=["load_id", "phase"], values="current_a")
    p = p.reindex(columns=pd.MultiIndex.from_product([names, PHASES])).sort_index()
    if p.isna().to_numpy().any():
        raise ValueError(f"{int(p.isna().to_numpy().sum())} missing DT currents")
    return pd.DatetimeIndex(p.index), p.to_numpy(float).reshape(len(p), len(names), len(PHASES))


def read_ambient(path: str | Path = ERA5_FILE) -> pd.Series:
    """ERA5 2 m temperature (degC) per 15-min IST step of 2024 (run sheet §6: 35,136 rows)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 file missing: {path}")
    e = pd.read_parquet(path, columns=["timestamp", "t2m_c"])
    if len(e) != STEPS_2024 or e.t2m_c.isna().any():
        raise ValueError(f"ERA5 file has {len(e)} rows ({int(e.t2m_c.isna().sum())} NaN), expected {STEPS_2024}")
    return e.set_index("timestamp").t2m_c


def load_factor(current: np.ndarray, i_rated: np.ndarray) -> np.ndarray:
    """K_p = I_p / I_r; ``current`` is [..., DT, phase]."""
    return current / i_rated[:, None]


def k_oil(k: np.ndarray) -> np.ndarray:
    """K_oil = sqrt(mean_p K_p^2) over the last (phase) axis."""
    return np.sqrt(np.mean(k ** 2, axis=-1))


def k_hs(k: np.ndarray) -> np.ndarray:
    """K_hs = max_p K_p over the last (phase) axis."""
    return np.max(k, axis=-1)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class State:
    theta_o: np.ndarray
    dh1: np.ndarray
    dh2: np.ndarray

    @property
    def theta_h(self) -> np.ndarray:
        return self.theta_o + self.dh1 - self.dh2


def ultimate(p: ThermalParams, ko, kh):
    """Ultimate top-oil rise and gradient terms for load factors K_oil, K_hs."""
    oil = ((1.0 + p.R * ko ** 2) / (1.0 + p.R)) ** p.x * p.dtheta_or
    khy = kh ** p.y
    return oil, p.k21 * p.dtheta_hr * khy, (p.k21 - 1.0) * p.dtheta_hr * khy


def steady_state(p: ThermalParams, ko, kh, theta_a) -> State:
    oil, h1, h2 = ultimate(p, np.asarray(ko, float), np.asarray(kh, float))
    return State(theta_o=theta_a + oil, dh1=h1, dh2=h2)


def ageing_rate(theta_h) -> np.ndarray:
    """Relative ageing rate V = 2^((theta_h - 98)/6) (eq. (2), non-upgraded paper)."""
    return 2.0 ** ((np.asarray(theta_h, float) - THETA_REF_C) / AGEING_DOUBLING_K)


def simulate(p: ThermalParams, ko: np.ndarray, kh: np.ndarray, theta_a: np.ndarray, state0: State,
             substeps: int = SUBSTEPS, dt: float = DT_MIN, thresholds=THRESHOLDS_C) -> dict:
    """Cl. 8.2.3 difference equations over T input steps (axis 0), each held for
    ``substeps`` sub-steps of ``dt`` minutes.  ``ko``, ``kh`` and ``theta_a`` are
    broadcast to a common shape (T, ...).

    Per input step [T, ...]: ``theta_h_end`` (theta_h at the end of the step),
    ``theta_h_max`` (largest sub-step value), ``lol_h`` (sum of V dt over the
    step's sub-steps, hours), ``cross_min[thr]`` (minutes from the step start to
    the end of the first sub-step above ``thr``; NaN if none), and the state at
    the end of the step (``theta_o``, ``dh1``, ``dh2``)."""
    ko, kh, theta_a = np.broadcast_arrays(np.asarray(ko, float), np.asarray(kh, float),
                                          np.asarray(theta_a, float))
    oil, h1u, h2u = ultimate(p, ko, kh)
    shape = ko.shape
    a_o = dt / (p.k11 * p.tau_o)
    a_1 = dt / (p.k22 * p.tau_w)
    a_2 = dt / (p.tau_o / p.k22)
    m = np.arange(1, substeps + 1, dtype=float).reshape((substeps,) + (1,) * (len(shape) - 1))
    d_o, d_1, d_2 = (1.0 - a_o) ** m, (1.0 - a_1) ** m, (1.0 - a_2) ** m
    t_sub = (m * dt).reshape(-1)
    out = {k: np.empty(shape) for k in ("theta_h_end", "theta_h_max", "lol_h", "theta_o", "dh1", "dh2")}
    cross = {thr: np.empty(shape) for thr in thresholds}
    so = np.broadcast_to(state0.theta_o, shape[1:]).astype(float)
    s1 = np.broadcast_to(state0.dh1, shape[1:]).astype(float)
    s2 = np.broadcast_to(state0.dh2, shape[1:]).astype(float)
    for j in range(shape[0]):
        uo = oil[j] + theta_a[j]
        o = uo + (so - uo) * d_o
        g1 = h1u[j] + (s1 - h1u[j]) * d_1
        g2 = h2u[j] + (s2 - h2u[j]) * d_2
        th = o + g1 - g2
        so, s1, s2 = o[-1], g1[-1], g2[-1]
        out["theta_o"][j], out["dh1"][j], out["dh2"][j] = so, s1, s2
        out["theta_h_end"][j] = th[-1]
        out["theta_h_max"][j] = th.max(axis=0)
        out["lol_h"][j] = ageing_rate(th).sum(axis=0) * dt / 60.0
        for thr in thresholds:
            above = th > thr
            cross[thr][j] = np.where(above.any(axis=0), t_sub[np.argmax(above, axis=0)], np.nan)
    return {**out, "cross_min": cross}


def simulate_loop(p: ThermalParams, ko, kh, theta_a, state0: State,
                  substeps: int = SUBSTEPS, dt: float = DT_MIN) -> np.ndarray:
    """Reference implementation: the Cl. 8.2.3 difference equations one sub-step
    at a time; theta_h [T, substeps, ...] (used by the tests to check :func:`simulate`)."""
    ko, kh, theta_a = np.broadcast_arrays(np.asarray(ko, float), np.asarray(kh, float),
                                          np.asarray(theta_a, float))
    oil, h1u, h2u = ultimate(p, ko, kh)
    so, s1, s2 = (np.array(np.broadcast_to(v, ko.shape[1:]), float)
                  for v in (state0.theta_o, state0.dh1, state0.dh2))
    out = np.empty((ko.shape[0], substeps) + ko.shape[1:])
    for j in range(ko.shape[0]):
        for i in range(substeps):
            so = so + dt / (p.k11 * p.tau_o) * (oil[j] - (so - theta_a[j]))
            s1 = s1 + dt / (p.k22 * p.tau_w) * (h1u[j] - s1)
            s2 = s2 + dt / (p.tau_o / p.k22) * (h2u[j] - s2)
            out[j, i] = so + s1 - s2
    return out


def annex_e_example(dt: float = DT_MIN) -> dict:
    """The Annex E worked example with the Fig. 10 method: K = 0.8 steady,
    1.4 for 30 min, then 0.8; theta_a = 20 degC; one input per minute, held for
    1/dt sub-steps."""
    p = annex_e_params()
    sub = int(round(1.0 / dt))
    k = np.full(ANNEX_E_END_MIN, ANNEX_E_K[2])
    k[:ANNEX_E_STEP_MIN] = ANNEX_E_K[1]
    s0 = steady_state(p, ANNEX_E_K[0], ANNEX_E_K[0], ANNEX_E_THETA_A)
    r = simulate(p, k, k, np.full(ANNEX_E_END_MIN, ANNEX_E_THETA_A), s0, substeps=sub, dt=dt,
                 thresholds=())
    th, to = r["theta_h_end"], r["theta_o"]
    return {"dt_min": dt, "dtheta_oi_k": float(s0.theta_o - ANNEX_E_THETA_A),
            "dtheta_hi_k": float(s0.dh1 - s0.dh2),
            "theta_h_c": {t: float(th[t - 1]) for t in ANNEX_E_EXPECTED},
            "theta_o_c": {t: float(to[t - 1]) for t in ANNEX_E_EXPECTED},
            "max_hot_spot_rise_k": float(r["theta_h_max"].max() - ANNEX_E_THETA_A)}


def annex_e_passes(res: dict) -> bool:
    return all(round(res["theta_h_c"][t], 1) == v for t, v in ANNEX_E_EXPECTED.items())


# --------------------------------------------------------------------------
# Trajectories, loss of life, events
# --------------------------------------------------------------------------

def reference_run(p: ThermalParams, k: np.ndarray, theta_a: np.ndarray) -> dict:
    """Continuous run over the year from steady state at the first step's load
    and ambient (run sheet §3).  ``k`` is K_p [T, DT, phase], ``theta_a`` [T]."""
    ko, kh = k_oil(k), k_hs(k)
    ta = np.broadcast_to(np.asarray(theta_a, float)[:, None], ko.shape)
    s0 = steady_state(p, ko[0], kh[0], ta[0])
    return simulate(p, ko, kh, ta, s0)


def forecast_run(p: ThermalParams, k: np.ndarray, theta_a: np.ndarray, state0: State) -> dict:
    """Forecast trajectories: ``k`` is K_p [origin, 96, DT, phase], ``theta_a``
    [origin, 96]; ``state0`` the reference state at 00:00 of each origin
    [origin, DT].  Returns the :func:`simulate` outputs as [origin, 96, DT]."""
    ko = np.moveaxis(k_oil(k), 1, 0)                         # [96, origin, DT]
    kh = np.moveaxis(k_hs(k), 1, 0)
    ta = np.broadcast_to(np.moveaxis(np.asarray(theta_a, float), 1, 0)[..., None], ko.shape)
    r = simulate(p, ko, kh, ta, state0)
    out = {k: np.moveaxis(v, 0, 1) for k, v in r.items() if k != "cross_min"}
    out["cross_min"] = {t: np.moveaxis(v, 0, 1) for t, v in r["cross_min"].items()}
    return out


def state_at(res: dict, step) -> State:
    """The state at the end of input step(s) ``step`` of a :func:`simulate` result."""
    return State(theta_o=res["theta_o"][step], dh1=res["dh1"][step], dh2=res["dh2"][step])


def per_day(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Split the 15-min step axis into (day, 96), moved to the front."""
    x = np.moveaxis(x, axis, 0)
    if x.shape[0] % STEPS_PER_DAY:
        raise ValueError(f"{x.shape[0]} steps is not a whole number of days")
    return x.reshape((x.shape[0] // STEPS_PER_DAY, STEPS_PER_DAY) + x.shape[1:])


def day_summary(res: dict, axis: int = 0) -> dict:
    """Per calendar day, from a :func:`simulate`-style result whose step axis is
    ``axis`` (days moved to the front): peak theta_h, loss of life (h/day), and
    for every threshold the first crossing in minutes after 00:00 (NaN = none)."""
    out = {"theta_h_max": per_day(res["theta_h_max"], axis).max(axis=1),
           "lol_h": per_day(res["lol_h"], axis).sum(axis=1), "first_cross_min": {}}
    for thr, c in res["cross_min"].items():
        t = per_day(c, axis)
        t = t + (np.arange(STEPS_PER_DAY) * STEP_MIN).reshape((1, -1) + (1,) * (t.ndim - 2))
        any_ = ~np.isnan(t).all(axis=1)
        out["first_cross_min"][thr] = np.where(any_, np.nanmin(np.where(np.isnan(t), np.inf, t), axis=1),
                                               np.nan)
    return out
