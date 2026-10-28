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

Consumer detector (plan §2 N2; run sheet §4): Jokar et al. 2016 per-household
SVM, leave-one-attack-out; see the section at the end of this module.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_source.feeder_model import build_network, run_timeseries
from src.data_source.theft_injector import ATTACKS, apply_attack

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


# --------------------------------------------------------------------------
# Consumer detector: Jokar et al. 2016 (plan §2 N2, N4; run sheet §4)
# --------------------------------------------------------------------------
# Input: the household AMI export (15-min metered Wh), summed to 24 hourly
# values per day (Jokar §IV-A step 2; h1-h6 are defined on hourly vectors; a
# stated deviation from the plan's "96 values").  Days with any missing sample
# are skipped.  Per household and fold (Jokar §IV-A, Experiment 2):
#   1. benign set = the household's online training days (onsets unknown);
#   2. features scaled to [-1, 1] per hour from those days (Jokar, Experiment 1);
#      k-means, k* = argmax mean silhouette over K_RANGE; clusters below
#      MIN_CLUSTER_SHARE of the benign days are dropped (their days leave the
#      benign set); k' clusters remain;
#   3. synthetic malicious set: every benign training day through each of the
#      fold's training attacks (theft_injector.apply_attack);
#   4. SVC, RBF, k' + 1 classes (one-vs-one), benign classes oversampled by
#      replication until benign and malicious counts are equal;
#   5. a day is flagged if the predicted class is the malicious one; the
#      household score is the share of its online scored days flagged.
# Day score for the (C, gamma) objective: the malicious class's column of
# SVC.decision_function, decision_function_shape="ovr" [our choice, user
# decision 27 Oct].

JOKAR_SEED = 20261027                                    # run sheet §4 [our seed]
K_RANGE = (2, 3, 4)                                      # [our choice of range]
MIN_CLUSTER_SHARE = 0.05                                 # [our choice; Jokar: "few members"]
MIN_TRAIN_DAYS = 30                                      # [our choice] skipped below this
# (C, gamma) grid [our choice; Jokar's grid not stated].  Amended 27 Oct before D was frozen
# (user decision, tuning data only): the first grid put the pick on its corner in 5 of 7 folds.
GRID_C_FIRST = tuple(2.0 ** k for k in (0, 2, 4, 6, 8))
GRID_GAMMA_FIRST = tuple(2.0 ** k for k in (-7, -5, -3, -1))
GRID_C = tuple(2.0 ** k for k in (-6, -4, -2, 0, 2, 4, 6, 8))
GRID_GAMMA = tuple(2.0 ** k for k in (-13, -11, -9, -7, -5, -3, -1, 1))
N_TUNE_HOUSEHOLDS = 300                                  # run sheet §4 tuning step 2
HOURS = 24
FOLDS = {**{a: tuple(b for b in ATTACKS if b != a) for a in ATTACKS},   # leave-one-attack-out
         "circular": ATTACKS}                            # graded on its own generator: upper bound only
# Phases: (training days, scored days)
PHASES_JOKAR = {
    "validation": (("2024-01-01", "2024-04-30"), ("2024-05-01", "2024-06-30")),
    "test": (("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")),
}
_PHASE_CODE = {"validation": 0, "test": 1}


def ami_hourly(ami_dir: str | Path, households=None) -> tuple[list[str], pd.DatetimeIndex, np.ndarray]:
    """(household ids, days, X) with X (households, days, 24) hourly kWh,
    NaN on every day with a missing 15-min sample.  Reads the monthly AMI
    export files (Wh, null = offline)."""
    files = sorted(Path(ami_dir).glob("ami_*.parquet"))
    blocks, days, ids = [], [], None
    for f in files:
        t = pd.read_parquet(f) if households is None else pd.read_parquet(f, columns=["timestamp", *households])
        ts = pd.DatetimeIndex(t.pop("timestamp"))
        if ids is None:
            ids = list(t.columns)
        spd = int(round(pd.Timedelta(days=1) / (ts[1] - ts[0])))
        if len(ts) % spd or ts[0] != ts[0].normalize():
            raise ValueError(f"{f.name}: not whole days")
        x = t[ids].to_numpy(dtype=float, na_value=np.nan).T / 1e3            # (hh, samples) kWh
        n_d = len(ts) // spd
        x = x.reshape(len(ids), n_d, HOURS, spd // HOURS).sum(axis=3)      # NaN propagates
        blocks.append(x.astype(np.float32))
        days.append(ts[::spd])
    return ids, pd.DatetimeIndex(np.concatenate(days)), np.concatenate(blocks, axis=1)


def window_mask(days: pd.DatetimeIndex, w) -> np.ndarray:
    return np.asarray((days >= pd.Timestamp(w[0])) & (days <= pd.Timestamp(w[1])))


def scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-hour min and max over the benign training days."""
    return x.min(axis=0), x.max(axis=0)


def scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map [lo, hi] to [-1, 1] per hour (a constant hour maps to 0)."""
    span = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(span > 0, 2.0 * (x - lo) / np.where(span > 0, span, 1.0) - 1.0, 0.0)


def benign_clusters(z: np.ndarray) -> dict:
    """k-means on the scaled benign days: k*, the retained days and their labels."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = None
    distinct = len(np.unique(z.round(12), axis=0))
    for k in K_RANGE:
        if k >= len(z) or k > distinct:
            continue
        lab = KMeans(n_clusters=k, random_state=JOKAR_SEED).fit_predict(z)
        if len(np.unique(lab)) < 2:
            continue
        s = float(silhouette_score(z, lab))
        if best is None or s > best[0]:                      # ties: the smaller k
            best = (s, k, lab)
    if best is None:                                         # silhouette undefined: one cluster [our choice]
        k_star, lab, sil = 1, np.zeros(len(z), int), None
    else:
        sil, k_star, lab = best
    counts = np.bincount(lab, minlength=k_star)
    kept = np.flatnonzero(counts >= MIN_CLUSTER_SHARE * len(z))
    keep = np.isin(lab, kept)
    relabel = {c: i for i, c in enumerate(kept)}
    return {"k_star": int(k_star), "k_prime": int(len(kept)), "silhouette": sil, "keep": keep,
            "labels": np.array([relabel[c] for c in lab[keep]], int)}


def attacked_days(x: np.ndarray, attacks, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Every day of ``x`` (n, 24) through each attack, one draw per day."""
    return {a: apply_attack(a, x, rng)[0] for a in attacks}


def _rng(row: int, phase: str, stream: int) -> np.random.Generator:
    """Per-household child stream of JOKAR_SEED [our choice: results do not
    depend on the order the workers run in]; never the feeder's thief stream."""
    return np.random.default_rng(np.random.SeedSequence(JOKAR_SEED, spawn_key=(row, _PHASE_CODE[phase], stream)))


def fit_svm(zb: np.ndarray, lb: np.ndarray, zm: np.ndarray, c: float, gamma: float):
    """SVC on k' benign classes + 1 malicious class (label k'); the benign
    days are replicated (cycled) until benign and malicious counts are equal."""
    from sklearn.svm import SVC
    k_prime = int(lb.max()) + 1
    n = max(len(zm), len(zb))
    idx = np.resize(np.arange(len(zb)), n)
    x = np.vstack([zb[idx], zm])
    y = np.concatenate([lb[idx], np.full(len(zm), k_prime)])
    return SVC(C=c, kernel="rbf", gamma=gamma, decision_function_shape="ovr").fit(x, y), k_prime


def malicious_score(svm, k_prime: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(day score, flag): the malicious class's ovr decision value and predict == malicious."""
    d = svm.decision_function(z)
    score = d if d.ndim == 1 else d[:, list(svm.classes_).index(k_prime)]
    return score, svm.predict(z) == k_prime


def prepare_household(x_year: np.ndarray, days: pd.DatetimeIndex, row: int, phase: str) -> Optional[dict]:
    """Benign set, scaling, clusters and the synthetic attacked training days
    (all six attacks) for one household and phase; None if skipped."""
    tr_w, sc_w = PHASES_JOKAR[phase]
    online = ~np.isnan(x_year).any(axis=1)
    tr = window_mask(days, tr_w) & online
    if tr.sum() < MIN_TRAIN_DAYS:
        return None
    xb = x_year[tr].astype(float)
    lo, hi = scale_fit(xb)
    zb_all = scale(xb, lo, hi)
    cl = benign_clusters(zb_all)
    mal = attacked_days(xb, ATTACKS, _rng(row, phase, 0))
    sc = window_mask(days, sc_w)
    return {"lo": lo, "hi": hi, "zb": zb_all[cl["keep"]], "lb": cl["labels"],
            "zm": {a: scale(v, lo, hi) for a, v in mal.items()}, "clusters": cl,
            "n_train_days": int(tr.sum()), "scored": sc & online, "score_window": sc, "x_score": x_year[sc & online]}


def grid_auc(prep: dict, row: int, fold: str, grid) -> list[float]:
    """Per (C, gamma) in ``grid``: the day-level ROC AUC separating the
    household's scored (validation) days from synthetic attacked versions of
    them made with the fold's training attacks."""
    from sklearn.metrics import roc_auc_score
    xv = prep["x_score"].astype(float)
    if len(xv) == 0:
        return [float("nan")] * len(grid)
    fold_attacks = FOLDS[fold]
    att = attacked_days(xv, fold_attacks, _rng(row, "validation", 1 + list(FOLDS).index(fold)))
    zv = scale(np.vstack([xv, *att.values()]), prep["lo"], prep["hi"])
    y = np.r_[np.zeros(len(xv)), np.ones(len(xv) * len(fold_attacks))]
    zm = np.vstack([prep["zm"][a] for a in fold_attacks])
    out = []
    for c, g in grid:
        svm, kp = fit_svm(prep["zb"], prep["lb"], zm, c, g)
        out.append(float(roc_auc_score(y, malicious_score(svm, kp, zv)[0])))
    return out


def score_household(prep: dict, fold_attacks, c: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Day scores and flags over the phase's scored window (NaN / -1 on
    offline days) for one fold's (C, gamma)."""
    zm = np.vstack([prep["zm"][a] for a in fold_attacks])
    svm, kp = fit_svm(prep["zb"], prep["lb"], zm, c, gamma)
    n = int(prep["score_window"].sum())
    s = np.full(n, np.nan, np.float32)
    f = np.full(n, -1, np.int8)
    on = prep["scored"][prep["score_window"]]
    if on.any():
        sc, fl = malicious_score(svm, kp, scale(prep["x_score"].astype(float), prep["lo"], prep["hi"]))
        s[on], f[on] = sc, fl
    return s, f


def household_score(flags: np.ndarray, gate: Optional[np.ndarray] = None) -> float:
    """Share of online scored days flagged (with ``gate``: flagged and gate open)."""
    on = flags >= 0
    if not on.any():
        return float("nan")
    hit = flags == 1
    if gate is not None:
        hit = hit & gate
    return float(hit[on].mean())


def select_svm_params(auc: np.ndarray, grid) -> tuple[float, float, float]:
    """(C, gamma, mean AUC) maximising the mean per-household AUC; ties: smaller C, then smaller gamma."""
    m = np.nanmean(auc, axis=0)
    best = max(m)
    c, g = min(p for p, v in zip(grid, m) if v == best)
    return c, g, float(best)
