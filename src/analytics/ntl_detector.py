"""Jokar consumer detector, Stage 4 (theft-engine fixes; Stage 4 run sheet §4).

Every Stage 2 rule stays (:mod:`ntl_engine`): benign set = the household's
online training days; per-household scaling to [-1, 1] on the benign set
(Jokar, Experiment 1); k-means with k* = argmax silhouette over k = 2, 3, 4,
clusters below 5 % dropped; synthetic malicious set = every benign day through
each training attack (``theft_injector.apply_attack``, per-household child
streams of rng 20261027); benign classes oversampled to the malicious count;
multi-class RBF SVC; day score = the malicious class's ovr decision value;
household score = share of flagged online days; leave-one-attack-out folds
plus the circular reference; at least 30 training days.  Changes:

- **B1, monthly benign update [our choice: Jokar §IV-B step 4 without its DT
  condition].**  Variant ``update``: at the start of each month M in the
  scored window, the household's online days of month M - 1 that the current
  model classified benign join its benign set; their attack patterns are made
  with the fold's training attacks (the same per-household generator family:
  child stream 20 + M - 1 of rng 20261027 [our choice]) and join the attack
  set.  Then the scaling [our choice: refitted on the grown benign set, as
  Stage 2 scales on the training days], the k-means rule and the SVC (frozen
  C, gamma) are refitted.  The benign set grows (expanding window).  Variant
  ``static`` is the Stage 2 behaviour (no update), kept as the ablation.  The
  benign sets grow per fold (each fold's model classifies on its own).
- **B9a, grid:** C in 2^-6, 2^-4, ..., 2^12 and gamma in 2^-17, 2^-15, ..., 2^3.
- **B9b, resolution:** 24 hourly or 96 quarter-hourly values per day; at 96
  the attacks keep Jokar's hourly parameters and whole-hour off windows
  (``theft_injector``); chosen on the tuning sample and frozen.

The validation phase (train January-April) also predicts its own January-April
training days, for the drift diagnostic (prediction only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_source.theft_injector import ATTACKS

from . import ntl_engine as N

GRID_C = tuple(2.0 ** k for k in range(-6, 13, 2))                 # B9a [our choice of grid]
GRID_GAMMA = tuple(2.0 ** k for k in range(-17, 4, 2))
GRID = tuple((c, g) for c in GRID_C for g in GRID_GAMMA)
RESOLUTIONS = (24, 96)                                              # B9b; ties keep 24 (Jokar)
VARIANTS = ("static", "update")
FOLDS = N.FOLDS
LOO_FOLDS = tuple(f for f in FOLDS if f != "circular")
UPDATE_STREAM = 20                                                  # child stream 20 + month [our choice]
# (training days, scored days, days predicted for the output)
PHASES = {
    "validation": (("2024-01-01", "2024-04-30"), ("2024-05-01", "2024-06-30"), ("2024-01-01", "2024-06-30")),
    "test": (("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"), ("2024-07-01", "2024-12-31")),
}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def ami_vectors(ami_dir: str | Path, samples: int = 24) -> tuple[list[str], pd.DatetimeIndex, np.ndarray]:
    """(household ids, days, X) with X (households, days, ``samples``) kWh per
    hour (24) or per 15 min (96); NaN on every day with a missing sample."""
    if samples == 24:
        return N.ami_hourly(ami_dir)
    if samples != 96:
        raise ValueError(f"samples must be 24 or 96, got {samples}")
    files = sorted(Path(ami_dir).glob("ami_*.parquet"))
    blocks, days, ids = [], [], None
    for f in files:
        t = pd.read_parquet(f)
        ts = pd.DatetimeIndex(t.pop("timestamp"))
        ids = list(t.columns) if ids is None else ids
        spd = int(round(pd.Timedelta(days=1) / (ts[1] - ts[0])))
        if spd != 96 or len(ts) % spd or ts[0] != ts[0].normalize():
            raise ValueError(f"{f.name}: not whole days of 96 samples")
        x = t[ids].to_numpy(dtype=np.float32, na_value=np.nan).T / 1e3
        blocks.append(x.reshape(len(ids), len(ts) // spd, spd))
        days.append(ts[::spd])
    return ids, pd.DatetimeIndex(np.concatenate(days)), np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------
# One household
# --------------------------------------------------------------------------

def fit_model(xb: np.ndarray, mal: np.ndarray, c: float, gamma: float, base: Optional[dict] = None) -> dict:
    """Scaling on the benign set ``xb`` (raw), k-means rule, SVC on the kept
    benign clusters + the scaled malicious set ``mal`` (raw).  ``base``:
    precomputed scaling and clusters of the same ``xb``."""
    b = base or benign_base(xb)
    zm = N.scale(mal, b["lo"], b["hi"])
    svm, kp = N.fit_svm(b["zb"][b["cl"]["keep"]], b["cl"]["labels"], zm, c, gamma)
    return {"svm": svm, "kp": kp, "lo": b["lo"], "hi": b["hi"], "k_star": b["cl"]["k_star"],
            "k_prime": b["cl"]["k_prime"]}


def benign_base(xb: np.ndarray) -> dict:
    lo, hi = N.scale_fit(xb)
    zb = N.scale(xb, lo, hi)
    return {"lo": lo, "hi": hi, "zb": zb, "cl": N.benign_clusters(zb)}


def predict(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(day score, flag) of raw days ``x``."""
    if len(x) == 0:
        return np.zeros(0), np.zeros(0, bool)
    return N.malicious_score(model["svm"], model["kp"], N.scale(x.astype(float), model["lo"], model["hi"]))


def prepare(x_year: np.ndarray, days: pd.DatetimeIndex, row: int, phase: str) -> Optional[dict]:
    """Benign training set, its attacked versions (all six attacks, stream 0
    as Stage 2), the scaling and clusters; None below MIN_TRAIN_DAYS."""
    tr_w, sc_w, out_w = PHASES[phase]
    online = ~np.isnan(x_year).any(axis=1)
    tr = N.window_mask(days, tr_w) & online
    if tr.sum() < N.MIN_TRAIN_DAYS:
        return None
    xb = x_year[tr].astype(float)
    return {"x": x_year, "days": days, "row": row, "phase": phase, "online": online, "xb": xb,
            "mal0": N.attacked_days(xb, ATTACKS, N._rng(row, phase, 0)), "base": benign_base(xb),
            "scored": N.window_mask(days, sc_w), "out": N.window_mask(days, out_w),
            "n_train_days": int(tr.sum()), "_month_att": {}}


def month_attacks(prep: dict, month: int) -> tuple[np.ndarray, dict]:
    """(day indices, attacked versions by attack) of the online scored days of
    ``month``; one draw per day on child stream UPDATE_STREAM + month, so the
    patterns of a day do not depend on the fold or on which days are added."""
    if month not in prep["_month_att"]:
        d = np.flatnonzero(prep["scored"] & prep["online"] & (prep["days"].month == month))
        att = N.attacked_days(prep["x"][d].astype(float), ATTACKS,
                              N._rng(prep["row"], prep["phase"], UPDATE_STREAM + month))
        prep["_month_att"][month] = (d, att)
    return prep["_month_att"][month]


def run_fold(prep: dict, fold_attacks, c: float, gamma: float, variant: str) -> dict:
    """Score the phase's output window with one fold's training attacks.

    Returns ``score`` (float32, NaN offline), ``flag`` (int8: 1 flagged,
    0 benign, -1 offline / not predicted) over the output window, ``added``
    (bool: day joined the benign set by B1) and ``k_prime`` per scored month.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    days, x, online = prep["days"], prep["x"], prep["online"]
    out_idx = np.flatnonzero(prep["out"])
    pos = {d: k for k, d in enumerate(out_idx)}
    score = np.full(len(out_idx), np.nan, np.float32)
    flag = np.full(len(out_idx), -1, np.int8)
    added = np.zeros(len(out_idx), bool)
    xb = prep["xb"]
    mal = {a: prep["mal0"][a] for a in fold_attacks}
    model = fit_model(xb, np.vstack([mal[a] for a in fold_attacks]), c, gamma, prep["base"])
    k_prime = {}
    # days predicted before the scored window (validation phase: its own training days)
    pre = np.flatnonzero(prep["out"] & ~prep["scored"] & online)
    s, f = predict(model, x[pre])
    for d, sv, fv in zip(pre, s, f):
        score[pos[d]], flag[pos[d]] = sv, fv
    months = list(pd.unique(days[prep["scored"]].month))
    for k, m in enumerate(months):
        k_prime[int(m)] = model["k_prime"]
        d, att = month_attacks(prep, int(m))
        s, f = predict(model, x[d])
        for dd, sv, fv in zip(d, s, f):
            score[pos[dd]], flag[pos[dd]] = sv, fv
        if variant == "update" and k < len(months) - 1:
            ben = ~f
            if ben.any():
                xb = np.vstack([xb, x[d[ben]].astype(float)])
                for a in fold_attacks:
                    mal[a] = np.vstack([mal[a], att[a][ben]])
                added[[pos[dd] for dd in d[ben]]] = True
                model = fit_model(xb, np.vstack([mal[a] for a in fold_attacks]), c, gamma)
    return {"score": score, "flag": flag, "added": added, "k_prime": k_prime}


def grid_auc(prep: dict, fold: str, grid, variants=VARIANTS) -> dict[str, list[float]]:
    """Tuning objective per (C, gamma) and variant (validation phase): the
    day-level ROC AUC separating the household's May-June online days from
    synthetic attacked versions of them made with the fold's training attacks
    (stream 1 + fold index, as Stage 2).  ``update``: May scored by the
    initial model; at the start of June May's benign-classified days join the
    benign set (B1) and June is scored by the refitted model."""
    from sklearn.metrics import roc_auc_score
    if prep["phase"] != "validation":
        raise ValueError("grid_auc runs on the validation phase")
    fa = FOLDS[fold]
    d = np.flatnonzero(prep["scored"] & prep["online"])
    if len(d) == 0:
        return {v: [float("nan")] * len(grid) for v in variants}
    xv = prep["x"][d].astype(float)
    att = N.attacked_days(xv, fa, N._rng(prep["row"], "validation", 1 + list(FOLDS).index(fold)))
    month = prep["days"][d].month.to_numpy()
    y = np.r_[np.zeros(len(xv)), np.ones(len(xv) * len(fa))]
    m_all = np.tile(month, 1 + len(fa))
    xall = np.vstack([xv, *[att[a] for a in fa]])
    first = month == month.min()
    mal0 = np.vstack([prep["mal0"][a] for a in fa])
    out = {v: [] for v in variants}
    for c, g in grid:
        model = fit_model(prep["xb"], mal0, c, g, prep["base"])
        s0, _ = predict(model, xall)
        if "static" in variants:
            out["static"].append(float(roc_auc_score(y, s0)))
        if "update" in variants:
            ben = ~predict(model, xv[first])[1]
            s = s0.copy()
            if ben.any():
                dm, matt = month_attacks(prep, int(month.min()))
                pick = np.isin(dm, d[first][ben])
                xb = np.vstack([prep["xb"], xv[first][ben]])
                mal = np.vstack([np.vstack([prep["mal0"][a], matt[a][pick]]) for a in fa])
                later = m_all != month.min()
                s[later] = predict(fit_model(xb, mal, c, g), xall[later])[0]
            out["update"].append(float(roc_auc_score(y, s)))
    return out


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def resolution_choice(best_mean_auc: dict[int, dict[str, float]], folds=LOO_FOLDS) -> dict:
    """B9b: mean over ``folds`` of each fold's best mean AUC, per resolution;
    96 only if strictly higher, ties keep 24 (Jokar's resolution)."""
    avg = {r: float(np.mean([best_mean_auc[r][f] for f in folds])) for r in best_mean_auc}
    pick = 96 if 96 in avg and avg[96] > avg[24] else 24
    return {"average_over_folds": avg, "folds": list(folds), "samples_per_day": pick}


def edge(c: float, g: float) -> list[str]:
    """Grid edges a pick lies on (reported, never extended; B9a)."""
    return [s for s, on in (("C_low", c == min(GRID_C)), ("C_high", c == max(GRID_C)),
                            ("gamma_low", g == min(GRID_GAMMA)), ("gamma_high", g == max(GRID_GAMMA))) if on]
