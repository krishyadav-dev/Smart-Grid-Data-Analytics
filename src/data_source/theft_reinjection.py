"""
Second theft injection (Layer 3 Stage 4 run sheet §2, decision D1).

Layer 1 code: it rebuilds the household population exactly as
:func:`simulation.run_year` does (:func:`simulation.build_population`), then
draws the thieves, attack types and onsets again with
:func:`metering.assign_theft` on the named streams

    config.rng("theft.pick/<tag>"), config.rng("theft.attack/<tag>")

[our choice: derived from the master seed, no hand-picked seed].  With
``tag = ""`` the stream names are the v1 ones (``theft.pick``,
``theft.attack``), so the same code must reproduce the v1 theft-dependent
files exactly (:func:`reproduction_check`).  Prevalence p is unchanged.

Theft does not change the network: :func:`metering.assign_theft` alters only
what the thieves' meters report.  The true household loads, the power flow,
the meter frames, the DT and head meter readings, ``state.parquet``,
``dts.parquet``, the SGCC pools, L^_tech and the lambda_true DT inputs are
reused from ``results/year_2024/`` and ``results/layer3/ntl/``
(:func:`theft_independence_checks` verifies the inputs they are built from).

Written (theft-dependent files only), under ``out``:

- ``households.parquet``, ``household_daily.parquet``,
  ``household_daily_labels.parquet`` (as :func:`simulation.run_year`);
- ``ami/`` (the household AMI export, :func:`simulation.write_ami_export`);
- ``lambda_sweep/`` (the un-netted sweep's metrics and DT-day table depend on
  the theft truth; :func:`simulation.lambda_sweep` with the v1 true frame);
- ``reinject_info.json`` (the theft-dependent part of ``run_info.json``:
  thieves per attack, streams; a label file, read by the evaluator only);
- ideal-meter household inputs (every gain 1.0, :func:`metering.daily_table`)
  under ``ideal_out``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from . import metering as M
from . import simulation as S

logger = logging.getLogger(__name__)

V1_CACHE = config.YEAR_CACHE_DIR
V1_NTL = config.PROJECT_ROOT / "results" / "layer3" / "ntl"
INFO_FILE = "reinject_info.json"
THEFT_FILES = ("households.parquet", "household_daily.parquet", S.LABELS_FILE)
# never written by this module (v1 results stay as they are; Stage 4 run sheet §0)
PROTECTED = (V1_CACHE, V1_NTL, config.PROJECT_ROOT / "results" / "layer3" / "sgcc")


def stream_names(tag: str) -> tuple[str, str]:
    """(pick, attack) stream names; ``tag = ""`` gives the v1 names."""
    suffix = f"/{tag}" if tag else ""
    return f"theft.pick{suffix}", f"theft.attack{suffix}"


def _check_out(path: Path) -> None:
    p = Path(path).resolve()
    for q in PROTECTED:
        q = Path(q).resolve()
        if p == q or q in p.parents:
            raise ValueError(f"refusing to write into the v1 results: {p}")


def redraw(pop: S.Population, tag: str) -> M.Theft:
    """The population's theft drawn again on the tagged streams."""
    pick, attack = stream_names(tag)
    return M.assign_theft(pop.hh, pop.matrix.theft_prevalence, config.rng(pick), config.rng(attack))


def theft_info(pop: S.Population, theft: M.Theft, tag: str) -> dict:
    return {
        "tag": tag, "streams": list(stream_names(tag)), "seed": config.MASTER_SEED,
        "theft_prevalence_p": pop.matrix.theft_prevalence, "households": int(len(pop.hh.hh_id)),
        "thieves": int(theft.thief.sum()),
        "thieves_per_attack": {k: int(v) for k, v in zip(*np.unique(theft.attack[theft.thief],
                                                                      return_counts=True))},
        "thieves_mostly_zero": int((theft.thief & pop.hh.mostly_zero).sum()),
        "thief_series_mb": round(theft.reported_kw.nbytes / 1e6, 1),
    }


def write(pop: S.Population, theft: M.Theft, out: Path, ideal_out: Optional[Path], tag: str,
          v1_cache: Path = V1_CACHE, write_ami: bool = True, sweep: bool = True) -> dict:
    """Write every theft-dependent file for ``theft`` (see the module docstring)."""
    out = Path(out)
    _check_out(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    M.ground_truth(pop.hh, theft, pop.gain_household).to_parquet(out / "households.parquet", index=False)
    M.daily_table(pop.hh, theft, pop.gain_household).to_parquet(out / "household_daily.parquet", index=False)
    M.daily_labels(pop.hh, theft).to_parquet(out / S.LABELS_FILE, index=False)
    if write_ami:
        S.write_ami_export(dataclasses.replace(pop, theft=theft), out / "ami")
    if sweep:
        S.lambda_sweep(out, out_dir=out / "lambda_sweep", frame_cache=v1_cache)
    if ideal_out is not None:
        ideal_out = Path(ideal_out)
        _check_out(ideal_out)
        ideal_out.mkdir(parents=True, exist_ok=True)
        ones = np.ones(len(pop.hh.hh_id))
        M.daily_table(pop.hh, theft, ones)[["household", "date", "metered_kwh"]].to_parquet(
            ideal_out / "household_daily.parquet", index=False)
    info = theft_info(pop, theft, tag)
    info["reused_from_v1"] = {
        "results/year_2024": ["meter_frame_true.parquet", "meter_frame_metered.parquet", "state.parquet",
                              "dts.parquet", "meter_points.parquet", "acceptance.json",
                              "sgcc_detector_train.parquet", "sgcc_detector_test.parquet"],
        "results/layer3/ntl": ["ltech_real.parquet", "ltech_ideal.parquet", "lambda_true_{0,2,4}/",
                               "ideal_meter/meter_frame_metered.parquet", "consumer_index.parquet"]}
    info["written"] = [*THEFT_FILES, "ami/" if write_ami else None, "lambda_sweep/" if sweep else None,
                       INFO_FILE]
    info["write_s"] = round(time.perf_counter() - t0, 1)
    (out / INFO_FILE).write_text(json.dumps(info, indent=2, default=S._json), encoding="utf-8")
    return info


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_parquet(a: Path, b: Path) -> dict:
    """Byte hash and exact frame equality of two Parquet files."""
    rec = {"file": Path(a).name, "sha256_equal": sha256(a) == sha256(b)}
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)
        rec["frame_equal"] = True
    except AssertionError as e:                       # recorded, then raised by the caller
        rec["frame_equal"] = False
        rec["difference"] = str(e)[:500]
    return rec


def reproduction_check(out: Path, ideal_out: Optional[Path], v1_cache: Path = V1_CACHE,
                       v1_ntl: Path = V1_NTL) -> dict:
    """Files written with ``tag = ""`` against the v1 files (stop if any differs)."""
    out, v1_cache = Path(out), Path(v1_cache)
    rows = [_same_parquet(out / f, v1_cache / f) for f in THEFT_FILES]
    for f in sorted((v1_cache / "ami").glob("ami_*.parquet")):
        rows.append(_same_parquet(out / "ami" / f.name, f))
    for f in ("dt_day_balance.parquet",):
        rows.append(_same_parquet(out / "lambda_sweep" / f, v1_cache / "lambda_sweep" / f))
    m_new = json.loads((out / "lambda_sweep" / "metrics.json").read_text(encoding="utf-8"))
    m_old = json.loads((v1_cache / "lambda_sweep" / "metrics.json").read_text(encoding="utf-8"))
    rows.append({"file": "lambda_sweep/metrics.json", "json_equal": m_new == m_old})
    if ideal_out is not None:
        rows.append(_same_parquet(Path(ideal_out) / "household_daily.parquet",
                                  Path(v1_ntl) / "ideal_meter" / "household_daily.parquet"))
    passed = all(r.get("frame_equal", r.get("json_equal")) for r in rows)
    return {"files": rows, "all_equal": bool(passed),
            "all_sha256_equal": bool(all(r.get("sha256_equal", True) for r in rows))}


def theft_independence_checks(pop: S.Population, theft: M.Theft, v1_cache: Path = V1_CACHE,
                              v1_ntl: Path = V1_NTL) -> dict:
    """Evidence that the reused v1 inputs do not depend on the theft draw.

    The meter frames, state and L^_tech are functions of the true household
    load, the meter gains and the network only (:func:`simulation.run_year`
    drives the power flow with ``Population.dt_loads``, i.e. ``true_kwh`` and
    the shapes).  So: (a) the rebuilt true loads, offline days, shapes and
    every meter gain equal the v1 cache's; (b) the true DT meter energy per
    DT-day equals the sum of the households' true kWh (the DT load is the
    household load; no theft term); (c) the consumer index equals the
    rebuilt (household, DT, bus).  Run after the redraw, (a) also shows that
    assign_theft left the household set untouched.
    """
    v1_cache, v1_ntl = Path(v1_cache), Path(v1_ntl)
    hh = pop.hh
    d1 = pd.read_parquet(v1_cache / "household_daily.parquet", columns=["household", "date", "true_kwh",
                                                                         "metered_kwh"])
    n, d = hh.true_kwh.shape
    rec = {}
    rec["true_kwh_equal_v1"] = bool(np.array_equal(d1.true_kwh.to_numpy().reshape(n, d), hh.true_kwh))
    rec["offline_equal_v1"] = bool(np.array_equal(d1.metered_kwh.isna().to_numpy().reshape(n, d), hh.offline))
    g1 = pd.read_parquet(v1_cache / "households.parquet", columns=["household", "meter_gain", "dt", "bus"])
    rec["household_gains_equal_v1"] = bool(np.array_equal(g1.meter_gain.to_numpy(), pop.gain_household))
    dts = pd.read_parquet(v1_cache / "dts.parquet")
    rec["dt_gains_equal_v1"] = bool(np.array_equal(dts.meter_gain.to_numpy(), pop.gain_dt))
    g_fh = json.loads((v1_cache / "run_info.json").read_text(encoding="utf-8"))["feeder_head_gain"]
    rec["feeder_head_gain_equal_v1"] = bool(g_fh == pop.gain_feeder_head)
    idx = pd.read_parquet(v1_ntl / "consumer_index.parquet")
    rec["consumer_index_equal_rebuilt"] = bool(
        list(idx.household) == list(hh.hh_id) and list(idx.dt) == list(hh.dt) and list(idx.bus) == list(hh.bus))
    # (b) DT true energy per DT-day vs the households' true kWh
    fr = pd.read_parquet(v1_cache / "meter_frame_true.parquet", columns=["timestamp", "load_id",
                                                                           "active_power_kw"],
                         filters=[("load_id", "in", list(dts.name))])
    e = fr.pivot_table(index="timestamp", columns="load_id", values="active_power_kw", aggfunc="sum")[dts.name]
    e = (e * S.STEP_H).groupby(pd.DatetimeIndex(e.index).normalize()).sum()
    hh_dt = pd.DataFrame(hh.true_kwh.astype(float), index=hh.dt).groupby(level=0).sum().T
    hh_dt.index = e.index
    rel = float((e - hh_dt[e.columns]).abs().to_numpy().max() / e.to_numpy().max())
    rec["dt_true_energy_vs_household_true_kwh_max_rel"] = rel
    rec["dt_true_energy_matches_households"] = bool(rel < 1e-6)
    rec["passed"] = all(v for k, v in rec.items() if isinstance(v, bool))
    return rec
