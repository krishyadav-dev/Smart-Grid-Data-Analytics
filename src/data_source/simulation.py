"""
Feeder population and the simulated year (Part 2C with brief v2 X.1/X.2).

:func:`build_population` goes SGCC -> eligibility -> h_bar and N -> pools ->
households on DTs and LV phases -> theft -> meter gains, with every random
stream taken from ``config.rng(name)`` (master seed).  :func:`run_year` runs
the power flow once over the window with the DT loads = sum of **true**
household load, applies the meter errors and writes the cache
(``results/year_2024/``).  :func:`lambda_sweep` is the un-netted DT-level
energy-balance baseline over the LV-loss fraction lambda.

Performance (X.2): households are stored as daily kWh plus the shared
intraday shape; 15-minute values are generated on demand per DT and per day.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from . import emarc
from . import households as H
from . import metering as M
from . import sgcc
from .feeder_model import build_network, meter_points, run_timeseries

logger = logging.getLogger(__name__)

STEP_H = 0.25
LAMBDAS = (0.0, 0.01, 0.02, 0.03, 0.05)          # brief v2 X.2
LABELS_FILE = "household_daily_labels.parquet"   # Layer 3 P3 (N5): evaluation-only labels
SWEEP_LABEL = "un-netted DT-level baseline (Evaluation Plan), not the Layer 3 engine"


@dataclass
class Population:
    net: object
    days: pd.DatetimeIndex
    matrix: sgcc.SGCCMatrix
    eligibility: sgcc.Eligibility
    h_bar_kw: float
    s_design: float
    counts: dict
    pools: sgcc.Pools
    hh: H.HouseholdSet
    theft: M.Theft
    gain_household: np.ndarray
    gain_dt: np.ndarray
    gain_feeder_head: float
    info: dict = field(default_factory=dict)

    def dt_loads(self, start: pd.Timestamp):
        """``run_timeseries`` callback: DT LV load = sum of true household load.

        Built once per day (households grouped by profile id) and indexed by step.
        """
        spd = self.hh.steps_per_day
        day0 = self.days[0]
        cache: dict = {}

        def f(i, ts):
            d = (ts.normalize() - day0).days
            if cache.get("day") != d:
                cache["day"] = d
                cache["p"], cache["q"] = self.hh.dt_phase_kw_day(d)
            s = int((ts - ts.normalize()) / pd.Timedelta(hours=24 / spd))
            return cache["p"][:, :, s], cache["q"][:, :, s]
        return f


def simulated_days(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")


def calendar_days() -> pd.DatetimeIndex:
    """The simulated calendar year (brief v2 X.2)."""
    y = config.SIMULATION_YEAR
    return simulated_days(f"{y}-01-01", f"{y}-12-31")


def assign_shapes(m: sgcc.SGCCMatrix, el: sgcc.Eligibility, lib: emarc.ShapeLibrary,
                  rng: np.random.Generator) -> np.ndarray:
    """eMARC household (library row) of every eligible SGCC consumer; -1 elsewhere.

    Drawn once for the eligible pool in matrix order (stream ``households.shape``),
    so the pool-average series and the simulated households use the same
    assignment; a simulated household inherits its consumer's eMARC household.
    """
    out = np.full(len(m.flag), -1, int)
    idx = np.flatnonzero(el.eligible)
    out[idx] = emarc.assign(len(idx), lib, rng)
    return out


def design_peak(m: sgcc.SGCCMatrix, el: sgcc.Eligibility, lib: emarc.ShapeLibrary,
                emarc_row: np.ndarray, days: pd.DatetimeIndex, h_bar_kw: float) -> dict:
    """s_design (brief v2 X.3 item 1, with the X.3.3 eMARC shapes).

    Pool-average 15-min demand over the calendar: the mean over the eligible
    pool of each consumer's feeder-rule kWh / 24 times its own eMARC shape for
    the day's month.  s_design = its yearly maximum / h_bar.  Also returns
    the timestamp of the maximum and the P99.9 value of the same series.
    """
    idx = np.flatnonzero(el.eligible)
    x = sgcc.feeder_preprocess(m.kwh[idx])[:, el.window_cols] / 24.0      # (n, days) kW
    valid = np.isfinite(x)
    g = np.zeros((lib.n_households, x.shape[1]))
    np.add.at(g, emarc_row[idx], np.where(valid, x, 0.0))                  # kW per eMARC group
    n_valid = valid.sum(axis=0)
    months = days.month.to_numpy() - 1
    spd = lib.profiles.shape[1]
    series = np.empty((len(days), spd))
    for d in range(len(days)):
        series[d] = g[:, d] @ lib.profiles[lib.month_profile[:, months[d]]] / n_valid[d]
    flat = series.ravel()
    i = int(np.argmax(flat))
    step = pd.Timedelta(hours=24 / spd)
    d, s = divmod(i, spd)
    peak = float(flat[i])
    p999 = float(np.percentile(flat, 99.9))
    return {"s_design": peak / h_bar_kw, "pool_avg_peak_kw": peak,
            "timestamp": str(days[d] + s * step), "p99_9_kw": p999,
            "p99_9_over_h_bar": p999 / h_bar_kw,
            "pool_avg_mean_kw": float(flat.mean()), "steps": int(flat.size)}


def build_population(start="2024-01-01", end="2024-12-31 23:45",
                     sgcc_path: Optional[Path | str] = None, net=None,
                     matrix: Optional[sgcc.SGCCMatrix] = None) -> Population:
    """Everything the window needs except the power flow (seeded, logged).

    Eligibility, h_bar, the eMARC shape assignment, s_design, N, the pools and
    the placement are always computed on the full calendar year, so a shorter
    window (June, a test week) simulates the year's own households on its days.
    """
    net = build_network("B") if net is None else net
    days = simulated_days(start, end)
    year = calendar_days()
    if days[0] < year[0] or days[-1] > year[-1]:
        raise ValueError(f"window {days[0].date()}..{days[-1].date()} outside {config.SIMULATION_YEAR}")
    m = sgcc.load_matrix(sgcc_path) if matrix is None else matrix
    el = sgcc.eligibility(m, year)
    lib = emarc.load_library()
    emarc_all = assign_shapes(m, el, lib, config.rng("households.shape"))
    h_bar = sgcc.mean_household_kw(m, el)
    design = design_peak(m, el, lib, emarc_all, year, h_bar)
    s_design = design["s_design"]
    bus_p = {b: float(v[0].sum()) for b, v in H.lv_phase_loads().items()}
    counts_rule = H.household_counts(bus_p, h_bar, s_design)
    # Layer 3 P3: N pinned in config/household_counts.yaml (within 1 of the rule per bus)
    counts = H.pinned_household_counts()
    counts_diff = H.check_pinned_counts(counts, counts_rule)
    # Effect of the top-5 % exclusion on h_bar and N (reported; same s_design)
    no_top = (m.flag == 0) & (el.completeness >= sgcc.COMPLETENESS_MIN)
    h_bar_all = sgcc.mean_household_kw(m, el, no_top)
    counts_all = H.household_counts(bus_p, h_bar_all, s_design)
    pools = sgcc.build_pools(m, el, sum(counts.values()), config.rng("sgcc.feeder_pool"),
                             config.rng("sgcc.detector_split"))
    daily = sgcc.feeder_preprocess(m.kwh[pools.feeder])[:, el.window_cols]
    i0 = (days[0] - year[0]).days
    mostly_zero = sgcc.mostly_zero(m, el)
    e_row = emarc_all[pools.feeder]
    hh = H.build_households(daily, pools.feeder, m.consumer_id[pools.feeder], counts,
                            net["dts"], days, config.rng("households.placement"),
                            lib.profiles, emarc.profile_ids(lib, e_row, days), e_row,
                            day_slice=slice(i0, i0 + len(days)),
                            mostly_zero=mostly_zero[pools.feeder])
    theft = M.assign_theft(hh, m.theft_prevalence, config.rng("theft.pick"),
                           config.rng("theft.attack"))
    g_hh = M.meter_gains(len(hh.hh_id), M.CLASS_HOUSEHOLD, config.rng("meters.household"))
    g_dt = M.meter_gains(len(net["dts"]), M.CLASS_DT, config.rng("meters.dt"))
    g_fh = float(M.meter_gains(1, M.CLASS_FEEDER_HEAD, config.rng("meters.feeder_head"))[0])
    dts = net["dts"]
    info = {
        "eligibility": el.report, "h_bar_kw": h_bar,
        "design_peak": design, "s_design": s_design,
        "counts": counts, "households": int(sum(counts.values())),
        "counts_source": "pinned, config/household_counts.yaml (Layer 3 P3)",
        "counts_rule": counts_rule, "counts_pinned_minus_rule": counts_diff,
        "feeder_mostly_zero": int(hh.mostly_zero.sum()),
        "mostly_zero_per_dt": {str(n): int(hh.mostly_zero[hh.dt_index == k].sum())
                               for k, n in enumerate(dts.name)},
        "h_bar_kw_without_top5_exclusion": h_bar_all, "counts_without_top5_exclusion": counts_all,
        "theft_prevalence_p": m.theft_prevalence, "thieves": int(theft.thief.sum()),
        "thieves_per_attack": {k: int(v) for k, v in zip(*np.unique(
            theft.attack[theft.thief], return_counts=True))},
        "thieves_mostly_zero": int((theft.thief & hh.mostly_zero).sum()),
        "pools": {"feeder": len(pools.feeder), "detector_train": len(pools.detector_train),
                  "detector_test": len(pools.detector_test),
                  "detector_train_flag1": int(m.flag[pools.detector_train].sum()),
                  "detector_test_flag1": int(m.flag[pools.detector_test].sum())},
        "emarc": lib.info,
        "emarc_households_used_by_feeder": int(len(np.unique(e_row))),
        "households_info": hh.info,
        "thief_series_mb": round(theft.reported_kw.nbytes / 1e6, 1),
        "seed": config.MASTER_SEED,
        "streams": ["households.shape", "sgcc.feeder_pool", "sgcc.detector_split",
                    "households.placement", "theft.pick", "theft.attack", "meters.household",
                    "meters.dt", "meters.feeder_head"],
    }
    logger.info("Population: h_bar %.4f kW, s_design %.4f (peak %s, P99.9 %.4f kW), N %s (total %d)",
                h_bar, s_design, design["timestamp"], design["p99_9_kw"], counts, info["households"])
    return Population(net, days, m, el, h_bar, s_design, counts, pools, hh, theft,
                      g_hh, g_dt, g_fh, info)


# --------------------------------------------------------------------------
# Meter errors
# --------------------------------------------------------------------------

def apply_meter_errors(frame: pd.DataFrame, gain_dt: dict, gain_feeder_head: float) -> pd.DataFrame:
    """Metered frame: feeder head and DT rows x their gains (P, Q, I); the
    bus_virtual rows are recomputed as the sum of their (metered) DT rows."""
    out = frame.copy()
    cols = ["current_a", "active_power_kw", "reactive_power_kvar", "load_current_a"]
    g = out.load_id.map(gain_dt).astype(float)
    g[out.load_id == "feeder_head_650"] = gain_feeder_head
    is_meter = g.notna()
    out.loc[is_meter, cols] = out.loc[is_meter, cols].mul(g[is_meter], axis=0)
    dt = out[out.load_id.str.startswith("dt_")]
    dtsum = (dt.assign(lp=dt.load_id.str.rsplit("_", n=1).str[0])
             .groupby(["timestamp", "lp", "phase"])[["active_power_kw", "reactive_power_kvar",
                                                     "current_a"]].sum())
    bv = out.load_id.str.startswith("load_")
    key = pd.MultiIndex.from_arrays([out.loc[bv, "timestamp"],
                                     "dt_" + out.loc[bv, "bus"], out.loc[bv, "phase"]])
    s = dtsum.reindex(key)
    out.loc[bv, "active_power_kw"] = s["active_power_kw"].to_numpy()
    out.loc[bv, "reactive_power_kvar"] = s["reactive_power_kvar"].to_numpy()
    out.loc[bv, "current_a"] = s["current_a"].to_numpy()
    v = out.loc[bv, "voltage_v"].to_numpy()
    out.loc[bv, "load_current_a"] = np.where(
        v > 0, np.hypot(out.loc[bv, "active_power_kw"], out.loc[bv, "reactive_power_kvar"]) * 1e3 / v, 0.0)
    return out


# --------------------------------------------------------------------------
# The year
# --------------------------------------------------------------------------

def peak_memory_mb() -> Optional[float]:
    """Peak resident memory of this process (MB), or None if unavailable."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # Linux: kB
    except ImportError:
        pass
    try:                                       # Windows: PeakWorkingSetSize
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if get_info(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1e6
    except Exception:          # noqa: BLE001  (diagnostic only)
        pass
    return None


def run_year(out_dir: Path | str = config.YEAR_CACHE_DIR, start="2024-01-01",
             end="2024-12-31 23:45", sgcc_path=None, write_ami: bool = True,
             progress_every: int = 960) -> dict:
    """Run the window once and write the cache.  Returns the run info."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pop = build_population(start, end, sgcc_path)
    t_pop = time.perf_counter() - t0
    t1 = time.perf_counter()
    frame, state = run_timeseries(start, end, net=pop.net, return_state=True,
                                  dt_loads=pop.dt_loads(pd.Timestamp(start)),
                                  progress_every=progress_every)
    t_pf = time.perf_counter() - t1
    dts = pop.net["dts"]
    metered = apply_meter_errors(frame, dict(zip(dts.name, pop.gain_dt)), pop.gain_feeder_head)
    frame.to_parquet(out / "meter_frame_true.parquet", index=False)
    metered.to_parquet(out / "meter_frame_metered.parquet", index=False)
    state.to_parquet(out / "state.parquet", index=False)
    dts.drop(columns=["trafo", "hv_bus", "lv_bus", "load"]).assign(
        meter_gain=pop.gain_dt).to_parquet(out / "dts.parquet", index=False)
    meter_points(pop.net).to_parquet(out / "meter_points.parquet", index=False)
    gt = M.ground_truth(pop.hh, pop.theft, pop.gain_household)
    gt.to_parquet(out / "households.parquet", index=False)
    M.daily_table(pop.hh, pop.theft, pop.gain_household).to_parquet(
        out / "household_daily.parquet", index=False)
    # Layer 3 P3 (N5): evaluation-only labels, kept out of household_daily and the AMI export
    M.daily_labels(pop.hh, pop.theft).to_parquet(out / LABELS_FILE, index=False)
    write_detector_pools(pop, out)
    if write_ami:
        write_ami_export(pop, out / "ami")
    t_total = time.perf_counter() - t0
    info = dict(pop.info)
    info.update({
        "window": [str(start), str(end)], "steps": int(state.shape[0]),
        "time_population_s": round(t_pop, 1), "time_power_flow_s": round(t_pf, 1),
        "s_per_step": round(t_pf / max(len(state), 1), 4), "time_total_s": round(t_total, 1),
        "peak_memory_mb": peak_memory_mb(), "feeder_head_gain": pop.gain_feeder_head,
        "sweep_label": SWEEP_LABEL,
        "design_precheck": design_precheck(pop).to_dict("records"),
    })
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=_json), encoding="utf-8")
    return info


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def write_detector_pools(pop: Population, out: Path) -> None:
    """Detector-train / detector-test consumers, Zheng Eq. (1)-(3) applied."""
    m = pop.matrix
    cols = m.dates.strftime("%Y-%m-%d")
    for name, rows in (("train", pop.pools.detector_train), ("test", pop.pools.detector_test)):
        x = sgcc.zheng_preprocess(m.kwh[rows])
        df = pd.DataFrame(x, columns=cols)
        df.insert(0, "FLAG", m.flag[rows])
        df.insert(0, "CONS_NO", m.consumer_id[rows])
        df.to_parquet(out / f"sgcc_detector_{name}.parquet", index=False)


def write_ami_export(pop: Population, out: Path) -> None:
    """Simulated AMI head-end export: metered 15-min energy per household in
    **Wh** (integer, 1 Wh resolution; null = meter offline), one Parquet file
    per month (rows = timestamps, columns = households)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    hh = pop.hh
    spd = hh.steps_per_day
    step = pd.Timedelta(hours=24 / spd)
    for month, idx in pd.Series(np.arange(len(pop.days)), index=pop.days).groupby(
            pop.days.to_period("M")):
        blocks = [M.reported_kw_day(hh, pop.theft, int(d)) for d in idx.values]
        wh = np.concatenate(blocks, axis=1) * STEP_H * 1e3 * pop.gain_household[:, None]
        missing = np.isnan(wh)
        if np.nanmax(wh, initial=0.0) >= 65535:
            raise ValueError("15-min household energy does not fit uint16 Wh")
        whi = np.where(missing, 0, np.rint(wh)).astype(np.uint16)
        ts = pd.date_range(pop.days[idx.values[0]], periods=wh.shape[1], freq=step)
        arrays = [pa.array(ts.values)] + [pa.array(whi[i], mask=missing[i])
                                          for i in range(len(hh.hh_id))]
        table = pa.Table.from_arrays(arrays, names=["timestamp"] + list(hh.hh_id))
        pq.write_table(table, out / f"ami_{month}.parquet", compression="zstd")


# --------------------------------------------------------------------------
# Year acceptance (CP3) and the DT loading table
# --------------------------------------------------------------------------

BAND_SOURCED = (0.90, 1.10)     # IS 17036:2018 Table 4, MV: Un +/- 10 %
LOSS_LIMIT = 0.05               # MV network (lines + substation transformer), 2B criterion as revised in X.3
DT_LOSS_TOL = 0.05              # X.3.2 item 14: total DT loss vs IS 1180 P0 v^2 + Pk (K0^2+K1^2+K2^2)
# X.3.1 item 1: IS 2026 (Part 7):2009 = IEC 60076-7:2005, Table 4 (PDF p. 17, printed p. 12),
# distribution transformers, normal cyclic loading: current 1.5 pu.  Every DT, every step.
IS2026_7_NORMAL_CYCLIC_PU = 1.5
VUF_LIMIT_PCT = 2.0             # IS 17036:2018 voltage unbalance limit (reported, X.3 item 4d)


def nominal_dt_table(net=None) -> pd.DataFrame:
    """Part 2C.1 table at IEEE nominal (design peak) load: bus, S_bus, k, rating, loading."""
    import pandapower as pp
    from .feeder_model import _Extractor
    net = build_network("B") if net is None else net
    pp.runpp_3ph(net)
    st = _Extractor(net).state(net, None, 0, float("nan"))
    d = net["dts"]
    rows = []
    for r in d.itertuples():
        s_bus = float(np.hypot(sum(getattr(r, f"p_{p}_kw") for p in "abc"),
                               sum(getattr(r, f"q_{p}_kvar") for p in "abc"))) * r.k
        s_ph = max(float(np.hypot(getattr(r, f"p_{p}_kw"), getattr(r, f"q_{p}_kvar"))) for p in "abc")
        ph = [st[f"load_pct_{r.name}_{p}"] for p in "abc"]
        rows.append({"dt": r.name, "bus": r.bus, "S_bus_kva": round(s_bus, 1),
                     "S_phase_max_kva": round(s_ph * r.k, 1), "k": r.k,
                     "rating_kva": r.rating_kva, "vk_percent": r.vk_percent,
                     "p0_w": round(r.p0_w, 1), "pk_w": round(r.pk_w, 1),
                     "nominal_S_share_pct": round(100 * s_bus / r.k / r.rating_kva, 1),
                     "nominal_load_a_pct": round(ph[0], 1), "nominal_load_b_pct": round(ph[1], 1),
                     "nominal_load_c_pct": round(ph[2], 1)})
    return pd.DataFrame(rows)


def dt_loss_check(state: pd.DataFrame, dts: pd.DataFrame, tol: float = DT_LOSS_TOL) -> dict:
    """X.3 check c as revised by X.3.2 item 14, at each DT's peak step.

    Asserted: modelled DT loss vs IS 1180 **total** loss P0 v_hv^2 +
    Pk (K0^2 + K1^2 + K2^2) within ``tol``.  K0/K1/K2 are the sequence
    components of the LV current per unit of rated current (their squares
    sum to the mean of the per-phase K^2).  The DT's peak step is the step of
    its maximum mean phase loading.  P0 and Pk are the IS 1180 level-1 values
    of the rating (``stage_b.dt_losses_w``).  The sequence breakdown is reported.
    """
    rows = []
    for r in dts.itertuples():
        cols = [f"load_pct_{r.name}_{p}" for p in "abc"]
        load = state[cols].to_numpy() / 100.0
        i = int(np.argmax(load.mean(axis=1)))
        k0, k1, k2 = (float(state[f"{c}_{r.name}"].iloc[i]) for c in ("k0", "k1", "k2"))
        v2 = float(state[f"vhv2_{r.name}"].iloc[i])
        parts_w = {"p0_v2": r.p0_w * v2, "pk_k1": r.pk_w * k1 ** 2,
                   "pk_k2": r.pk_w * k2 ** 2, "pk_k0": r.pk_w * k0 ** 2}
        expected = sum(parts_w.values()) / 1e3
        model = float(state[f"ploss_kw_{r.name}"].iloc[i])
        err = model / expected - 1.0
        rows.append({"dt": r.name, "rating_kva": r.rating_kva,
                     "timestamp": str(state.timestamp.iloc[i]) if "timestamp" in state else None,
                     "K_phases": [round(float(x), 4) for x in load[i]],
                     "K0": round(k0, 4), "K1": round(k1, 4), "K2": round(k2, 4),
                     "v_hv2": round(v2, 4), "p0_w": r.p0_w, "pk_w": r.pk_w,
                     **{f"is1180_{k}_w": round(v, 1) for k, v in parts_w.items()},
                     "expected_kw": round(expected, 4), "model_kw": round(model, 4),
                     "rel_error": round(err, 4), "passed": bool(abs(err) <= tol)})
    return {"tolerance": tol, "rows": rows, "max_abs_rel_error": max(abs(x["rel_error"]) for x in rows),
            "passed": all(x["passed"] for x in rows)}


def nominal_vuf(phasing: str = "x3") -> dict:
    """VUF (|V2/V1|, %) at the 11 kV buses at design peak (IEEE nominal load).

    ``phasing="x3"``: the network as built (X.3 LV phasing).  ``"ieee"``: the
    same DTs with the IEEE phasing put back at LV (single-/two-phase buses on
    their IEEE phase), the OLTC re-settled.  Reported against the IS 17036
    2 % limit (X.3 item 4d).
    """
    from . import oltc, stage_b
    net = build_network("B")
    if phasing == "ieee":
        loads = H.ieee_phase_loads()
        al = net.asymmetric_load
        for r in net["dts"].itertuples():
            p, q = loads[r.bus]
            for j, ph in enumerate("abc"):
                for col, v in ((f"p_{ph}_mw", p[j]), (f"p0_{ph}_mw", p[j]),
                               (f"q_{ph}_mvar", q[j]), (f"q0_{ph}_mvar", q[j])):
                    al.at[r.load, col] = v / r.k / 1e3
        oltc.settle(net, solve=stage_b._solve)
    elif phasing != "x3":
        raise ValueError(phasing)
    stage_b._solve(net)
    mv = net.bus.index[net.bus.vn_kv == config.FEEDER_VOLTAGE_KV]
    ub = net.res_bus_3ph.loc[mv, "unbalance_percent"]
    by_bus = {str(net.bus.at[b, "name"]): round(float(v), 4) for b, v in ub.items()}
    return {"phasing": phasing, "vuf_pct": by_bus, "max_pct": float(ub.max()),
            "limit_pct": VUF_LIMIT_PCT, "within_limit": bool(ub.max() <= VUF_LIMIT_PCT)}


def year_acceptance(state: pd.DataFrame, dts: pd.DataFrame) -> dict:
    """CP3 checks on the cached state frame (brief v2 X.3 item 4).

    Asserted: (a) IS 17036 +/- 10 % on every 11 kV bus-phase at every step;
    (b) X.3.1 item 1: no DT phase at any step above the IS 2026-7 Table 4
    normal-cyclic current limit (1.5 pu), every DT; (c) X.3.2 item 14: total
    DT loss at each DT's peak step = P0 v^2 + Pk (K0^2 + K1^2 + K2^2) within
    5 %.  Reported: max loading and hours above 1.0 pu per DT and phase,
    annual energy losses by part, losses at the peak step, LV voltages, OLTC
    statistics, VUF.
    """
    st = state.reset_index(drop=True)
    lo, hi = BAND_SOURCED
    out_band = (st.v_min_11kv_pu < lo) | (st.v_max_11kv_pu > hi)
    ip = int(st.p_load_mw.idxmax())
    peak = st.loc[ip]
    day = pd.DatetimeIndex(st.timestamp).normalize()
    limit_h = st.at_tap_limit.groupby(day).sum() * STEP_H
    ops = st.tap_moved.groupby(day).sum()
    limit_pct = 100.0 * IS2026_7_NORMAL_CYCLIC_PU
    dt_rows, over_limit = [], []
    for r in dts.itertuples():
        cols = [f"load_pct_{r.name}_{p}" for p in "abc"]
        mx = st[cols].max()
        over = (st[cols] > 100).sum() * STEP_H
        at_peak = {p: float(peak[c]) for p, c in zip("abc", cols)}
        over_limit += [f"{r.name}_{p}" for p, c in zip("abc", cols) if mx[c] > limit_pct]
        dt_rows.append({"dt": r.name, "bus": r.bus, "rating_kva": r.rating_kva,
                        **{f"max_load_{p}_pu": round(float(mx[c]) / 100, 3) for p, c in zip("abc", cols)},
                        **{f"hours_over_1pu_{p}": float(over[c]) for p, c in zip("abc", cols)},
                        **{f"peak_step_load_{p}_pu": round(v / 100, 3) for p, v in at_peak.items()}})
    e_in = float(st.p_in_mw.sum())
    dt_load = st.dt_losses_mw - st.dt_noload_mw
    parts = {"sub": st.sub_losses_mw, "lines": st.line_losses_mw,
             "dt_noload": st.dt_noload_mw, "dt_load": dt_load}
    mv_peak = float((peak.line_losses_mw + peak.sub_losses_mw) / peak.p_in_mw)
    dtl = dt_loss_check(st, dts)
    res = {
        "steps": int(len(st)),
        "band": {"asserted": [lo, hi], "v_min_11kv_pu": float(st.v_min_11kv_pu.min()),
                 "v_max_11kv_pu": float(st.v_max_11kv_pu.max()),
                 "steps_out_of_band": int(out_band.sum()),
                 "hours_out_of_band": float(out_band.sum() * STEP_H),
                 "days_with_violation": int(out_band.groupby(day).any().sum()),
                 "first_violation": str(st.timestamp[out_band].iloc[0]) if out_band.any() else None,
                 "passed": bool(not out_band.any())},
        "dt_loading_limit": {
            "limit_pu": IS2026_7_NORMAL_CYCLIC_PU,
            "source": "IS 2026 (Part 7):2009 = IEC 60076-7:2005, Table 4, PDF p. 17 (printed p. 12), "
                      "distribution transformers, normal cyclic loading",
            "max_loading_pu": float(max(max(v for k, v in row.items() if k.startswith("max_load_"))
                                        for row in dt_rows)),
            "phases_over_limit": over_limit, "passed": not over_limit},
        "dt_losses": dtl,
        "target_095_105_steps_with_violation": int((st.v_below_095 + st.v_above_105).gt(0).sum()),
        "lv": {"v_min_pu": float(st.v_min_lv_pu.min()), "v_max_pu": float(st.v_max_lv_pu.max())},
        "losses_energy": {
            "e_in_mwh": e_in * STEP_H, "fraction_total": float(st.losses_mw.sum() / e_in),
            **{f"fraction_{k}": float(v.sum() / e_in) for k, v in parts.items()},
            **{f"{k}_mwh": float(v.sum() * STEP_H) for k, v in parts.items()},
            "dt_noload_note": "P0 x mean(v_hv^2) per DT (estimate); dt_load = DT total - no-load"},
        "losses_at_peak": {"timestamp": str(peak.timestamp), "p_load_mw": float(peak.p_load_mw),
                           "p_in_mw": float(peak.p_in_mw), "losses_mw": float(peak.losses_mw),
                           "line_losses_mw": float(peak.line_losses_mw),
                           "dt_losses_mw": float(peak.dt_losses_mw),
                           "dt_noload_mw": float(peak.dt_noload_mw),
                           "sub_losses_mw": float(peak.sub_losses_mw),
                           "fraction_total": float(peak.losses_mw / peak.p_in_mw),
                           "fraction_mv": mv_peak},
        "vuf_max_11kv_pct_over_window": float(st.vuf_max_11kv_pct.max()),
        "feeder_balance_max_abs_mw": float((st.p_head_mw - st.p_dt_hv_mw - st.line_losses_mw).abs().max()),
        "oltc": {"tap_min": float(st.oltc_tap.min()), "tap_max": float(st.oltc_tap.max()),
                 "operations_per_day_mean": float(ops.mean()), "operations_per_day_max": int(ops.max()),
                 "hours_at_limit_per_day_mean": float(limit_h.mean()),
                 "hours_at_limit_per_day_max": float(limit_h.max()),
                 "days_reaching_limit": int((limit_h > 0).sum()),
                 "max_moves_per_step": int(st.tap_moved.max())},
        "dt_loading": dt_rows,
    }
    res["passed"] = res["band"]["passed"] and res["dt_loading_limit"]["passed"] and dtl["passed"]
    return res


SLDC_LABEL = "Delhi system-level reference, not residential"


def load_sldc_reference(path: Path | str = config.SLDC_REFERENCE_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(load-duration curve, monthly peaks) from the SLDC Delhi 2024-25 transcription."""
    t = pd.read_csv(path, comment="#")
    ldc = t[t.table == "ldc"][["level_mw", "hours", "pct_time"]].astype(float).reset_index(drop=True)
    pk = t[t.table == "monthly_peak"][["date", "time", "demand_met_mw", "unrestricted_mw"]]
    return ldc, pk.reset_index(drop=True)


def shape_validation(state: pd.DataFrame, info: Optional[dict] = None) -> dict:
    """Brief v2 X.3.3 item 1 validation (reported only, never asserted).

    (i) eMARC median daily kWh by region vs the SGCC eligible-pool median;
    (ii) normalised (/ peak) feeder load-duration curve vs SLDC 2024-25 (pp. 79-80);
    (iii) monthly peak time of the feeder load vs SLDC (p. 67).
    (ii) and (iii) are a Delhi system-level reference, not residential.
    """
    out: dict = {}
    if info is not None:
        out["i_daily_kwh"] = {
            "emarc_median_by_region": info["emarc"]["median_daily_kwh_by_region"],
            "emarc_median_all": info["emarc"]["median_daily_kwh_all"],
            "sgcc_eligible_median": info["eligibility"]["median_kwh_eligible"]["median"]}
    ldc, pk = load_sldc_reference()
    p = state.set_index(pd.DatetimeIndex(state.timestamp)).p_load_mw
    rel = ldc.level_mw / 8656.0                                   # SLDC year peak (p. 79)
    sim_pct = [float(100 * (p / p.max() > r).mean()) for r in rel]
    diff = np.array(sim_pct) - ldc.pct_time.to_numpy()
    out["ii_load_duration"] = {
        "label": SLDC_LABEL, "feeder_peak_mw": float(p.max()),
        "feeder_min_over_peak": float(p.min() / p.max()), "sldc_min_over_peak": 1450 / 8656,
        "pct_time_above": [{"level_over_peak": round(float(r), 4), "sldc_pct": float(s),
                            "feeder_pct": round(f, 2)}
                           for r, s, f in zip(rel[::6], ldc.pct_time[::6], sim_pct[::6])],
        "max_abs_diff_pct_time": float(np.abs(diff).max()),
        "mean_abs_diff_pct_time": float(np.abs(diff).mean())}
    months = []
    for r in pk.itertuples():
        m = pd.Timestamp(r.date).month
        pm = p[p.index.month == m]
        if pm.empty:
            continue
        t = pm.idxmax()
        months.append({"month": m, "feeder_peak_time": t.strftime("%H:%M"),
                       "feeder_peak_date": str(t.date()), "sldc_peak_time": r.time[:5]})
    out["iii_monthly_peak_time"] = {"label": SLDC_LABEL, "months": months}
    return out


def design_precheck(pop: Population) -> pd.DataFrame:
    """Per DT phase, the maximum over the window of |S| = |sum of household
    load| (each household's own eMARC shape) vs rating/3, before any power
    flow (no voltage drop), with the step at which it occurs."""
    dts = pop.net["dts"]
    best = np.zeros((len(dts), 3))
    when = np.empty((len(dts), 3), dtype=object)
    spd = pop.hh.steps_per_day
    for d in range(len(pop.days)):
        p, q = pop.hh.dt_phase_kw_day(d)
        s = np.hypot(p, q)                                           # (dt, 3, steps)
        mx, arg = s.max(axis=2), s.argmax(axis=2)
        upd = mx > best
        best[upd] = mx[upd]
        for k, j in zip(*np.nonzero(upd)):
            when[k, j] = str(pop.days[d] + arg[k, j] * pd.Timedelta(hours=24 / spd))
    rows = []
    for k, r in enumerate(dts.itertuples()):
        rows.append({"dt": r.name, "rating_kva": r.rating_kva,
                     **{f"{ph}_pct": round(float(100 * best[k, j] / (r.rating_kva / 3)), 1)
                        for j, ph in enumerate("abc")},
                     **{f"{ph}_at": when[k, j] for j, ph in enumerate("abc")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# lambda sweep: un-netted DT-level energy balance
# --------------------------------------------------------------------------

def lv_loss_kw(p_kw: np.ndarray, lam: float) -> np.ndarray:
    """LV technical loss at fraction ``lam``: I^2R-shaped (proportional to P^2),
    scaled so the loss over the window is ``lam`` x the DT's energy.
    ``p_kw``: (steps, n_dt) DT total LV load."""
    s2 = (p_kw ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(s2 > 0, lam * p_kw.sum(axis=0) / s2, 0.0)
    return p_kw ** 2 * k


def _pr_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    order = np.argsort(-score, kind="stable")
    y = label[order].astype(float)
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def lambda_sweep(cache: Path | str = config.YEAR_CACHE_DIR, lambdas=LAMBDAS,
                 out_dir: Optional[Path | str] = None) -> dict:
    """DT-day energy balance dE = E_DT - sum(E_consumer) for each lambda.

    Primary result: dE against the true theft energy per DT-day (bias, MAE,
    correlation).  Theft truth (Layer 3 P3, N5): true - would-have-reported on
    attack-active days, offline days included; offline energy = the
    would-have-reported value on offline days (``household_daily_labels``).  The share of DT-days with active theft is reported first;
    PR / AUC-PR only where it is below 90 %.  Labelled as the un-netted
    DT-level baseline of the Evaluation Plan, not the Layer 3 engine.
    Reported, not asserted (brief v2 X.3.1 item 6).  Every metric is given
    for three label sets (X.3.1 item 4 / X.3.2 item 20): ``all_attacks``,
    ``without_h6`` and ``without_h5_h6`` (the load-shifting attacks); in a
    set without an attack its thieves count as neither active nor theft.
    """
    cache = Path(cache)
    out = Path(out_dir) if out_dir else cache / "lambda_sweep"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(cache / "meter_frame_true.parquet")
    dts = pd.read_parquet(cache / "dts.parquet")
    daily = pd.read_parquet(cache / "household_daily.parquet")
    hhs = pd.read_parquet(cache / "households.parquet")[["household", "dt", "attack"]]
    dt_rows = frame[frame.load_id.str.startswith("dt_")]
    p = (dt_rows.pivot_table(index="timestamp", columns="load_id", values="active_power_kw",
                             aggfunc="sum")[dts.name])
    ts = p.index
    day = pd.DatetimeIndex(ts).normalize()
    daily = daily.merge(hhs, on="household")
    # Layer 3 P3 (N5): theft truth = true - would-have-reported on attack-active days
    # (offline days included); the offline category is the would-have-reported value.
    lab = pd.read_parquet(cache / LABELS_FILE, columns=["household", "date", "offline",
                                                        "theft_kwh", "offline_kwh"])
    daily = daily.merge(lab, on=["household", "date"], how="left", validate="one_to_one")
    if daily.theft_kwh.isna().any():
        raise ValueError(f"{LABELS_FILE} does not cover household_daily")
    label_sets = {"all_attacks": (), "without_h6": ("h6",), "without_h5_h6": ("h5", "h6")}
    for name, excl in label_sets.items():
        act = daily.attack_active & ~daily.attack.isin(excl)
        daily[f"active_{name}"] = act
        daily[f"theft_kwh_{name}"] = np.where(act, daily.theft_kwh, 0.0)
    grp = daily.groupby(["dt", "date"])
    cons = grp.metered_kwh.sum(min_count=1).fillna(0.0)       # head-end: offline = no data
    truth = pd.DataFrame({"consumer_metered_kwh": cons,
                          "offline_kwh": grp.offline_kwh.sum(),
                          "offline_meters": grp.offline.sum(),
                          **{f"theft_kwh_{n}": grp[f"theft_kwh_{n}"].sum() for n in label_sets},
                          **{f"thieves_active_{n}": grp[f"active_{n}"].sum() for n in label_sets}})
    results = {"label": SWEEP_LABEL, "asserted": False, "lambdas": list(lambdas),
               "label_sets": {k: list(v) for k, v in label_sets.items()}}
    tables = []
    for lam in lambdas:
        e_dt = (p.to_numpy() + lv_loss_kw(p.to_numpy(), lam)) * STEP_H * dts.meter_gain.to_numpy()
        e_dt = pd.DataFrame(e_dt, index=ts, columns=dts.name)
        e_dt.to_parquet(out / f"dt_meter_kwh_lambda_{lam:.2f}.parquet")
        e_day = e_dt.groupby(day).sum().stack().rename("dt_metered_kwh")
        e_day.index.names = ["date", "dt"]
        t = truth.join(e_day.swaplevel().rename_axis(["dt", "date"]), how="inner")
        t["delta_e_kwh"] = t.dt_metered_kwh - t.consumer_metered_kwh
        t["lambda"] = lam
        tables.append(t.reset_index())
        res = {}
        for ls in label_sets:
            res[ls] = {}
            for subset, mask in (("all_dt_days", np.ones(len(t), bool)),
                                 ("dt_days_all_meters_online", (t.offline_meters == 0).to_numpy())):
                s = t[mask]
                if len(s) < 2:
                    res[ls][subset] = {"n": int(len(s))}
                    continue
                theft = s[f"theft_kwh_{ls}"]
                err = s.delta_e_kwh - theft
                active = s[f"thieves_active_{ls}"] > 0
                share = float(active.mean())
                r = {"n": int(len(s)), "share_with_active_theft": share,
                     "bias_kwh": float(err.mean()), "mae_kwh": float(err.abs().mean()),
                     "corr": float(np.corrcoef(s.delta_e_kwh, theft)[0, 1])
                     if theft.std() > 0 else float("nan"),
                     "mean_theft_kwh": float(theft.mean()),
                     "mean_offline_kwh": float(s.offline_kwh.mean())}
                if share < 0.9 and active.any():
                    r["auc_pr"] = _pr_auc(s.delta_e_kwh.to_numpy(), active.to_numpy())
                    r["auc_pr_baseline"] = share
                else:
                    r["auc_pr"] = "uninformative at DT level (share of DT-days with active theft >= 90 %)"
                res[ls][subset] = r
        results[f"{lam:.2f}"] = res
    pd.concat(tables).to_parquet(out / "dt_day_balance.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
