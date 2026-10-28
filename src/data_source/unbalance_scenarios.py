"""Voltage-unbalance test scenarios (Layer 3 P2; Evaluation Plan §1).

No 11 kV bus exceeds the IS 17036 2 % limit in the 2024 run, so the unbalance
alarm is tested on scenarios with deliberately asymmetric LV phase loading:

- **Week:** one representative week of the CP3 cache, the complete Monday-Sunday
  week of 2024 whose feeder-head energy is the median of the 52 complete weeks
  (lower middle; :func:`representative_week`): typical loading, neither the
  winter design peak nor the summer overload weeks.
- **Base load:** each DT's LV load per phase and step from the cached true
  frame (the DT meter at the LV terminals carries exactly the DT load, as in
  :func:`faults.steps_from_cache`).
- **Asymmetry:** for each selected DT a fraction s of the load on its other two
  LV phases moves onto one phase (P and Q alike), so the DT's total load is
  unchanged.  s and the choice of DTs and phases are **scenario parameters,
  not claims about real feeders** (stated in the metadata).  The sweep is
  seeded (stream ``unbalance.scenarios``): s in {0, 10, 20, 30, 40, 60} %; for
  each s > 0, three scenarios shift every DT onto the same phase (a, b, c) and
  three draw a random DT subset with a random phase per DT.
- **Power flow:** :func:`feeder_model.run_timeseries` (``runpp_3ph``, OLTC
  active, starting from the cached tap of the step before the week).
- **Labels** (evaluation only, never engine inputs): per step and physical
  meter the true V_N / V_P from the full line-to-line phasors; per scenario
  and meter the IS 17036 weekly flag, VU > 2 % for more than 5 % of the week's
  15-min steps (Table 8, 95 % of each week).
- **Engine inputs:** the metered frame in the ``meter_frame_metered.parquet``
  format (meter errors as :func:`simulation.apply_meter_errors`, with the
  year's own meter gains).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

OUT_DIR = config.PROJECT_ROOT / "results" / "unbalance_scenarios"
S_SWEEP = (0.0, 0.10, 0.20, 0.30, 0.40, 0.60)          # scenario parameters (prompt P2 example)
N_COMMON, N_RANDOM = 3, 3                               # per s > 0: all DTs onto a / b / c; random subsets
VU_LIMIT_PCT = 2.0                                      # IS 17036:2018 Cl. 4.2.4, Table 8, PDF p. 7
WEEK_EXCEED_SHARE = 0.05                                # Table 8: limit holds for 95 % of each week
PHYSICAL_ROLES = ("feeder_head", "dt")
LABELS_FILE = "unbalance_labels.parquet"                # per step and meter (evaluation only)
WEEKLY_LABELS_FILE = "unbalance_weekly_labels.parquet"  # per scenario and meter (evaluation only)
SCENARIO_NOTE = ("s and the choice of DTs and phases are scenario parameters for testing the "
                 "IS 17036 unbalance alarm, not claims about real feeders")
_A = np.exp(2j * np.pi / 3)


@dataclass
class Scenario:
    scenario_id: str
    s: float
    kind: str                         # baseline | common_phase | random
    shift: dict = field(default_factory=dict)       # DT name -> LV phase receiving the load


def representative_week(cache: Path = config.YEAR_CACHE_DIR) -> tuple[pd.Timestamp, pd.Timestamp, dict]:
    """(first step, last step, info) of the median-energy complete Mon-Sun week."""
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "active_power_kw"],
                        filters=[("load_id", "==", "feeder_head_650")])
    p = f.groupby("timestamp").active_power_kw.sum()
    monday = p.index.normalize() - pd.to_timedelta(p.index.weekday, unit="D")
    e = (p * 0.25).groupby(monday).agg(["sum", "size"])
    e = e[e["size"] == 7 * 96]["sum"].sort_values()
    start = pd.Timestamp(e.index[(len(e) - 1) // 2])
    info = {"rule": "complete Monday-Sunday week of 2024 with the median feeder-head energy "
                    "(lower middle of the sorted weekly energies)",
            "complete_weeks": int(len(e)), "week_energy_mwh": float(e.loc[start] / 1e3),
            "min_week_mwh": float(e.iloc[0] / 1e3), "max_week_mwh": float(e.iloc[-1] / 1e3)}
    return start, start + pd.Timedelta(days=7) - pd.Timedelta(minutes=15), info


def make_scenarios(dt_names: list[str], rng: np.random.Generator | None = None) -> list[Scenario]:
    rng = config.rng("unbalance.scenarios") if rng is None else rng
    out = [Scenario("s00_base", 0.0, "baseline")]
    for s in S_SWEEP[1:]:
        tag = f"s{int(round(s * 100)):02d}"
        for ph in "abc"[:N_COMMON]:
            out.append(Scenario(f"{tag}_all_{ph}", s, "common_phase", {n: ph for n in dt_names}))
        for k in range(N_RANDOM):
            n_sel = int(rng.integers(1, len(dt_names) + 1))
            sel = sorted(rng.choice(dt_names, n_sel, replace=False).tolist(), key=dt_names.index)
            out.append(Scenario(f"{tag}_rand{k + 1}", s, "random",
                                {n: str(rng.choice(list("abc"))) for n in sel}))
    return out


def shift_loads(p: np.ndarray, q: np.ndarray, rows: np.ndarray, phase: np.ndarray,
                s: float) -> tuple[np.ndarray, np.ndarray]:
    """Move fraction ``s`` of the other two phases' load onto ``phase`` for DT ``rows``."""
    p, q = p.copy(), q.copy()
    for r, j in zip(rows, phase):
        for x in (p, q):
            others = [k for k in range(3) if k != j]
            moved = s * x[r, others]
            x[r, others] -= moved
            x[r, j] += moved.sum()
    return p, q


def base_week(net, start: pd.Timestamp, end: pd.Timestamp, cache: Path = config.YEAR_CACHE_DIR):
    """(timestamps, P (steps, n_dt, 3) kW, Q kvar, tap of the step before ``start``)."""
    names = list(net["dts"].name)
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "phase", "active_power_kw", "reactive_power_kvar"],
                        filters=[("timestamp", ">=", start), ("timestamp", "<=", end), ("load_id", "in", names)])
    ts = pd.date_range(start, end, freq=config.SIMULATION_FREQ)
    out = []
    for col in ("active_power_kw", "reactive_power_kvar"):
        w = f.pivot_table(index="timestamp", columns=["load_id", "phase"], values=col, aggfunc="first")
        w = w.reindex(index=ts, columns=pd.MultiIndex.from_product([names, list("abc")]))
        out.append(w.to_numpy(float).reshape(len(ts), len(names), 3))
    prev = start - pd.Timedelta(minutes=15)
    st = pd.read_parquet(Path(cache) / "state.parquet", columns=["timestamp", "oltc_tap"],
                         filters=[("timestamp", "==", prev)])
    return ts, out[0], out[1], float(st.oltc_tap.iloc[0])


def run_scenario(net, scen: Scenario, ts, p0, q0, tap0: float):
    """Solve the week for one scenario; returns (true frame, state)."""
    from .feeder_model import run_timeseries
    names = list(net["dts"].name)
    rows = np.array([names.index(n) for n in scen.shift], int)
    phase = np.array(["abc".index(ph) for ph in scen.shift.values()], int)

    def dt_loads(i, _ts):
        return shift_loads(p0[i], q0[i], rows, phase, scen.s)

    t = int(net["oltc"].trafo.iloc[0])
    net.trafo.at[t, "tap_pos"] = tap0
    return run_timeseries(ts[0], ts[-1], net=net, return_state=True, dt_loads=dt_loads)


def vu_true(frame: pd.DataFrame) -> pd.DataFrame:
    """Per step and physical meter: V_N / V_P x 100 % from the full line-to-line phasors."""
    from .feeder_model import FEEDER_HEAD_ID
    phys = frame[(frame.load_id == FEEDER_HEAD_ID) | frame.load_id.str.startswith("dt_")]
    v = phys.voltage_v * np.exp(1j * np.deg2rad(phys.voltage_angle_deg))
    w = phys.assign(v=v).pivot(index=["load_id", "timestamp"], columns="phase", values="v")
    va, vb, vc = (w[p].to_numpy() for p in "abc")
    vab, vbc, vca = va - vb, vb - vc, vc - va
    v1 = (vab + _A * vbc + _A ** 2 * vca) / 3
    v2 = (vab + _A ** 2 * vbc + _A * vca) / 3
    return pd.DataFrame({"load_id": w.index.get_level_values(0), "timestamp": w.index.get_level_values(1),
                         "vu_true_pct": 100 * np.abs(v2) / np.abs(v1)})


def weekly_flags(vu: pd.DataFrame) -> pd.DataFrame:
    """Per scenario and meter: share of steps above 2 % and the IS 17036 weekly flag."""
    g = vu.groupby(["scenario_id", "load_id"])
    out = g.vu_true_pct.agg(vu_max_pct="max", vu_p95_pct=lambda x: float(np.percentile(x, 95)),
                            steps="size").reset_index()
    out["share_above_limit"] = g.vu_true_pct.apply(lambda x: float((x > VU_LIMIT_PCT).mean())).to_numpy()
    out["exceeds_limit"] = out.share_above_limit > WEEK_EXCEED_SHARE
    return out


def generate(out_dir: Path | str = OUT_DIR, cache: Path = config.YEAR_CACHE_DIR) -> dict:
    """Generate every scenario once and write ``out_dir`` (prompt P2 item 7)."""
    import time
    import warnings

    from .feeder_model import build_network
    from .simulation import apply_meter_errors
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore", FutureWarning)
    net = build_network("B")
    names = list(net["dts"].name)
    start, end, week_info = representative_week(cache)
    ts, p0, q0, tap0 = base_week(net, start, end, cache)
    dts = pd.read_parquet(Path(cache) / "dts.parquet", columns=["name", "meter_gain"])
    gains = dict(zip(dts.name, dts.meter_gain))
    g_head = float(json.loads((Path(cache) / "run_info.json").read_text(encoding="utf-8"))["feeder_head_gain"])
    scens = make_scenarios(names)
    vus, states, timing = [], [], {}
    for sc in scens:
        t0 = time.perf_counter()
        frame, state = run_scenario(net, sc, ts, p0, q0, tap0)
        timing[sc.scenario_id] = round(time.perf_counter() - t0, 1)
        apply_meter_errors(frame, gains, g_head).to_parquet(out / "frames" / f"{sc.scenario_id}.parquet",
                                                             index=False)
        if sc.kind == "baseline":
            frame.to_parquet(out / "baseline_true_frame.parquet", index=False)
        vus.append(vu_true(frame).assign(scenario_id=sc.scenario_id))
        states.append(state.assign(scenario_id=sc.scenario_id))
        logger.info("scenario %s done (%.1f s)", sc.scenario_id, timing[sc.scenario_id])
    vu = pd.concat(vus, ignore_index=True)
    vu.to_parquet(out / LABELS_FILE, index=False)
    weekly = weekly_flags(vu)
    weekly.to_parquet(out / WEEKLY_LABELS_FILE, index=False)
    pd.concat(states, ignore_index=True).to_parquet(out / "states.parquet", index=False)
    meta = {"note": SCENARIO_NOTE, "week": [str(start), str(end)], "week_choice": week_info,
            "initial_tap": tap0, "s_sweep": list(S_SWEEP), "seed_stream": "unbalance.scenarios",
            "master_seed": config.MASTER_SEED,
            "limit": {"vu_pct": VU_LIMIT_PCT, "exceed_share": WEEK_EXCEED_SHARE,
                      "source": "IS 17036:2018 Cl. 4.2.4, Table 8, PDF p. 7 (95 % of each week)"},
            "scenarios": [{"scenario_id": s.scenario_id, "s": s.s, "kind": s.kind, "shift": s.shift}
                          for s in scens],
            "time_s": timing,
            "inputs": "frames/<scenario_id>.parquet (meter_frame_metered format; engine inputs)",
            "labels": [LABELS_FILE, WEEKLY_LABELS_FILE, "baseline_true_frame.parquet", "states.parquet"]}
    (out / "scenarios.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
