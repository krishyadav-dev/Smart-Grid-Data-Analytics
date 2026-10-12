"""Phase 2B report run: Stage B step by step, sensitivity and two simulated weeks.

- Writes a pandapower JSON snapshot after every step to results/stage_b_snapshots/.
- Prints each step's effect at nominal load (max |dV| vs the previous step,
  input P/Q, losses, charging kvar).
- B9 sensitivity: final Stage B with the Stage A Z/I load models.
- Nominal metrics (loadings, voltages, 3-phase fault currents).
- Synthetic week and SGCC-driven week (small SGCC file, 2024-01-01..07):
  state frames to results/stage_b_week_{synthetic,sgcc}.csv; voltage band
  (IS 17036 +/-10 % and the +/-5 % design target), tap operations per day,
  hours at the tap limit, steps out of band at the tap limit.

    python scripts/run_stage_b.py
"""

from __future__ import annotations

import copy
import logging
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandapower as pp  # noqa: E402
import pandapower.shortcircuit as sc  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_source import config, dataset_loader, oltc  # noqa: E402
from src.data_source.feeder_model import (  # noqa: E402
    build_network, get_load_points, run_timeseries, solve_with_load_models,
)
from src.data_source.stage_b import STEPS, build_stage_b  # noqa: E402

OUT = ROOT / "results"
VCOLS = ["vm_a_pu", "vm_b_pu", "vm_c_pu"]


def _named_v(net):
    v = net.res_bus_3ph[VCOLS].copy()
    v.index = net.bus.name.loc[v.index]
    return v


def _sum3(df, fmt):
    return float(sum(df[fmt.format(p)].sum() for p in "abc"))


def _charging_kvar(net):
    return sum(net.bus.vn_kv[r.from_bus] ** 2 * 2 * math.pi * net.f_hz * r.c_nf_per_km
               * 1e-9 * r.length_km * 1e3 for _, r in net.line.iterrows())


def step_table() -> pd.DataFrame:
    rows, prev = [], None
    a = build_network("A")
    solve_with_load_models(a)
    nets = [("A", a)] + [(s, build_stage_b(upto=s, snapshot_dir=OUT / "stage_b_snapshots"))
                         for s in STEPS]
    for name, net in nets:
        if name != "A":
            solve_with_load_models(net)
        v = _named_v(net)
        p_in = _sum3(net.res_ext_grid_3ph, "p_{}_mw")
        p_ld = _sum3(net.asymmetric_load[net.asymmetric_load.name.str.startswith("load_")], "p_{}_mw")
        dv = np.nan if prev is None else float(np.nanmax(np.abs(
            100 * (v.loc[prev.index] - prev) / prev).values))
        rows.append({"step": name, "max_dV_vs_prev_pct": dv,
                     "v_min": float(np.nanmin(v.drop(index=["sourcebus"], errors="ignore").values)),
                     "v_max": float(np.nanmax(v.drop(index=["sourcebus"], errors="ignore").values)),
                     "p_in_kw": p_in * 1e3,
                     "q_in_kvar": _sum3(net.res_ext_grid_3ph, "q_{}_mvar") * 1e3,
                     "losses_pct": 100 * (p_in - p_ld) / p_in,
                     "charging_kvar": _charging_kvar(net)})
        prev = v
    return pd.DataFrame(rows)


def nominal_metrics(net) -> dict:
    pp.runpp_3ph(net)
    v = _named_v(net)
    lp = dict(zip(net.trafo.name, net.res_trafo_3ph.loading_percent))
    p_in = _sum3(net.res_ext_grid_3ph, "p_{}_mw")
    p_ld = _sum3(net.res_asymmetric_load_3ph, "p_{}_mw")
    out = {"losses_pct": 100 * (p_in - p_ld) / p_in, "sub_loading_pct": lp["Sub_XFMR"],
           "xfm1_loading_pct": lp["XFM-1"],
           "oltc_tap": float(net.trafo.at[int(net["oltc"].trafo.iloc[0]), "tap_pos"]),
           "v_relay_pu": oltc.relay_voltage(net)[0]}
    for b in ("632", "634", "675"):
        out[f"v_{b}"] = v.loc[b].round(4).tolist()
    sc.calc_sc(net, fault="3ph", case="max")
    names = net.bus.name
    for b in ("632", "634", "675", "652"):
        out[f"ik3_{b}_ka"] = float(net.res_bus_sc.at[names[names == b].index[0], "ikss_ka"])
    return out


def week(profiles: str) -> pd.DataFrame:
    net = build_network("B")
    start, end = "2024-01-01 00:00", "2024-01-07 23:45"
    prof = None
    if profiles == "sgcc":
        sgcc = dataset_loader.load_sgcc(str(config.SGCC_SMALL_FALLBACK_PATH))
        prof = dataset_loader.sample_load_profiles(get_load_points(net), sgcc, start=start, end=end)
    _, st = run_timeseries(start, end, net=net, profiles=prof, return_state=True)
    st.to_csv(OUT / f"stage_b_week_{profiles}.csv", index=False)
    return st


def summarise_week(name: str, st: pd.DataFrame) -> None:
    days = st.groupby(st.timestamp.dt.date)
    stuck = st[st.at_tap_limit & ((st.v_relay_pu - oltc.SET_POINT_PU).abs() > oltc.HALF_BAND_PU)]
    viol_steps = (st.v_below_095 + st.v_above_105).gt(0)
    print(f"\n== {name} week ==")
    print(f"11 kV V min {st.v_min_11kv_pu.min():.4f} pu, max {st.v_max_11kv_pu.max():.4f} pu; "
          f"outside 0.90-1.10: {int(((st.v_min_11kv_pu < 0.90) | (st.v_max_11kv_pu > 1.10)).sum())} steps")
    print(f"+/-5 % target: {int(viol_steps.sum())} steps with violations, "
          f"{int(st.v_below_095.sum())} bus-phase-steps < 0.95, {int(st.v_above_105.sum())} > 1.05")
    print(f"tap range {st.oltc_tap.min():+.0f}..{st.oltc_tap.max():+.0f}; operations/day "
          f"{days.tap_moved.sum().astype(int).tolist()}")
    print(f"hours/day at tap limit {(days.at_tap_limit.sum() * 0.25).tolist()}; "
          f"steps out of band at the limit: {len(stuck)}")
    print(f"losses {100 * (st.losses_mw / st.p_in_mw).min():.2f}-{100 * (st.losses_mw / st.p_in_mw).max():.2f} %; "
          f"max |energy balance| {(st.losses_mw - st.branch_losses_mw).abs().max():.2e} MW")


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.WARNING)
    pd.set_option("display.width", 160)
    OUT.mkdir(exist_ok=True)
    print("== Steps (nominal load, loads with their own models) ==")
    print(step_table().round(4).to_string(index=False))

    final = build_network("B")
    pp.runpp_3ph(final)
    zi = copy.deepcopy(final)
    a = build_network("A")
    zi.asymmetric_load["load_model"] = zi.asymmetric_load.name.map(
        dict(zip(a.asymmetric_load.name, a.asymmetric_load.load_model)))
    solve_with_load_models(zi)
    dv = 100 * (_named_v(zi) - _named_v(final)) / _named_v(final)
    loss = lambda n: _sum3(n.res_ext_grid_3ph, "p_{}_mw") - _sum3(n.asymmetric_load, "p_{}_mw")  # noqa: E731
    print(f"\n== B9 sensitivity == max |dV| {np.nanmax(np.abs(dv.values)):.3f} %, "
          f"losses PQ {loss(final) * 1e3:.1f} kW vs Z/I {loss(zi) * 1e3:.1f} kW")

    print("\n== Stage B nominal ==")
    for k, v in nominal_metrics(build_network("B")).items():
        print(f"{k}: {v}")

    for name in ("synthetic", "sgcc"):
        summarise_week(name, week(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
