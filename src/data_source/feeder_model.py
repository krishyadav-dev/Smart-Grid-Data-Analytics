"""
IEEE 13-node feeder — unbalanced three-phase power flow (pandapower ``runpp_3ph``).

Two modelling stages (docs/Phase_2_Modeling_Decisions_Brief.md):

- **Stage A** — the IEEE 13-node test feeder as published (4.16 kV, 60 Hz),
  built only from :mod:`ieee13_data` and validated against Kersting's
  published power flow (Phase 2A, docs/build_report_2A.md).
- **Stage B** — the Indian 11 kV feeder, derived from Stage A by the
  documented transformations B1–B9 in :mod:`stage_b` (Phase 2B): 11 kV, line
  lengths x (11/4.16)^2, 50 Hz, 66/11 kV substation transformer with a
  controlled OLTC and line-drop compensation (:mod:`oltc`), three-phase
  laterals and capacitors, constant-PQ loads.  This is the network the meters,
  :func:`run_timeseries` and Layer 2 use.

The network is genuinely **unbalanced**; the solver is always
``pandapower.runpp_3ph`` — **never** the balanced ``runpp()``.

References
----------
1.  Kersting, W.H., "Radial Distribution Test Feeders," IEEE Trans. Power
    Systems, vol. 6, no. 3, pp. 975–985, 1991.
2.  IEEE PES Distribution System Analysis Subcommittee, IEEE 13 Node Test
    Feeder (2004).
3.  pandapower ``runpp_3ph`` and ``shortcircuit.calc_sc`` documentation.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from . import config
from . import ieee13_data as ieee
from . import oltc
from .line_sequence import sequence_params

logger = logging.getLogger(__name__)

# Mapping from node name → bus index of the last Stage B network built
# (used by inject_fault to resolve bus names).
_NODE_INDEX: Dict[str, int] = {}


# ===================================================================== #
#                        NETWORK CONSTRUCTION                            #
# ===================================================================== #

def build_ieee13_network() -> pp.pandapowerNet:
    """Build the Stage B feeder (alias of ``build_network("B")``).

    Raises
    ------
    AssertionError
        If total connected load deviates from 3.466 MW / 2.102 Mvar by more
        than 0.5 %.
    """
    return build_network("B")


def _check_load_totals(net: pp.pandapowerNet) -> None:
    """Total connected load = IEEE spot + distributed load (Kersting 1991)."""
    loads = net.asymmetric_load[net.asymmetric_load.name.str.startswith("load_")]
    total_p = loads[["p0_a_mw", "p0_b_mw", "p0_c_mw"]].sum().sum()
    total_q = loads[["q0_a_mvar", "q0_b_mvar", "q0_c_mvar"]].sum().sum()
    for label, total, expected, unit in (
        ("load", total_p, config.TOTAL_LOAD_MW_ASSERTION, "MW"),
        ("reactive load", total_q, config.TOTAL_LOAD_MVAR_ASSERTION, "Mvar"),
    ):
        tolerance = config.LOAD_TOLERANCE_FRACTION * expected
        assert abs(total - expected) < tolerance, (
            f"Total connected {label} {total:.3f} {unit} deviates from "
            f"expected {expected} {unit} by more than "
            f"{config.LOAD_TOLERANCE_FRACTION * 100:.1f} %"
        )


# ===================================================================== #
#              STAGE A: IEEE 13-NODE FEEDER AS PUBLISHED (2A)            #
# ===================================================================== #

FT_TO_KM = 0.0003048

# Regulator 650 -> RG60 as one YNyn transformer.  runpp_3ph has one tap per
# transformer, so the published per-phase taps 10/8/11 (ieee13_data,
# l. 259-261) are replaced by their mean: ratio 1 + 0.00625 * 29/3 = 1.0604.
# The IEEE regulator has no impedance (published loss 0, l. 275), but
# pandapower needs vk > 0: numerical stand-in for an ideal regulator.
REG_STANDIN_SN_MVA = 5.0       # numerical stand-in for an ideal regulator
REG_STANDIN_VK_PERCENT = 0.01  # numerical stand-in for an ideal regulator


def build_network(stage: str = "B") -> pp.pandapowerNet:
    """Build the feeder for one modelling stage.

    ``"A"``: the IEEE 13-node feeder as published (4.16 kV, 60 Hz), the
    validation base.  ``"B"``: the Indian 11 kV feeder used by the meters and
    the time series, derived from stage A by :func:`stage_b.build_stage_b`
    (also returned by :func:`build_ieee13_network`).
    """
    if stage == "A":
        return _build_stage_a()
    if stage == "B":
        from .stage_b import build_stage_b
        net = build_stage_b()
        _check_load_totals(net)
        _NODE_INDEX.clear()
        _NODE_INDEX.update({str(n): int(i) for i, n in net.bus.name.items()})
        logger.info(
            "Stage B network built: %d buses, %d lines, %.3f MW total load",
            len(net.bus), len(net.line),
            net.asymmetric_load[["p0_a_mw", "p0_b_mw", "p0_c_mw"]].sum().sum(),
        )
        return net
    raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")


def _phase_str(p: tuple, q: tuple) -> str:
    return "".join(ph for ph, pv, qv in zip("abc", p, q) if pv or qv) or "abc"


def _add_modelled_load(net, bus: int, name: str, conn: str, model: str,
                       p_kw: tuple, q_kvar: tuple) -> int:
    """Asymmetric load with nominal values and model kept for 2A.2."""
    idx = pp.create_asymmetric_load(
        net, bus=bus,
        p_a_mw=p_kw[0] / 1e3, p_b_mw=p_kw[1] / 1e3, p_c_mw=p_kw[2] / 1e3,
        q_a_mvar=q_kvar[0] / 1e3, q_b_mvar=q_kvar[1] / 1e3,
        q_c_mvar=q_kvar[2] / 1e3,
        type="wye" if conn == "Y" else "delta", name=name,
    )
    for ph, pv, qv in zip("abc", p_kw, q_kvar):
        net.asymmetric_load.at[idx, f"p0_{ph}_mw"] = pv / 1e3
        net.asymmetric_load.at[idx, f"q0_{ph}_mvar"] = qv / 1e3
    net.asymmetric_load.at[idx, "load_model"] = model
    return idx


def _build_stage_a() -> pp.pandapowerNet:
    """IEEE 13-node feeder from the published data only (Phase 2A).

    Source: external grid directly at 650, 1.0 pu, 4.16 kV, as in the
    published power flow (650 is the source node, l. 235, 271-272); no
    substation transformer (it enters in Stage B, step B4).  All values come
    from :mod:`ieee13_data`.
    """
    net = pp.create_empty_network(
        name="IEEE 13-node (Stage A, 4.16 kV, 60 Hz)",
        f_hz=ieee.FREQUENCY_HZ, sn_mva=1.0,
    )
    names = ["650", "RG60", "632", "633", "634", "645", "646", "670", "671",
             "680", "684", "611", "652", "692", "675"]
    B = {
        n: pp.create_bus(
            net, name=n,
            vn_kv=ieee.V_XFM1_LV_KV if n == "634" else ieee.V_PRIMARY_KV)
        for n in names
    }

    # Short-circuit data is required by runpp_3ph but not published: verify
    # (same assumed values as stage B, audit §5).
    pp.create_ext_grid(
        net, bus=B["650"], vm_pu=1.0, name="Source_650",
        s_sc_max_mva=1000.0, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1,
        s_sc_min_mva=800.0, rx_min=0.1, x0x_min=1.0, r0x0_min=0.1,
    )

    taps = ieee.REGULATOR["published_taps"]
    pp.create_transformer_from_parameters(
        net, hv_bus=B["650"], lv_bus=B["RG60"], sn_mva=REG_STANDIN_SN_MVA,
        vn_hv_kv=ieee.V_PRIMARY_KV, vn_lv_kv=ieee.V_PRIMARY_KV,
        vkr_percent=0.0, vk_percent=REG_STANDIN_VK_PERCENT,
        pfe_kw=0.0, i0_percent=0.0, vector_group="YNyn",
        vk0_percent=REG_STANDIN_VK_PERCENT, vkr0_percent=0.0,
        mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9, shift_degree=0,
        tap_side="lv", tap_neutral=0, tap_min=-16, tap_max=16,
        tap_step_percent=ieee.REGULATOR_STEP_PU * 100,
        tap_pos=sum(taps) / len(taps), tap_changer_type="Ratio",
        name="Regulator_650_RG60",
    )

    x1 = ieee.XFM1
    vk = (x1["r_pct"] ** 2 + x1["x_pct"] ** 2) ** 0.5
    # Zero-sequence data is not published; vk0 = vk is an assumption: verify.
    pp.create_transformer_from_parameters(
        net, hv_bus=B["633"], lv_bus=B["634"], sn_mva=x1["kva"] / 1e3,
        vn_hv_kv=x1["kv_high"], vn_lv_kv=x1["kv_low"],
        vkr_percent=x1["r_pct"], vk_percent=vk, pfe_kw=0.0, i0_percent=0.0,
        vector_group="YNyn", vk0_percent=vk, vkr0_percent=x1["r_pct"],
        mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9, shift_degree=0,
        name="XFM-1",
    )

    for a, b, length_ft, cfg in ieee.LINE_SEGMENTS:
        if cfg == "XFM-1":
            continue
        if cfg == "Switch":
            pp.create_switch(net, bus=B[a], element=B[b], et="b", closed=True,
                             name=f"Switch_{a}_{b}")
            continue
        if (a, b) == ("650", "632"):
            a = "RG60"  # the regulator sits at the 650 end of this segment
        segs = [(a, b, length_ft)]
        if (a, b) == ("632", "671"):
            # Distributed load lumped at 670, one third along (Phase 1).
            segs = [("632", "670", 667), ("670", "671", 1333)]
        _, z_up, b_up = ieee.LINE_CONFIGS[cfg]
        sp = sequence_params(z_up, b_up, ieee.CONFIG_PHASES[cfg], net.f_hz)
        for fa, fb, ft in segs:
            pp.create_line_from_parameters(
                net, from_bus=B[fa], to_bus=B[fb], length_km=ft * FT_TO_KM,
                r_ohm_per_km=sp.r1_ohm_per_km, x_ohm_per_km=sp.x1_ohm_per_km,
                c_nf_per_km=sp.c1_nf_per_km, r0_ohm_per_km=sp.r0_ohm_per_km,
                x0_ohm_per_km=sp.x0_ohm_per_km, c0_nf_per_km=sp.c0_nf_per_km,
                max_i_ka=ieee.CONFIG_AMPACITY_KA[cfg],
                name=f"Line_{fa}_{fb}", config=cfg,
            )

    for bus, (conn, model, p_kw, q_kvar) in ieee.SPOT_LOADS.items():
        _add_modelled_load(net, B[bus], f"load_{bus}_{_phase_str(p_kw, q_kvar)}",
                           conn, model, p_kw, q_kvar)
    _, _, conn, model, p_kw, q_kvar = ieee.DISTRIBUTED_LOAD
    _add_modelled_load(net, B["670"], f"load_670_{_phase_str(p_kw, q_kvar)}",
                       conn, model, p_kw, q_kvar)

    # Capacitors: 675 is balanced -> shunt (constant-Z in runpp_3ph).  611 is
    # phase C only and runpp_3ph shunts are balanced, so it is a constant-Z
    # phase-C element with Q0 = -100 kvar, scaled by the load-model loop.
    cap675 = ieee.CAPACITORS["675"]
    pp.create_shunt(net, bus=B["675"], q_mvar=-sum(cap675) / 1e3, p_mw=0.0,
                    name="Cap_675")
    cap611 = ieee.CAPACITORS["611"]
    _add_modelled_load(net, B["611"], "cap_611_c", "Y", "Z",
                       (0.0, 0.0, 0.0), tuple(-q for q in cap611))
    return net


def _load_voltage_pu(net, idx: int) -> Tuple[float, float, float]:
    """Per-phase voltage magnitude seen by a load, in pu.

    Wye: line-to-neutral.  Delta: |Vx - Vy| / sqrt(3) for the pair that
    pandapower assigns to p_a / p_b / p_c (AB / BC / CA).
    """
    bus = int(net.asymmetric_load.at[idx, "bus"])
    r = net.res_bus_3ph.loc[bus]
    v = [r[f"vm_{p}_pu"] * np.exp(1j * np.deg2rad(r[f"va_{p}_degree"]))
         for p in "abc"]
    if net.asymmetric_load.at[idx, "type"] == "delta":
        return tuple(abs(v[i] - v[(i + 1) % 3]) / np.sqrt(3) for i in range(3))
    return tuple(abs(x) for x in v)


def solve_with_load_models(net: pp.pandapowerNet, tol: float = 1e-3,
                           max_iter: int = 10) -> int:
    """Run runpp_3ph with constant-Z / constant-I loads (Phase 2A.2).

    Outer iteration: solve, set P = P0 V^k and Q = Q0 V^k (k = 2 for Z, 1 for
    I, 0 for PQ) from the solved voltages, repeat until the largest change in
    any load P is below ``tol`` x total P.  Returns the number of power flows.
    Raises RuntimeError if it does not converge (no silent fallback).
    """
    exp = {"PQ": 0, "Z": 2, "I": 1}
    al = net.asymmetric_load
    p_total = float(al[[f"p0_{p}_mw" for p in "abc"]].abs().sum().sum())
    for n_pf in range(1, max_iter + 1):
        pp.runpp_3ph(net)
        max_dp = 0.0
        updates = []
        for idx in al.index:
            k = exp[al.at[idx, "load_model"]]
            vpu = _load_voltage_pu(net, idx)
            for ph, v in zip("abc", vpu):
                p_new = al.at[idx, f"p0_{ph}_mw"] * v ** k
                q_new = al.at[idx, f"q0_{ph}_mvar"] * v ** k
                max_dp = max(max_dp, abs(p_new - al.at[idx, f"p_{ph}_mw"]),
                             abs(q_new - al.at[idx, f"q_{ph}_mvar"]))
                updates.append((idx, ph, p_new, q_new))
        if max_dp < tol * p_total:
            logger.info("Load-model iteration converged after %d power flows", n_pf)
            return n_pf
        for idx, ph, p_new, q_new in updates:
            al.at[idx, f"p_{ph}_mw"] = p_new
            al.at[idx, f"q_{ph}_mvar"] = q_new
    raise RuntimeError(
        f"Load-model iteration did not converge in {max_iter} power flows")


# ===================================================================== #
#                          LOAD POINTS                                    #
# ===================================================================== #

def get_load_points(net: pp.pandapowerNet | None = None) -> pd.DataFrame:
    """Return a table of every load point in the network.

    This helper lets Part B (dataset_loader, meter_server) look up each
    load point's expected demand without importing pandapower.

    Parameters
    ----------
    net : pandapowerNet, optional
        Network.  If *None*, builds a fresh IEEE 13-bus network.

    Returns
    -------
    pd.DataFrame
        Columns: ``load_id`` (str, unique), ``bus`` (str), ``phase`` (str:
        ``'a'``, ``'b'``, ``'c'``, or ``'abc'``), ``nominal_kw`` (float),
        ``nominal_kvar`` (float).  One row per load point.
    """
    if net is None:
        net = build_ieee13_network()

    rows: list[dict] = []
    # Stage A keeps nominal values in p0_*/q0_* (p_*/q_* move with the load
    # models); stage B has no such columns.
    nom = "p0_a_mw" in net.asymmetric_load.columns
    for idx in net.asymmetric_load.index:
        load_id = net.asymmetric_load.at[idx, "name"]
        if not str(load_id).startswith("load_"):
            continue  # e.g. Stage A's single-phase capacitor cap_611_c
        bus_idx = int(net.asymmetric_load.at[idx, "bus"])
        bus_name = net.bus.at[bus_idx, "name"]

        pfx_p, pfx_q = ("p0", "q0") if nom else ("p", "q")
        pa = float(net.asymmetric_load.at[idx, f"{pfx_p}_a_mw"])
        pb = float(net.asymmetric_load.at[idx, f"{pfx_p}_b_mw"])
        pc = float(net.asymmetric_load.at[idx, f"{pfx_p}_c_mw"])
        qa = float(net.asymmetric_load.at[idx, f"{pfx_q}_a_mvar"])
        qb = float(net.asymmetric_load.at[idx, f"{pfx_q}_b_mvar"])
        qc = float(net.asymmetric_load.at[idx, f"{pfx_q}_c_mvar"])

        # Determine phase
        phases_present = []
        if pa > 0 or qa > 0:
            phases_present.append("a")
        if pb > 0 or qb > 0:
            phases_present.append("b")
        if pc > 0 or qc > 0:
            phases_present.append("c")
        phase = "".join(phases_present) if phases_present else "abc"

        total_kw = (pa + pb + pc) * 1000.0
        total_kvar = (qa + qb + qc) * 1000.0

        rows.append({
            "load_id": load_id,
            "bus": bus_name,
            "phase": phase,
            "nominal_kw": total_kw,
            "nominal_kvar": total_kvar,
        })

    return pd.DataFrame(rows)


# ===================================================================== #
#                          POWER FLOW                                    #
# ===================================================================== #

def run_snapshot(
    net: pp.pandapowerNet | None = None,
    load_scaling: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Run a single three-phase unbalanced power-flow snapshot.

    Parameters
    ----------
    net : pandapowerNet, optional
        Network to solve.  If *None*, builds a fresh IEEE 13-bus network.
    load_scaling : dict, optional
        Mapping ``{load_id: scaling_factor}`` to apply before solving.
        Every load is reset to 1.0 first, so a missing load_id runs at
        1.0 regardless of earlier calls.  Unknown load_ids raise KeyError.

    Returns
    -------
    dict
        ``bus_voltages``  — DataFrame of per-phase bus voltages (pu),
        ``line_currents`` — DataFrame of per-phase line currents (kA),
        ``line_powers``   — DataFrame of per-phase line active powers (MW),
        ``net``           — the solved network object.
    """
    if net is None:
        net = build_ieee13_network()

    # Reset every load to 1.0 first so a partial dict never leaves stale
    # scaling from a previous call on the same net.
    net.asymmetric_load["scaling"] = 1.0
    if load_scaling:
        unknown = set(load_scaling) - set(net.asymmetric_load["name"])
        if unknown:
            raise KeyError(f"Unknown load_id(s) in load_scaling: {sorted(unknown)}")
        for idx in net.asymmetric_load.index:
            load_id = net.asymmetric_load.at[idx, "name"]
            if load_id in load_scaling:
                net.asymmetric_load.at[idx, "scaling"] = load_scaling[load_id]

    # IMPORTANT: Use runpp_3ph — NEVER the balanced runpp()!
    # The balanced solver silently produces wrong results for this network.
    pp.runpp_3ph(net)

    result = {
        "bus_voltages": net.res_bus_3ph.copy() if hasattr(net, "res_bus_3ph") else pd.DataFrame(),
        "line_currents": net.res_line_3ph.copy() if hasattr(net, "res_line_3ph") else pd.DataFrame(),
        "line_powers": pd.DataFrame(),
        "net": net,
    }

    # Extract power results if available
    if hasattr(net, "res_line_3ph") and not net.res_line_3ph.empty:
        power_cols = [c for c in net.res_line_3ph.columns if "p_" in c.lower()]
        if power_cols:
            result["line_powers"] = net.res_line_3ph[power_cols].copy()

    return result


def _synthetic_daily_profile(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Generate a simple synthetic daily load shape.

    Returns scaling factors (0.4–1.0) that follow a double-hump daily
    pattern: morning peak ~08:00, evening peak ~19:00, overnight trough.
    Used when ``run_timeseries`` is called without external profiles so
    the function works and can be tested standalone.
    """
    hour = timestamps.hour + timestamps.minute / 60.0
    # Double-hump: morning peak at 8, evening peak at 19
    profile = (
        0.4
        + 0.3 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
        + 0.3 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
    )
    return pd.Series(profile, index=timestamps)


def run_timeseries(
    start: str | datetime = "2024-01-01",
    end: str | datetime = "2024-01-02",
    freq: str | None = None,
    net: pp.pandapowerNet | None = None,
    profiles: pd.DataFrame | None = None,
    return_state: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Step through a time range, applying load profiles at each step.

    If the network has a controlled OLTC (Stage B), each step is solved, the
    tap moves at most one step towards the band and the step is re-solved.

    Parameters
    ----------
    start, end : str or datetime
        Inclusive bounds of the simulation window.
    freq : str, optional
        Pandas frequency string (default: ``config.SIMULATION_FREQ``).
    net : pandapowerNet, optional
        Network.  Built from scratch if *None*.
    profiles : DataFrame, optional
        Index = timestamps, columns = load_id strings, values = multiplicative
        scaling factors versus nominal.  Must cover every simulated timestamp
        and every load_id.  If *None*, a built-in synthetic daily shape is
        used so the function works standalone.
    return_state : bool
        If True, also return a per-step state frame: ``timestamp, oltc_tap,
        tap_moved, at_tap_limit, v_relay_pu, v_min_11kv_pu, v_max_11kv_pu,
        p_in_mw, p_load_mw, losses_mw, branch_losses_mw, v_below_095,
        v_above_105`` (the last two count 11 kV bus-phases outside the
        0.95-1.05 pu design target).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with one row per ``(timestamp, load_id, phase)``
        and columns: ``timestamp, load_id, bus, phase, voltage_v,
        current_a, active_power_kw, reactive_power_kvar`` (unchanged since
        Phase 1) plus, for schema 2.0, ``voltage_angle_deg`` (phase angle of
        the bus voltage) and ``load_current_a`` (the load's own current,
        |S|/|V| per phase).  Timestamps are timezone-naive IST wall time.
        With ``return_state`` a ``(frame, state)`` tuple.

    Raises
    ------
    KeyError
        If ``profiles`` is missing a simulated timestamp or a load_id.
    RuntimeError
        If the power flow fails at any step (no step is skipped).
    """
    if net is None:
        net = build_ieee13_network()
    if freq is None:
        freq = config.SIMULATION_FREQ

    timestamps = pd.date_range(start=start, end=end, freq=freq)

    # Load point metadata
    lp_df = get_load_points(net)
    load_ids = list(lp_df["load_id"])

    # Build profiles if not provided
    if profiles is None:
        synth = _synthetic_daily_profile(timestamps)
        profiles = pd.DataFrame(
            {lid: synth.values for lid in load_ids},
            index=timestamps,
        )

    missing_ids = [lid for lid in load_ids if lid not in profiles.columns]
    if missing_ids:
        raise KeyError(f"profiles is missing load_id column(s): {missing_ids}")
    missing_ts = timestamps.difference(profiles.index)
    if len(missing_ts):
        raise KeyError(
            f"profiles is missing {len(missing_ts)} timestamp(s), "
            f"first {missing_ts[0]}"
        )

    # Bus voltage nominal values (for converting pu → V)
    bus_vn: Dict[int, float] = {}
    for bus_idx in net.bus.index:
        bus_vn[bus_idx] = float(net.bus.at[bus_idx, "vn_kv"]) * 1000.0  # kV → V

    records: list[dict] = []
    states: list[dict] = []
    controlled = oltc.has_oltc(net)

    for ts in timestamps:
        # Scaling for this timestamp (coverage checked above)
        scaling: Dict[str, float] = {
            lid: float(profiles.at[ts, lid]) for lid in load_ids
        }

        try:
            snap = run_snapshot(net, load_scaling=scaling)
            if controlled:
                if ts == timestamps[0]:
                    # Initial condition: settle the tap to the first step's
                    # load (not counted as tap operations).
                    oltc.regulate(net, max_moves=40)
                # At most one OLTC move per time step (delay ignored at 15 min)
                moves, v_relay = oltc.regulate(net, max_moves=1)
                snap["bus_voltages"] = net.res_bus_3ph.copy()
                snap["line_currents"] = net.res_line_3ph.copy()
            bus_v = snap["bus_voltages"]
            line_i = snap["line_currents"]

            if bus_v.empty:
                raise RuntimeError("no 3ph voltage results")
            if return_state:
                states.append(_state_row(net, ts, moves if controlled else 0,
                                         v_relay if controlled else np.nan))

            # For each load point, extract per-phase results
            for _, lp_row in lp_df.iterrows():
                load_id = lp_row["load_id"]
                bus_name = lp_row["bus"]
                phase_str = lp_row["phase"]

                # Find bus index
                bus_idx = None
                for bi in net.bus.index:
                    if net.bus.at[bi, "name"] == bus_name:
                        bus_idx = bi
                        break
                if bus_idx is None:
                    raise RuntimeError(f"bus {bus_name!r} not in network")

                vn_v = bus_vn[bus_idx]
                # Line-to-neutral nominal
                vn_ln = vn_v / np.sqrt(3)

                # Determine phases to report
                if phase_str == "abc":
                    phases = ["a", "b", "c"]
                else:
                    phases = list(phase_str)

                for ph in phases:
                    row: dict[str, Any] = {
                        "timestamp": ts,
                        "load_id": load_id,
                        "bus": bus_name,
                        "phase": ph,
                    }

                    # Voltage (line-to-neutral in V)
                    vm_col = [c for c in bus_v.columns
                              if f"vm_{ph}" in c.lower() and "pu" in c.lower()]
                    if vm_col and bus_idx in bus_v.index:
                        vm_pu = float(bus_v.at[bus_idx, vm_col[0]])
                        row["voltage_v"] = vm_pu * vn_ln
                        row["voltage_angle_deg"] = float(
                            bus_v.at[bus_idx, f"va_{ph}_degree"])
                    else:
                        row["voltage_v"] = np.nan
                        row["voltage_angle_deg"] = np.nan

                    # Current (kA → A) — max over lines connected to this
                    # bus and transformer windings on this bus (bus 634 is
                    # fed only by XFM-1)
                    branch_i: list[float] = []
                    if not line_i.empty:
                        connected = net.line[
                            (net.line["from_bus"] == bus_idx)
                            | (net.line["to_bus"] == bus_idx)
                        ].index
                        i_cols = [c for c in line_i.columns
                                  if f"i_{ph}" in c.lower() and "ka" in c.lower()]
                        if i_cols and len(connected) > 0:
                            branch_i.append(float(
                                line_i.loc[connected, i_cols[0]].max()))
                    for side in ("hv", "lv"):
                        on_bus = net.trafo.index[net.trafo[f"{side}_bus"] == bus_idx]
                        if len(on_bus) > 0:
                            branch_i.append(float(net.res_trafo_3ph.loc[
                                on_bus, f"i_{ph}_{side}_ka"].max()))
                    row["current_a"] = (
                        max(branch_i) * 1000.0 if branch_i else np.nan  # kA → A
                    )

                    # Active power (MW → kW)
                    # Find the asymmetric load index
                    p_val = np.nan
                    q_val = np.nan
                    for li in net.asymmetric_load.index:
                        if net.asymmetric_load.at[li, "name"] == load_id:
                            p_col = f"p_{ph}_mw"
                            q_col = f"q_{ph}_mvar"
                            if p_col in net.asymmetric_load.columns:
                                p_mw = float(net.asymmetric_load.at[li, p_col])
                                s_factor = float(
                                    net.asymmetric_load.at[li, "scaling"]
                                )
                                p_val = p_mw * s_factor * 1000.0  # MW → kW
                            if q_col in net.asymmetric_load.columns:
                                q_mvar = float(
                                    net.asymmetric_load.at[li, q_col]
                                )
                                s_factor = float(
                                    net.asymmetric_load.at[li, "scaling"]
                                )
                                q_val = q_mvar * s_factor * 1000.0  # Mvar → kvar
                            break
                    row["active_power_kw"] = p_val
                    row["reactive_power_kvar"] = q_val
                    # The load's own current (schema 2.0, audit L1-15): |S|/|V|
                    # per phase from the solved voltage (current_a keeps its
                    # Phase 1 meaning, the largest connected branch current).
                    v_ph = row["voltage_v"]
                    row["load_current_a"] = (
                        math.hypot(p_val, q_val) * 1000.0 / v_ph if v_ph > 0 else 0.0)

                    records.append(row)

        except Exception as exc:
            raise RuntimeError(f"Power flow failed at {ts}: {exc}") from exc

    df = pd.DataFrame(records)
    if df.empty:
        # Return empty frame with correct columns
        df = pd.DataFrame(columns=[
            "timestamp", "load_id", "bus", "phase",
            "voltage_v", "current_a", "active_power_kw", "reactive_power_kvar",
            "voltage_angle_deg", "load_current_a",
        ])
    value_cols = ["voltage_v", "current_a", "active_power_kw", "reactive_power_kvar",
                  "voltage_angle_deg", "load_current_a"]
    nan_rows = df[df[value_cols].isna().any(axis=1)]
    if not nan_rows.empty:
        raise RuntimeError(
            f"run_timeseries produced NaN values for "
            f"{sorted(nan_rows['load_id'].unique())}"
        )
    if return_state:
        return df, pd.DataFrame(states)
    return df


def _state_row(net: pp.pandapowerNet, ts, moves: int, v_relay: float) -> dict:
    """Network-level state of one solved time step (OLTC, voltages, losses)."""
    mv = net.bus.index[net.bus.vn_kv == config.FEEDER_VOLTAGE_KV]
    vm = net.res_bus_3ph.loc[mv, ["vm_a_pu", "vm_b_pu", "vm_c_pu"]]
    al = net.asymmetric_load
    p_load = float(sum((al[f"p_{p}_mw"] * al["scaling"]).sum() for p in "abc"))
    p_in = float(sum(net.res_ext_grid_3ph[f"p_{p}_mw"].sum() for p in "abc"))
    row = {
        "timestamp": ts, "oltc_tap": np.nan, "tap_moved": moves,
        "at_tap_limit": False, "v_relay_pu": v_relay,
        "v_min_11kv_pu": float(np.nanmin(vm.values)),
        "v_max_11kv_pu": float(np.nanmax(vm.values)),
        "p_in_mw": p_in, "p_load_mw": p_load, "losses_mw": p_in - p_load,
        "branch_losses_mw": float(
            sum(net.res_line_3ph[f"pl_{p}_mw"].sum() for p in "abc")
            + sum(net.res_trafo_3ph[f"pl_{p}_mw"].sum() for p in "abc")),
        "v_below_095": int((vm < 0.95).sum().sum()),
        "v_above_105": int((vm > 1.05).sum().sum()),
    }
    if oltc.has_oltc(net):
        t = int(net["oltc"].trafo.iloc[0])
        tap = float(net.trafo.at[t, "tap_pos"])
        row["oltc_tap"] = tap
        row["at_tap_limit"] = tap in (net.trafo.at[t, "tap_min"], net.trafo.at[t, "tap_max"])
    return row


# ===================================================================== #
#                      SHORT-CIRCUIT / FAULT INJECTION                   #
# ===================================================================== #

def inject_fault(
    bus: int | str,
    fault_type: str = "SLG",
    timestamp: str | datetime | None = None,
    net: pp.pandapowerNet | None = None,
) -> Dict[str, Any]:
    """Run an IEC 60909 short-circuit study and return results for one bus.

    Wraps ``pandapower.shortcircuit.calc_sc``.  Internally the full-network
    study runs at all buses; results are then filtered to the requested bus.

    Parameters
    ----------
    bus : int or str
        Bus index or node name (e.g. ``"671"``).
    fault_type : str
        ``"SLG"`` (single-line-to-ground), ``"LL"`` (line-to-line),
        or ``"3PH"`` (three-phase bolted).
    timestamp : str or datetime, optional
        If given, a pre-fault ``run_snapshot`` is executed first so that
        pre-fault currents are available for comparison.
    net : pandapowerNet, optional
        Network.  Built from scratch if *None*.

    Returns
    -------
    dict
        ``timestamp``          — Echo of the requested timestamp (or None),
        ``bus``                — Bus name (str),
        ``fault_type``         — Echo of the requested fault type,
        ``fault_current_ka``   — Initial symmetrical SC current (magnitude),
        ``affected_phases``    — List of ``'a'``/``'b'``/``'c'``,
        ``prefault_current_ka``— Pre-fault current magnitude (if timestamp
            was provided).
    """
    if net is None:
        net = build_ieee13_network()

    # Resolve bus name → index
    if isinstance(bus, str):
        if bus not in _NODE_INDEX:
            matches = net.bus[net.bus["name"] == bus].index
            if len(matches) == 0:
                raise ValueError(f"Unknown bus name: {bus}")
            bus_idx = int(matches[0])
            bus_name = bus
        else:
            bus_idx = _NODE_INDEX[bus]
            bus_name = bus
    else:
        bus_idx = int(bus)
        bus_name = net.bus.at[bus_idx, "name"]

    # Pre-fault snapshot (optional)
    prefault_i: Optional[float] = None
    if timestamp is not None:
        try:
            snap = run_snapshot(net)
            line_i = snap["line_currents"]
            if not line_i.empty:
                connected = net.line[
                    (net.line["from_bus"] == bus_idx)
                    | (net.line["to_bus"] == bus_idx)
                ].index
                i_cols = [c for c in line_i.columns
                          if "i_" in c.lower() and "ka" in c.lower()]
                if i_cols and len(connected) > 0:
                    prefault_i = float(
                        line_i.loc[connected, i_cols].values.max()
                    )
        except Exception as exc:
            logger.warning("Pre-fault snapshot failed: %s", exc)

    # Map contract fault types to pandapower calc_sc "fault" parameter
    FAULT_TYPE_MAP = {
        "SLG": "1ph",   # single-line-to-ground
        "LL":  "2ph",   # line-to-line
        "3PH": "3ph",   # three-phase bolted
    }
    sc_fault = FAULT_TYPE_MAP.get(fault_type)
    if sc_fault is None:
        raise ValueError(
            f"fault_type must be one of {list(FAULT_TYPE_MAP)}, got {fault_type!r}"
        )

    # Determine affected phases
    phase_map = {
        "SLG": ["a"],              # single-line-to-ground
        "LL":  ["a", "b"],         # line-to-line
        "3PH": ["a", "b", "c"],   # three-phase
    }

    fault_i = 0.0
    try:
        pp.shortcircuit.calc_sc(net, fault=sc_fault, case="max")
        sc_result = net.res_bus_sc.loc[bus_idx]
        # ikss_ka is the initial symmetrical short-circuit current
        fault_i = float(sc_result.get("ikss_ka", 0.0))
    except Exception as exc:
        logger.error("Short-circuit calc failed for %s at bus %s: %s",
                      fault_type, bus_name, exc)

    return {
        "timestamp": timestamp,
        "bus": bus_name,
        "fault_type": fault_type,
        "fault_current_ka": fault_i,
        "affected_phases": phase_map.get(fault_type, []),
        "prefault_current_ka": prefault_i,
    }


# ===================================================================== #
#                              MAIN                                      #
# ===================================================================== #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Building IEEE 13-bus network at 11 kV ...")
    network = build_ieee13_network()

    print("\n=== Load Points ===")
    lp = get_load_points(network)
    print(lp.to_string(index=False))

    print("\nRunning unbalanced power flow snapshot ...")
    result = run_snapshot(network)

    print("\n=== Bus Voltages (3-phase) ===")
    if not result["bus_voltages"].empty:
        print(result["bus_voltages"].to_string())
    else:
        print("(no 3ph voltage results — check pandapower version)")

    print("\n=== Line Currents (3-phase) ===")
    if not result["line_currents"].empty:
        print(result["line_currents"].to_string())
    else:
        print("(no 3ph current results)")

    print("\nRunning short time-series (1 hour) ...")
    ts_df = run_timeseries(
        start="2024-01-01 00:00",
        end="2024-01-01 01:00",
        freq="30min",
        net=network,
    )
    print(f"\nTimeseries result: {len(ts_df)} rows")
    if not ts_df.empty:
        print(ts_df.head(20).to_string(index=False))

    print("\nRunning 3PH fault at bus 671 ...")
    fault_res = inject_fault("671", "3PH", timestamp="2024-01-01", net=network)
    print(fault_res)

    print("\nDone.")
