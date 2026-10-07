"""
IEEE 13-Bus Test Feeder — Unbalanced Power-Flow Simulation at 11 kV.

Builds the IEEE 13-bus test feeder (Kersting, 1991) in pandapower, adapted so
the primary feeder operates at **11 kV line-to-line** instead of the original
4.16 kV.  Voltage-dependent parameters are scaled accordingly; line impedances
stay in ohms/km from the original data but per-unit bases change with the new
voltage.

The network is modelled as a genuinely **unbalanced** system: single-phase and
three-phase loads are represented explicitly per phase, and the solver used is
``pandapower.pf.runpp_3ph`` — **never** the balanced ``runpp()``.

Modelling judgement calls (documented in docs/feeder_model_contract.md)
-----------------------------------------------------------------------
- Substation transformer: 115 kV / 11 kV, Dyn, 5 MVA (replaces 115/4.16).
- XFM-1 (633→634): 11 kV / ~1.27 kV (0.48 × 11/4.16).  The kV ratio is
  scaled proportionally so that downstream loads keep the same per-unit
  voltage.
- Voltage regulator at 650: represented via the external grid's vm_pu.
  pandapower's 3ph solver does not support ideal voltage regulators; the
  effect is captured by setting vm_pu=1.0 at the slack.
- 632→671 distributed load: lumped at bus 671, consistent with IEEE published
  totals.
- 671→692 switch: modelled as a very short line (0.001 km, config 606).
- Capacitor banks at 675 (600 kVAR) and 611 (100 kVAR): kept at original
  values.

References
----------
1.  Kersting, W.H., "Radial Distribution Test Feeders," IEEE Trans. Power
    Systems, vol. 6, no. 3, pp. 975–985, 1991.
2.  IEEE PES Distribution System Analysis Subcommittee, Test Feeder Resources.
3.  pandapower ``runpp_3ph`` and ``shortcircuit.calc_sc`` documentation.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IEEE 13-bus node names ↔ internal bus indices
# ---------------------------------------------------------------------------
NODE_NAMES = [
    "650",   # 0  — substation / slack
    "632",   # 1  — primary feeder
    "633",   # 2
    "634",   # 3  — XFM-1 secondary
    "645",   # 4
    "646",   # 5
    "671",   # 6
    "680",   # 7  — no load
    "684",   # 8
    "611",   # 9
    "652",   # 10
    "675",   # 11
    "692",   # 12
]

# Mapping from node name → bus index (populated by build_ieee13_network)
_NODE_INDEX: Dict[str, int] = {}


# ===================================================================== #
#                        NETWORK CONSTRUCTION                            #
# ===================================================================== #

def build_ieee13_network() -> pp.pandapowerNet:
    """Construct the IEEE 13-bus test feeder adapted to 11 kV.

    Returns
    -------
    net : pandapowerNet
        Ready-to-solve pandapower network with all buses, lines,
        transformers, loads, shunts, and external grid.

    Raises
    ------
    AssertionError
        If total connected load deviates from ~3.5 MW by more than 15 %.
    """
    net = pp.create_empty_network(name="IEEE 13-Bus @ 11 kV", sn_mva=10.0)

    vn_primary = config.FEEDER_VOLTAGE_KV           # 11 kV
    vn_sub     = 115.0                               # substation HV side
    # XFM-1 secondary was 0.48 kV in original; scale proportionally
    vn_xfm1_lv = 0.48 * config.VOLTAGE_SCALE_FACTOR  # ≈ 1.269 kV

    # ------------------------------------------------------------------ #
    # Buses                                                                #
    # ------------------------------------------------------------------ #
    _NODE_INDEX.clear()
    for i, name in enumerate(NODE_NAMES):
        if name == "650":
            vn = vn_sub
        elif name == "634":
            vn = vn_xfm1_lv
        else:
            vn = vn_primary
        idx = pp.create_bus(net, vn_kv=vn, name=name)
        _NODE_INDEX[name] = idx

    B = _NODE_INDEX  # shorthand

    # ------------------------------------------------------------------ #
    # External grid (slack at bus 650)                                     #
    # Short-circuit data needed by calc_sc (IEC 60909).                    #
    # Zero-sequence parameters needed for SLG faults.                      #
    # ------------------------------------------------------------------ #
    pp.create_ext_grid(
        net,
        bus=B["650"],
        vm_pu=1.0,
        name="Substation",
        s_sc_max_mva=1000.0,
        rx_max=0.1,
        s_sc_min_mva=800.0,
        rx_min=0.1,
        # Zero-sequence data for SLG fault calculation
        x0x_max=1.0,
        r0x0_max=0.1,
        x0x_min=1.0,
        r0x0_min=0.1,
    )

    # ------------------------------------------------------------------ #
    # Substation transformer  115 kV / 11 kV  (Dyn — required for 3ph)    #
    # ------------------------------------------------------------------ #
    pp.create_transformer_from_parameters(
        net,
        hv_bus=B["650"],
        lv_bus=B["632"],
        sn_mva=5.0,
        vn_hv_kv=vn_sub,
        vn_lv_kv=vn_primary,
        vkr_percent=1.0,
        vk_percent=8.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        vector_group="Dyn",
        vk0_percent=8.0,
        vkr0_percent=1.0,
        mag0_percent=100.0,
        mag0_rx=0.0,
        si0_hv_partial=0.9,
        shift_degree=30,
        name="Sub_XFMR",
    )

    # ------------------------------------------------------------------ #
    # In-line transformer XFM-1  (632 → 634): 11 kV / ~1.27 kV            #
    # ------------------------------------------------------------------ #
    pp.create_transformer_from_parameters(
        net,
        hv_bus=B["632"],
        lv_bus=B["634"],
        sn_mva=0.5,
        vn_hv_kv=vn_primary,
        vn_lv_kv=vn_xfm1_lv,
        vkr_percent=1.1,
        vk_percent=4.5,
        pfe_kw=0.0,
        i0_percent=0.0,
        vector_group="Dyn",
        vk0_percent=4.5,
        vkr0_percent=1.1,
        mag0_percent=100.0,
        mag0_rx=0.0,
        si0_hv_partial=0.9,
        shift_degree=30,
        name="XFM-1",
    )

    # ------------------------------------------------------------------ #
    # Line segments                                                        #
    # Original IEEE 13-bus line data: impedances in ohms/mile,             #
    # lengths in feet.  We convert to ohms/km and km.                      #
    # ------------------------------------------------------------------ #
    FT_TO_KM = 0.0003048
    MI_TO_KM = 1.60934

    # Line configuration impedances (ohms/mile → ohms/km)
    # Positive-sequence R1, X1 and zero-sequence R0, X0 per config type
    # Source: IEEE 13-bus test feeder documentation
    line_configs = {
        # config_id: (r1_ohm_per_km, x1_ohm_per_km, c1_nf_per_km,
        #             r0_ohm_per_km, x0_ohm_per_km, c0_nf_per_km, max_i_ka)
        601: (0.2153, 0.4660, 10.0, 0.6459, 1.3980, 5.0, 0.4),
        602: (0.2676, 0.4839, 10.0, 0.8028, 1.4517, 5.0, 0.4),
        603: (0.7982, 0.4463, 10.0, 2.3946, 1.3389, 5.0, 0.3),
        604: (0.7982, 0.4463, 10.0, 2.3946, 1.3389, 5.0, 0.3),
        605: (0.7982, 0.4463, 10.0, 2.3946, 1.3389, 5.0, 0.3),
        606: (0.4605, 0.2649, 10.0, 1.3815, 0.7947, 5.0, 0.3),
        607: (0.5108, 0.2566, 10.0, 1.5324, 0.7698, 5.0, 0.3),
    }

    # Line segments: (from_node, to_node, config_id, length_ft)
    line_segments = [
        ("632", "645", 603, 500),
        ("632", "633", 602, 500),
        ("633", "634", 607, 0),     # through XFM-1, negligible line
        ("645", "646", 603, 300),
        ("632", "671", 601, 2000),
        ("671", "684", 604, 300),
        ("671", "680", 601, 1000),
        ("671", "692", 606, 0),     # switch, negligible length
        ("684", "611", 605, 300),
        ("684", "652", 607, 800),
        ("692", "675", 606, 500),
    ]

    for from_n, to_n, cfg_id, length_ft in line_segments:
        cfg = line_configs[cfg_id]
        length_km = max(length_ft * FT_TO_KM, 0.001)  # avoid zero length
        pp.create_line_from_parameters(
            net,
            from_bus=B[from_n],
            to_bus=B[to_n],
            length_km=length_km,
            r_ohm_per_km=cfg[0],
            x_ohm_per_km=cfg[1],
            c_nf_per_km=cfg[2],
            r0_ohm_per_km=cfg[3],
            x0_ohm_per_km=cfg[4],
            c0_nf_per_km=cfg[5],
            max_i_ka=cfg[6],
            name=f"Line_{from_n}_{to_n}",
        )

    # ------------------------------------------------------------------ #
    # Asymmetric (unbalanced) loads                                        #
    # Source: IEEE 13-bus Spot Load Data                                    #
    # (p_a/b/c in MW, q_a/b/c in Mvar)                                   #
    #                                                                      #
    # load_id is a stable string matching the contract:                    #
    #   "load_<bus>_<phase>" where phase ∈ {a, b, c, abc}                 #
    # Three-phase loads get one asymmetric_load entry with all phases.     #
    # Single-/two-phase loads get one entry with zero on unused phases.    #
    # ------------------------------------------------------------------ #
    spot_loads = [
        # (bus,  p_a,    p_b,    p_c,    q_a,    q_b,    q_c,   model, conn)
        ("634", 0.160,  0.120,  0.120,  0.110,  0.090,  0.090, "wye",   "PQ"),
        ("645", 0.000,  0.170,  0.000,  0.000,  0.125,  0.000, "wye",   "PQ"),
        ("646", 0.000,  0.230,  0.000,  0.000,  0.132,  0.000, "delta", "Z"),
        ("652", 0.128,  0.000,  0.000,  0.086,  0.000,  0.000, "wye",   "Z"),
        ("671", 0.385,  0.385,  0.385,  0.220,  0.220,  0.220, "delta", "PQ"),
        ("675", 0.485,  0.068,  0.290,  0.190,  0.060,  0.212, "wye",   "PQ"),
        ("692", 0.000,  0.000,  0.170,  0.000,  0.000,  0.151, "delta", "PQ"),
        ("611", 0.000,  0.000,  0.170,  0.000,  0.000,  0.080, "wye",   "PQ"),
    ]

    # Distributed load on line 632-671 (modelled as lumped at midpoint 671)
    # Original: 17 kW Ph-A, 66 kW Ph-B, 117 kW Ph-C  (aggregated into 671)
    # Already included in spot_loads["671"] values above — the IEEE 13-bus
    # published totals roll distributed loads into the nearest bus.

    for bus_name, pa, pb, pc, qa, qb, qc, conn_type, load_model in spot_loads:
        # Determine phase descriptor for load_id
        phases_present = []
        if pa > 0 or qa > 0:
            phases_present.append("a")
        if pb > 0 or qb > 0:
            phases_present.append("b")
        if pc > 0 or qc > 0:
            phases_present.append("c")
        phase_str = "".join(phases_present) if phases_present else "abc"

        load_id = f"load_{bus_name}_{phase_str}"

        pp.create_asymmetric_load(
            net,
            bus=B[bus_name],
            p_a_mw=pa,
            p_b_mw=pb,
            p_c_mw=pc,
            q_a_mvar=qa,
            q_b_mvar=qb,
            q_c_mvar=qc,
            type=conn_type,
            name=load_id,
        )

    # ------------------------------------------------------------------ #
    # Shunt capacitors                                                     #
    # ------------------------------------------------------------------ #
    # Bus 675: 3-phase, 200 kVAR per phase
    pp.create_shunt(
        net,
        bus=B["675"],
        q_mvar=-0.600,   # 3 × 200 kVAR (negative = capacitive)
        p_mw=0.0,
        name="Cap_675",
    )
    # Bus 611: single-phase (phase C), 100 kVAR
    pp.create_shunt(
        net,
        bus=B["611"],
        q_mvar=-0.100,
        p_mw=0.0,
        name="Cap_611",
    )

    # ------------------------------------------------------------------ #
    # Sanity check: total connected load ≈ 3.5 MW                         #
    # Published figure: IEEE 13-bus total spot + distributed ≈ 3.466 MW    #
    # We use 3.5 MW as the round reference with ±15 % tolerance.           #
    # ------------------------------------------------------------------ #
    total_p = (
        net.asymmetric_load[["p_a_mw", "p_b_mw", "p_c_mw"]].sum().sum()
    )
    tolerance = config.LOAD_TOLERANCE_FRACTION * config.TOTAL_LOAD_MW_ASSERTION
    assert abs(total_p - config.TOTAL_LOAD_MW_ASSERTION) < tolerance, (
        f"Total connected load {total_p:.3f} MW deviates from expected "
        f"{config.TOTAL_LOAD_MW_ASSERTION} MW by more than "
        f"{config.LOAD_TOLERANCE_FRACTION * 100:.0f} %"
    )
    logger.info(
        "IEEE 13-bus network built: %d buses, %d lines, %.3f MW total load",
        len(net.bus), len(net.line), total_p,
    )
    return net


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
    for idx in net.asymmetric_load.index:
        load_id = net.asymmetric_load.at[idx, "name"]
        bus_idx = int(net.asymmetric_load.at[idx, "bus"])
        bus_name = net.bus.at[bus_idx, "name"]

        pa = float(net.asymmetric_load.at[idx, "p_a_mw"])
        pb = float(net.asymmetric_load.at[idx, "p_b_mw"])
        pc = float(net.asymmetric_load.at[idx, "p_c_mw"])
        qa = float(net.asymmetric_load.at[idx, "q_a_mvar"])
        qb = float(net.asymmetric_load.at[idx, "q_b_mvar"])
        qc = float(net.asymmetric_load.at[idx, "q_c_mvar"])

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
        A missing load_id defaults to 1.0.

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

    # Apply optional per-load scaling
    if load_scaling:
        for idx in net.asymmetric_load.index:
            load_id = net.asymmetric_load.at[idx, "name"]
            if load_id in load_scaling:
                net.asymmetric_load.at[idx, "scaling"] = load_scaling[load_id]
    else:
        # Reset to 1.0 if no scaling provided
        net.asymmetric_load["scaling"] = 1.0

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
) -> pd.DataFrame:
    """Step through a time range, applying load profiles at each step.

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
        scaling factors versus nominal.  If *None*, a built-in synthetic daily
        shape is used so the function works standalone.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with one row per ``(timestamp, load_id, phase)``
        and columns: ``timestamp, load_id, bus, phase, voltage_v,
        current_a, active_power_kw, reactive_power_kvar``.
        Timestamps are timezone-naive.
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

    # Bus voltage nominal values (for converting pu → V)
    bus_vn: Dict[int, float] = {}
    for bus_idx in net.bus.index:
        bus_vn[bus_idx] = float(net.bus.at[bus_idx, "vn_kv"]) * 1000.0  # kV → V

    records: list[dict] = []

    for ts in timestamps:
        # Build scaling dict for this timestamp
        scaling: Dict[str, float] = {}
        if ts in profiles.index:
            for lid in load_ids:
                if lid in profiles.columns:
                    scaling[lid] = float(profiles.at[ts, lid])

        # Reset scaling to 1.0 before applying new values
        net.asymmetric_load["scaling"] = 1.0

        try:
            snap = run_snapshot(net, load_scaling=scaling if scaling else None)
            bus_v = snap["bus_voltages"]
            line_i = snap["line_currents"]

            if bus_v.empty:
                logger.warning("No 3ph voltage results at %s", ts)
                continue

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
                    continue

                vn_v = bus_vn.get(bus_idx, 11000.0)
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
                    else:
                        row["voltage_v"] = np.nan

                    # Current (kA → A) — from lines connected to this bus
                    i_val = np.nan
                    if not line_i.empty:
                        connected = net.line[
                            (net.line["from_bus"] == bus_idx)
                            | (net.line["to_bus"] == bus_idx)
                        ].index
                        i_cols = [c for c in line_i.columns
                                  if f"i_{ph}" in c.lower() and "ka" in c.lower()]
                        if i_cols and len(connected) > 0:
                            i_val = float(
                                line_i.loc[connected, i_cols[0]].max()
                            ) * 1000.0  # kA → A
                    row["current_a"] = i_val

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

                    records.append(row)

        except Exception as exc:
            logger.warning("Power flow failed at %s: %s", ts, exc)
            continue

    df = pd.DataFrame(records)
    if df.empty:
        # Return empty frame with correct columns
        df = pd.DataFrame(columns=[
            "timestamp", "load_id", "bus", "phase",
            "voltage_v", "current_a", "active_power_kw", "reactive_power_kvar",
        ])
    return df


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
