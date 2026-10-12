"""
Stage B: the Indian 11 kV feeder, derived from Stage A (Phase 2B).

``build_stage_b()`` starts from the validated Stage A network
(:func:`feeder_model._build_stage_a`) and applies the brief's
transformations in order, logging each one:

B1 voltage 4.16 -> 11 kV          B5 laterals three-phase, three-wire
B2 lengths x (11/4.16)^2          B6 capacitors three-phase
B3 50 Hz (X x 5/6)                B7 controlled substation OLTC with LDC
B4 substation transformer         B8 distribution transformers -> Part 2C
                                  B9 constant-PQ loads

See docs/Phase_2_Modeling_Decisions_Brief.md §2B and docs/build_report_2B.md.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Dict, Optional

import pandapower as pp

from . import config
from . import ieee13_data as ieee
from . import oltc
from .line_sequence import (
    SequenceParams, sequence_params_from_self_mutual, upper_to_matrix,
)

logger = logging.getLogger(__name__)

V_STAGE_B_KV = config.FEEDER_VOLTAGE_KV                     # 11 kV
LENGTH_FACTOR = (V_STAGE_B_KV / ieee.V_PRIMARY_KV) ** 2     # 6.992
F_STAGE_B_HZ = 50.0
# B4: with the substation transformer in series the Stage A regulator
# stand-in (vk 0.01 %) does not converge; 0.5 % does.  Numerical stand-in for
# an ideal regulator, removed again at B7.
REG_STANDIN_VK_PERCENT_B4 = 0.5
SUB_TAP_STEP_PERCENT = ieee.REGULATOR_STEP_PU * 100         # IEEE regulator step
SUB_TAP_RANGE = 16                                          # IEEE regulator +/-16
MONITORED_LINE = "Line_RG60_632"                            # leaves the regulated bus
STEPS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9")


def _solve(net: pp.pandapowerNet) -> int:
    """Power flow honouring any constant-Z / constant-I load models."""
    from .feeder_model import solve_with_load_models
    if (net.asymmetric_load.load_model != "PQ").any():
        return solve_with_load_models(net)
    pp.runpp_3ph(net)
    return 1


def b1_voltage(net) -> str:
    primary = net.bus.vn_kv == ieee.V_PRIMARY_KV
    net.bus.loc[primary, "vn_kv"] = V_STAGE_B_KV
    for col in ("vn_hv_kv", "vn_lv_kv"):
        net.trafo.loc[net.trafo[col] == ieee.V_PRIMARY_KV, col] = V_STAGE_B_KV
    net.shunt["vn_kv"] = V_STAGE_B_KV  # shunt kvar is rated at vn_kv
    return (f"{int(primary.sum())} buses 4.16 -> {V_STAGE_B_KV} kV; XFM-1 "
            f"{V_STAGE_B_KV}/{ieee.V_XFM1_LV_KV} kV; shunts rated at {V_STAGE_B_KV} kV")


def b2_lengths(net) -> str:
    net.line["length_km"] *= LENGTH_FACTOR
    return f"line lengths x {LENGTH_FACTOR:.3f}"


def b3_frequency(net) -> str:
    k = F_STAGE_B_HZ / net.f_hz
    net.line["x_ohm_per_km"] *= k
    net.line["x0_ohm_per_km"] *= k
    net.f_hz = F_STAGE_B_HZ  # c_nf unchanged: B = 2 pi f C scales by k
    return f"f_hz {F_STAGE_B_HZ}; line X and B x {k:.4f}"


def b4_substation_transformer(net) -> str:
    st = ieee.SUBSTATION_TRANSFORMER
    sb = pp.create_bus(net, vn_kv=st["kv_high"], name="sourcebus")
    net.ext_grid.at[net.ext_grid.index[0], "bus"] = sb
    net.ext_grid.at[net.ext_grid.index[0], "name"] = "Grid_115kV"
    vk = math.hypot(st["r_pct"], st["x_pct"])
    b650 = int(net.bus.index[net.bus.name == "650"][0])
    # Zero-sequence data is not published: vk0 = vk, mag0 100 % (verify).
    pp.create_transformer_from_parameters(
        net, hv_bus=sb, lv_bus=b650, sn_mva=st["kva"] / 1e3,
        vn_hv_kv=st["kv_high"], vn_lv_kv=V_STAGE_B_KV,
        vkr_percent=st["r_pct"], vk_percent=vk, pfe_kw=0.0, i0_percent=0.0,
        vector_group="Dyn", vk0_percent=vk, vkr0_percent=st["r_pct"],
        mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9, shift_degree=30,
        tap_side="lv", tap_neutral=0, tap_min=-SUB_TAP_RANGE,
        tap_max=SUB_TAP_RANGE, tap_step_percent=SUB_TAP_STEP_PERCENT,
        tap_pos=0, tap_changer_type="Ratio", name="Sub_XFMR",
    )
    reg = net.trafo.name == "Regulator_650_RG60"
    net.trafo.loc[reg, ["vk_percent", "vk0_percent"]] = REG_STANDIN_VK_PERCENT_B4
    return (f"115 kV sourcebus -> Sub_XFMR {st['kva']} kVA Dyn vk {vk:.3f} % -> 650; "
            f"regulator stand-in vk {REG_STANDIN_VK_PERCENT_B4} %")


def config_605_three_phase(f_hz: float) -> SequenceParams:
    """Three-phase sequence data for config 605 (1/0 ACSR) at ``f_hz``.

    605 is single-phase in the IEEE data (Z1 = Z0 = Zcc in 2A.1).  As a
    three-phase lateral it takes Zs (and Bs) from its own matrix and Zm (and
    Bm) from the 603/604 matrices: same 1/0 ACSR conductor, but spacing ID 510
    vs 505 (l. 32-34), so the mutual terms are an approximation (stated).
    Published data is at 60 Hz; X is scaled by f/60 as in B3.
    """
    _, z605, b605 = ieee.LINE_CONFIGS[605]
    _, z603, b603 = ieee.LINE_CONFIGS[603]
    zs = upper_to_matrix(z605)[2, 2]          # c-c of 605
    zm = upper_to_matrix(z603)[1, 2]          # b-c of 603 (= a-c of 604)
    bs = upper_to_matrix(b605)[2, 2]
    bm = upper_to_matrix(b603)[1, 2]
    sp = sequence_params_from_self_mutual(zs, zm, bs, bm, ieee.FREQUENCY_HZ)
    k = f_hz / ieee.FREQUENCY_HZ
    return SequenceParams(sp.r1_ohm_per_km, sp.x1_ohm_per_km * k,
                          sp.r0_ohm_per_km, sp.x0_ohm_per_km * k,
                          sp.c1_nf_per_km, sp.c0_nf_per_km)


def b5_laterals(net) -> str:
    # pandapower lines are symmetric three-phase.  Configs 603/604 already
    # carry two-phase-derived three-phase data (2A.1).  605 gets three-phase
    # data from its own Zs and the 603/604 Zm.  607 (1/0 AA cable) keeps
    # Z1 = Z0 = Zii: no three-phase data for that cable exists (stated).
    sp = config_605_three_phase(net.f_hz)
    rows = net.line.config == 605
    net.line.loc[rows, ["r_ohm_per_km", "x_ohm_per_km", "r0_ohm_per_km",
                        "x0_ohm_per_km", "c_nf_per_km", "c0_nf_per_km"]] = [
        sp.r1_ohm_per_km, sp.x1_ohm_per_km, sp.r0_ohm_per_km,
        sp.x0_ohm_per_km, sp.c1_nf_per_km, sp.c0_nf_per_km]
    return (f"laterals three-phase, three-wire (stated deviation); 605 -> "
            f"Z1 {sp.r1_ohm_per_km:.4f}+j{sp.x1_ohm_per_km:.4f} ohm/km from its Zs and "
            f"603/604 Zm ({int(rows.sum())} line); 607 keeps Z1 = Z0 = Zii")


def b6_capacitors(net) -> str:
    cap = net.asymmetric_load.index[net.asymmetric_load.name == "cap_611_c"]
    net.asymmetric_load.drop(cap, inplace=True)
    q611 = sum(ieee.CAPACITORS["611"])
    pp.create_shunt(net, bus=int(net.bus.index[net.bus.name == "611"][0]),
                    q_mvar=-q611 / 1e3, p_mw=0.0, vn_kv=V_STAGE_B_KV, name="Cap_611")
    return f"611: {q611:.0f} kvar three-phase ({q611 / 3:.1f} per phase); 675 unchanged"


def b7_oltc(net) -> str:
    reg = net.trafo.index[net.trafo.name == "Regulator_650_RG60"]
    hv, lv = int(net.trafo.at[reg[0], "hv_bus"]), int(net.trafo.at[reg[0], "lv_bus"])
    net.trafo.drop(reg, inplace=True)
    pp.create_switch(net, bus=hv, element=lv, et="b", closed=True, name="Switch_650_RG60")
    oltc.attach_oltc(net, "Sub_XFMR", MONITORED_LINE)
    moves, v = oltc.settle(net, solve=_solve)
    tap = net.trafo.at[int(net["oltc"].trafo.iloc[0]), "tap_pos"]
    return (f"regulator stand-in -> closed switch; Sub_XFMR OLTC controlled "
            f"(LDC {complex(net['oltc'].r_ldc_ohm.iloc[0], net['oltc'].x_ldc_ohm.iloc[0]):.4f} ohm); "
            f"settled at tap {tap:+.0f} ({moves} moves), V_relay {v:.4f} pu")


def b8_distribution_transformers(net) -> str:
    return "deferred to Part 2C (DT sizing, households and meters together)"


def b9_constant_pq(net) -> str:
    al = net.asymmetric_load
    al["load_model"] = "PQ"
    for p in "abc":
        al[f"p_{p}_mw"] = al[f"p0_{p}_mw"]
        al[f"q_{p}_mvar"] = al[f"q0_{p}_mvar"]
    moves, v = oltc.settle(net, solve=_solve)
    tap = net.trafo.at[int(net["oltc"].trafo.iloc[0]), "tap_pos"]
    return f"all loads constant-PQ; OLTC re-settled at tap {tap:+.0f} ({moves} moves), V_relay {v:.4f} pu"


_STEP_FUNCS: Dict[str, Callable] = {
    "B1": b1_voltage, "B2": b2_lengths, "B3": b3_frequency,
    "B4": b4_substation_transformer, "B5": b5_laterals, "B6": b6_capacitors,
    "B7": b7_oltc, "B8": b8_distribution_transformers, "B9": b9_constant_pq,
}


def build_stage_b(upto: str = "B9",
                  snapshot_dir: Optional[Path] = None) -> pp.pandapowerNet:
    """Stage A with transformations B1..``upto`` applied in order."""
    from .feeder_model import _build_stage_a
    if upto not in STEPS:
        raise ValueError(f"upto must be one of {STEPS}, got {upto!r}")
    net = _build_stage_a()
    net.name = "IEEE 13-node -> Indian 11 kV feeder (Stage B)"
    for step in STEPS[: STEPS.index(upto) + 1]:
        msg = _STEP_FUNCS[step](net)
        logger.info("Stage B %s: %s", step, msg)
        if snapshot_dir is not None:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            pp.to_json(net, str(Path(snapshot_dir) / f"{step}.json"))
    return net
