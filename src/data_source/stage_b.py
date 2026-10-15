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

See docs/Phase_2_Modeling_Decisions_Brief_v2.md §2B and docs/build_report_2B.md.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandapower as pp
import pandas as pd

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
# OLTC (brief v2 X.3.2 item 21): GETCO/E/TS-XMER/R13 (2017), Data sheet A1
# item 12, p. 28: OLTC on the HV winding for LV variation, -5 % to +15 %
# (looking from the LV side), step 1.25 % -> taps -4 .. +12 (16 steps).
# Modelled as an LV-side ratio tap (a positive tap raises the LV voltage;
# tested).  The IEEE set point / band are kept (oltc.py); full band 1.67 % >
# step 1.25 % (GETCO p. 106: AVR bandwidth set to suit the 1.25 % step).
SUB_TAP_STEP_PERCENT = 1.25
SUB_TAP_MIN, SUB_TAP_MAX = -4, 12
MONITORED_LINE = "Line_RG60_632"                            # leaves the regulated bus
# B4 (brief v2 §2X, X.1): 66/11 kV substation.  CEA Manual on Transmission
# Planning Criteria 2025, §5.1.5 (PDF p. 35): 66 kV equipment rated 31.5 kA,
# maximum short-circuit level <= 80 % of it -> 25.2 kA at 66 kV.  The IEEE
# transformer data (5 MVA, R 1 %, X 8 %, vk 8.062 %) is kept: GETCO p. 12
# gives 8.0 % (+/- 10 % tolerance) for 5 MVA 66/11.55 kV, Dyn11, with the
# 11 kV system neutral solidly grounded (stated; no longer verify).
# Zero sequence (X.3.2 item 14): the zero-sequence magnetising branch of
# every Dyn unit is effectively open, |Z_m0| = 1000 |Z0k| (pandapower:
# Z_m0 = mag0_percent/100 x Z_sc, vk0 = vk) -- Grainger & Stevenson,
# Power System Analysis (1994) §11.8, Fig. 11.17, book p. 450 (magnetising
# path omitted).  The brief asks |Z_m0| >= 100 |Z0k| and R0,eff/R >= 0.99;
# at exactly 100 x the reactive branch still diverts ~1 % of I0 and
# Sub_XFMR gives R0,eff/R = 0.984, so 1000 x is used (R0,eff/R ~ 0.998).
# The former mag0 100 % / mag0_rx 0 had no source.
MAG0_PERCENT = 1.0e5
MAG0_RX = 0.0
SUB_HV_KV = 66.0
GRID_IK_MAX_KA = 0.8 * 31.5                                 # 25.2 kA
GRID_S_SC_MAX_MVA = math.sqrt(3) * SUB_HV_KV * GRID_IK_MAX_KA  # 2880.7 MVA
GRID_RX_MAX = 0.1                                           # as in 2B (verify)
# Minimum short-circuit level: verify (kept at the Stage A value).
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
    sb = pp.create_bus(net, vn_kv=SUB_HV_KV, name="sourcebus")
    eg = net.ext_grid.index[0]
    net.ext_grid.at[eg, "bus"] = sb
    net.ext_grid.at[eg, "name"] = "Grid_66kV"
    net.ext_grid.at[eg, "s_sc_max_mva"] = GRID_S_SC_MAX_MVA
    net.ext_grid.at[eg, "rx_max"] = GRID_RX_MAX
    vk = math.hypot(st["r_pct"], st["x_pct"])
    b650 = int(net.bus.index[net.bus.name == "650"][0])
    # Zero sequence: vk0 = vk, vkr0 = vkr (R0 = R1, verify), magnetising branch open (X.3.2 item 14).
    pp.create_transformer_from_parameters(
        net, hv_bus=sb, lv_bus=b650, sn_mva=st["kva"] / 1e3,
        vn_hv_kv=SUB_HV_KV, vn_lv_kv=V_STAGE_B_KV,
        vkr_percent=st["r_pct"], vk_percent=vk, pfe_kw=0.0, i0_percent=0.0,
        vector_group="Dyn", vk0_percent=vk, vkr0_percent=st["r_pct"],
        mag0_percent=MAG0_PERCENT, mag0_rx=MAG0_RX, si0_hv_partial=0.9, shift_degree=30,
        tap_side="lv", tap_neutral=0, tap_min=SUB_TAP_MIN,
        tap_max=SUB_TAP_MAX, tap_step_percent=SUB_TAP_STEP_PERCENT,
        tap_pos=0, tap_changer_type="Ratio", name="Sub_XFMR",
    )
    reg = net.trafo.name == "Regulator_650_RG60"
    net.trafo.loc[reg, ["vk_percent", "vk0_percent"]] = REG_STANDIN_VK_PERCENT_B4
    return (f"{SUB_HV_KV:.0f} kV sourcebus (S_sc {GRID_S_SC_MAX_MVA:.1f} MVA) -> Sub_XFMR "
            f"{st['kva']} kVA Dyn vk {vk:.3f} % -> 650; "
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


# --------------------------------------------------------------------------
# B8: distribution transformers (Part 2C.1, built in checkpoint 3)
# --------------------------------------------------------------------------
# IS 1180 (Part 1):2014, outdoor oil-immersed DTs up to 2500 kVA:
# - Cl. 6.1 Table 1 (PDF p. 4) and Cl. 7.1 Table 4 (PDF p. 5): ratings.  The
#   non-preferred ratings 6.3/10/20/40 kVA are left out (none is needed).
# - Cl. 6.5 / 7.5: no-load ratio 11 000/433 V.  Cl. 6.6 / 7.6: Dyn11.
# - Table 3 (PDF p. 5) / Table 6 (PDF p. 6): impedance 4.5 % up to 630 kVA
#   (5.0 % for 1000-1250 kVA) and the maximum total loss at 50 % / 100 % load,
#   energy-efficiency level 1 (the highest permissible losses).
# Losses split (stated interpretation): P_k = (T100 - T50) / 0.75,
# P_0 = T100 - P_k.  No-load current is not taken from the standard: the
# magnetising branch carries only P_0 (i0 = P_0 / S_r, stated).
# Zero-sequence data: vk0 = vk, vkr0 = vkr (R0 = R1, verify), si0_hv_partial 0.9,
# magnetising branch effectively open (MAG0_PERCENT; brief v2 X.3.2 item 14).
DT_LV_KV = 0.433
DT_VECTOR_GROUP = "Dyn"
DT_SHIFT_DEGREE = 330                 # Dyn11: LV leads HV by 30 deg (330 deg lag)
S_MAX_KVA = 630.0                     # brief v2 X.2 (verify vs Indian urban practice)
IS1180_RATINGS_KVA = (16, 25, 63, 100, 160, 200, 250, 315, 400, 500, 630,
                      1000, 1250, 1600, 2000, 2500)
IS1180_IMPEDANCE_PCT = {r: 4.5 for r in IS1180_RATINGS_KVA if r <= 630}
IS1180_IMPEDANCE_PCT.update({1000: 5.0, 1250: 5.0})
IS1180_LOSS_W = {   # level 1: (total loss at 50 % load, at 100 % load), W
    16: (150, 480), 25: (210, 695), 63: (380, 1250), 100: (520, 1800),
    160: (770, 2200), 200: (890, 2700), 250: (1050, 3150), 315: (1100, 3275),
    400: (1300, 3875), 500: (1600, 4750), 630: (2000, 5855),
}
XFM1_BUS = "634"                      # XFM-1 becomes one 500 kVA DT (Part 2C.1)
XFM1_DT_KVA = 500


def dt_losses_w(rating_kva: int) -> tuple[float, float]:
    """(P_0, P_k) in W from the IS 1180 level-1 total losses at 50 % / 100 %."""
    t50, t100 = IS1180_LOSS_W[rating_kva]
    pk = (t100 - t50) / 0.75
    return t100 - pk, pk


def r0_eff_pu(trafo: pd.Series) -> float:
    """Effective zero-sequence resistance seen from the LV side of a Dyn unit,
    per unit of the rating, from the pandapower Dyn T-model
    (``pandapower/pd2ppc_zero.py``, Dyn branch): Re{z2 + z1 || z3} with
    z1 = si0 x Z0k (HV winding, shorted by the delta), z2 = (1 - si0) x Z0k
    (LV winding), z3 = Z_m0 = mag0_percent/100 x Z_sc (reactive for mag0_rx 0).
    Divided by R0 = vkr0/100 it is 1 when the magnetising branch is open."""
    zk = complex(trafo.vkr0_percent, math.sqrt(trafo.vk0_percent ** 2 - trafo.vkr0_percent ** 2)) / 100
    zsc = complex(trafo.vkr_percent, math.sqrt(trafo.vk_percent ** 2 - trafo.vkr_percent ** 2)) / 100
    zm_abs = abs(zsc) * trafo.mag0_percent / 100
    rx = trafo.mag0_rx
    zm = complex(zm_abs / math.sqrt(rx ** 2 + 1) * rx, zm_abs / math.sqrt(rx ** 2 + 1))
    z1, z2 = trafo.si0_hv_partial * zk, (1 - trafo.si0_hv_partial) * zk
    return float((z2 + z1 * zm / (z1 + zm)).real)


def size_dts(s_phase_max_kva: float, s_max_kva: float = S_MAX_KVA) -> tuple[int, int]:
    """k and rating with rating/3 >= the bus's maximum per-phase kVA / k (brief v2 X.3 item 3).

    k = ceil(3 S_phase,max / S_max) DTs, each the smallest IS 1180 rating
    >= 3 S_phase,max / k.
    """
    s_need = 3.0 * s_phase_max_kva
    k = max(1, math.ceil(s_need / s_max_kva - 1e-9))
    need = s_need / k
    rating = next(r for r in IS1180_RATINGS_KVA if r >= need - 1e-9)
    return k, rating


def b8_distribution_transformers(net) -> str:
    """Replace every load by k DTs (11/0.433 kV Dyn11) with their own LV buses.

    The bus's design-peak per-phase P/Q at LV is split equally over its DTs
    (equal ratings at a bus) and put on one wye load per DT at the LV bus.
    LV phasing (brief v2 X.3 item 2, stated deviation from IEEE phasing):
    three-phase IEEE buses keep the IEEE per-phase values, single- and
    two-phase IEEE buses spread their load equally over a/b/c
    (:func:`households.lv_phasing`).  Delta pairs AB/BC/CA map to LV phases
    a/b/c (Dyn11: an LV phase-a load draws 11 kV current in lines A and B).
    DTs are sized on the maximum per-phase kVA (:func:`size_dts`), except
    XFM-1 (633-634), which stays one 500 kVA DT whose LV bus is node 634
    (thermal stress case).  The nine load points are kept in
    ``net["load_points"]`` (same load_ids), the DTs in ``net["dts"]``.
    """
    from .feeder_model import _add_modelled_load, get_load_points
    from .households import lv_phasing
    lp = get_load_points(net)
    al = net.asymmetric_load
    rows, msgs = [], []
    x1 = net.trafo.index[net.trafo.name == "XFM-1"]
    hv_634 = int(net.trafo.at[x1[0], "hv_bus"])
    b634 = int(net.trafo.at[x1[0], "lv_bus"])
    net.trafo.drop(x1, inplace=True)
    for _, r in lp.iterrows():
        li = int(al.index[al.name == r.load_id][0])
        bus = int(al.at[li, "bus"])
        p = np.array([al.at[li, f"p0_{ph}_mw"] for ph in "abc"]) * 1e3   # kW
        q = np.array([al.at[li, f"q0_{ph}_mvar"] for ph in "abc"]) * 1e3
        p, q = lv_phasing(r.bus, p, q)
        s_bus = math.hypot(p.sum(), q.sum())
        s_ph_max = float(np.hypot(p, q).max())
        if r.bus == XFM1_BUS:
            k, rating, hv = 1, XFM1_DT_KVA, hv_634
        else:
            (k, rating), hv = size_dts(s_ph_max), bus
        p0_w, pk_w = dt_losses_w(rating)
        vk = IS1180_IMPEDANCE_PCT[rating]
        vkr = pk_w / (rating * 1e3) * 100
        for j in range(1, k + 1):
            name = f"dt_{r.bus}_{j}"
            if r.bus == XFM1_BUS:
                lv = b634
                net.bus.at[lv, "vn_kv"] = DT_LV_KV
            else:
                lv = pp.create_bus(net, vn_kv=DT_LV_KV, name=f"{r.bus}_lv{j}")
            t = pp.create_transformer_from_parameters(
                net, hv_bus=hv, lv_bus=lv, sn_mva=rating / 1e3,
                vn_hv_kv=V_STAGE_B_KV, vn_lv_kv=DT_LV_KV, vkr_percent=vkr,
                vk_percent=vk, pfe_kw=p0_w / 1e3, i0_percent=p0_w / (rating * 1e3) * 100,
                vector_group=DT_VECTOR_GROUP, vk0_percent=vk, vkr0_percent=vkr,
                mag0_percent=MAG0_PERCENT, mag0_rx=MAG0_RX, si0_hv_partial=0.9,
                shift_degree=DT_SHIFT_DEGREE, name=name,
            )
            ld = _add_modelled_load(net, lv, f"dtload_{r.bus}_{j}", "Y", "PQ",
                                    tuple(p / k), tuple(q / k))
            rows.append({
                "name": name, "load_id": r.load_id, "bus": r.bus, "k": k, "index": j,
                "rating_kva": rating, "trafo": t, "hv_bus": hv, "lv_bus": lv, "load": ld,
                "vk_percent": vk, "p0_w": p0_w, "pk_w": pk_w,
                **{f"p_{ph}_kw": v / k for ph, v in zip("abc", p)},
                **{f"q_{ph}_kvar": v / k for ph, v in zip("abc", q)},
            })
        msgs.append(f"{r.bus}: S {s_bus:.0f} kVA, 3 x S_phase,max {3 * s_ph_max:.0f} kVA "
                    f"-> {k} x {rating} kVA")
        al.drop(li, inplace=True)
    net["load_points"] = lp.reset_index(drop=True)
    net["dts"] = pd.DataFrame(rows)
    return (f"{len(rows)} DTs 11/{DT_LV_KV} kV Dyn11 (IS 1180), S_max {S_MAX_KVA:.0f} kVA: "
            + "; ".join(msgs))


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
