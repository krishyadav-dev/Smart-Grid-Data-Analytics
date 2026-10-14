"""
Fault event bursts on the Stage B feeder (brief v2 §2D with X.1, X.3.1 items
8-10 and 12, X.3.2 items 11/18 and 14-17 and 23, X.3.3 items 2-4).

Method (stated): sequence-network superposition
-----------------------------------------------
Grainger & Stevenson, *Power System Analysis* (1994) §12.1-12.4, book
pp. 470-499.

- **Network matrices.**  Y0, Y1, Y2 are the sequence bus admittance matrices
  that ``pandapower.runpp_3ph`` itself builds from the network (same internal
  conversion, :func:`build_sequence_network`).  The external grid's sequence
  source admittance, Z_Q = c U_n^2 / S_sc'' with c = 1.1 as pandapower uses it
  (``verify``: the IEC 60909-0 voltage factor; the standard is not in
  docs/References, so it is not cited as a verified source), is added to Y1,
  Y2 and Y0 at the source bus before inverting (X.3.2 item 15); each
  matrix's condition number is checked.  **Loads are left out** of the fault
  (Delta V) network (brief X.3.2 item 16); shunt capacitors, line charging
  and DT magnetising branches stay in Y, so the pre-fault state and the
  fault network are the same network (user decision, default #4).
- **3PH hand check (§2D):** the hand formula (grid + Sub_XFMR at its tap +
  line RG60-632) is compared with the engine on the same network with all
  shunt elements removed (:func:`shunt_free_copy`); the shunt effect is
  reported separately as a capacitor cross-check (:func:`capacitor_check`).
  The Dyn units carry their +/-30 deg shifts and the open zero-sequence
  magnetising branch of X.3.2 item 14.
- **Pre-fault state** = ``runpp_3ph`` *with* load at the event's 15-min step
  (CP3 households, the step's OLTC tap).
- **Fault currents** at bus k from the Thevenin impedances Z0kk, Z1kk, Z2kk
  and the pre-fault positive-sequence voltage V_f (G&S §12.1, Eq. 12.1-12.3):
  SLG I0 = I1 = I2 = V_f / (Z0 + Z1 + Z2 + 3 Z_f); LL I1 = -I2 =
  V_f / (Z1 + Z2 + Z_f); LLG with Z_f to ground (G&S §12.4); 3PH I1 =
  V_f / (Z1 + Z_f).  The faulted phase (SLG) or pair (LL, LLG) is applied by
  relabelling the phases cyclically, which is exact because the network's
  sequence data are symmetric.  The pre-fault negative and zero sequence
  voltages (load unbalance, VUF <= 1.7 %) are carried by superposition but
  not used for V_f (stated).
- **During the fault** V = V_pre - Z(:, k) I_f per sequence, at every bus;
  branch currents = Yf V (so the pre-fault load flow is included).
- **DT meter voltages** (X.3.1 item 9): the 11 kV Delta V is carried through
  the Dyn11 DT explicitly, positive sequence x e^{+j30 deg}, negative
  x e^{-j30 deg}, zero sequence blocked, and added to the LV pre-fault
  phasors; the 11 kV per-unit values are not copied.  Cross-checked against
  the LV rows of Z (:func:`dyn11_transfer_check`).  The DT meter current is
  held at its pre-fault value during the fault (constant-PQ loads are out of
  the Delta V network; quasi-steady, stated).
- **Per unit** follows pandapower's three-phase convention (per-phase power
  on ``baseMVA``): I_base = sqrt(3) baseMVA / V_n, Z_base = V_n^2 / (3 baseMVA).
- **11 kV neutral**: Sub_XFMR is Dyn with the 11 kV star solidly earthed
  (GETCO/E/TS-XMER/R13, p. 12).

Protection, clearing and bursts
-------------------------------
- One relay at the feeder head (on the 650 -> 632 feeder, Line_RG60_632), NPAG
  §9.4 Table 9.1 IEC standard inverse t = TMS 0.14 / (I_r^0.02 - 1), TMS 0.1.
  Phase element Is = 1.2 x the maximum feeder-head load current of the CP3
  year (X.1).  Earth-fault element on the residual current 3 I0, pickup
  30 % of the same maximum load current (sourced: NPAG §9.16, PDF p. 145,
  typical 30-40 % of full-load current; lower end of the range), TMS 0.1
  (``verify``).  Limitation (stated): no downstream DT fuse grading is
  modelled; the feeder-head relay clears every fault it detects.
- Fault resistance by type (CP4 follow-up): SLG and LLG through Rf in
  {0, 5, 20, 50} ohm (Dagenhart p. C4-2, ground faults); 3PH and LL through
  Warrington's arc resistance R_a = 28 710 L / I^1.4 (NPAG §11, Eq. 11.4,
  PDF p. 188) at the fault's own current, arc length L = 1.0 m (``verify``).
- Earth-fault security over the CP3 year (max feeder-head |3I0|) is a
  **model-symmetry check, not evidence of relay security**: the residual is
  ~0 because pandapower lines use transposed (symmetric) sequence impedances
  while the IEEE 13 lines are untransposed, the capacitors are symmetric and
  every load is behind a Dyn DT.
- Clearing time = fastest element + breaker breaking time 60 ms (ABB VD4X
  manual p. 7, <= 60 ms); sensitivity case 80 ms (Siemens 3AH3 catalogue
  p. 44, first shunt release).  A fault below both pickups is **not cleared**
  and runs to the 60 s cap (IS 17036 Cl. 3.24).
- Bursts: 10 ms half-cycle RMS values (IS 17036 Cl. 3.18), quasi-steady (no
  transients, DC offset or CT saturation; stated): 0.5 s pre-fault + fault
  until clearing + 1.0 s after.  After clearing the feeder breaker is open: every
  meter downstream (DT and bus_virtual meters) shows an interruption (0 V,
  0 A); the feeder-head meter sits on the bus side of the breaker and shows
  its pre-fault voltage and zero current (stated).  No auto-reclose.
- Classification: IS 17036 Table 13 dip bins, Table 12 short interruption
  (PDF p. 9), thresholds 90 % / 5 % (Cl. 3.23, 3.19).  Polyphase rules (IEC
  61000-4-30, stated): a dip starts when any phase falls below 90 % and ends
  when all phases are back at or above 90 %; the residual is the lowest phase;
  an interruption needs all phases below 5 %.  No hysteresis (stated).
  Comparing each simulated burst's classification with the bins of its own
  simulated residual and duration is a **binning self-consistency check**,
  not validation against measured events.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pandapower as pp

from . import config

logger = logging.getLogger(__name__)

_a = np.exp(2j * np.pi / 3)
A = np.array([[1, 1, 1], [1, _a ** 2, _a], [1, _a, _a ** 2]])     # V_abc = A V_012
A_INV = np.linalg.inv(A)

FAULT_TYPES = ("SLG", "LL", "LLG", "3PH")
RF_OHM = (0.0, 5.0, 20.0, 50.0)          # Dagenhart 2000 p. C4-2 (X.3.2 item 23): ground faults
GROUND_FAULTS = ("SLG", "LLG")           # Rf from RF_OHM (Dagenhart's set is for ground faults)
ARC_FAULTS = ("3PH", "LL")               # arc resistance (Warrington), CP4 follow-up
# Warrington's empirical arc resistance, NPAG §11 Equation 11.4, PDF p. 188
# (A.R. van C. Warrington, Protective Relays, 1962): R_a = 28 710 L / I^1.4,
# R_a in ohm, L arc length in m, I arc current in A.
WARRINGTON_K = 28_710.0
ARC_LENGTH_M = 1.0                       # verify: ~11 kV phase spacing (IEEE spacing 500: 0.76-2.1 m)
SAMPLE_MS = 10
PRE_S, POST_S, CAP_S = 0.5, 1.0, 60.0
N_PRE, N_POST, N_CAP = 50, 100, 6000
TMS = 0.1
IS_FACTOR = 1.2                          # X.1: Is = 1.2 x max feeder-head load current
EF_FACTOR = 0.30                         # NPAG §9.16 p. 145: 30-40 % of full-load current
EF_TMS = 0.1
BREAKER_S = 0.060                        # ABB VD4X manual p. 7
BREAKER_SENS_S = 0.080                   # Siemens 3AH3 catalogue p. 44
AMPACITY_601_A = 730.0                   # config 601 (556.5 ACSR), IEEE 13 Table 3
REMOTE_MARGIN_MIN = 1.5
COND_MAX = 1e12
C_MAX = 1.1                              # grid voltage factor as pandapower uses it (verify: IEC 60909-0, not in References)
DIP_PCT, INTERRUPTION_PCT = 90.0, 5.0
U_BINS = ((80.0, 90.0), (70.0, 80.0), (40.0, 70.0), (5.0, 40.0))          # Table 13 rows
T_BINS_DIP = ((10, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 60000))
T_BINS_INT = ((10, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 180000))
EXCLUDED_BUSES = ("650", "RG60", "sourcebus")   # outside the relay zone / source (X.3.3 item 2)
HEAD_LINE = "Line_RG60_632"
DYN11_SHIFT = np.exp(1j * np.pi / 6)     # LV leads HV by 30 deg (pandapower shift_degree 330)


# --------------------------------------------------------------------------
# Sequence network
# --------------------------------------------------------------------------

@dataclass
class SequenceNetwork:
    z: list                       # dense Z0, Z1, Z2 (ppci order)
    yf: list                      # sparse Yf0, Yf1, Yf2 (branch from-side)
    lookup: np.ndarray            # pandapower bus -> ppci bus
    vn_kv: np.ndarray             # per ppci bus
    base_mva: float
    cond: list
    head_branch: int
    head_vn_kv: float

    def i_base_ka(self, k: int) -> float:
        return math.sqrt(3) * self.base_mva / self.vn_kv[k]

    def z_base_ohm(self, k: int) -> float:
        return self.vn_kv[k] ** 2 / (3 * self.base_mva)


def build_sequence_network(net) -> SequenceNetwork:
    """Y0/Y1/Y2 as ``runpp_3ph`` builds them, with the grid source admittance
    in all three (X.3.2 item 15), inverted to Z.  ``net`` must have been solved
    with ``runpp_3ph`` (its options are reused)."""
    from pandapower.build_bus import _add_ext_grid_sc_impedance
    from pandapower.pf.runpp_3ph import _pd2ppc_recycle
    from pandapower.pypower.makeYbus import makeYbus

    _, p1 = _pd2ppc_recycle(net, 1, recycle=None)
    _add_ext_grid_sc_impedance(net, p1)                 # positive-sequence source
    _, p2 = _pd2ppc_recycle(net, 2, recycle=None)
    _add_ext_grid_sc_impedance(net, p2)                 # negative-sequence source
    _, p0 = _pd2ppc_recycle(net, 0, recycle=None)       # includes the zero-sequence source
    z, yf, cond = [], [], []
    for ppc in (p0, p1, p2):
        y, f, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        yd = y.toarray()
        c = float(np.linalg.cond(yd))
        if not c < COND_MAX:
            raise ValueError(f"sequence admittance matrix ill-conditioned (cond {c:.3e})")
        cond.append(c)
        z.append(np.linalg.inv(yd))
        yf.append(f)
    lookup = np.asarray(net._pd2ppc_lookups["bus"])
    vn = np.zeros(p1["bus"].shape[0])
    vn[lookup[net.bus.index]] = net.bus.vn_kv.to_numpy()
    ln = int(net.line.index[net.line.name == HEAD_LINE][0])
    f0, _ = net._pd2ppc_lookups["branch"]["line"]
    head = int(f0 + net.line.index.get_loc(ln))
    return SequenceNetwork(z, yf, lookup, vn, float(p1["baseMVA"]), cond, head,
                           float(net.bus.at[net.line.at[ln, "from_bus"], "vn_kv"]))


def prefault_v012(net, sn: SequenceNetwork) -> np.ndarray:
    """(3, n_ppci) pre-fault sequence voltages (pu) from the solved ``runpp_3ph``."""
    rb = net.res_bus_3ph
    v = np.zeros((3, len(sn.vn_kv)), complex)
    vm = rb[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].to_numpy()
    va = np.deg2rad(rb[["va_a_degree", "va_b_degree", "va_c_degree"]].to_numpy())
    vabc = (vm * np.exp(1j * va)).T                          # (3, n_pp)
    v[:, sn.lookup[rb.index]] = A_INV @ vabc
    return v


# --------------------------------------------------------------------------
# Faults
# --------------------------------------------------------------------------

def _special_phase(ftype: str, phases: str) -> int:
    """Index of the phase that plays 'a' in G&S: the faulted phase (SLG) or
    the healthy phase (LL, LLG); 0 for 3PH."""
    if ftype == "SLG":
        return "abc".index(phases)
    if ftype in ("LL", "LLG"):
        return "abc".index(next(p for p in "abc" if p not in phases))
    return 0


def fault_currents_012(z0: complex, z1: complex, z2: complex, vf: complex, ftype: str,
                       phases: str, zf: complex) -> np.ndarray:
    """Sequence fault currents leaving the network at the fault bus (pu).

    G&S formulas with the special phase as reference, then relabelled back.
    """
    s = _special_phase(ftype, phases)
    vfr = vf * _a ** (-s)                                    # V_f seen from the special phase
    if ftype == "SLG":
        i0 = i1 = i2 = vfr / (z0 + z1 + z2 + 3 * zf)
    elif ftype == "LL":
        i1 = vfr / (z1 + z2 + zf)
        i2, i0 = -i1, 0.0
    elif ftype == "LLG":
        zg = z0 + 3 * zf
        i1 = vfr / (z1 + z2 * zg / (z2 + zg))
        i2 = -i1 * zg / (z2 + zg)
        i0 = -i1 * z2 / (z2 + zg)
    elif ftype == "3PH":
        i1, i2, i0 = vfr / (z1 + zf), 0.0, 0.0
    else:
        raise ValueError(ftype)
    i_rot = A @ np.array([i0, i1, i2], complex)              # phases in the relabelled frame
    i_abc = np.empty(3, complex)
    for j in range(3):
        i_abc[(j + s) % 3] = i_rot[j]
    return A_INV @ i_abc


@dataclass
class FaultResult:
    bus: str
    ftype: str
    phases: str
    rf_ohm: float
    i_fault_ka: np.ndarray        # phase magnitudes at the fault (kA)
    v_pre: np.ndarray             # (3, n_ppci) sequence pu
    v_fault: np.ndarray           # (3, n_ppci) sequence pu
    i_head_pre_a: np.ndarray      # phase magnitudes (A) at the relay
    i_head_fault_a: np.ndarray
    i_head_residual_a: float      # |3 I0| during the fault (A)


def simulate_fault(net, sn: SequenceNetwork, bus: str, ftype: str, phases: str,
                   rf_ohm: float, v_pre: Optional[np.ndarray] = None) -> FaultResult:
    """Fault-on state by superposition (module docstring)."""
    b = int(net.bus.index[net.bus.name == bus][0])
    k = int(sn.lookup[b])
    v_pre = prefault_v012(net, sn) if v_pre is None else v_pre
    zf = rf_ohm / sn.z_base_ohm(k)
    i012 = fault_currents_012(sn.z[0][k, k], sn.z[1][k, k], sn.z[2][k, k], v_pre[1, k],
                              ftype, phases, zf)
    dv = np.array([-sn.z[s][:, k] * i012[s] for s in range(3)])
    v_f = v_pre + dv
    i_fault = np.abs(A @ i012) * sn.i_base_ka(k)
    ib = math.sqrt(3) * sn.base_mva / sn.head_vn_kv * 1e3          # A
    h_pre = np.array([sn.yf[s][sn.head_branch] @ v_pre[s] for s in range(3)]).ravel()
    h_f = np.array([sn.yf[s][sn.head_branch] @ v_f[s] for s in range(3)]).ravel()
    return FaultResult(bus, ftype, phases, rf_ohm, i_fault, v_pre, v_f,
                       np.abs(A @ h_pre) * ib, np.abs(A @ h_f) * ib, float(abs(3 * h_f[0]) * ib))


def warrington_ohm(i_a: float, length_m: float = ARC_LENGTH_M) -> float:
    """Arc resistance R_a = 28 710 L / I^1.4 (NPAG Eq. 11.4, PDF p. 188)."""
    return WARRINGTON_K * length_m / i_a ** 1.4


def simulate_arc_fault(net, sn: SequenceNetwork, bus: str, ftype: str, phases: str,
                       v_pre: Optional[np.ndarray] = None, length_m: float = ARC_LENGTH_M,
                       tol_ohm: float = 1e-6, max_iter: int = 100) -> FaultResult:
    """3PH / LL fault through an arc: Rf = Warrington R_a at the fault's own current.

    Fixed-point iteration from a bolted fault: I(R) falls as R rises, so the
    iteration converges.  3PH: R_a in each phase to the fault point; LL: R_a
    between the two phases (G&S Z_f).  Arc length ``length_m`` is ``verify``.
    """
    if ftype not in ARC_FAULTS:
        raise ValueError(f"arc model applies to {ARC_FAULTS}, not {ftype}")
    v_pre = prefault_v012(net, sn) if v_pre is None else v_pre
    rf = 0.0
    for _ in range(max_iter):
        res = simulate_fault(net, sn, bus, ftype, phases, rf, v_pre)
        rf_new = warrington_ohm(1e3 * float(res.i_fault_ka.max()), length_m)
        if abs(rf_new - rf) < tol_ohm:
            break
        rf = rf_new
    else:
        raise RuntimeError(f"arc resistance did not converge at {bus} {ftype}")
    return simulate_fault(net, sn, bus, ftype, phases, rf_new, v_pre)


def dyn11_lv_v012(sn: SequenceNetwork, v_pre: np.ndarray, v_fault: np.ndarray,
                  hv: int, lv: int) -> np.ndarray:
    """LV sequence voltages during the fault through a Dyn11 DT (X.3.1 item 9):
    LV pre-fault + 11 kV Delta V with +30 deg (positive), -30 deg (negative),
    zero sequence blocked."""
    dv = v_fault[:, hv] - v_pre[:, hv]
    return v_pre[:, lv] + np.array([0.0, dv[1] * DYN11_SHIFT, dv[2] / DYN11_SHIFT])


def dyn11_transfer_check(net, sn: SequenceNetwork, res: FaultResult) -> float:
    """Max |V_LV(Dyn11 transform) - V_LV(Z rows)| over all DT LV buses (pu, phase)."""
    worst = 0.0
    for d in net["dts"].itertuples():
        hv, lv = int(sn.lookup[int(d.hv_bus)]), int(sn.lookup[int(d.lv_bus)])
        explicit = A @ dyn11_lv_v012(sn, res.v_pre, res.v_fault, hv, lv)
        direct = A @ res.v_fault[:, lv]
        worst = max(worst, float(np.max(np.abs(explicit - direct))))
    return worst


# --------------------------------------------------------------------------
# Relay
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RelaySettings:
    is_a: float
    tms: float = TMS
    ef_is_a: float = 0.0
    ef_tms: float = EF_TMS
    breaker_s: float = BREAKER_S


def idmt_time(i_a: float, is_a: float, tms: float) -> float:
    """NPAG Table 9.1 IEC standard inverse; inf at or below pickup."""
    r = i_a / is_a
    return math.inf if r <= 1.0 else tms * 0.14 / (r ** 0.02 - 1.0)


def relay_operate(relay: RelaySettings, i_phase_a: float, i_residual_a: float) -> tuple[float, str]:
    """(relay operating time s, element) of the fastest element; (inf, 'none') if neither picks up."""
    tp = idmt_time(i_phase_a, relay.is_a, relay.tms)
    te = idmt_time(i_residual_a, relay.ef_is_a, relay.ef_tms) if relay.ef_is_a > 0 else math.inf
    if math.isinf(tp) and math.isinf(te):
        return math.inf, "none"
    return (tp, "phase") if tp <= te else (te, "earth_fault")


def relay_settings(max_head_current_a: float, breaker_s: float = BREAKER_S) -> RelaySettings:
    return RelaySettings(is_a=IS_FACTOR * max_head_current_a, tms=TMS,
                         ef_is_a=EF_FACTOR * max_head_current_a, ef_tms=EF_TMS, breaker_s=breaker_s)


def max_head_current_a(cache=config.YEAR_CACHE_DIR) -> float:
    """Maximum feeder-head load current over the CP3 year (A, any phase)."""
    from pathlib import Path
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["load_id", "load_current_a"],
                        filters=[("load_id", "==", "feeder_head_650")])
    return float(f.load_current_a.max())


def fault_buses(net) -> list[str]:
    """11 kV buses downstream of the feeder relay (X.3.3 item 2)."""
    mv = net.bus[net.bus.vn_kv == config.FEEDER_VOLTAGE_KV]
    return [str(n) for n in mv.name if str(n) not in EXCLUDED_BUSES]


# --------------------------------------------------------------------------
# Meters during the fault, bursts and classification
# --------------------------------------------------------------------------

@dataclass
class MeterFault:
    load_id: str
    role: str
    v_base_ln_v: float
    v_pre: np.ndarray             # (3,) V
    v_fault: np.ndarray
    i_pre: np.ndarray             # (3,) A
    i_fault: np.ndarray
    downstream: bool              # loses supply when the feeder breaker opens


def meter_faults(net, sn: SequenceNetwork, res: FaultResult) -> dict[str, MeterFault]:
    """Fault-on V and I at every meter point (feeder head, DTs, bus_virtual)."""
    from .feeder_model import FEEDER_HEAD_ID, meter_points
    out: dict[str, MeterFault] = {}
    rt = net.res_trafo_3ph
    dts = net["dts"]
    dt_v: dict[str, tuple] = {}
    for d in dts.itertuples():
        hv, lv = int(sn.lookup[int(d.hv_bus)]), int(sn.lookup[int(d.lv_bus)])
        vb = sn.vn_kv[lv] * 1e3 / math.sqrt(3)
        v_pre = np.abs(A @ res.v_pre[:, lv]) * vb
        v_f = np.abs(A @ dyn11_lv_v012(sn, res.v_pre, res.v_fault, hv, lv)) * vb
        i = rt.loc[int(d.trafo), ["i_a_lv_ka", "i_b_lv_ka", "i_c_lv_ka"]].to_numpy(float) * 1e3
        out[d.name] = MeterFault(d.name, "dt", vb, v_pre, v_f, i, i.copy(), True)
        dt_v[d.name] = (v_pre, v_f, i)
    b650 = int(sn.lookup[int(net.bus.index[net.bus.name == "650"][0])])
    vb = sn.vn_kv[b650] * 1e3 / math.sqrt(3)
    out[FEEDER_HEAD_ID] = MeterFault(FEEDER_HEAD_ID, "feeder_head", vb,
                                     np.abs(A @ res.v_pre[:, b650]) * vb,
                                     np.abs(A @ res.v_fault[:, b650]) * vb,
                                     res.i_head_pre_a, res.i_head_fault_a, False)
    for r in meter_points(net).itertuples():
        if r.role != "bus_virtual":
            continue
        names = dts.name[dts.load_id == r.load_id]
        vp = np.mean([dt_v[n][0] for n in names], axis=0)
        vf = np.mean([dt_v[n][1] for n in names], axis=0)
        i = np.sum([dt_v[n][2] for n in names], axis=0)
        out[r.load_id] = MeterFault(r.load_id, "bus_virtual", out[names.iloc[0]].v_base_ln_v,
                                    vp, vf, i, i.copy(), True)
    return out


def burst_lengths(t_clear_s: float) -> tuple[int, int, int]:
    """(pre, fault, post) sample counts; a fault that is not cleared runs to the cap."""
    if math.isinf(t_clear_s) or t_clear_s >= CAP_S:
        return N_PRE, N_CAP, 0
    return N_PRE, max(1, int(math.ceil(round(t_clear_s * 1000 / SAMPLE_MS, 6)))), N_POST


def build_burst(m: MeterFault, t_clear_s: float) -> tuple[np.ndarray, np.ndarray]:
    """10 ms RMS samples (n, 3) of V and I for one meter."""
    n_pre, n_f, n_post = burst_lengths(t_clear_s)
    if m.downstream:
        v_post, i_post = np.zeros(3), np.zeros(3)
    else:
        v_post, i_post = m.v_pre, np.zeros(3)
    v = np.vstack([np.tile(m.v_pre, (n_pre, 1)), np.tile(m.v_fault, (n_f, 1)),
                   np.tile(v_post, (n_post, 1))])
    i = np.vstack([np.tile(m.i_pre, (n_pre, 1)), np.tile(m.i_fault, (n_f, 1)),
                   np.tile(i_post, (n_post, 1))])
    return v, i


def _bin(value: float, bins, upper_inclusive: bool = True) -> Optional[str]:
    for lo, hi in bins:
        if (lo < value <= hi) if upper_inclusive else (lo <= value < hi):
            return f"{lo:g}-{hi:g}"
    return None


def u_bin(residual_pct: float) -> str:
    """Table 13 residual-voltage row (lo <= u < hi); '<5' below the table."""
    if residual_pct < U_BINS[-1][0]:
        return "<5"
    return _bin(residual_pct, U_BINS, upper_inclusive=False)


def t_bin(duration_ms: float, bins=T_BINS_DIP) -> Optional[str]:
    """Table 12/13 duration column; the first column includes 10 ms (10 <= t <= 200)."""
    if bins[0][0] <= duration_ms <= bins[0][1]:
        return f"{bins[0][0]:g}-{bins[0][1]:g}"
    return _bin(duration_ms, bins[1:])


@dataclass
class Classification:
    kind: str                     # no_dip | dip | dip_not_cleared | interruption
    residual_pct: Optional[float] = None
    duration_ms: Optional[float] = None
    u_bin: Optional[str] = None
    t_bin: Optional[str] = None
    interruption_ms: Optional[float] = None     # interruption after the dip (or alone)
    interruption_open_ended: bool = False
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.kind, self.u_bin, self.t_bin, self.interruption_open_ended)


def classify(v_rms: np.ndarray, v_declared: float, interval_ms: int = SAMPLE_MS,
             phases: str = "abc") -> Classification:
    """IS 17036 classification of a burst's RMS samples (module docstring)."""
    cols = ["abc".index(p) for p in phases]
    u = 100.0 * np.asarray(v_rms, float)[:, cols] / v_declared
    n = len(u)
    below = u.min(axis=1) < DIP_PCT
    if not below.any():
        return Classification("no_dip")
    i0 = int(np.argmax(below))
    rec = np.flatnonzero(~below[i0:])
    i_end = i0 + int(rec[0]) if rec.size else n
    allint = (u[i0:i_end] < INTERRUPTION_PCT).all(axis=1)
    # interruption = the trailing run of all-phase < 5 % samples inside the event
    tail = allint[::-1]
    run = int(np.argmin(tail)) if not tail.all() else len(tail)
    j = i_end - run
    open_ended = i_end == n
    if j == i0:                                        # interruption only
        dur = (i_end - i0) * interval_ms
        return Classification("interruption", residual_pct=float(u[i0:i_end].min()),
                              interruption_ms=dur, t_bin=t_bin(dur, T_BINS_INT),
                              interruption_open_ended=open_ended)
    dur = (j - i0) * interval_ms
    resid = float(u[i0:j].min())
    kind = "dip"
    if j == i_end and open_ended and dur >= CAP_S * 1000:
        kind = "dip_not_cleared"
    return Classification(kind, residual_pct=resid, duration_ms=dur, u_bin=u_bin(resid),
                          t_bin=t_bin(dur), interruption_ms=(i_end - j) * interval_ms if j < i_end else None,
                          interruption_open_ended=open_ended and j < i_end)


def expected_classification(m: MeterFault, t_clear_s: float) -> Classification:
    """What the classifier must return for this meter, from the simulated fault-on
    values and the clearing time (X.3.1 item 10)."""
    n_pre, n_f, n_post = burst_lengths(t_clear_s)
    u_f = 100.0 * float(m.v_fault.min()) / m.v_base_ln_v
    all_int_f = bool((100.0 * m.v_fault / m.v_base_ln_v < INTERRUPTION_PCT).all())
    cleared = n_post > 0
    post_int = cleared and m.downstream
    if u_f >= DIP_PCT:
        if post_int:
            return Classification("interruption", residual_pct=0.0, interruption_ms=n_post * SAMPLE_MS,
                                  t_bin=t_bin(n_post * SAMPLE_MS, T_BINS_INT), interruption_open_ended=True)
        return Classification("no_dip")
    if all_int_f:
        dur = (n_f + (n_post if post_int else 0)) * SAMPLE_MS
        return Classification("interruption", residual_pct=u_f, interruption_ms=dur,
                              t_bin=t_bin(dur, T_BINS_INT),
                              interruption_open_ended=post_int or not cleared)
    dur = n_f * SAMPLE_MS
    kind = "dip_not_cleared" if not cleared else "dip"
    return Classification(kind, residual_pct=u_f, duration_ms=dur, u_bin=u_bin(u_f), t_bin=t_bin(dur),
                          interruption_ms=n_post * SAMPLE_MS if post_int else None,
                          interruption_open_ended=post_int)


# --------------------------------------------------------------------------
# Event set
# --------------------------------------------------------------------------

def make_event_set(net, n_faults: int = 200, n_quiet: int = 200,
                   rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """200 faults + 200 quiet windows, seeded (stream ``faults.events``).

    Fault type uniform (balanced by design for classifier evaluation, not
    field fault frequencies; X.3.3 item 4); bus uniform over the 11 kV buses
    downstream of the relay; SLG phase / LL-LLG pair uniform; time uniform
    over the 2024 steps plus a uniform offset inside the step.  Fault
    resistance by type (CP4 follow-up): SLG and LLG take Rf uniform over
    {0, 5, 20, 50} ohm (Dagenhart, ground faults); 3PH and LL are arc faults
    (``rf_model`` "arc", Warrington R_a at the fault's own current,
    :func:`simulate_arc_fault`).  The same random draws as before are made
    for every event (the Rf drawn for an arc fault is discarded), so with the
    same seed the buses, times, types and phases are unchanged.
    """
    rng = config.rng("faults.events") if rng is None else rng
    steps = pd.date_range(f"{config.SIMULATION_YEAR}-01-01", f"{config.SIMULATION_YEAR}-12-31 23:45",
                          freq=config.SIMULATION_FREQ)
    buses = fault_buses(net)
    rows = []
    for i in range(n_faults + n_quiet):
        fault = i < n_faults
        ftype = str(rng.choice(FAULT_TYPES)) if fault else "none"
        if ftype == "SLG":
            ph = str(rng.choice(list("abc")))
        elif ftype in ("LL", "LLG"):
            ph = str(rng.choice(["ab", "bc", "ca"]))
        else:
            ph = "abc" if fault else ""
        row = {"event_id": i + 1, "kind": "fault" if fault else "quiet",
               "step": steps[int(rng.integers(len(steps)))],
               "offset_s": float(np.round(rng.uniform(0, 900), 2)),
               "bus": str(rng.choice(buses)) if fault else "",
               "ftype": ftype, "phases": ph}
        rf = float(rng.choice(RF_OHM)) if fault else np.nan        # drawn for every fault (same stream)
        arc = ftype in ARC_FAULTS
        row["rf_model"] = "arc" if arc else ("fixed" if fault else "")
        row["rf_ohm"] = np.nan if arc else rf                      # arc: set by simulate_arc_fault
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Network-level checks: 3PH hand calculation, remote bus, relay margins
# --------------------------------------------------------------------------

def thevenin_ohm(sn: SequenceNetwork, net, bus: str) -> tuple[complex, complex, complex]:
    """(Z0, Z1, Z2) at ``bus`` in ohm (per phase)."""
    k = int(sn.lookup[int(net.bus.index[net.bus.name == bus][0])])
    zb = sn.z_base_ohm(k)
    return tuple(complex(sn.z[s][k, k]) * zb for s in range(3))


def three_phase_currents_ka(net, sn: SequenceNetwork, v_pre: np.ndarray) -> dict[str, float]:
    """Bolted 3PH fault current (kA) at every fault bus."""
    out = {}
    for b in fault_buses(net):
        out[b] = float(simulate_fault(net, sn, b, "3PH", "abc", 0.0, v_pre).i_fault_ka.max())
    return out


def hand_3ph_632_ka(net, v_pre_pu: complex) -> dict:
    """Hand calculation of the bolted 3PH current at 632 (X.3.1 item 8, §2D):
    I = |V_pre,632| / |Z_grid + Z_sub(tap) + Z_line(RG60-632)|, all referred to
    11 kV at the step's tap (ratio tap on the LV winding: impedances on that
    side scale with the tapped voltage squared).  Uses the same pre-fault
    positive-sequence voltage as the engine."""
    eg = net.ext_grid.iloc[0]
    sub = net.trafo[net.trafo.name == "Sub_XFMR"].iloc[0]
    v_lv_t = sub.vn_lv_kv * (1 + sub.tap_pos * sub.tap_step_percent / 100)
    zq = C_MAX * sub.vn_hv_kv ** 2 / eg.s_sc_max_mva     # ohm at 66 kV; c = 1.1 verify
    xq = zq / math.sqrt(1 + eg.rx_max ** 2)
    z_grid = complex(eg.rx_max * xq, xq) * (v_lv_t / sub.vn_hv_kv) ** 2
    zt = sub.vk_percent / 100 * v_lv_t ** 2 / sub.sn_mva
    rt = sub.vkr_percent / 100 * v_lv_t ** 2 / sub.sn_mva
    z_sub = complex(rt, math.sqrt(zt ** 2 - rt ** 2))
    ln = net.line[net.line.name == HEAD_LINE].iloc[0]
    z_line = complex(ln.r_ohm_per_km, ln.x_ohm_per_km) * ln.length_km
    v_ph = abs(v_pre_pu) * net.bus.at[int(net.bus.index[net.bus.name == "632"][0]), "vn_kv"] / math.sqrt(3)
    z = z_grid + z_sub + z_line
    return {"i_ka": v_ph / abs(z), "tap": int(sub.tap_pos), "v_pre_pu": abs(v_pre_pu),
            "z_grid_ohm": z_grid, "z_sub_ohm": z_sub, "z_line_ohm": z_line, "z_total_ohm": z}


def shunt_free_copy(net):
    """Copy of the network with every shunt element removed (capacitors, line
    charging, DT magnetising branches), same tap and loads, solved.  Used only
    for the §2D hand check; the event simulation keeps the full network."""
    import copy
    n = copy.deepcopy(net)
    n.shunt["in_service"] = False
    n.line[["c_nf_per_km", "c0_nf_per_km"]] = 0.0
    n.trafo[["i0_percent", "pfe_kw"]] = 0.0
    pp.runpp_3ph(n)
    return n


def check_3ph_632(net) -> dict:
    """§2D acceptance (user decision 2026-10-14, option b): engine with all
    shunts removed vs the hand formula, same pre-fault voltage and tap (from
    the full network), within 2 %.  ``net`` must be solved."""
    sn = build_sequence_network(net)
    v_pre = prefault_v012(net, sn)
    k = int(sn.lookup[int(net.bus.index[net.bus.name == "632"][0])])
    hand = hand_3ph_632_ka(net, v_pre[1, k])
    free = shunt_free_copy(net)
    sn_free = build_sequence_network(free)
    eng_free = simulate_fault(free, sn_free, "632", "3PH", "abc", 0.0, v_pre)
    eng_full = simulate_fault(net, sn, "632", "3PH", "abc", 0.0, v_pre)
    z_full = thevenin_ohm(sn, net, "632")[1]
    z_free = thevenin_ohm(sn_free, free, "632")[1]
    err = float(eng_free.i_fault_ka.max() / hand["i_ka"] - 1)
    return {"hand_ka": hand["i_ka"], "engine_no_shunt_ka": float(eng_free.i_fault_ka.max()),
            "engine_full_ka": float(eng_full.i_fault_ka.max()), "rel_error": err,
            "passed": abs(err) <= 0.02, "tap": hand["tap"], "v_pre_pu": hand["v_pre_pu"],
            "z_hand_ohm": hand["z_total_ohm"], "z1_no_shunt_ohm": z_free, "z1_full_ohm": z_full}


def capacitor_check(net, c: Optional[dict] = None) -> dict:
    """Reported cross-check of the shunt effect at 632 (not asserted):
    |Z_th,full| / |Z_th,no-shunt| vs 1 / (1 - |Z_src,632| / Xc,eq), Xc,eq = V^2 / Q
    of the 675 + 611 capacitors at their modelled kvar (V = 11 kV line-to-line)."""
    c = check_3ph_632(net) if c is None else c
    q_mvar = float(-net.shunt.loc[net.shunt.in_service, "q_mvar"].sum())
    vn = float(net.shunt.vn_kv.iloc[0])
    xc = vn ** 2 / q_mvar
    hand_ratio = 1.0 / (1.0 - abs(c["z_hand_ohm"]) / xc)
    return {"q_caps_mvar": q_mvar, "xc_eq_ohm": xc, "z_src_632_ohm": abs(c["z_hand_ohm"]),
            "engine_ratio": abs(c["z1_full_ohm"]) / abs(c["z1_no_shunt_ohm"]), "hand_ratio": hand_ratio}


def head_residual_from_cache(cache=config.YEAR_CACHE_DIR) -> dict:
    """Maximum feeder-head residual current |3 I0| over the CP3 year (A), from the
    cached meter frame: I_phase = conj(S / V) with the solved voltage angle."""
    from pathlib import Path
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "phase", "voltage_v", "voltage_angle_deg",
                                 "active_power_kw", "reactive_power_kvar"],
                        filters=[("load_id", "==", "feeder_head_650")])
    v = f.voltage_v.to_numpy() * np.exp(1j * np.deg2rad(f.voltage_angle_deg.to_numpy()))
    s = (f.active_power_kw.to_numpy() + 1j * f.reactive_power_kvar.to_numpy()) * 1e3
    f = f.assign(i=np.conj(s / v))
    res = f.groupby("timestamp").i.sum().abs()
    return {"max_3i0_a": float(res.max()), "at": str(res.idxmax()), "p99_3i0_a": float(res.quantile(0.99)),
            "steps": int(len(res))}


def relay_margins(net, sn: SequenceNetwork, v_pre: np.ndarray, relay: RelaySettings) -> dict:
    """Is < 730 A and remote 3PH I/Is >= 1.5 (asserted by the caller; X.3.1 item 12).
    The remote bus is the fault bus with the lowest bolted 3PH current (X.3.3 item 3)."""
    i3 = three_phase_currents_ka(net, sn, v_pre)
    remote = min(i3, key=i3.get)
    margin = i3[remote] * 1e3 / relay.is_a
    return {"is_a": relay.is_a, "ef_is_a": relay.ef_is_a, "tms": relay.tms, "ef_tms": relay.ef_tms,
            "ampacity_601_a": AMPACITY_601_A, "is_below_ampacity": relay.is_a < AMPACITY_601_A,
            "three_phase_ka": i3, "remote_bus": remote, "remote_3ph_ka": i3[remote],
            "remote_margin": margin, "margin_ok": margin >= REMOTE_MARGIN_MIN}


# --------------------------------------------------------------------------
# Running the event set
# --------------------------------------------------------------------------

def set_prefault(net, p_kw: np.ndarray, q_kvar: np.ndarray, tap: float) -> None:
    """DT loads (n_dt, 3) and the OLTC tap of a step, then ``runpp_3ph`` (with load)."""
    d = net["dts"]
    al = net.asymmetric_load
    idx = d.load.astype(int).to_numpy()
    for j, ph in enumerate("abc"):
        al.loc[idx, f"p_{ph}_mw"] = np.asarray(p_kw, float)[:, j] / 1e3
        al.loc[idx, f"q_{ph}_mvar"] = np.asarray(q_kvar, float)[:, j] / 1e3
    t = int(net["oltc"].trafo.iloc[0])
    net.trafo.at[t, "tap_pos"] = tap
    pp.runpp_3ph(net)


def step_from_cache(net, step, cache=config.YEAR_CACHE_DIR) -> tuple[np.ndarray, np.ndarray, float]:
    """(P kW, Q kvar) per DT and LV phase, and the OLTC tap, of one cached CP3 step.

    The DT meter (true frame) at the LV terminals carries exactly the DT load,
    so the step's pre-fault state can be rebuilt without the household
    population (used by the live burst test and CP5)."""
    from pathlib import Path
    step = pd.Timestamp(step)
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "phase", "active_power_kw", "reactive_power_kvar"],
                        filters=[("timestamp", "==", step)])
    names = list(net["dts"].name)
    f = f[f.load_id.isin(names)]
    p = f.pivot(index="load_id", columns="phase", values="active_power_kw").loc[names, list("abc")]
    q = f.pivot(index="load_id", columns="phase", values="reactive_power_kvar").loc[names, list("abc")]
    st = pd.read_parquet(Path(cache) / "state.parquet", columns=["timestamp", "oltc_tap"],
                         filters=[("timestamp", "==", step)])
    return p.to_numpy(float), q.to_numpy(float), float(st.oltc_tap.iloc[0])


def steps_from_cache(net, steps, cache=config.YEAR_CACHE_DIR) -> dict:
    """{step: (P kW, Q kvar, tap)} for many cached CP3 steps, read in one pass
    (same content as :func:`step_from_cache`)."""
    from pathlib import Path
    steps = sorted({pd.Timestamp(s) for s in steps})
    f = pd.read_parquet(Path(cache) / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "phase", "active_power_kw", "reactive_power_kvar"],
                        filters=[("timestamp", "in", steps)])
    names = list(net["dts"].name)
    f = f[f.load_id.isin(names)]
    st = pd.read_parquet(Path(cache) / "state.parquet", columns=["timestamp", "oltc_tap"],
                         filters=[("timestamp", "in", steps)]).set_index("timestamp").oltc_tap
    out = {}
    for ts, g in f.groupby("timestamp"):
        p = g.pivot(index="load_id", columns="phase", values="active_power_kw").loc[names, list("abc")]
        q = g.pivot(index="load_id", columns="phase", values="reactive_power_kvar").loc[names, list("abc")]
        out[pd.Timestamp(ts)] = (p.to_numpy(float), q.to_numpy(float), float(st.loc[ts]))
    return out


def steady_result(net, sn: SequenceNetwork, v_pre: np.ndarray) -> FaultResult:
    """A 'no fault' result (quiet window): fault-on values = pre-fault values."""
    ib = math.sqrt(3) * sn.base_mva / sn.head_vn_kv * 1e3
    h = np.array([sn.yf[s][sn.head_branch] @ v_pre[s] for s in range(3)]).ravel()
    i = np.abs(A @ h) * ib
    return FaultResult("", "none", "", math.nan, np.zeros(3), v_pre, v_pre, i, i, 0.0)


def run_event(net, sn: SequenceNetwork, ev, relay: RelaySettings, keep_bursts: bool = False) -> dict:
    """One event: fault (or quiet window) -> relay -> bursts -> classification per meter."""
    v_pre = prefault_v012(net, sn)
    out = {"event_id": int(ev.event_id), "kind": ev.kind}
    if ev.kind == "fault":
        if getattr(ev, "rf_model", "fixed") == "arc":
            res = simulate_arc_fault(net, sn, ev.bus, ev.ftype, ev.phases, v_pre)
        else:
            res = simulate_fault(net, sn, ev.bus, ev.ftype, ev.phases, float(ev.rf_ohm), v_pre)
        out["rf_ohm_effective"] = float(res.rf_ohm)
        t_rel, element = relay_operate(relay, float(res.i_head_fault_a.max()), res.i_head_residual_a)
        t_clear = t_rel + relay.breaker_s
        t_clear80 = t_rel + BREAKER_SENS_S
        out.update({"i_fault_ka_max": float(res.i_fault_ka.max()),
                    "i_relay_phase_a": float(res.i_head_fault_a.max()),
                    "i_relay_residual_a": res.i_head_residual_a, "element": element,
                    "t_relay_s": t_rel, "t_clear_s": t_clear, "t_clear_80ms_s": t_clear80,
                    "cleared": not math.isinf(t_rel),
                    "dyn11_transfer_max_diff_pu": dyn11_transfer_check(net, sn, res)})
    else:
        res = steady_result(net, sn, v_pre)
        t_clear = t_clear80 = math.inf
    meters = meter_faults(net, sn, res)
    rows, bursts = [], {}
    for lid, m in meters.items():
        if ev.kind == "fault":
            v, i = build_burst(m, t_clear)
            exp = expected_classification(m, t_clear)
            exp80 = expected_classification(m, t_clear80)
        else:                                            # 1.5 s steady window
            v, i = np.tile(m.v_pre, (N_PRE + N_POST, 1)), np.tile(m.i_pre, (N_PRE + N_POST, 1))
            exp = exp80 = Classification("no_dip")
        got = classify(v, m.v_base_ln_v)
        rows.append({"event_id": int(ev.event_id), "meter": lid, "role": m.role, "n_samples": len(v),
                     "residual_pct": 100 * float(m.v_fault.min()) / m.v_base_ln_v,
                     "class_kind": got.kind, "u_bin": got.u_bin, "t_bin": got.t_bin,
                     "interruption_open_ended": got.interruption_open_ended,
                     "expected_kind": exp.kind, "expected_u_bin": exp.u_bin, "expected_t_bin": exp.t_bin,
                     "match": got.key() == exp.key(),
                     "t_bin_80ms": exp80.t_bin, "kind_80ms": exp80.kind,
                     "changed_by_80ms": exp80.key() != exp.key()})
        if keep_bursts:
            bursts[lid] = (v, i, m.v_base_ln_v)
    out["meters"] = rows
    if keep_bursts:
        out["bursts"] = bursts
    return out
