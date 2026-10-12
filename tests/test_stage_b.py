"""
Phase 2B: Stage B, the Indian 11 kV feeder derived from Stage A (B1-B9).

One test per transformation step, the B1-B2 equivalence check, the OLTC/LDC
checks and Acceptance 2B.  The asserted voltage band is IS 17036:2018 Table 4,
MV: Un +/- 10 % (0.90-1.10 pu); the unsourced +/- 5 % is reported as a design
target.  Run with ``pytest -s`` to see the measured numbers.
"""

import copy
import math

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top
import pytest

from src.data_source import ieee13_data as ieee
from src.data_source import oltc
from src.data_source.feeder_model import (
    build_network, run_timeseries, solve_with_load_models,
)
from src.data_source.line_sequence import MILE_KM, sequence_params, upper_to_matrix
from src.data_source.stage_b import (
    LENGTH_FACTOR, MONITORED_LINE, build_stage_b, config_605_three_phase,
)

BAND_SOURCED = (0.90, 1.10)   # IS 17036:2018 Table 4, MV, 100 % of time
BAND_TARGET = (0.95, 1.05)    # design target, reported only
VCOLS = ["vm_a_pu", "vm_b_pu", "vm_c_pu"]


def _named_v(net):
    v = net.res_bus_3ph[VCOLS].copy()
    v.index = net.bus.name.loc[v.index]
    return v


def _p_in(net):
    return float(sum(net.res_ext_grid_3ph[f"p_{p}_mw"].sum() for p in "abc"))


@pytest.fixture(scope="module")
def stage_a():
    net = build_network("A")
    solve_with_load_models(net)
    return net


@pytest.fixture(scope="module")
def steps():
    """Solved network after each step (with the loads' own models)."""
    out = {}
    for s in ("B1", "B2", "B3", "B4", "B6"):
        net = build_stage_b(upto=s)
        solve_with_load_models(net)
        out[s] = net
    return out


@pytest.fixture(scope="module")
def final():
    net = build_network("B")
    pp.runpp_3ph(net)
    return net


# ===================================================================== #
#  B1-B6                                                                  #
# ===================================================================== #

def test_b1_voltage(steps):
    net = steps["B1"]
    names = dict(zip(net.bus.index, net.bus.name))
    assert all(vn == 11.0 for i, vn in net.bus.vn_kv.items() if names[i] != "634")
    assert net.bus.vn_kv[net.bus.name == "634"].iloc[0] == ieee.V_XFM1_LV_KV
    x1 = net.trafo[net.trafo.name == "XFM-1"].iloc[0]
    assert (x1.vn_hv_kv, x1.vn_lv_kv) == (11.0, 0.48)
    assert (net.shunt.vn_kv == 11.0).all()
    assert names[net.ext_grid.bus.iloc[0]] == "650"


def test_b2_lengths_and_equivalence_with_stage_a(stage_a, steps):
    a, b = stage_a, steps["B2"]
    assert LENGTH_FACTOR == pytest.approx((11 / 4.16) ** 2)
    assert np.allclose(b.line.length_km.values, a.line.length_km.values * LENGTH_FACTOR)
    va, vb = _named_v(a), _named_v(b).loc[_named_v(a).index]
    diff = 100 * (vb - va) / va
    worst = float(np.nanmax(np.abs(diff.values)))
    chg = lambda n: sum(  # noqa: E731  three-phase charging kvar at nominal voltage
        n.bus.vn_kv[r.from_bus] ** 2 * 2 * math.pi * n.f_hz * r.c_nf_per_km * 1e-9
        * r.length_km * 1e3 for _, r in n.line.iterrows())
    print(f"\nB1-B2 vs Stage A: max |dV| {worst:.3f} %; charging {chg(a):.1f} -> {chg(b):.1f} kvar")
    assert worst <= 0.5


def test_b3_frequency(steps):
    b2, b3 = steps["B2"], steps["B3"]
    assert b3.f_hz == 50.0
    assert np.allclose(b3.line.x_ohm_per_km, b2.line.x_ohm_per_km * 5 / 6)
    assert np.allclose(b3.line.x0_ohm_per_km, b2.line.x0_ohm_per_km * 5 / 6)
    assert np.allclose(b3.line.c_nf_per_km, b2.line.c_nf_per_km)
    d = 100 * (_named_v(b3) - _named_v(b2)) / _named_v(b2)
    print(f"\nB3 effect: max |dV| {np.nanmax(np.abs(d.values)):.3f} %, "
          f"input {_p_in(b2) * 1e3:.1f} -> {_p_in(b3) * 1e3:.1f} kW")


def test_b4_substation_transformer(steps):
    b3, b4 = steps["B3"], steps["B4"]
    names = dict(zip(b4.bus.index, b4.bus.name))
    assert names[b4.ext_grid.bus.iloc[0]] == "sourcebus"
    assert b4.bus.vn_kv[b4.bus.name == "sourcebus"].iloc[0] == 66.0
    # CEA TPC 2025 §5.1.5: 0.8 x 31.5 kA at 66 kV
    eg = b4.ext_grid.iloc[0]
    assert eg.s_sc_max_mva == pytest.approx(math.sqrt(3) * 66.0 * 25.2)
    assert eg.rx_max == 0.1
    t = b4.trafo[b4.trafo.name == "Sub_XFMR"].iloc[0]
    assert (t.vector_group, t.sn_mva, t.vn_hv_kv, t.vn_lv_kv) == ("Dyn", 5.0, 66.0, 11.0)
    assert t.vk_percent == pytest.approx(math.hypot(1, 8))
    assert t.vkr_percent == 1.0
    d = 100 * (_named_v(b4).loc[_named_v(b3).index] - _named_v(b3)) / _named_v(b3)
    print(f"\nB4 effect: max |dV| {np.nanmax(np.abs(d.values)):.3f} %; "
          f"650 now {_named_v(b4).loc['650'].round(4).tolist()} pu")


def test_b5_line_data_three_phase():
    net = build_stage_b(upto="B5")
    for _, ln in net.line.iterrows():
        cfg = int(ln.config)
        if cfg == 605:
            sp = config_605_three_phase(50.0)
        else:
            _, z, b = ieee.LINE_CONFIGS[cfg]
            sp60 = sequence_params(z, b, ieee.CONFIG_PHASES[cfg], 60.0)
            sp = sp60.__class__(sp60.r1_ohm_per_km, sp60.x1_ohm_per_km * 5 / 6,
                                sp60.r0_ohm_per_km, sp60.x0_ohm_per_km * 5 / 6,
                                sp60.c1_nf_per_km, sp60.c0_nf_per_km)
        assert ln.r_ohm_per_km == pytest.approx(sp.r1_ohm_per_km), ln["name"]
        assert ln.x_ohm_per_km == pytest.approx(sp.x1_ohm_per_km), ln["name"]
        assert ln.r0_ohm_per_km == pytest.approx(sp.r0_ohm_per_km), ln["name"]
        assert ln.c_nf_per_km == pytest.approx(sp.c1_nf_per_km), ln["name"]


def test_b5_config_605_derivation():
    """605: Zs from its own matrix (c-c), Zm from 603/604 (same 1/0 ACSR)."""
    zs = upper_to_matrix(ieee.LINE_CONFIGS[605][1])[2, 2]
    zm = upper_to_matrix(ieee.LINE_CONFIGS[603][1])[1, 2]
    assert zm == upper_to_matrix(ieee.LINE_CONFIGS[604][1])[0, 2]
    sp = config_605_three_phase(60.0)
    assert complex(sp.r1_ohm_per_km, sp.x1_ohm_per_km) * MILE_KM == pytest.approx(zs - zm)
    assert complex(sp.r0_ohm_per_km, sp.x0_ohm_per_km) * MILE_KM == pytest.approx(zs + 2 * zm)
    sp50 = config_605_three_phase(50.0)
    assert sp50.x1_ohm_per_km == pytest.approx(sp.x1_ohm_per_km * 5 / 6)


def test_b6_capacitors(steps):
    net = steps["B6"]
    assert "cap_611_c" not in set(net.asymmetric_load.name)
    sh = net.shunt.set_index("name")
    assert sh.at["Cap_611", "q_mvar"] == pytest.approx(-0.1)
    assert sh.at["Cap_675", "q_mvar"] == pytest.approx(-0.6)
    assert (net.shunt.vn_kv == 11.0).all()


# ===================================================================== #
#  B7 OLTC with LDC                                                       #
# ===================================================================== #

def test_ldc_conversion():
    z416 = (3 + 9j) * 20 / 700
    assert oltc.ldc_impedance_ohm(4.16, 60.0) == pytest.approx(z416)
    z11 = oltc.ldc_impedance_ohm(11.0, 60.0)
    assert z11 == pytest.approx(z416 * (11 / 4.16) ** 2)
    z11_50 = oltc.ldc_impedance_ohm(11.0, 50.0)
    assert z11_50 == pytest.approx(complex(z11.real, z11.imag * 5 / 6))
    assert oltc.SET_POINT_PU == pytest.approx(122 / 120)
    assert oltc.HALF_BAND_PU == pytest.approx(1 / 120)
    print(f"\nZ_LDC: 4.16 kV {z416:.4f} ohm; 11 kV {z11:.4f} ohm; 11 kV 50 Hz {z11_50:.4f} ohm")


def test_b7_oltc_replaces_regulator(final):
    names = dict(zip(final.bus.index, final.bus.name))
    assert "Regulator_650_RG60" not in set(final.trafo.name)
    sw = final.switch[final.switch.name == "Switch_650_RG60"].iloc[0]
    assert sw.closed and {names[sw.bus], names[sw.element]} == {"650", "RG60"}
    c = final["oltc"].iloc[0]
    assert final.trafo.at[int(c.trafo), "name"] == "Sub_XFMR"
    assert final.line.at[int(c.line), "name"] == MONITORED_LINE
    v, _ = oltc.relay_voltage(final)
    assert abs(v - c.set_point_pu) <= c.half_band_pu + 1e-9


def test_controller_self_check_on_stage_a():
    """Same controller on the published case lands inside the published taps."""
    net = build_network("A")
    oltc.attach_oltc(net, "Regulator_650_RG60", MONITORED_LINE)
    t = net.trafo.index[net.trafo.name == "Regulator_650_RG60"][0]
    net.trafo.at[t, "tap_pos"] = 0
    oltc.settle(net, solve=solve_with_load_models)
    tap = net.trafo.at[t, "tap_pos"]
    print(f"\nStage A controller settles at tap {tap:+.0f}; published {ieee.REGULATOR['published_taps']}")
    assert min(ieee.REGULATOR["published_taps"]) <= tap <= max(ieee.REGULATOR["published_taps"])


# ===================================================================== #
#  B9 constant-PQ and sensitivity                                         #
# ===================================================================== #

def test_b9_constant_pq_and_sensitivity(final, stage_a):
    al = final.asymmetric_load
    assert (al.load_model == "PQ").all()
    for p in "abc":
        assert np.allclose(al[f"p_{p}_mw"], al[f"p0_{p}_mw"])
    zi = copy.deepcopy(final)
    models = dict(zip(stage_a.asymmetric_load.name, stage_a.asymmetric_load.load_model))
    zi.asymmetric_load["load_model"] = zi.asymmetric_load.name.map(models)
    solve_with_load_models(zi)
    dv = 100 * (_named_v(zi) - _named_v(final)) / _named_v(final)
    loss = lambda n: _p_in(n) - float(sum(n.asymmetric_load[f"p_{p}_mw"].sum() for p in "abc"))  # noqa: E731
    print(f"\nB9 sensitivity (Stage A Z/I models vs constant-PQ): max |dV| "
          f"{np.nanmax(np.abs(dv.values)):.3f} %, losses {loss(final) * 1e3:.1f} -> {loss(zi) * 1e3:.1f} kW")


# ===================================================================== #
#  Acceptance 2B                                                          #
# ===================================================================== #

def test_nominal_snapshot(final):
    mv = final.bus.index[final.bus.vn_kv == 11.0]
    v = final.res_bus_3ph.loc[mv, VCOLS]
    p_in = _p_in(final)
    p_load = float(sum(final.res_asymmetric_load_3ph[f"p_{p}_mw"].sum() for p in "abc"))
    p_branch = float(sum(final.res_line_3ph[f"pl_{p}_mw"].sum() for p in "abc")
                     + sum(final.res_trafo_3ph[f"pl_{p}_mw"].sum() for p in "abc"))
    out = (v < BAND_TARGET[0]) | (v > BAND_TARGET[1])
    worst = (v - 1.0).abs().stack().idxmax()
    print(f"\nnominal: V {v.values.min():.4f}-{v.values.max():.4f} pu, losses "
          f"{100 * (p_in - p_load) / p_in:.2f} %; +/-5 % target: {int(out.sum().sum())} "
          f"bus-phases outside, worst {final.bus.name[worst[0]]}/{worst[1][3]} "
          f"{v.loc[worst[0], worst[1]]:.4f} pu")
    assert BAND_SOURCED[0] <= v.values.min() and v.values.max() <= BAND_SOURCED[1]
    assert 100 * (p_in - p_load) / p_in < 5.0
    assert p_in == pytest.approx(p_load + p_branch, abs=1e-3)
    g = top.create_nxgraph(final, multi=True)
    assert nx.number_connected_components(g) == 1
    assert g.number_of_edges() == g.number_of_nodes() - 1


@pytest.fixture(scope="module")
def synthetic_week():
    net = build_network("B")
    _, state = run_timeseries("2024-01-01", "2024-01-07 23:45", net=net, return_state=True)
    return state


def test_synthetic_week(synthetic_week):
    st = synthetic_week
    days = st.groupby(st.timestamp.dt.date)
    ops = days.tap_moved.sum()
    limit_h = days.at_tap_limit.sum() * 0.25
    stuck = st[st.at_tap_limit & ((st.v_relay_pu - 122 / 120).abs() > 1 / 120)]
    print(f"\nsynthetic week: V {st.v_min_11kv_pu.min():.4f}-{st.v_max_11kv_pu.max():.4f} pu; "
          f"tap {st.oltc_tap.min():+.0f}..{st.oltc_tap.max():+.0f}; tap operations/day "
          f"{ops.tolist()}; hours/day at tap limit {limit_h.tolist()}; out of band at "
          f"limit: {len(stuck)} steps; +/-5 % target: {int((st.v_below_095 + st.v_above_105).gt(0).sum())} "
          f"steps with violations")
    assert len(st) == 7 * 96
    assert st.v_min_11kv_pu.min() >= BAND_SOURCED[0]
    assert st.v_max_11kv_pu.max() <= BAND_SOURCED[1]
    assert (st.losses_mw - st.branch_losses_mw).abs().max() < 1e-3
    assert (st.losses_mw / st.p_in_mw).max() < 0.05
    assert (st.tap_moved <= 1).all()
