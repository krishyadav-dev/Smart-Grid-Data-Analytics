"""
Phase 2A: Stage A (IEEE 13-node feeder as published) — construction and validation.

Acceptance 2A: total input P and Q within ±2 % of the published radial-flow
summary; phase-average voltage magnitude at each bus within ±1 % of the
published profile (single/two-phase buses on the phases present).  Per-phase
voltages and angles are printed, not asserted (run with ``pytest -s``).
"""

import math

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top
import pytest

from src.data_source import ieee13_data as ieee
from src.data_source.feeder_model import build_network
from src.data_source.line_sequence import (
    MILE_KM, sequence_params, upper_to_matrix,
)
from src.data_source.stage_a_validation import validate_stage_a

A_OP = np.exp(2j * np.pi / 3)
A_MAT = np.array([[1, 1, 1], [1, A_OP ** 2, A_OP], [1, A_OP, A_OP ** 2]])


@pytest.fixture(scope="module")
def result():
    return validate_stage_a()


# ===================================================================== #
#  2A.1 sequence impedances                                               #
# ===================================================================== #

@pytest.mark.parametrize("cfg", [601, 602, 606])
def test_three_phase_config_matches_fortescue_diagonal(cfg):
    _, z_up, b_up = ieee.LINE_CONFIGS[cfg]
    zseq = np.linalg.inv(A_MAT) @ upper_to_matrix(z_up) @ A_MAT
    sp = sequence_params(z_up, b_up, "abc", 60.0)
    z0 = complex(sp.r0_ohm_per_km, sp.x0_ohm_per_km) * MILE_KM
    z1 = complex(sp.r1_ohm_per_km, sp.x1_ohm_per_km) * MILE_KM
    assert z0 == pytest.approx(zseq[0, 0], abs=1e-12)
    assert z1 == pytest.approx(zseq[1, 1], abs=1e-12)
    assert z1 == pytest.approx(zseq[2, 2], abs=1e-12)


@pytest.mark.parametrize("cfg,phase", [(605, "c"), (607, "a")])
def test_single_phase_config_keeps_self_impedance(cfg, phase):
    _, z_up, b_up = ieee.LINE_CONFIGS[cfg]
    zii = upper_to_matrix(z_up)["abc".index(phase), "abc".index(phase)]
    sp = sequence_params(z_up, b_up, ieee.CONFIG_PHASES[cfg], 60.0)
    assert complex(sp.r1_ohm_per_km, sp.x1_ohm_per_km) * MILE_KM == pytest.approx(zii)
    assert complex(sp.r0_ohm_per_km, sp.x0_ohm_per_km) * MILE_KM == pytest.approx(zii)


def test_two_phase_config_uses_present_block():
    _, z_up, b_up = ieee.LINE_CONFIGS[603]  # phases b, c
    z = upper_to_matrix(z_up)
    zs, zm = (z[1, 1] + z[2, 2]) / 2, z[1, 2]
    sp = sequence_params(z_up, b_up, "bc", 60.0)
    assert complex(sp.r1_ohm_per_km, sp.x1_ohm_per_km) * MILE_KM == pytest.approx(zs - zm)
    assert complex(sp.r0_ohm_per_km, sp.x0_ohm_per_km) * MILE_KM == pytest.approx(zs + 2 * zm)


def test_capacitance_unit_conversion():
    _, z_up, b_up = ieee.LINE_CONFIGS[606]  # diagonal B = 96.8897 uS/mile
    sp = sequence_params(z_up, b_up, "abc", 60.0)
    expected_nf_per_km = 96.8897e-6 / (2 * math.pi * 60.0) * 1e9 / MILE_KM
    assert sp.c1_nf_per_km == pytest.approx(expected_nf_per_km)
    assert sp.c0_nf_per_km == pytest.approx(expected_nf_per_km)


def test_every_config_traces_to_source_lines():
    for cfg, (src_lines, z_up, b_up) in ieee.LINE_CONFIGS.items():
        first, last = src_lines
        assert 1 <= first < last, cfg
        assert len(z_up) == 6 and len(b_up) == 6, cfg
        assert cfg in ieee.CONFIG_PHASES and cfg in ieee.CONFIG_AMPACITY_KA, cfg


def test_network_lines_use_published_configs(result):
    net = result["net"]
    assert set(net.line.config) <= set(ieee.LINE_CONFIGS)
    for _, ln in net.line.iterrows():
        assert ln.max_i_ka == ieee.CONFIG_AMPACITY_KA[ln.config]


# ===================================================================== #
#  Topology and source                                                    #
# ===================================================================== #

def test_stage_a_topology(result):
    net = result["net"]
    names = dict(zip(net.bus.index, net.bus.name))
    assert net.f_hz == 60.0
    assert len(net.ext_grid) == 1 and names[net.ext_grid.bus.iloc[0]] == "650"
    assert "sourcebus" not in set(net.bus.name)
    assert set(net.trafo.name) == {"Regulator_650_RG60", "XFM-1"}
    assert (net.trafo.vector_group == "YNyn").all()
    sw = net.switch.iloc[0]
    assert len(net.switch) == 1 and sw.closed and sw.et == "b"
    assert {names[sw.bus], names[sw.element]} == {"671", "692"}
    pairs = {frozenset((names[a], names[b])) for a, b in zip(net.line.from_bus, net.line.to_bus)}
    assert frozenset(("671", "692")) not in pairs
    g = top.create_nxgraph(net, multi=True)
    assert nx.number_connected_components(g) == 1
    assert g.number_of_edges() == g.number_of_nodes() - 1


# ===================================================================== #
#  2A.2 load models and delta mapping                                     #
# ===================================================================== #

def test_load_models_follow_solved_voltage(result):
    net = result["net"]
    al = net.asymmetric_load.set_index("name")
    v = net.res_bus_3ph
    bus = dict(zip(net.bus.name, net.bus.index))
    va = lambda b, p: v.at[bus[b], f"vm_{p}_pu"] * np.exp(1j * np.deg2rad(v.at[bus[b], f"va_{p}_degree"]))  # noqa: E731
    # Y-Z at 652 phase a: P = P0 V^2
    assert al.at["load_652_a", "p_a_mw"] == pytest.approx(
        al.at["load_652_a", "p0_a_mw"] * v.at[bus["652"], "vm_a_pu"] ** 2, rel=2e-3)
    # Y-I at 611 phase c: P = P0 V
    assert al.at["load_611_c", "p_c_mw"] == pytest.approx(
        al.at["load_611_c", "p0_c_mw"] * v.at[bus["611"], "vm_c_pu"], rel=2e-3)
    # D-Z at 646 on BC: P = P0 (|Vb - Vc| / sqrt3)^2
    vbc = abs(va("646", "b") - va("646", "c")) / np.sqrt(3)
    assert al.at["load_646_b", "p_b_mw"] == pytest.approx(al.at["load_646_b", "p0_b_mw"] * vbc ** 2, rel=2e-3)
    # PQ loads keep their nominal values
    assert al.at["load_671_abc", "p_a_mw"] == al.at["load_671_abc", "p0_a_mw"]
    assert 1 <= result["n_power_flows"] <= 10


def test_delta_load_on_bc_draws_current_only_in_b_and_c():
    """pandapower maps a delta load's p_b to the BC pair (646, 671, 692 rely on it)."""
    net = pp.create_empty_network(f_hz=60.0)
    b0 = pp.create_bus(net, vn_kv=4.16)
    b1 = pp.create_bus(net, vn_kv=4.16)
    pp.create_ext_grid(net, b0, vm_pu=1.0, s_sc_max_mva=1000.0, rx_max=0.1,
                       x0x_max=1.0, r0x0_max=0.1)
    pp.create_line_from_parameters(net, b0, b1, length_km=0.1, r_ohm_per_km=0.2,
                                   x_ohm_per_km=0.4, c_nf_per_km=0.0, max_i_ka=1.0,
                                   r0_ohm_per_km=0.6, x0_ohm_per_km=1.2, c0_nf_per_km=0.0)
    pp.create_asymmetric_load(net, b1, p_b_mw=0.2, q_b_mvar=0.1, type="delta")
    pp.runpp_3ph(net)
    r = net.res_line_3ph.iloc[0]
    assert r.i_a_from_ka == pytest.approx(0.0, abs=1e-6)
    assert r.i_b_from_ka > 0.01
    assert r.i_b_from_ka == pytest.approx(r.i_c_from_ka, rel=1e-6)


# ===================================================================== #
#  Acceptance 2A                                                          #
# ===================================================================== #

def test_total_input_within_2_percent(result):
    t = result["totals"].set_index("quantity")
    print(f"\ninput: {t.at['input kW', 'model']:.1f} kW ({t.at['input kW', 'error_pct']:+.2f} %), "
          f"{t.at['input kvar', 'model']:.1f} kvar ({t.at['input kvar', 'error_pct']:+.2f} %); "
          f"losses {t.at['losses kW', 'model']:.1f} kW vs {ieee.PUBLISHED_LOSS_KW_TOTAL}")
    assert abs(t.at["input kW", "error_pct"]) <= 2.0
    assert abs(t.at["input kvar", "error_pct"]) <= 2.0


def test_phase_average_voltage_within_1_percent(result):
    avg = result["bus_avg"]
    print("\n" + avg.round(4).to_string(index=False))
    print("\n" + result["per_phase"].round(4).to_string(index=False))
    assert (avg.avg_error_pct.abs() <= 1.0).all(), avg[avg.avg_error_pct.abs() > 1.0]
