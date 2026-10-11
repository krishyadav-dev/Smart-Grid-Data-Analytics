"""
Physical-sanity tests for the IEEE 13-bus feeder at nominal load (runpp_3ph).

Covers Fix Brief Phase 1 §4 (audit L1-1, L1-7, L1-8): energy balance,
losses, voltage band, element loadings, radial topology and load totals.
Measured values are printed; run with ``pytest -s`` to see them.
"""

import networkx as nx
import pandapower as pp
import pandapower.topology as top
import pytest

from src.data_source.feeder_model import build_ieee13_network

PHASES = "abc"


def _sum3(df, fmt):
    return float(sum(df[fmt.format(p)].sum() for p in PHASES))


@pytest.fixture(scope="module")
def net():
    n = build_ieee13_network()
    pp.runpp_3ph(n)
    assert n.converged
    return n


def test_energy_balance(net):
    p_grid = _sum3(net.res_ext_grid_3ph, "p_{}_mw")
    p_load = _sum3(net.res_asymmetric_load_3ph, "p_{}_mw")
    p_loss = (_sum3(net.res_line_3ph, "pl_{}_mw")
              + _sum3(net.res_trafo_3ph, "pl_{}_mw"))
    print(f"\nP_grid={p_grid:.4f} MW  P_load={p_load:.4f} MW  P_loss={p_loss:.4f} MW")
    assert p_grid == pytest.approx(p_load + p_loss, abs=1e-3)


def test_losses_below_5_percent(net):
    p_grid = _sum3(net.res_ext_grid_3ph, "p_{}_mw")
    p_load = _sum3(net.res_asymmetric_load_3ph, "p_{}_mw")
    loss_pct = 100.0 * (p_grid - p_load) / p_grid
    print(f"\nlosses = {loss_pct:.2f} % of input")
    assert 0.0 < loss_pct < 5.0


def test_11kv_voltages_in_band(net):
    vm = net.res_bus_3ph[[f"vm_{p}_pu" for p in PHASES]]
    mv = net.bus.index[net.bus.vn_kv == 11.0]
    lo, hi = float(vm.loc[mv].values.min()), float(vm.loc[mv].values.max())
    print(f"\n11 kV phase voltages: min {lo:.4f} pu, max {hi:.4f} pu")
    v634 = vm.loc[net.bus.index[net.bus.name == "634"][0]].round(4).tolist()
    print(f"bus 634 (XFM-1 LV, not asserted): {v634}")
    assert 0.95 <= lo and hi <= 1.05


def test_bus_632_regulated_band(net):
    i632 = net.bus.index[net.bus.name == "632"][0]
    v = [float(net.res_bus_3ph.at[i632, f"vm_{p}_pu"]) for p in PHASES]
    print(f"\nbus 632: {[round(x, 4) for x in v]} pu")
    assert all(1.00 <= x <= 1.04 for x in v)


def test_loadings(net):
    trafo = dict(zip(net.trafo.name, net.res_trafo_3ph.loading_percent))
    lines = dict(zip(net.line.name, net.res_line_3ph.loading_percent))
    print(f"\nSub_XFMR loading {trafo['Sub_XFMR']:.1f} %")
    # XFM-1 is reported, not asserted: its IEEE load (400 kW + j290 kvar,
    # 194 kVA on phase A) sits at or above the 500 kVA rating at nominal.
    print(f"XFM-1 loading {trafo['XFM-1']:.1f} % (reported only)")
    worst = max(lines, key=lines.get)
    print(f"max line loading {lines[worst]:.1f} % ({worst})")
    assert trafo["Sub_XFMR"] < 100.0
    assert all(v < 100.0 for v in lines.values()), lines


def test_network_is_radial(net):
    g = top.create_nxgraph(net, multi=True)
    assert nx.number_connected_components(g) == 1
    # A connected graph is a tree iff |E| = |V| - 1 (parallel edges count).
    assert g.number_of_edges() == g.number_of_nodes() - 1
    assert "Line_633_634" not in set(net.line.name)


def test_load_totals(net):
    p = _sum3(net.asymmetric_load, "p_{}_mw")
    q = _sum3(net.asymmetric_load, "q_{}_mvar")
    print(f"\nload totals: {p:.4f} MW, {q:.4f} Mvar")
    assert p == pytest.approx(3.466, rel=0.005)
    assert q == pytest.approx(2.102, rel=0.005)
