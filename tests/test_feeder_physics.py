"""
Physical-sanity tests for the Stage B feeder at nominal load (runpp_3ph).

Phase 1 checks (energy balance, losses, voltages, loadings, radial topology,
load totals), updated for Stage B (Phase 2B):

- The asserted voltage band is IS 17036:2018 Table 4, MV: Un +/- 10 %
  (0.90-1.10 pu, 100 % of the time).  The former +/- 5 % (0.95-1.05 pu) has no
  source; it is reported as a design target, not asserted.
- The bus-632 fixed-tap check became: the OLTC's compensated (relay) voltage
  is inside its band (set point 1.0167 pu +/- 0.0083 pu); 632 is reported.

Measured values are printed; run with ``pytest -s`` to see them.
"""

import networkx as nx
import pandapower as pp
import pandapower.topology as top
import pytest

from src.data_source import oltc
from src.data_source.feeder_model import build_ieee13_network

PHASES = "abc"
BAND_SOURCED = (0.90, 1.10)   # IS 17036:2018 Table 4, MV, 100 % of time
BAND_TARGET = (0.95, 1.05)    # design target, reported only


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
    """MV-network losses (11 kV lines + substation transformer, DTs excluded)
    < 5 % at nominal (brief v2 X.3 item 4e; the DT losses are checked against
    IS 1180 in test_stage_b.py::test_nominal_dt_losses_is1180)."""
    p_grid = _sum3(net.res_ext_grid_3ph, "p_{}_mw")
    p_load = _sum3(net.res_asymmetric_load_3ph, "p_{}_mw")
    sub = net.trafo.index[net.trafo.name == "Sub_XFMR"]
    p_mv = (_sum3(net.res_line_3ph, "pl_{}_mw")
            + _sum3(net.res_trafo_3ph.loc[sub], "pl_{}_mw"))
    total_pct = 100.0 * (p_grid - p_load) / p_grid
    mv_pct = 100.0 * p_mv / p_grid
    print(f"\nMV losses = {mv_pct:.2f} % of input (asserted); total incl. DTs {total_pct:.2f} %")
    assert 0.0 < mv_pct < 5.0


def test_11kv_voltages_in_band(net):
    vm = net.res_bus_3ph[[f"vm_{p}_pu" for p in PHASES]]
    mv = net.bus.index[net.bus.vn_kv == 11.0]
    v = vm.loc[mv]
    lo, hi = float(v.values.min()), float(v.values.max())
    outside = int(((v < BAND_TARGET[0]) | (v > BAND_TARGET[1])).sum().sum())
    print(f"\n11 kV phase voltages: min {lo:.4f} pu, max {hi:.4f} pu; "
          f"{outside} bus-phases outside the 0.95-1.05 target (reported)")
    v634 = vm.loc[net.bus.index[net.bus.name == "634"][0]].round(4).tolist()
    lv = vm.loc[net.bus.index[net.bus.vn_kv < 1.0]]
    print(f"bus 634 (dt_634_1 LV, not asserted): {v634}; DT LV buses "
          f"{float(lv.values.min()):.4f}-{float(lv.values.max()):.4f} pu (not asserted)")
    assert BAND_SOURCED[0] <= lo and hi <= BAND_SOURCED[1]


def test_oltc_relay_voltage_in_band(net):
    v, per_phase = oltc.relay_voltage(net)
    c = net["oltc"].iloc[0]
    i632 = net.bus.index[net.bus.name == "632"][0]
    v632 = [round(float(net.res_bus_3ph.at[i632, f"vm_{p}_pu"]), 4) for p in PHASES]
    print(f"\nOLTC tap {net.trafo.at[int(c.trafo), 'tap_pos']:+.0f}, relay {v:.4f} pu "
          f"(per phase {[round(x, 4) for x in per_phase]}); bus 632: {v632} pu")
    assert abs(v - c.set_point_pu) <= c.half_band_pu + 1e-9


def test_loadings(net):
    trafo = dict(zip(net.trafo.name, net.res_trafo_3ph.loading_percent))
    lines = dict(zip(net.line.name, net.res_line_3ph.loading_percent))
    print(f"\nSub_XFMR loading {trafo['Sub_XFMR']:.1f} %")
    # DTs are reported, not asserted (Part 2C.1): XFM-1's load on dt_634_1
    # (194 kVA on phase A vs 167 kVA per phase) is the thermal stress case,
    # and DTs at IEEE single-phase buses carry their whole load on one phase.
    dts = {n: v for n, v in trafo.items() if n.startswith("dt_")}
    print("DT loading (pandapower, max phase): "
          + ", ".join(f"{n} {v:.0f} %" for n, v in dts.items()) + " (reported only)")
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
    loads = net.asymmetric_load[net.asymmetric_load.name.str.startswith("dtload_")]
    p = _sum3(loads, "p_{}_mw")
    q = _sum3(loads, "q_{}_mvar")
    print(f"\nload totals: {p:.4f} MW, {q:.4f} Mvar")
    assert p == pytest.approx(3.466, rel=0.005)
    assert q == pytest.approx(2.102, rel=0.005)
