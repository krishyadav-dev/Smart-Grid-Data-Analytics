"""
Fault event bursts (brief v2 §2D; X.3.1 items 8-10, 12; X.3.2 items 11/18,
14-17, 23; X.3.3 items 2-4 and the CP4 decisions in X.3.3 item 6).

- Sequence matrices well-conditioned; Dyn11 transfer of the DT meter voltages.
- §2D: 3PH hand check at 632 (engine without shunts vs the hand formula, 2 %);
  the capacitor cross-check is reported.
- SLG shows the dip on the faulted phase; every fault type gives a 10 ms
  burst whose IS 17036 classification matches the simulated event.
- Clearing time = IDMT + breaker (60 ms; 80 ms sensitivity).
- Relay margins (asserted): Is < 730 A, remote 3PH I/Is >= 1.5 (remote = lowest
  3PH bus); earth-fault security over the CP3 year.
- Classifier on synthetic waveforms, one per Table 13 bin, plus no dip, not
  cleared and Table 12 interruptions (independent of the fault simulation).

Run with ``pytest -s`` to see the numbers.  The relay settings use the CP3
year cache (results/year_2024).
"""

import math

import numpy as np
import pandapower as pp
import pytest

from src.data_source import faults as F
from src.data_source.feeder_model import build_network


@pytest.fixture(scope="module")
def nominal():
    net = build_network("B")
    pp.runpp_3ph(net)
    sn = F.build_sequence_network(net)
    return net, sn, F.prefault_v012(net, sn)


@pytest.fixture(scope="module")
def relay():
    return F.relay_settings(F.max_head_current_a())


def _bus(net, sn, name):
    return int(sn.lookup[int(net.bus.index[net.bus.name == name][0])])


# --------------------------------------------------------------------------
# Network and method
# --------------------------------------------------------------------------

def test_sequence_matrices_well_conditioned(nominal):
    net, sn, _ = nominal
    print(f"\ncond Y0 {sn.cond[0]:.3e}, Y1 {sn.cond[1]:.3e}, Y2 {sn.cond[2]:.3e}")
    assert max(sn.cond) < F.COND_MAX


def test_hand_3ph_632_within_2_percent(nominal):
    """§2D acceptance (CP4 decision, option b): engine with all shunts removed
    vs the hand formula, same pre-fault voltage and tap, within 2 %."""
    net, _, _ = nominal
    c = F.check_3ph_632(net)
    cap = F.capacitor_check(net, c)
    print(f"\n3PH 632: hand {c['hand_ka']:.4f} kA, engine without shunts {c['engine_no_shunt_ka']:.4f} kA "
          f"({100 * c['rel_error']:+.3f} %), engine full network {c['engine_full_ka']:.4f} kA; tap {c['tap']}; "
          f"Z hand {c['z_hand_ohm']:.4f}, Z1 full {c['z1_full_ohm']:.4f} ohm\n"
          f"capacitor cross-check (reported): engine |Z_full|/|Z_no-shunt| {cap['engine_ratio']:.5f}, "
          f"1/(1 - |Z_src|/Xc,eq) {cap['hand_ratio']:.5f} (Xc,eq {cap['xc_eq_ohm']:.2f} ohm, "
          f"Q {1e3 * cap['q_caps_mvar']:.0f} kvar)")
    assert abs(c["rel_error"]) <= 0.02
    assert cap["engine_ratio"] > 1 and cap["hand_ratio"] > 1


def test_dyn11_transfer_of_dt_meter_voltages(nominal):
    """X.3.1 item 9: explicit Dyn11 transform (+30 deg / -30 deg / zero blocked)
    agrees with the LV rows of the network solution."""
    net, sn, v_pre = nominal
    worst = 0.0
    for ftype, ph in (("SLG", "a"), ("LL", "bc"), ("LLG", "ca"), ("3PH", "abc")):
        res = F.simulate_fault(net, sn, "671", ftype, ph, 0.0, v_pre)
        worst = max(worst, F.dyn11_transfer_check(net, sn, res))
    print(f"\nDyn11 transfer vs network LV rows: max |dV| {worst:.2e} pu")
    assert worst < 1e-3


def test_slg_dip_on_faulted_phase(nominal):
    net, sn, v_pre = nominal
    k = _bus(net, sn, "652")
    for ph in "abc":
        res = F.simulate_fault(net, sn, "652", "SLG", ph, 0.0, v_pre)
        v = np.abs(F.A @ res.v_fault[:, k])
        j = "abc".index(ph)
        print(f"\nSLG {ph} at 652: V {np.round(v, 4)} pu, I_f {np.round(res.i_fault_ka, 4)} kA", end="")
        assert int(np.argmin(v)) == j and v[j] < 0.05
        assert all(v[i] > 0.9 for i in range(3) if i != j)
        assert int(np.argmax(res.i_fault_ka)) == j and np.count_nonzero(res.i_fault_ka > 1e-9) == 1


def test_ll_has_no_residual_and_3ph_is_balanced(nominal):
    net, sn, v_pre = nominal
    ll = F.simulate_fault(net, sn, "675", "LL", "ab", 5.0, v_pre)
    assert ll.i_head_residual_a < 1e-6 and ll.i_fault_ka[2] < 1e-9
    assert ll.i_fault_ka[0] == pytest.approx(ll.i_fault_ka[1])
    p3 = F.simulate_fault(net, sn, "675", "3PH", "abc", 0.0, v_pre)
    assert np.ptp(p3.i_fault_ka) < 1e-9


# --------------------------------------------------------------------------
# Relay
# --------------------------------------------------------------------------

def test_relay_margins(nominal, relay):
    """X.3.1 item 12 (asserted): Is < 730 A; remote (lowest 3PH current) I/Is >= 1.5."""
    net, sn, v_pre = nominal
    m = F.relay_margins(net, sn, v_pre, relay)
    print(f"\nIs {m['is_a']:.1f} A (1.2 x {m['is_a'] / 1.2:.1f} A), EF pickup {m['ef_is_a']:.1f} A, "
          f"TMS {m['tms']}; remote bus {m['remote_bus']} 3PH {m['remote_3ph_ka']:.4f} kA, "
          f"margin I/Is {m['remote_margin']:.3f}; 3PH kA {m['three_phase_ka']}")
    assert m["is_a"] < F.AMPACITY_601_A
    assert m["remote_bus"] == min(m["three_phase_ka"], key=m["three_phase_ka"].get)
    assert m["remote_margin"] >= F.REMOTE_MARGIN_MIN


def test_earth_fault_security_over_the_year(relay):
    """CP4 decision: max feeder-head |3I0| over the CP3 year < 0.5 x EF pickup."""
    r = F.head_residual_from_cache()
    print(f"\nmax feeder-head |3I0| over 2024: {r['max_3i0_a']:.3e} A at {r['at']} "
          f"(limit {0.5 * relay.ef_is_a:.1f} A)")
    assert r["max_3i0_a"] < 0.5 * relay.ef_is_a


def test_earth_fault_sensitivity_slg_652_50_ohm(nominal, relay):
    """Reported: SLG at 652, Rf 50 ohm, and its margin to the EF pickup."""
    net, sn, v_pre = nominal
    res = F.simulate_fault(net, sn, "652", "SLG", "a", 50.0, v_pre)
    t, el = F.relay_operate(relay, float(res.i_head_fault_a.max()), res.i_head_residual_a)
    print(f"\nSLG 652 Rf 50: I_f {1e3 * res.i_fault_ka.max():.1f} A, relay |3I0| "
          f"{res.i_head_residual_a:.1f} A (EF margin {res.i_head_residual_a / relay.ef_is_a:.3f}), "
          f"phase max {res.i_head_fault_a.max():.1f} A; trips by {el} in {t:.3f} s")
    assert res.i_head_residual_a > 0


def test_clearing_time_idmt_plus_breaker(nominal, relay):
    net, sn, v_pre = nominal
    res = F.simulate_fault(net, sn, "671", "LLG", "bc", 5.0, v_pre)
    i_ph, i_res = float(res.i_head_fault_a.max()), res.i_head_residual_a
    t, el = F.relay_operate(relay, i_ph, i_res)
    t_phase = 0.1 * 0.14 / ((i_ph / relay.is_a) ** 0.02 - 1)
    t_ef = 0.1 * 0.14 / ((i_res / relay.ef_is_a) ** 0.02 - 1)
    assert t == pytest.approx(min(t_phase, t_ef))
    for breaker in (F.BREAKER_S, F.BREAKER_SENS_S):
        n_pre, n_f, n_post = F.burst_lengths(t + breaker)
        assert (n_pre, n_post) == (50, 100)
        assert n_f == math.ceil(round((t + breaker) * 100, 6))
    print(f"\nLLG bc 671 Rf 5: relay {t:.4f} s ({el}); clearing {t + F.BREAKER_S:.4f} s "
          f"(80 ms case {t + F.BREAKER_SENS_S:.4f} s)")
    assert F.idmt_time(0.99 * relay.is_a, relay.is_a, 0.1) == math.inf


def test_each_fault_type_burst_classification_matches(nominal, relay):
    """§2D / X.3.1 item 10: each fault type gives a 10 ms burst whose
    classification equals the one of the simulated residual and duration."""
    net, sn, _ = nominal
    import pandas as pd
    for ftype, ph in (("SLG", "b"), ("LL", "ca"), ("LLG", "ab"), ("3PH", "abc")):
        for rf in F.RF_OHM:
            ev = pd.Series({"event_id": 1, "kind": "fault", "bus": "652", "ftype": ftype,
                            "phases": ph, "rf_ohm": rf})
            out = F.run_event(net, sn, ev, relay)
            assert all(m["match"] for m in out["meters"]), (ftype, rf)
            n = {m["n_samples"] for m in out["meters"]}
            assert n == {sum(F.burst_lengths(out["t_clear_s"]))}
    quiet = F.run_event(net, sn, pd.Series({"event_id": 2, "kind": "quiet"}), relay)
    assert all(m["class_kind"] == "no_dip" for m in quiet["meters"])


def test_event_set_reproducible_and_scoped():
    net = build_network("B")
    a, b = F.make_event_set(net), F.make_event_set(net)
    assert a.equals(b) and len(a) == 400
    fl = a[a.kind == "fault"]
    assert len(fl) == 200 and set(fl.ftype) == set(F.FAULT_TYPES) and set(fl.rf_ohm) == set(F.RF_OHM)
    assert not set(fl.bus) & set(F.EXCLUDED_BUSES)
    assert set(fl.bus) <= set(F.fault_buses(net))
    assert (fl[fl.ftype == "SLG"].phases.str.len() == 1).all()
    assert fl[fl.ftype.isin(["LL", "LLG"])].phases.isin(["ab", "bc", "ca"]).all()


# --------------------------------------------------------------------------
# Classifier on synthetic waveforms (X.3.2 item 17)
# --------------------------------------------------------------------------

U_MID = {"80-90": 85.0, "70-80": 75.0, "40-70": 55.0, "5-40": 20.0}
T_MID = {"10-200": 100, "200-500": 300, "500-1000": 750, "1000-5000": 3000, "5000-60000": 30000}
VB = 6350.85


def _wave(u_pct_phases, dur_ms, n_pre=50, n_post=100, post_pct=100.0):
    n = dur_ms // 10
    pre = np.full((n_pre, 3), VB)
    dip = np.tile(np.asarray(u_pct_phases, float) / 100 * VB, (n, 1))
    post = np.full((n_post, 3), post_pct / 100 * VB)
    return np.vstack([pre, dip, post])


@pytest.mark.parametrize("ub", list(U_MID))
@pytest.mark.parametrize("tb", list(T_MID))
def test_classifier_table13_bins(ub, tb):
    v = _wave([U_MID[ub], 100.0, 100.0], T_MID[tb])           # single-phase dip (polyphase rule)
    c = F.classify(v, VB)
    assert (c.kind, c.u_bin, c.t_bin) == ("dip", ub, tb)
    assert c.residual_pct == pytest.approx(U_MID[ub]) and c.duration_ms == T_MID[tb]


def test_classifier_bin_edges():
    assert F.t_bin(10) == "10-200" and F.t_bin(200) == "10-200" and F.t_bin(210) == "200-500"
    assert F.t_bin(60000) == "5000-60000" and F.t_bin(60010) is None
    assert F.u_bin(80.0) == "80-90" and F.u_bin(79.99) == "70-80" and F.u_bin(5.0) == "5-40"
    assert F.u_bin(4.99) == "<5"


def test_classifier_no_dip():
    v = _wave([92.0, 95.0, 100.0], 500)
    assert F.classify(v, VB).kind == "no_dip"


def test_classifier_not_cleared():
    """Dip lasting to the 60 s cap with no recovery = not cleared."""
    v = _wave([60.0, 100.0, 100.0], 60000, n_post=0)
    c = F.classify(v, VB)
    assert (c.kind, c.u_bin, c.t_bin) == ("dip_not_cleared", "40-70", "5000-60000")


@pytest.mark.parametrize("tb,dur", [("10-200", 100), ("200-500", 300), ("500-1000", 750),
                                    ("1000-5000", 3000), ("5000-180000", 90000)])
def test_classifier_table12_interruption(tb, dur):
    v = _wave([2.0, 3.0, 1.0], dur)                           # all phases < 5 %
    c = F.classify(v, VB)
    assert (c.kind, c.t_bin, c.interruption_open_ended) == ("interruption", tb, False)


def test_classifier_dip_then_interruption_after_clearing():
    """A downstream meter: dip during the fault, then the breaker opens (0 V)."""
    v = _wave([40.0, 90.0, 95.0], 300, post_pct=0.0)
    c = F.classify(v, VB)
    assert (c.kind, c.u_bin, c.t_bin) == ("dip", "40-70", "200-500")
    assert c.interruption_ms == 1000 and c.interruption_open_ended
