"""
Part 2C acceptance on one simulated week (brief v2 §2C, X.1/X.2; plan CP3).

- DT and household table produced; sum of household true load = DT LV load
  at every step (exact).
- Feeder head = sum of DT HV inputs + 11 kV line losses within 1e-3 MW.
- For every thief and every day: reported <= true; severity matches its
  definition.
- Pools disjoint; feeder pool FLAG = 0 only.
- bus_virtual = sum of DT meters; meter classes (CEA 2019 amendment).
- Delta MV loads behind Dyn11 DTs draw 11 kV current in the IEEE pair.

The year-level checks (brief v2 X.3 item 4: IS 17036 band at every step of
2024, DT loading at the yearly peak, DT losses vs IS 1180; reported losses,
OLTC, VUF) run in ``scripts/run_year.py`` on the cached year.  Run with
``pytest -s`` to see the numbers.
"""

import math

import numpy as np
import pandapower as pp
import pandas as pd
import pytest

from src.data_source import households as H
from src.data_source import metering as M
from src.data_source import simulation as S
from src.data_source.feeder_model import build_network, run_timeseries
from src.data_source.stage_b import (
    DT_LV_KV, DT_SHIFT_DEGREE, IS1180_LOSS_W, MAG0_PERCENT, MAG0_RX, S_MAX_KVA,
    dt_losses_w, r0_eff_pu, size_dts,
)

START, END = "2024-01-01", "2024-01-07 23:45"

# DT sizing (brief v2 X.3 item 3: rating/3 >= max per-phase kVA at design peak;
# 634 keeps XFM-1 500 kVA), worked out by hand in docs/phase2_progress.md
EXPECTED_DTS = {"634": (1, 500), "645": (1, 250), "646": (1, 315), "652": (1, 160),
                "671": (3, 500), "675": (3, 630), "692": (1, 250), "611": (1, 200),
                "670": (1, 500)}


@pytest.fixture(scope="module")
def pop():
    return S.build_population(START, END)


@pytest.fixture(scope="module")
def week(pop):
    frame, state = run_timeseries(START, END, net=pop.net, return_state=True,
                                  dt_loads=pop.dt_loads(pd.Timestamp(START)))
    return frame, state


# --------------------------------------------------------------------------
# 2C.1 distribution transformers
# --------------------------------------------------------------------------

def test_dt_table(pop):
    d = pop.net["dts"]
    print("\n" + S.nominal_dt_table().to_string(index=False))
    assert len(d) == 13
    for bus, (k, rating) in EXPECTED_DTS.items():
        rows = d[d.bus == bus]
        assert len(rows) == k and (rows.rating_kva == rating).all(), bus
    x1 = d[d.bus == "634"].iloc[0]
    assert x1.rating_kva == 500 and x1["name"] == "dt_634_1"
    t = pop.net.trafo.loc[d.trafo.astype(int)]
    assert (t.vn_hv_kv == 11.0).all() and (t.vn_lv_kv == DT_LV_KV).all()
    assert (t.vector_group == "Dyn").all() and (t.shift_degree == DT_SHIFT_DEGREE).all()
    assert (t.vk_percent == 4.5).all()                      # IS 1180 Table 3 / Table 6
    for r in d.itertuples():
        p0, pk = dt_losses_w(r.rating_kva)
        t50, t100 = IS1180_LOSS_W[r.rating_kva]
        assert p0 + pk == pytest.approx(t100) and p0 + 0.25 * pk == pytest.approx(t50)
    assert "XFM-1" not in set(pop.net.trafo.name)


def test_size_dts_rule():
    """Argument: the bus's maximum per-phase kVA; k x rating with rating/3 >= it / k."""
    assert size_dts(443.4) == (3, 500)         # 671: 3 x 443 = 1330 -> ceil(1330/630) = 3 -> 500
    assert size_dts(520.9) == (3, 630)         # 675 phase a: 1563 -> 3 x 521 -> 630
    assert size_dts(135.3) == (1, 500)         # 670 phase c: 406 -> 500
    assert size_dts(51.4) == (1, 160)          # 652 spread over a/b/c: 154 -> 160
    assert size_dts(210.0) == (1, 630)
    assert size_dts(210.4) == (2, 400)
    assert S_MAX_KVA == 630.0


def test_lv_phasing():
    """X.3 item 2: single-/two-phase IEEE buses spread equally, others unchanged."""
    ieee_l, lv = H.ieee_phase_loads(), H.lv_phase_loads()
    for bus, (p, q) in ieee_l.items():
        lp, lq = lv[bus]
        assert lp.sum() == pytest.approx(p.sum()) and lq.sum() == pytest.approx(q.sum())
        if bus in H.EQUAL_SPLIT_BUSES:
            assert np.allclose(lp, p.sum() / 3) and np.allclose(lq, q.sum() / 3)
        else:
            assert np.array_equal(lp, p) and np.array_equal(lq, q)
    assert set(H.EQUAL_SPLIT_BUSES) == {b for b, (p, _) in ieee_l.items() if (p > 0).sum() < 3}


def test_delta_loads_mv_currents():
    """Dyn11: an LV phase a/b/c load draws 11 kV current in A-B / B-C / C-A
    (the IEEE delta pair index is the LV phase index).  One DT at a time
    carries a load on a single LV phase; the 11 kV phase not in its pair is idle."""
    for lv_ph, idle in (("a", "c"), ("b", "a"), ("c", "b")):
        net = build_network("B")
        d = net["dts"].set_index("name")
        ld = int(d.at["dt_646_1", "load"])
        al = net.asymmetric_load
        for p in "abc":
            al.at[ld, f"p_{p}_mw"] = 0.1 if p == lv_ph else 0.0
            al.at[ld, f"q_{p}_mvar"] = 0.05 if p == lv_ph else 0.0
        pp.runpp_3ph(net)
        r = net.res_trafo_3ph.loc[int(d.at["dt_646_1", "trafo"])]
        cur = {p: r[f"i_{p}_hv_ka"] for p in "abc"}
        busy = [p for p in "abc" if p != idle]
        print(f"\ndt_646_1 LV {lv_ph}: HV currents {({p: round(v * 1e3, 2) for p, v in cur.items()})} A")
        # the idle line carries only the magnetising current (i0 = P0/S_r = 0.12 % of rated)
        assert cur[idle] < 1e-2 * cur[busy[0]]
        assert cur[busy[0]] == pytest.approx(cur[busy[1]], rel=1e-2)


def test_r0_eff_dyn_units():
    """X.3.2 item 14: zero-sequence magnetising branch open -> R0,eff / R >= 0.99
    for Sub_XFMR and every DT (vkr0 = vkr reported, X.3.1 item 2a)."""
    net = build_network("B")
    names = ["Sub_XFMR"] + list(net["dts"].name)
    for name in names:
        t = net.trafo.loc[net.trafo.name == name].iloc[0]
        ratio = r0_eff_pu(t) / (t.vkr0_percent / 100)
        former = t.copy()
        former["mag0_percent"] = 100.0
        old = r0_eff_pu(former) / (t.vkr0_percent / 100)
        print(f"\n{name}: vkr {t.vkr_percent:.4f} % vkr0 {t.vkr0_percent:.4f} %, "
              f"R0,eff/R {ratio:.4f} (with the former mag0 100 %: {old:.3f})", end="")
        assert t.vkr0_percent == pytest.approx(t.vkr_percent)
        assert t.mag0_percent == MAG0_PERCENT and t.mag0_rx == MAG0_RX
        assert ratio >= 0.99


def test_dyn11_single_phase_losses():
    """X.3.1 item 2b: one 250 kVA Dyn11 DT (B8 parameters), single-phase LV load.

    Hand calculation in sequence components of the LV current:
    P = P0 v_hv^2 + Pk (K0^2 + K1^2 + K2^2) with R0 = R1 (open magnetising
    branch).  Asserted within 1 %; the ratio is reported.
    """
    rating = 250
    p0_w, pk_w = dt_losses_w(rating)
    vkr = pk_w / (rating * 1e3) * 100
    net = pp.create_empty_network(f_hz=50.0)
    hv = pp.create_bus(net, vn_kv=11.0)
    lv = pp.create_bus(net, vn_kv=DT_LV_KV)
    pp.create_ext_grid(net, hv, vm_pu=1.0, s_sc_max_mva=1000.0, rx_max=0.1, x0x_max=1.0,
                       r0x0_max=0.1)
    pp.create_transformer_from_parameters(
        net, hv_bus=hv, lv_bus=lv, sn_mva=rating / 1e3, vn_hv_kv=11.0, vn_lv_kv=DT_LV_KV,
        vkr_percent=vkr, vk_percent=4.5, pfe_kw=p0_w / 1e3, i0_percent=p0_w / (rating * 1e3) * 100,
        vector_group="Dyn", vk0_percent=4.5, vkr0_percent=vkr, mag0_percent=MAG0_PERCENT,
        mag0_rx=MAG0_RX, si0_hv_partial=0.9, shift_degree=DT_SHIFT_DEGREE)
    pp.create_asymmetric_load(net, lv, p_a_mw=0.05, q_a_mvar=0.02, type="wye")
    pp.runpp_3ph(net)
    rt = net.res_trafo_3ph.iloc[0]
    model_w = sum(rt[f"pl_{p}_mw"] for p in "abc") * 1e6
    rb = net.res_bus_3ph.loc[lv]
    v = np.array([rb[f"vm_{p}_pu"] * np.exp(1j * np.radians(rb[f"va_{p}_degree"]))
                  for p in "abc"]) * DT_LV_KV * 1e3 / math.sqrt(3)
    s = -np.array([rt[f"p_{p}_lv_mw"] + 1j * rt[f"q_{p}_lv_mvar"] for p in "abc"]) * 1e6
    i = np.where(np.abs(v) > 0, np.conj(s / v), 0.0)
    a = np.exp(2j * np.pi / 3)
    i012 = np.array([[1, 1, 1], [1, a, a * a], [1, a * a, a]]) @ i / 3
    k012 = np.abs(i012) / (rating * 1e3 / (math.sqrt(3) * DT_LV_KV * 1e3))
    vh = net.res_bus_3ph.loc[hv]
    v_hv2 = np.mean([vh[f"vm_{p}_pu"] ** 2 for p in "abc"])
    hand_w = p0_w * v_hv2 + pk_w * float((k012 ** 2).sum())
    print(f"\nDyn11 250 kVA, 50 kW + 20 kvar on LV phase a: K012 {k012.round(4)}, "
          f"pandapower {model_w:.1f} W, hand {hand_w:.1f} W, ratio {model_w / hand_w:.4f}")
    assert model_w / hand_w == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------
# 2C.2 households
# --------------------------------------------------------------------------

def test_household_counts_and_split(pop):
    hh, info = pop.hh, pop.info
    dp = info["design_peak"]
    # X.3 item 1: s_design = yearly max of the pool-average 15-min demand / h_bar
    assert pop.s_design == pytest.approx(dp["pool_avg_peak_kw"] / pop.h_bar_kw)
    assert dp["steps"] == len(S.calendar_days()) * 96            # whole 2024 calendar
    assert dp["p99_9_kw"] <= dp["pool_avg_peak_kw"]
    assert pop.s_design > 1.0                                     # peak above the mean
    bus_p = {b: float(v[0].sum()) for b, v in H.lv_phase_loads().items()}
    for bus, n in pop.counts.items():
        assert n == round(bus_p[bus] / (pop.h_bar_kw * pop.s_design)), bus
        assert int((hh.bus == bus).sum()) == n
    print(f"\nh_bar {pop.h_bar_kw:.4f} kW, s_design {pop.s_design:.4f} (peak {dp['timestamp']}, "
          f"P99.9 {dp['p99_9_kw']:.4f} kW), N {pop.counts} (total {info['households']}); "
          f"memory {hh.info['memory_mb']} MB + thief series {info['thief_series_mb']} MB")
    d = pop.net["dts"]
    loads = H.lv_phase_loads()
    for bus in pop.counts:
        on_bus = hh.bus == bus
        per_dt = np.bincount(hh.dt_index[on_bus], minlength=len(d))[d.index[d.bus == bus]]
        assert per_dt.max() - per_dt.min() <= 1                 # equal ratings at a bus
        assert set(np.unique(hh.phase[on_bus])) == set(np.flatnonzero(loads[bus][0] > 0)), bus
        if bus in H.EQUAL_SPLIT_BUSES:                           # X.3 item 2: equal over a/b/c
            for k in d.index[d.bus == bus]:
                n_ph = np.bincount(hh.phase[on_bus & (hh.dt_index == k)], minlength=3)
                assert n_ph.max() - n_ph.min() <= 1, bus
    assert hh.true_kwh.dtype == np.float32


def test_sum_household_true_equals_dt_load(pop, week):
    """DT LV load = sum of true household load at every step.

    Exact for the load the power flow is given (floating-point identity);
    the DT meter (solved LV terminal flow) matches it within the runpp_3ph
    convergence tolerance (1e-8 MVA per bus).
    """
    frame, _ = week
    hh = pop.hh
    dt_names = pop.net["dts"].name.to_numpy()
    f = pop.dt_loads(pd.Timestamp(START))
    ts = pd.date_range(START, END, freq="15min")
    rows = frame[frame.load_id.str.startswith("dt_")]
    p = rows.pivot_table(index="timestamp", columns=["load_id", "phase"],
                         values="active_power_kw", aggfunc="first")
    worst_in = worst_meter = 0.0
    for day in range(len(hh.days)):
        kw = hh.true_kw_day(day)                               # (n_hh, 96)
        given = np.stack([f(day * 96 + s, ts[day * 96 + s])[0] for s in range(96)])
        for k, name in enumerate(dt_names):
            for ph in range(3):
                members = (hh.dt_index == k) & (hh.phase == ph)
                expected = kw[members].sum(axis=0)
                worst_in = max(worst_in, float(np.max(np.abs(given[:, k, ph] - expected)
                                                      / np.maximum(expected, 1.0))))
                got = p[(name, "abc"[ph])].to_numpy()[day * 96:(day + 1) * 96]
                worst_meter = max(worst_meter, float(np.max(np.abs(got - expected))))
    print(f"\nload given to the power flow vs sum household true: max rel {worst_in:.1e}; "
          f"DT meter (solved) vs sum household true: max {worst_meter:.1e} kW")
    assert worst_in < 1e-12
    assert worst_meter < 1e-4


def test_theft_never_touches_the_power_flow(pop):
    """The DT loads are built from true kWh only."""
    p_day = pop.hh.dt_phase_daily_kw()
    assert p_day.sum() == pytest.approx(pop.hh.true_kwh.astype(float).sum() / 24.0)


# --------------------------------------------------------------------------
# 2C.3 meters
# --------------------------------------------------------------------------

def test_feeder_head_balance(week):
    frame, st = week
    bal = (st.p_head_mw - st.p_dt_hv_mw - st.line_losses_mw).abs()
    print(f"\nfeeder head - (sum DT HV + 11 kV line losses): max {bal.max():.2e} MW")
    assert bal.max() < 1e-3
    fh = frame[frame.load_id == "feeder_head_650"].groupby("timestamp").active_power_kw.sum()
    assert np.allclose(fh.to_numpy() / 1e3, st.p_head_mw.to_numpy(), atol=1e-9)


def test_bus_virtual_is_sum_of_dts(pop, week):
    frame, _ = week
    d = pop.net["dts"]
    gains = dict(zip(d.name, pop.gain_dt))
    for fr in (frame, S.apply_meter_errors(frame, gains, pop.gain_feeder_head)):
        dt = fr[fr.load_id.str.startswith("dt_")].copy()
        dt["lp"] = dt.load_id.map(dict(zip(d.name, d.load_id)))
        s = dt.groupby(["timestamp", "lp", "phase"])[["active_power_kw", "reactive_power_kvar"]].sum()
        bv = fr[fr.load_id.str.startswith("load_")].set_index(["timestamp", "load_id", "phase"])
        exp = s.reindex(bv.index)
        assert np.allclose(bv.active_power_kw, exp.active_power_kw, atol=1e-9)
        assert np.allclose(bv.reactive_power_kvar, exp.reactive_power_kvar, atol=1e-9)


def test_meter_classes_and_gains(pop):
    assert (M.CLASS_HOUSEHOLD, M.CLASS_DT, M.CLASS_FEEDER_HEAD) == (1.0, 0.5, 0.5)
    assert np.abs(pop.gain_household - 1).max() <= 0.01
    assert np.abs(pop.gain_dt - 1).max() <= 0.005
    assert abs(pop.gain_feeder_head - 1) <= 0.005


def test_metered_frame_applies_dt_gain(pop, week):
    frame, _ = week
    d = pop.net["dts"]
    met = S.apply_meter_errors(frame, dict(zip(d.name, pop.gain_dt)), pop.gain_feeder_head)
    for name, g in zip(d.name, pop.gain_dt):
        a = frame[frame.load_id == name].active_power_kw.to_numpy()
        b = met[met.load_id == name].active_power_kw.to_numpy()
        assert np.allclose(b, a * g)
    assert np.allclose(met.voltage_v, frame.voltage_v)        # voltage not scaled


# --------------------------------------------------------------------------
# 2C.4 SGCC split and theft
# --------------------------------------------------------------------------

def test_pools_disjoint_and_feeder_flag0(pop):
    m, pools = pop.matrix, pop.pools
    pools.assert_valid(m)
    f = set(pools.feeder)
    assert not f & set(pools.detector_train) and not f & set(pools.detector_test)
    assert not set(pools.detector_train) & set(pools.detector_test)
    assert (m.flag[pools.feeder] == 0).all()
    assert len(pools.feeder) == len(pop.hh.hh_id)
    print(f"\npools: {pop.info['pools']}; p = {m.theft_prevalence:.4f}")


def test_thief_count_and_attacks(pop):
    """X.3.1 item 3: count = round(p x N_total), thieves never mostly-zero."""
    th, hh = pop.theft, pop.hh
    assert th.thief.sum() == round(pop.matrix.theft_prevalence * len(hh.hh_id))
    assert not (th.thief & hh.mostly_zero).any()
    assert set(th.attack[th.thief]) == set(M.ATTACKS)
    assert (th.onset_day[th.thief] >= 0).all() and (th.onset_day[~th.thief] == -1).all()
    gt = M.ground_truth(hh, th, pop.gain_household)
    assert set(gt.attack_class[gt.attack.isin(["h5", "h6"])]) == {"load-shifting"}
    assert set(gt.attack_class[gt.attack.isin(["h1", "h2", "h3", "h4"])]) == {"energy-reducing"}
    print(f"\nthieves per attack {pop.info['thieves_per_attack']}; mostly-zero households "
          f"{int(hh.mostly_zero.sum())} (per DT {pop.info['mostly_zero_per_dt']})")


def test_meter_error_within_class(pop):
    """X.3.1 item 5 / X.3.2 item 19: |metered - reported| <= class x max(reported, 1 Wh)
    per household-day (meter error tested apart from theft)."""
    t = M.daily_table(pop.hh, pop.theft, pop.gain_household)
    on = t.reported_kwh.notna()
    err = (t.metered_kwh[on] - t.reported_kwh[on]).abs().to_numpy(float)
    bound = M.CLASS_HOUSEHOLD / 100 * np.maximum(t.reported_kwh[on].to_numpy(float), 1e-3)
    assert (err <= bound * (1 + 1e-5)).all()


def test_reported_le_true_every_thief_every_day(pop):
    """Reported (the attacked value, before meter error) <= true (X.3.1 item 5)."""
    hh, th = pop.hh, pop.theft
    rep = M.reported_daily_kwh(hh, th)
    true = hh.true_kwh.astype(float)
    r, t = rep[th.thief_rows], true[th.thief_rows]
    online = ~np.isnan(r)
    assert np.all(r[online] <= t[online] * (1 + 1e-6) + 1e-9)
    # before onset the meter reports the truth
    for j, i in enumerate(th.thief_rows):
        a = th.onset_day[i]
        pre = online[j, :a]
        assert np.allclose(r[j, :a][pre], t[j, :a][pre], rtol=1e-5)


def test_severity_matches_definition(pop):
    gt = M.ground_truth(pop.hh, pop.theft, pop.gain_household)
    rep = M.reported_daily_kwh(pop.hh, pop.theft)
    true = pop.hh.true_kwh.astype(float)
    for i in pop.theft.thief_rows[:200]:
        a = pop.theft.onset_day[i]
        on = ~np.isnan(rep[i, a:])
        expected = 1 - rep[i, a:][on].sum() / true[i, a:][on].sum() if true[i, a:][on].sum() > 0 else 0
        assert gt.severity[i] == pytest.approx(expected, abs=1e-9)
    assert (gt.severity[~gt.thief] == 0).all()
    assert set(gt.columns) >= {"household", "dt", "bus", "thief", "attack", "onset", "severity",
                               "true_kwh", "reported_kwh"}
    sev = gt[gt.thief].groupby("attack").severity.mean().round(3).to_dict()
    print(f"\nmean severity by attack (week window): {sev}")
