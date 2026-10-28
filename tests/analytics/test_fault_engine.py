"""Fault engine unit tests (run sheet Step C) on hand-built bursts."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from src.analytics import fault_engine as E
from src.analytics import meters as M

HEAD_V = 6350.85
DT_V = 249.99
T0 = pd.Timestamp("2024-03-11 10:00:00", tz="Asia/Kolkata")
P = E.Params(rho_star=1.5, coincidence_samples=3, balance_tol=0.2, delta_pp=1.0)


def doc(meter_id, v, i, t0=T0, t_poll="2026-10-27T00:00:00Z"):
    v, i = np.asarray(v, float), np.asarray(i, float)
    return {"meter_id": meter_id, "t_start": t0.isoformat(timespec="milliseconds"), "n_samples": len(v),
            "samples": {"voltage_v": {p: v[:, k].tolist() for k, p in enumerate("abc")},
                        "current_a": {p: i[:, k].tolist() for k, p in enumerate("abc")}},
            "t_poll_utc": t_poll}


def head_burst(v_fault, i_fault, n_fault=30, i_pre=(100.0, 100.0, 100.0)):
    """Feeder head: 50 pre samples, the fault, then the breaker open (V pre, I 0)."""
    vp, ip = np.full(3, HEAD_V), np.asarray(i_pre, float)
    v = np.vstack([np.tile(vp, (50, 1)), np.tile(v_fault, (n_fault, 1)), np.tile(vp, (100, 1))])
    i = np.vstack([np.tile(ip, (50, 1)), np.tile(i_fault, (n_fault, 1)), np.zeros((100, 3))])
    return v, i


def dt_burst(u_fault_pct, n_fault=30, interrupt=True):
    """DT meter: 50 pre samples at 100 %, the fault, then the interruption (0 V)."""
    vp = np.full(3, DT_V)
    vf = np.asarray(u_fault_pct, float) / 100 * DT_V
    post = np.zeros((100, 3)) if interrupt else np.tile(vp, (100, 1))
    v = np.vstack([np.tile(vp, (50, 1)), np.tile(vf, (n_fault, 1)), post])
    i = np.full((len(v), 3), 120.0)
    i[50 + n_fault:] = 0.0 if interrupt else 120.0
    return v, i


def analyse(meter_id, v, i, t0=T0, rho=P.rho_star):
    return E.analyse(E.burst_from_doc(doc(meter_id, v, i, t0)), rho)


# -- Step 4: IDMT worked example (plan §1 Step 4, post-P3 values) -------------

def test_idmt_worked_example():
    r = E.load_relay()
    assert r.is_a == pytest.approx(261.9461, abs=1e-4) and r.ef_is_a == pytest.approx(65.4865, abs=1e-4)
    assert r.tms == r.ef_tms == 0.1 and r.breaker_s == 0.060
    t_ef = E.idmt_time_s(123.93, r.ef_is_a, r.ef_tms, r.k, r.alpha)
    t_ph = E.idmt_time_s(334.45, r.is_a, r.tms, r.k, r.alpha)
    assert round(t_ef, 4) == 1.0904
    assert round(t_ph, 2) == 2.86
    assert E.idmt_time_s(r.is_a, r.is_a, r.tms) == float("inf")


# -- Step 3: type rules on synthetic feeder-head bursts ----------------------

@pytest.mark.parametrize("ftype,v_fault,i_fault", [
    ("SLG", (0.6, 1.0, 1.0), (500, 100, 100)),
    ("LL", (1.0, 0.7, 0.7), (100, 800, 760)),
    ("LLG", (1.0, 0.55, 0.6), (100, 900, 500)),
    ("3PH", (0.5, 0.5, 0.5), (900, 880, 910)),
])
def test_fault_type_rules(ftype, v_fault, i_fault):
    v, i = head_burst(np.asarray(v_fault) * HEAD_V, i_fault)
    a = analyse("m10", v, i)
    assert a.detected and a.onset_k == 50
    ev = E.finalise(E.merge([a], P.coincidence_samples)[0], P, E.load_relay())
    assert ev.ftype == ftype and ev.type_source == "feeder_head"
    assert ev.clearing_ms == 300.0 and ev.clearing_source == "feeder_head"


def test_type_uses_dips_when_no_phase_rises():
    v, i = head_burst(np.array([0.6, 1.0, 1.0]) * HEAD_V, (120, 100, 100))    # rho 1.2 < rho*
    a = analyse("m10", v, i)
    assert a.oc_phases == "" and a.dip_phases == "a"
    assert E.fault_type(a, P.balance_tol) == "SLG"


# -- Step 2: no-dip current rise ---------------------------------------------

def test_no_dip_current_rise():
    v, i = head_burst(np.full(3, HEAD_V), (250, 100, 100))                     # rho 2.5, no dip
    a = analyse("m10", v, i)
    assert a.kind == "no_dip" and a.by == ("current_rise",) and a.oc_phases == "a"
    assert a.rho[0] == pytest.approx(2.5) and a.onset_k == 50 and a.last_k == 79
    ev = E.finalise(E.merge([a], 3)[0], P, E.load_relay())
    assert ev.ftype == "SLG" and ev.head_no_dip and ev.head_rise_only
    assert not analyse("m10", v, i, rho=3.0).detected


def test_quiet_window_no_detection():
    v = np.tile([HEAD_V, HEAD_V, HEAD_V], (150, 1))
    i = np.tile([100.0, 90.0, 95.0], (150, 1))
    a = analyse("m10", v, i, rho=1.05)
    assert not a.detected and a.kind == "no_dip"
    assert E.flat_alerts_burst(E.burst_from_doc(doc("m10", v, i))) == []


# -- Step 1: polyphase start / end / residual --------------------------------

def test_polyphase_rule_hand_built():
    u = np.full((200, 3), 100.0)
    u[50:61, 0] = 70.0            # a below 90 % for samples 50..60
    u[55:71, 1] = 85.0            # b below 90 % for samples 55..70
    u[60:65, 2] = 62.0            # c lowest: 62 %
    v = u / 100 * DT_V
    i = np.full((200, 3), 120.0)
    a = analyse("m15", v, i)
    assert a.kind == "dip" and a.onset_k == 50 and a.last_k == 70
    assert a.residual_pct == pytest.approx(62.0, abs=1e-3)
    assert a.duration_ms == 210.0 and a.u_bin == "40-70" and a.t_bin == "200-500"
    assert a.dip_pp == pytest.approx(38.0, abs=1e-3)


def test_interruption_excluded_from_dip_depth():
    v, i = dt_burst((80, 85, 90))
    a = analyse("m15", v, i)
    assert a.kind == "dip" and a.dip_pp == pytest.approx(20.0, abs=1e-3)
    assert a.clear_k == 80 and a.last_k == 179


# -- Step 6: coincidence window ----------------------------------------------

def _shifted(meter_id, samples):
    v, i = dt_burst((80, 80, 80), n_fault=10, interrupt=False)
    return analyse(meter_id, v, i, t0=T0 + pd.Timedelta(milliseconds=10 * samples))


def test_merge_inside_and_outside_window():
    a = _shifted("m15", 0)                       # disturbed samples 50..59 -> last at +590 ms
    inside = _shifted("m16", 12)                 # onset +620 ms <= 590 + 30 ms window
    outside = _shifted("m17", 14)                # onset +640 ms > 620 ms
    assert len(E.merge([a, inside], 3)) == 1
    assert len(E.merge([a, outside], 3)) == 2
    # strict onset mode: only onsets within one window of the first onset
    near = _shifted("m16", 2)
    assert len(E.merge([a, near], 3, mode="strict")) == 1
    assert len(E.merge([a, inside], 3, mode="strict")) == 2


def test_open_event_absorbs_post_clearing_interruption():
    """A DT that did not dip detects only the interruption when the breaker
    opens; it joins the event of the meters that dipped (decision 27 Oct)."""
    dipped = analyse("m14", *dt_burst((60, 95, 95)))
    v = np.vstack([np.tile(np.full(3, DT_V), (80, 1)), np.zeros((100, 3))])
    late = analyse("m11", v, np.full((180, 3), 50.0))
    assert late.kind == "interruption" and late.onset_k == 80
    assert len(E.merge([dipped, late], 1)) == 1
    assert len(E.merge([dipped, late], 10, mode="strict")) == 2


def test_merge_is_order_independent():
    ans = [_shifted(m, s) for m, s in (("m15", 0), ("m16", 5), ("m17", 40), ("m18", 41))]
    ref = [e.meters for e in E.merge(ans, 3)]
    for seed in range(5):
        random.Random(seed).shuffle(ans)
        assert [e.meters for e in E.merge(ans, 3)] == ref


# -- Step 7: zone location ---------------------------------------------------

@pytest.mark.parametrize("deep,node,zone", [
    (("m12", "m15"), "632", "A"),               # dt_645_1 + dt_671_1
    (("m22", "m18"), "671", "E"),               # dt_611_1 + dt_675_1
    (("m22", "m14"), "684", "G"),               # dt_611_1 + dt_652_1
    (("m14",), "652", "G"),
])
def test_zone_location(deep, node, zone):
    ans = []
    for m in M.dt_meters():
        u = 60.0 if m.meter_id in deep else 75.0       # others 15 pp shallower
        ans.append(analyse(m.meter_id, *dt_burst((u, 95, 95))))
    loc = E.locate(ans, delta_pp=1.0)
    assert loc["node"] == node and loc["zone"] == zone
    assert set(loc["set"]) == set(deep)
    assert loc["baseline_bus"] in {"645", "671", "611", "675", "652"}


def test_bus_virtual_burst_rejected():
    v, i = dt_burst((80, 80, 80))
    with pytest.raises(M.NotPhysicalMeter):
        E.burst_from_doc(doc("m05", v, i))


# -- Step 5 and the 15-min flat baseline -------------------------------------

def test_weekly_unbalance_and_flat_stream():
    t = pd.date_range("2024-09-02", periods=672, freq="15min")
    ll = np.full((672, 3), 11000.0)
    ll[:40, 0] = 11500.0                      # 40 / 672 = 6 % of the week above the limit
    rows = pd.DataFrame({"meter_id": "m10", "t_sim": t.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                         **{f"voltage_ll_v_{p}": ll[:, k] for k, p in enumerate(("ab", "bc", "ca"))},
                         **{f"voltage_v_{p}": HEAD_V for p in "abc"}})
    rows.loc[100:102, "voltage_v_a"] = 0.85 * HEAD_V            # one 3-sample excursion
    s = E.stream_frame(rows)
    w = E.weekly_unbalance(s)
    assert len(w) == 1 and bool(w.alarm.iloc[0]) and w.samples.iloc[0] == 672
    rows.loc[:, "voltage_ll_v_ab"] = 11000.0
    rows.loc[:20, "voltage_ll_v_ab"] = 11500.0               # 21 / 672 = 3.1 % -> no alarm
    assert not bool(E.weekly_unbalance(E.stream_frame(rows)).alarm.iloc[0])
    assert len(E.flat_alerts_stream(s)) == 1
