"""CP5: one simulated week end to end with the CP3 population and CP4 bursts.

Meter server (CP3 year cache frame, ``meter_frame_metered.parquet``: CP3
households on the 13 DTs, 23 meter points, meter errors applied) plus the CP4
fault bursts of that week -> gateway -> Mosquitto -> listener, all real
processes (``scripts/devstack.py``).

The week is the Monday-Sunday week of 2024 with the most CP4 fault events
(2024-03-11 .. 2024-03-17: 11 faults, every fault type).  Each burst uses its
event's own pre-fault state (DT loads and OLTC tap of its step, from the CP3
cache), the CP4 relay and the 60 ms breaker, and is written to all 23 meters
right after its step.

Checks (exit 1 if any fails):
- every (meter, 15-min step) of the week delivered exactly once (validated);
- every decoded value equals the served frame within register quantisation;
- every burst arrives at the listener with every sample (count = expected),
  values within quantisation.
Reported: flags (GAP etc.), latency, burst reads.  Writes
``results/cp5_week_<speed>x.json``.

PowerShell::

    python scripts/run_week_e2e.py                 # ~4 min at 3600x
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from devstack import Recorder, Stack, stop, wait_for  # noqa: E402

WEEK = ("2024-03-11 00:00", "2024-03-17 23:45")
CACHE_FRAME = ROOT / "results" / "year_2024" / "meter_frame_metered.parquet"
IST = "+05:30"
TOL = {"voltage_v": 0.005 + 1e-9, "load_current_a": 0.0005 + 1e-9,
       "active_power_kw": 0.0005 + 1e-9, "reactive_power_kvar": 0.0005 + 1e-9}
KEY = {"voltage_v": "voltage_v", "load_current_a": "current_a",
       "active_power_kw": "active_power_kw", "reactive_power_kvar": "reactive_power_kvar"}


def build_bursts(start: str, end: str) -> tuple[list[dict], pd.DataFrame]:
    """CP4 bursts of the events whose step lies in [start, end]."""
    from src.data_source import faults as F
    from src.data_source.feeder_model import build_network
    net = build_network("B")
    relay = F.relay_settings(F.max_head_current_a())
    ev = F.make_event_set(net)
    ev = ev[(ev.kind == "fault") & (ev.step >= pd.Timestamp(start)) & (ev.step <= pd.Timestamp(end))]
    cache = F.steps_from_cache(net, ev.step)
    bursts, rows = [], []
    for e in ev.itertuples():
        F.set_prefault(net, *cache[pd.Timestamp(e.step)])
        sn = F.build_sequence_network(net)
        res = (F.simulate_arc_fault(net, sn, e.bus, e.ftype, e.phases) if e.rf_model == "arc"
               else F.simulate_fault(net, sn, e.bus, e.ftype, e.phases, float(e.rf_ohm)))
        t_rel, element = F.relay_operate(relay, float(res.i_head_fault_a.max()), res.i_head_residual_a)
        t_clear = t_rel + relay.breaker_s
        t_start = pd.Timestamp(e.step) + pd.Timedelta(seconds=float(e.offset_s)) - pd.Timedelta(seconds=F.PRE_S)
        for lid, m in F.meter_faults(net, sn, res).items():
            v, i = F.build_burst(m, t_clear)
            bursts.append({"trigger": e.step, "load_id": lid, "event_id": int(e.event_id),
                           "t_start": t_start, "v": v, "i": i, "v_base_ln_v": m.v_base_ln_v,
                           "phases": "abc"})
        rows.append({"event_id": int(e.event_id), "step": str(e.step), "bus": e.bus, "ftype": e.ftype,
                     "phases": e.phases, "rf_ohm": float(res.rf_ohm), "t_clear_s": t_clear,
                     "element": element, "n_samples": int(sum(F.burst_lengths(t_clear)))})
    return bursts, pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speed", type=float, default=3600.0)
    ap.add_argument("--poll-interval", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="default results/cp5_week_<speed>x.json")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore", FutureWarning)
    from run_meter_server import load_cache_frame
    from src.data_source import meter_server as ms
    from src.ingestion.config import load_meters

    t0 = time.monotonic()
    frame = load_cache_frame(str(CACHE_FRAME), *WEEK)
    bursts, events = build_bursts(*WEEK)
    prep_s = time.monotonic() - t0
    meters = load_meters().meters
    lid_of = {m.meter_id: m.load_id for m in meters}
    steps = sorted(frame.timestamp.unique())
    st = Stack()
    rec = None
    try:
        ev_path = st.workdir / "bursts.npz"
        ms.save_event_file(ev_path, bursts)
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        assert rec.connected.wait(10)
        st.start_listener(persistent=True, client_id="cp5-week-listener")
        st.start_gateway(poll_interval=args.poll_interval, playback_speed=args.speed,
                         series_start=WEEK[0].replace(" ", "T") + ":00+05:30")
        st.start_meter(args.speed, start_delay=3.0, events=ev_path, cache_frame=CACHE_FRAME,
                       start=WEEK[0], end=WEEK[1])
        t_play = time.monotonic()
        expected = len(steps) * len(meters)
        n_ev = len(events) * len(meters)

        def done():
            val = rec.snapshot(contains="/validated/")
            tel = {(m.json()["meter_id"], m.json()["t_sim"]) for m in val}
            evs = [m for m in rec.snapshot(suffix="/event") if "/validated/" not in m.topic]
            return len(tel) >= expected and len(evs) >= n_ev

        timed_out = False
        try:
            wait_for(done, timeout=len(steps) * 900 / args.speed + 300, interval=2.0, what="the week")
        except TimeoutError:
            timed_out = True        # report what arrived; the checks below fail
        time.sleep(2.0)
        wall = time.monotonic() - t_play
        stats = st.http("/api/v1/stats")
        stop(st.gateway)
        gw_log = st.log_text("gateway")
        val = [m.json() for m in rec.snapshot(contains="/validated/")]
        raw_ev = [m.json() for m in rec.snapshot(suffix="/event") if "/validated/" not in m.topic]
    finally:
        if rec:
            rec.close()
        st.close()

    # -- delivery ---------------------------------------------------------------
    got = [(v["meter_id"], v["t_sim"]) for v in val]
    exp_keys = {(m.meter_id, pd.Timestamp(t).isoformat() + IST) for m in meters for t in steps}
    missing = sorted(exp_keys - set(got))
    dupes = len(got) - len(set(got))
    # -- values ---------------------------------------------------------------
    fr = frame.set_index(["load_id", "timestamp", "phase"])
    worst = defaultdict(float)
    bad_values = 0
    for v in val:
        lid = lid_of[v["meter_id"]]
        ts = pd.Timestamp(v["t_sim"][:-6])
        meas = v["measurements"]
        for ph in meas["voltage_v"]:
            if ph == "total":
                continue
            row = fr.loc[(lid, ts, ph)]
            for col, key in KEY.items():
                err = abs(meas[key][ph] - float(row[col]))
                worst[key] = max(worst[key], err)
                bad_values += err > TOL[col]
    # -- bursts ---------------------------------------------------------------
    exp_b = {(b["load_id"], b["event_id"]): b for b in bursts}
    burst_ok, burst_bad, samples = 0, [], 0
    for e in raw_ev:
        lid = lid_of[e["meter_id"]]
        b = exp_b.get((lid, e["event_id"]))
        if b is None:
            burst_bad.append((e["meter_id"], e["event_id"], "unexpected"))
            continue
        vb = b["v_base_ln_v"]
        ok = e["n_samples"] == len(b["v"])
        for k, p in enumerate("abc"):
            gv = np.asarray(e["samples"]["voltage_v"][p])
            ok &= len(gv) == len(b["v"]) and bool(np.all(
                np.abs(gv - b["v"][:, k]) <= 0.5e-4 * vb + np.abs(b["v"][:, k]) * 0.005 / vb + 1e-3 + 1e-9))
        samples += e["n_samples"]
        if ok:
            burst_ok += 1
        else:
            burst_bad.append((e["meter_id"], e["event_id"], "values"))
    import re
    reads = [(int(r[2]), int(r[3]), float(r[4])) for r in re.findall(
        r"(m\d+): event (\d+) read: (\d+) samples, (\d+) reads, ([\d.]+) s", gw_log)]
    flags = defaultdict(int)
    for v in val:
        for f in v["quality"]["flags"]:
            flags[f] += 1
    lat = sorted(v["ingest"]["latency_ms"]["total"] for v in val)
    summary = {
        "week": WEEK, "speed": args.speed, "poll_interval_s": args.poll_interval, "timed_out": timed_out,
        "missing_per_meter": {mid: n for mid, n in sorted(
            pd.Series([m for m, _ in missing]).value_counts().items())} if missing else {},
        "prep_s": round(prep_s, 1), "wall_s": round(wall, 1),
        "meters": len(meters), "steps": len(steps), "expected_messages": expected,
        "validated_unique": len(set(got)), "missing": len(missing), "missing_list": missing[:500],
        "duplicates": dupes, "flags": dict(flags), "listener_stats": stats,
        "latency_total_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * len(lat))], "max": lat[-1]},
        "value_check": {"bad": int(bad_values), "worst_abs_error": dict(worst), "tolerance": TOL},
        "events": events.to_dict("records"),
        "bursts": {"expected": len(exp_b), "received": len(raw_ev), "ok": burst_ok,
                   "bad": burst_bad[:10], "samples": samples,
                   "expected_samples": int(sum(len(b["v"]) for b in bursts)),
                   "gateway_reads": {"n": len(reads),
                                     "reads_per_burst": sorted({r for _, r, _ in reads}),
                                     "wall_s_median": float(np.median([s for *_, s in reads])) if reads else None,
                                     "wall_s_max": max((s for *_, s in reads), default=None)}},
    }
    summary["passed"] = bool(not missing and dupes == 0 and bad_values == 0
                             and burst_ok == len(exp_b) == len(raw_ev))
    out = Path(args.out) if args.out else ROOT / "results" / f"cp5_week_{int(args.speed)}x.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("listener_stats", "events")},
                     indent=2, default=str))
    if not summary["passed"]:
        print("CP5 WEEK END-TO-END FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
