"""End-to-end ingestion benchmark (D6; CP5 characterisation, brief v2 X.3.1 item 13).

For each playback speed, runs the whole chain for one simulated day — real
meter server fed by real ``run_timeseries`` output, gateway, Mosquitto,
listener — and measures per-hop latency, throughput, loss, duplicates and
gaps.  No latency budget is asserted (``LATENCY_BUDGET_MS`` stays ``None``).

CP5: all 23 meter points, and CP4 fault bursts injected during the day (the
event set's longest bursts, each written to every meter right after its
trigger step).  For every burst and meter the **event drain** is the window
from the burst being written to the gateway publishing it; the 15-min
interval reads whose step was written in that window (or one step after)
are counted **late** (polled more than one step period after the step was
written) or **missed** (never delivered), with the GAP flags in the window.
Reported for every speed.

Assertions (brief v2 X.3.3 item 9, CP5 decision option b):

* **900x: no interval lost**, in each of ``--asserted-runs`` (default 3)
  repeated runs of the benchmark day.  Each run reports the gateway's longest
  poll cycle (start-to-start between two successful polls of one meter) as
  headroom against the 1.0 s wall step.
* **Every speed: no silent loss.**  Per meter, the intervals lost must equal
  the intervals flagged (sum of ``gap_steps`` on GAP-flagged messages).  The
  gateway gets ``series_start`` = the day's first step (commissioning time),
  so intervals lost before a meter's first read are flagged too (X.3.3 item 10).
* Requested vs achieved speed: the actual step is the fitted mean wall time
  between server writes; the longest poll cycle is compared with it.
* 3600x and above: losses, their timestamps and the poll-cycle maximum are a
  measured characteristic.  A meter holds only its latest interval, so a
  gateway stall longer than one wall step overwrites an interval unread.

The highest acceleration with zero loss is reported (characterisation).

Writes ``results/bench_<timestamp>.json`` and ``results/bench_<timestamp>.csv``
(one row per validated message).

PowerShell::

    python scripts/bench_ingestion.py                     # 900x x3, 3600x, 7200x, 14400x
    python scripts/bench_ingestion.py --profiles sgcc     # SGCC-driven load profiles
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from devstack import Recorder, Stack, find_mosquitto, stop, wait_for  # noqa: E402
from src.ingestion.config import load_meters  # noqa: E402

STEP_S = 900.0
N_METERS = len(load_meters().meters)


def percentiles(vals: list[float]) -> dict:
    from src.ingestion.listener.stats import percentile

    s = sorted(vals)
    return {"n": len(s), "p50": percentile(s, 50), "p95": percentile(s, 95),
            "p99": percentile(s, 99), "max": s[-1] if s else None}


def versions() -> dict:
    out = {}
    for pkg in ("pymodbus", "aiomqtt", "paho-mqtt", "fastapi", "uvicorn", "pydantic",
                "pydantic-settings", "pandapower", "pandas", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    exe = find_mosquitto()
    if exe:
        r = subprocess.run([exe, "-h"], capture_output=True, text=True)
        out["mosquitto"] = (r.stdout or r.stderr).splitlines()[0].strip()
    return out


# Brief v2 X.3.3 item 9 (CP5 decision, option b): zero loss asserted at 900x
# only, in each of NO_LOSS_RUNS repeated runs; 3600x and above are a measured
# characteristic.  No silent loss is asserted at every speed.
ASSERT_NO_LOSS_SPEEDS = (900.0,)
NO_LOSS_RUNS = 3
_CYCLE_RE = re.compile(r"poll cycle max ([0-9.]+) s \((m\d+)\)")


def gateway_poll_cycle_max(log: str) -> dict:
    """Longest start-to-start time between two successful polls of one meter,
    over the whole gateway run (from the gateway's "poll cycle max" log lines)."""
    hits = _CYCLE_RE.findall(log)
    if not hits:
        return {"max_s": None, "meter_id": None}
    s, m = hits[-1]
    return {"max_s": float(s), "meter_id": m}


_PLAY_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) INFO run_meter_server: "
                      r"(starting playback|playback finished)", re.M)


def playback_span_s(log: str) -> float | None:
    """Wall seconds from the meter server's "starting playback" to its
    "playback finished" log line (None if either is missing)."""
    t = {what: datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f") for ts, what in _PLAY_RE.findall(log)}
    if len(t) < 2:
        return None
    return (t["playback finished"] - t["starting playback"]).total_seconds()


def loss_accounting(val: list[dict], step_times: list[str]) -> dict:
    """Per meter: intervals lost (expected steps never delivered) and intervals
    flagged (sum of ``gap_steps`` on its GAP-flagged messages).  A loss is
    silent when lost != flagged for that meter."""
    expected = [datetime.fromisoformat(t) for t in step_times]
    exp_set = set(expected)
    got, flagged = defaultdict(set), defaultdict(int)
    for v in val:
        t = datetime.fromisoformat(v["t_sim"]).replace(tzinfo=None)
        got[v["meter_id"]].add(t)
        if "GAP" in v["quality"]["flags"]:
            flagged[v["meter_id"]] += v["quality"]["gap_steps"]
    per_meter, lost_at = {}, []
    for m in sorted(x.meter_id for x in load_meters().meters):
        missing = sorted(exp_set - got[m])
        per_meter[m] = {"lost": len(missing), "flagged": flagged[m]}
        lost_at += [(m, t.strftime("%H:%M")) for t in missing]
    silent = {m: d for m, d in per_meter.items() if d["lost"] != d["flagged"]}
    by_step = defaultdict(list)
    for m, t in lost_at:
        by_step[t].append(m)
    return {"per_meter": per_meter, "silent": silent, "no_silent_loss": not silent,
            "lost_total": len(lost_at), "flagged_total": sum(flagged.values()),
            "lost_by_step": {t: sorted(ms) for t, ms in sorted(by_step.items())}}


def build_bench_bursts(frame_steps: list, trigger_idx=(24, 48, 72), n: int = 3) -> tuple[list[dict], list[dict]]:
    """The CP4 event set's ``n`` longest bursts (each with its own pre-fault
    state from the CP3 cache), re-triggered at the given steps of the
    benchmark day (stated: burst data unchanged, trigger time moved)."""
    import pandas as pd
    from src.data_source import faults as F
    from src.data_source.feeder_model import build_network
    net = build_network("B")
    relay = F.relay_settings(F.max_head_current_a())
    ev = F.make_event_set(net)
    fl = ev[ev.kind == "fault"]
    cache = F.steps_from_cache(net, fl.step)
    scored = []
    for e in fl.itertuples():
        F.set_prefault(net, *cache[pd.Timestamp(e.step)])
        sn = F.build_sequence_network(net)
        res = (F.simulate_arc_fault(net, sn, e.bus, e.ftype, e.phases) if e.rf_model == "arc"
               else F.simulate_fault(net, sn, e.bus, e.ftype, e.phases, float(e.rf_ohm)))
        t_rel, _ = F.relay_operate(relay, float(res.i_head_fault_a.max()), res.i_head_residual_a)
        scored.append((sum(F.burst_lengths(t_rel + relay.breaker_s)), e.event_id))
    chosen = [eid for _, eid in sorted(scored, key=lambda x: (-x[0], x[1]))[:n]]
    bursts, info = [], []
    for eid, k in zip(chosen, trigger_idx):
        e = fl[fl.event_id == eid].iloc[0]
        F.set_prefault(net, *cache[pd.Timestamp(e.step)])
        sn = F.build_sequence_network(net)
        res = (F.simulate_arc_fault(net, sn, e.bus, e.ftype, e.phases) if e.rf_model == "arc"
               else F.simulate_fault(net, sn, e.bus, e.ftype, e.phases, float(e.rf_ohm)))
        t_rel, el = F.relay_operate(relay, float(res.i_head_fault_a.max()), res.i_head_residual_a)
        t_clear = t_rel + relay.breaker_s
        trig = pd.Timestamp(frame_steps[k])
        for lid, m in F.meter_faults(net, sn, res).items():
            v, i = F.build_burst(m, t_clear)
            bursts.append({"trigger": trig, "load_id": lid, "event_id": int(eid), "t_start": trig,
                           "v": v, "i": i, "v_base_ln_v": m.v_base_ln_v, "phases": "abc"})
        info.append({"event_id": int(eid), "trigger_step": k, "bus": e.bus, "ftype": e.ftype,
                     "n_samples": int(sum(F.burst_lengths(t_clear))), "t_clear_s": t_clear})
    return bursts, info


def drain_metrics(speed: float, steps: int, val: list[dict], events: list[dict],
                  burst_info: list[dict]) -> dict:
    """Late / missed 15-min intervals per meter during each event drain."""
    import pandas as pd
    dt = STEP_S / speed
    t_sims = sorted({v["t_sim"] for v in val})
    k_of = {t: i for i, t in enumerate(t_sims)}
    poll = defaultdict(dict)
    for v in val:
        poll[v["meter_id"]][k_of[v["t_sim"]]] = (
            datetime.fromisoformat(v["t_poll_utc"].replace("Z", "+00:00")).timestamp(),
            "GAP" in v["quality"]["flags"])
    # Wall time at which step k was written: the meter server sleeps a fixed time
    # after writing each step, so the real period is a little longer than dt.
    # Fit w_k = a + b k to the earliest poll of each step (all meters), leaving
    # out the steps near the burst triggers, where every poller is draining.
    trig = {b["event_id"]: b["trigger_step"] for b in burst_info}
    near = {k for k0 in trig.values() for k in range(k0, k0 + 4)}
    first = defaultdict(lambda: math.inf)
    for m in poll.values():
        for k, (p, _) in m.items():
            first[k] = min(first[k], p)
    ks = np.array([k for k in sorted(first) if k not in near], float)
    ps = np.array([first[int(k)] for k in ks])
    b_fit, a_fit = np.polyfit(ks, ps, 1) if len(ks) > 1 else (dt, ps[0] if len(ps) else 0.0)

    def w(k):
        return a_fit + b_fit * k
    per_meter = defaultdict(lambda: {"late": 0, "missed": 0, "gap_flags": 0, "intervals": 0})
    drains = []
    for e in events:
        k_e = trig.get(e["event_id"])
        if k_e is None:
            continue
        start = w(k_e)
        end = datetime.fromisoformat(e["t_poll_utc"].replace("Z", "+00:00")).timestamp()
        drains.append(end - start)
        m = per_meter[e["meter_id"]]
        k_hi = int(math.floor((end - a_fit) / b_fit)) + 1
        for k in range(k_e + 1, min(k_hi, steps - 1) + 1):
            m["intervals"] += 1
            got = poll[e["meter_id"]].get(k)
            if got is None:
                m["missed"] += 1
            else:
                m["late"] += int(got[0] - w(k) > b_fit)
                m["gap_flags"] += int(got[1])
    tot = {k: sum(m[k] for m in per_meter.values()) for k in ("intervals", "late", "missed", "gap_flags")}
    # When most steps are read late or missed the earliest poll per step lags
    # the write, the fit's write times run late, and drains come out negative:
    # the window counts are then not meaningful.  A single slightly negative
    # drain is poll jitter around the fit (a burst read quickly), so the test
    # is on the median.
    valid = bool(not drains or float(np.median(drains)) >= 0.0)
    return {"valid": valid,
            "note": None if valid else "fitted write times later than burst publication "
                                       "(most steps missed); drain counts not meaningful",
            "step_wall_s": dt, "fitted_step_wall_s": float(b_fit), "drain_s": {"n": len(drains),
                                           "median": float(pd.Series(drains).median()) if drains else None,
                                           "max": max(drains) if drains else None},
            "totals": tot, "per_meter": dict(per_meter)}


def run_speed(speed: float, frame_csv: Path, step_times: list[str], poll_interval: float,
              events_path: Path | None = None, burst_info: list | None = None) -> tuple[dict, list[dict]]:
    steps = len(step_times)
    st = Stack()
    rec = None
    try:
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        rec.connected.wait(10)
        st.start_listener(persistent=True, client_id=f"bench-listener-{int(speed)}")
        # The head-end knows when the series starts (commissioning time).
        st.start_gateway(poll_interval=poll_interval, playback_speed=speed,
                         series_start=step_times[0] + "+05:30")
        st.start_meter(speed, frame_in=frame_csv, start_delay=2.0, events=events_path)
        t_start = time.monotonic()
        play_s = steps * STEP_S / speed
        # Wait for playback to finish, then for the stream to go quiet.
        time.sleep(2.0 + play_s)

        def quiet():
            n = len(rec.snapshot(contains="/validated/"))
            time.sleep(1.0)
            return n == len(rec.snapshot(contains="/validated/"))

        wait_for(quiet, 60, interval=0.0, what="stream quiet")
        wall = time.monotonic() - t_start
        stats = st.http("/api/v1/stats")
        stop(st.gateway)
        cycle = gateway_poll_cycle_max(st.log_text("gateway"))
        play = playback_span_s(st.log_text("meter"))
        tel = [m.json() for m in rec.snapshot(suffix="/telemetry")]
        val_rec = rec.snapshot(contains="/validated/")
        val = [m.json() for m in val_rec]
        raw_events = [m.json() for m in rec.snapshot(suffix="/event") if "/validated" not in m.topic]
    finally:
        if rec:
            rec.close()
        st.close()

    rows = []
    hops = defaultdict(list)
    for r, v in zip(val_rec, val):
        lat = v["ingest"]["latency_ms"]
        for h, x in lat.items():
            hops[h].append(x)
        t_recv = datetime.fromisoformat(v["ingest"]["t_recv_utc"].replace("Z", "+00:00"))
        to_consumer = (r.t_recv - t_recv.timestamp()) * 1000.0
        hops["recv_to_validated_subscriber"].append(to_consumer)
        rows.append({
            "speed": speed, "meter_id": v["meter_id"], "t_sim": v["t_sim"], "seq": v["seq"],
            "ingest_seq": v["ingest"]["ingest_seq"], "flags": "|".join(v["quality"]["flags"]),
            "gap_steps": v["quality"]["gap_steps"], "poll_to_pub_ms": lat["poll_to_pub"],
            "pub_to_recv_ms": lat["pub_to_recv"], "total_ms": lat["total"],
            "recv_to_validated_subscriber_ms": round(to_consumer, 3),
        })

    expected = steps * N_METERS
    unique = {(v["meter_id"], v["t_sim"]) for v in val}
    val_times = [m.t_recv for m in val_rec]
    span = (max(val_times) - min(val_times)) if len(val_times) > 1 else 0.0
    gap_msgs = [v for v in val if "GAP" in v["quality"]["flags"]]
    step_wall = STEP_S / speed
    result = {
        "playback_speed": speed,
        "wall_s_per_step": step_wall,
        "poll_interval_s": poll_interval,
        "gateway_poll_cycle_max": {**cycle, "headroom_s": (round(step_wall - cycle["max_s"], 3)
                                                           if cycle["max_s"] is not None else None)},
        "loss_accounting": loss_accounting(val, step_times),
        "wall_duration_s": round(wall, 2),
        "expected_messages": expected,
        "gateway_published": len(tel),
        "listener_received": stats["received"],
        "accepted": stats["accepted"],
        "validated_delivered": len(val),
        "unique_meter_steps": len(unique),
        "loss_steps": expected - len(unique),
        "loss_fraction": round((expected - len(unique)) / expected, 4),
        "duplicates_rejected": stats["rejects"]["DUPLICATE"],
        "duplicates_in_validated": len(val) - len(unique),
        "gap_messages": len(gap_msgs),
        "gap_steps_total": sum(v["quality"]["gap_steps"] for v in gap_msgs),
        "rejects": stats["rejects"],
        "flags": stats["flags"],
        "throughput_msgs_per_s": round(len(val) / span, 2) if span else None,
        "latency_ms": {h: percentiles(v) for h, v in hops.items()},
        "bursts": {"expected": N_METERS * len(burst_info or []), "received": len(raw_events),
                   "samples": sum(e["n_samples"] for e in raw_events)},
        "event_drain": drain_metrics(speed, steps, val, raw_events, burst_info or []) if burst_info else None,
    }
    # Requested vs achieved speed: the actual step is the meter server's own
    # playback span / steps (its log), longer than 900/speed because the
    # server sleeps a fixed time after each write.
    if play is not None:
        actual = play / steps
        mx = cycle["max_s"]
        result["speed_achieved"] = {
            "requested_x": speed, "nominal_step_s": step_wall, "actual_step_s": round(actual, 4),
            "achieved_x": round(STEP_S / actual, 1),
            "poll_cycle_max_s": mx,
            "headroom_vs_actual_s": round(actual - mx, 4) if mx is not None else None,
            "headroom_vs_actual_pct": round(100 * (actual - mx) / actual, 1) if mx is not None else None,
        }
    return result, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speeds", type=float, nargs="+", default=[900, 3600, 7200, 14400])
    ap.add_argument("--asserted-runs", type=int, default=NO_LOSS_RUNS,
                    help="repeated runs at each asserted no-loss speed (900x)")
    ap.add_argument("--host-note", default="", help="free text stored in the metadata "
                    "(e.g. that OneDrive sync was paused)")
    ap.add_argument("--poll-interval", type=float, default=0.25,
                    help="gateway poll_interval_s (default = config default 0.25)")
    ap.add_argument("--profiles", choices=["synthetic", "sgcc"], default="synthetic")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--no-bursts", action="store_true", help="skip the CP4 burst injection")
    args = ap.parse_args(argv)

    if find_mosquitto() is None:
        print("mosquitto not found; set SG_MOSQUITTO_BIN", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    from run_meter_server import build_frame
    t0 = time.monotonic()
    day = "2024-01-01"
    frame = build_frame(f"{day} 00:00", f"{day} 23:45", args.profiles)
    solve_s = time.monotonic() - t0
    frame_csv = out / f"bench_{stamp}_frame.csv"
    frame.to_csv(frame_csv, index=False)
    import pandas as pd
    step_times = [pd.Timestamp(t).tz_localize(None).isoformat() if pd.Timestamp(t).tzinfo
                  else pd.Timestamp(t).isoformat() for t in sorted(frame["timestamp"].unique())]
    steps = len(step_times)
    events_path, burst_info = None, []
    if not args.no_bursts:
        from src.data_source import meter_server as ms
        bursts, burst_info = build_bench_bursts(sorted(frame["timestamp"].unique()))
        events_path = out / f"bench_{stamp}_bursts.npz"
        ms.save_event_file(events_path, bursts)
        print(f"bursts: {burst_info}", flush=True)

    plan = [(s, r) for s in args.speeds
            for r in range(args.asserted_runs if s in ASSERT_NO_LOSS_SPEEDS else 1)]
    results, all_rows = [], []
    for speed, rep in plan:
        print(f"speed {speed:g}x run {rep + 1} ({STEP_S / speed:.3f} s/step) ...", flush=True)
        res, rows = run_speed(speed, frame_csv, step_times, args.poll_interval, events_path, burst_info)
        res["run"] = rep + 1
        results.append(res)
        all_rows.extend({**r, "run": rep + 1} for r in rows)
        lt = res["latency_ms"]["total"]
        dr = (res.get("event_drain") or {}).get("totals")
        la, cy = res["loss_accounting"], res["gateway_poll_cycle_max"]
        print(f"  delivered {res['unique_meter_steps']}/{res['expected_messages']} "
              f"loss={res['loss_steps']} flagged={la['flagged_total']} silent={la['silent']} "
              f"gaps={res['gap_messages']} "
              f"total p50/p95/p99 = {lt['p50']}/{lt['p95']}/{lt['p99']} ms; "
              f"bursts {res['bursts']['received']}/{res['bursts']['expected']}; drain {dr}; "
              f"poll cycle max {cy['max_s']} s ({cy['meter_id']}); achieved {res.get('speed_achieved')}; "
              f"lost at {la['lost_by_step']}", flush=True)

    from src.ingestion.config import ListenerSettings
    doc = {
        "created_utc": stamp,
        "metadata": {
            "python": sys.version.split()[0],
            "os": platform.platform(),
            "machine": platform.machine(),
            "versions": versions(),
            "profiles": args.profiles,
            "simulated_window": f"{day} 00:00 .. 23:45 (one day)",
            "steps": int(steps),
            "meters": N_METERS,
            "run_timeseries_solve_s": round(solve_s, 2),
            "gateway_poll_interval_s": args.poll_interval,
            "speeds": args.speeds,
            "LATENCY_BUDGET_MS": ListenerSettings(mqtt_enabled=False).LATENCY_BUDGET_MS,
            "clock_note": "gateway and listener on the same host; wall clocks comparable",
        },
        "results": results,
    }
    zero = [s for s in {r["playback_speed"] for r in results}
            if all(r["loss_steps"] == 0 and r["bursts"]["received"] == r["bursts"]["expected"]
                   for r in results if r["playback_speed"] == s)]
    doc["characterisation"] = {
        "highest_zero_loss_speed": max(zero) if zero else None,
        "note": "characterisation, not pass/fail (brief v2 X.3.1 item 13)",
        "bursts": burst_info,
    }
    asserted = {f"{s:g}x_run{r['run']}_no_interval_lost":
                r["loss_steps"] == 0 and r["bursts"]["received"] == r["bursts"]["expected"]
                for r in results for s in [r["playback_speed"]] if s in ASSERT_NO_LOSS_SPEEDS}
    asserted.update({f"{r['playback_speed']:g}x_run{r['run']}_no_silent_loss":
                     r["loss_accounting"]["no_silent_loss"] for r in results})
    doc["asserted"] = asserted
    doc["metadata"]["host_note"] = args.host_note
    jpath = out / f"bench_{stamp}.json"
    jpath.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    cpath = out / f"bench_{stamp}.csv"
    with cpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()) if all_rows else ["speed"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {jpath}\nwrote {cpath}")
    print(f"highest zero-loss speed: {doc['characterisation']['highest_zero_loss_speed']}; asserted {asserted}")
    if not all(asserted.values()):
        print(f"BENCHMARK: assertion failed: {[k for k, v in asserted.items() if not v]}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
