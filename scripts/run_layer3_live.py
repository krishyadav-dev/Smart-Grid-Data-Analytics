"""Layer 3 Step F: live replay of the fault engine (run sheet Step F, F3).

Real processes (``scripts/devstack.py``): Mosquitto, the Layer 1 meter server
(CP3 metered frame of the week + the event records of the week's events, quiet
windows included), the Layer 2 gateway (``config/gateway.yaml`` operating
poll interval 0.25 s) and listener.  The engine subscribes in-process to
``validated_event/{meter}`` and ``validated/{meter}`` of the 14 physical
meters and runs :class:`fault_engine.LiveEngine` on every burst as it arrives.

Scope (fixed in the run sheet):
- 900x: the 4 ISO weeks with the most test-half faults (ties: earliest), plus
  2 further weeks drawn with ``numpy.random.default_rng(20261025)`` from the
  remaining weeks (sorted by start date).  Every event of those weeks is
  replayed (tuning and test half alike).
- 1x: a 2-hour window (8 steps) around the first test-half fault of the first
  of those weeks.

For every replayed event the live and batch outputs (detection, type, zone,
residual, duration) must be identical (rule 5): any difference exits 1.
Reported: processing latency (``t_poll_utc`` -> alarm) p50 / p95 / p99, GAP
intervals, live false alarms (engine events matching no fault, flat-baseline
alerts on bursts and 15-min samples, weekly unbalance alarms).

PowerShell::

    python scripts/run_layer3_live.py plan          # print the weeks, write live/plan.json
    python scripts/run_layer3_live.py run           # 6 weeks at 900x (~75 min) + 1x (2 h)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "results" / "layer3" / "fault"
LIVE = OUT / "live"
RECORDS = ROOT / "results" / "faults_2024" / "event_records.npz"
METERED = ROOT / "results" / "year_2024" / "meter_frame_metered.parquet"
SEED = 20261025
N_TOP, N_DRAW = 4, 2
FEEDER = "ieee13_11kv"
COMPARE = ("ftype", "zone", "zone_node", "residual_pct", "duration_ms")


def iso_week_start(ms: int) -> pd.Timestamp:
    t = pd.Timestamp(ms, unit="ms", tz="Asia/Kolkata").tz_localize(None)
    return t.normalize() - pd.Timedelta(days=t.weekday())


def plan_weeks() -> dict:
    from src.analytics import evaluate as V
    split, labels = V.load_split(), V.load_labels()
    lab = labels.assign(week=[iso_week_start(t) for t in labels.t_event_ms])
    weeks = pd.date_range("2024-01-01", "2024-12-30", freq="7D")        # every Monday of 2024
    half = {**{e: "test" for e in split["faults_test"]}, **{e: "tune" for e in split["faults_tune"]}}
    lab["half"] = [half.get(e, "quiet") for e in lab.index]
    cnt = pd.DataFrame({"test": lab[lab.half == "test"].groupby("week").size(),
                        "tune": lab[lab.half == "tune"].groupby("week").size(),
                        "quiet": lab[lab.kind == "quiet"].groupby("week").size()}).reindex(weeks).fillna(0).astype(int)
    faults_per_week = cnt.test + cnt.tune
    top = sorted(weeks, key=lambda w: (-cnt.test[w], w))[:N_TOP]
    rest = sorted(w for w in weeks if w not in top)
    rng = np.random.default_rng(SEED)
    drawn = [rest[int(k)] for k in rng.choice(len(rest), N_DRAW, replace=False)]
    chosen = top + drawn
    first = chosen[0]
    tf = lab[(lab.half == "test") & (lab.week == first)].sort_values("t_event_ms")
    one_x = int(tf.index[0])
    t_fault = pd.Timestamp(int(tf.t_event_ms.iloc[0]), unit="ms", tz="Asia/Kolkata").tz_localize(None)
    step = t_fault.floor("15min")
    return {"weeks": [{"start": str(w.date()), "end": str((w + pd.Timedelta(days=6)).date()),
                       "chosen_by": "top test-fault count" if w in top else f"rng({SEED}) draw",
                       "test_faults": int(cnt.test[w]), "tuning_faults": int(cnt.tune[w]),
                       "quiet_windows": int(cnt.quiet[w])} for w in chosen],
            "weeks_without_any_fault": int((faults_per_week == 0).sum()),
            "one_x": {"event_id": one_x, "fault_time": str(t_fault),
                      "start": str(step - pd.Timedelta(hours=1)), "end": str(step + pd.Timedelta(minutes=45)),
                      "steps": 8},
            "seed": SEED, "rule": "4 ISO weeks with the most test-half faults (ties earliest) + 2 drawn "
                                  "from the remaining weeks with default_rng(seed).choice(len, 2, replace=False)"}


class EngineSubscriber:
    """In-process engine on the validated topics of the physical meters."""

    def __init__(self, port: int, params):
        import paho.mqtt.client as mqtt

        from src.analytics import fault_engine as E
        from src.analytics import meters as M
        self.E = E
        self.engine = E.LiveEngine(params)
        self.phys = [m.meter_id for m in M.physical_meters()]
        self.tel: list[dict] = []
        self.flat_burst: list[int] = []
        self.errors: list[str] = []
        self.n_bursts = 0
        self.burst_latency_ms: list[float] = []      # t_poll_utc -> burst analysed (every burst)
        self._lock = threading.Lock()
        self.connected = threading.Event()
        self._c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"layer3-engine-{time.time_ns()}",
                              clean_session=True)
        self._c.on_connect = self._on_connect
        self._c.on_message = self._on_message
        self._c.connect_async("127.0.0.1", port, keepalive=10)
        self._c.loop_start()

    def _on_connect(self, client, userdata, flags, rc, props):
        for m in self.phys:
            client.subscribe(f"sg/v2/{FEEDER}/validated_event/{m}", qos=1)
            client.subscribe(f"sg/v2/{FEEDER}/validated/{m}", qos=1)
        self.connected.set()

    def _on_message(self, client, userdata, msg):
        from src.offline_pipeline import flatten_doc
        try:
            doc = json.loads(msg.payload)
            with self._lock:
                if "/validated_event/" in msg.topic:
                    self.n_bursts += 1
                    self.engine.on_burst(doc)
                    self.burst_latency_ms.append(
                        (pd.Timestamp.now(tz="UTC") - pd.Timestamp(doc["t_poll_utc"])).total_seconds() * 1e3)
                    self.flat_burst += self.E.flat_alerts_burst(self.E.burst_from_doc(doc))
                else:
                    self.tel.append(flatten_doc(doc))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def close(self):
        self._c.loop_stop()
        self._c.disconnect()


def replay(start: str, end: str, speed: float, params, events_path: Path, n_bursts: int) -> dict:
    from devstack import Stack, stop, wait_for
    steps = len(pd.date_range(start, end, freq="15min"))
    st = Stack()
    sub = None
    try:
        st.start_broker()
        sub = EngineSubscriber(st.mqtt_port, params)
        assert sub.connected.wait(10)
        st.start_listener(persistent=True, client_id=f"layer3-listener-{int(speed)}")
        st.start_gateway(poll_interval=0.25, playback_speed=speed,
                         series_start=pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%S") + "+05:30")
        st.start_meter(speed, start_delay=3.0, events=events_path, cache_frame=METERED, start=start, end=end)
        t0 = time.monotonic()
        expected = steps * len(sub.phys)

        def done():
            with sub._lock:
                return len({(r["meter_id"], r["t_sim"]) for r in sub.tel}) >= expected and sub.n_bursts >= n_bursts
        timed_out = False
        try:
            wait_for(done, timeout=steps * 900 / speed + 300, interval=1.0, what="the replay")
        except TimeoutError:
            timed_out = True
        time.sleep(3.0)
        wall = time.monotonic() - t0
        stop(st.gateway)
        gw_log = st.log_text("gateway")
    finally:
        if sub:
            sub.close()
        st.close()
    tel = pd.DataFrame(sub.tel)
    return {"sub": sub, "tel": tel, "wall_s": wall, "timed_out": timed_out, "steps": steps,
            "expected_telemetry": expected,
            "events_missed_in_gateway_log": gw_log.count("missed")}


def _pct(x: np.ndarray) -> dict:
    return {"n": int(len(x)), **({"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                                  "p99": float(np.percentile(x, 99)), "max": float(x.max())} if len(x) else {})}


def compare(live_ev: pd.DataFrame, batch_ev: pd.DataFrame) -> dict:
    """Rule 5: every replayed event identical in live and batch."""
    key = ["onset_ms"]
    m = live_ev.merge(batch_ev, on=key, how="outer", suffixes=("_live", "_batch"), indicator=True)
    diffs = []
    for _, r in m.iterrows():
        if r["_merge"] != "both":
            diffs.append({"onset_ms": int(r.onset_ms), "only_in": r["_merge"]})
            continue
        for c in COMPARE + ("meters",):
            a, b = r[f"{c}_live"], r[f"{c}_batch"]
            same = (pd.isna(a) and pd.isna(b)) or a == b
            if not same:
                diffs.append({"onset_ms": int(r.onset_ms), "field": c, "live": a, "batch": b})
    return {"events_live": int(len(live_ev)), "events_batch": int(len(batch_ev)),
            "differences": diffs, "identical": not diffs}


def run_one(name: str, start: str, end: str, speed: float, params, labels, docs_all, records) -> dict:
    from src.analytics import evaluate as V
    from src.analytics import fault_engine as E
    from src.data_source import meter_server as ms
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    week_records = [r for r in records if t0 <= pd.Timestamp(r["trigger"]) <= t1]
    ids = sorted({r["event_id"] for r in week_records})
    path = LIVE / f"records_{name}.npz"
    ms.save_event_file(path, week_records)
    n_bursts = len(ids) * 14
    r = replay(start, end, speed, params, path, n_bursts)
    sub = r["sub"]
    live_events = sub.engine.events()
    live_tab = V.events_table(live_events, labels)
    batch_events, _ = E.run_batch(V.select_docs(docs_all, ids), params)
    batch_tab = V.events_table(batch_events, labels)
    cmp_ = compare(live_tab, batch_tab)
    lat = np.array([a["latency_ms"] for a in sub.engine.alarms if a["latency_ms"] is not None], float)
    tel = r["tel"]
    gaps = int(tel["flags"].str.contains("GAP").sum()) if len(tel) else 0
    stream = E.stream_frame(tel) if len(tel) else tel
    weekly = E.weekly_unbalance(stream) if len(tel) else pd.DataFrame()
    flat_stream = E.flat_alerts_stream(stream) if len(tel) else pd.DataFrame()
    fault_ids = [e for e in ids if labels.kind[e] == "fault"]
    quiet_ids = [e for e in ids if labels.kind[e] == "quiet"]
    unmatched = live_tab[~live_tab.matched.isin(fault_ids)]
    flat_m = V.match_onsets(sub.flat_burst, labels)
    out = {"name": name, "start": start, "end": end, "speed": speed, "wall_s": round(r["wall_s"], 1),
           "timed_out": r["timed_out"], "events_replayed": len(ids), "faults": len(fault_ids),
           "quiet_windows": len(quiet_ids), "bursts_expected": n_bursts, "bursts_received": sub.n_bursts,
           "telemetry_expected": r["expected_telemetry"],
           "telemetry_received_unique": int(len({(a, b) for a, b in zip(tel.meter_id, tel.t_sim)})) if len(tel) else 0,
           "gap_flagged_messages": gaps,
           "gap_steps_total": int(tel.gap_steps.sum()) if len(tel) else 0,
           "gateway_events_missed_log_lines": r["events_missed_in_gateway_log"],
           "subscriber_errors": sub.errors[:10],
           "alarms": len(sub.engine.alarms),
           "latency_ms": {"n": int(len(lat)), **({"p50": float(np.percentile(lat, 50)),
                                                   "p95": float(np.percentile(lat, 95)),
                                                   "p99": float(np.percentile(lat, 99)),
                                                   "max": float(lat.max())} if len(lat) else {})},
           "alarm_latencies_ms": [float(x) for x in lat],
           "burst_processing_latency_ms": _pct(np.array(sub.burst_latency_ms, float)),
           "burst_latencies_ms": [float(x) for x in sub.burst_latency_ms],
           "live_batch": cmp_,
           "live_false_alarms": {"engine_events_matching_no_fault": int(len(unmatched)),
                                 "engine_events_in_quiet_windows": int(live_tab.matched.isin(quiet_ids).sum()),
                                 "flat_burst_alerts_matching_no_fault": int(sum(1 for x in flat_m if x not in set(fault_ids))),
                                 "flat_stream_alerts": int(len(flat_stream)),
                                 "unbalance_weekly_alarms": int(weekly.alarm.sum()) if len(weekly) else 0},
           "detected_faults_live": int(live_tab.matched.isin(fault_ids).groupby(live_tab.matched).any().sum())
           if len(live_tab) else 0}
    (LIVE / f"{name}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    live_tab.to_parquet(LIVE / f"{name}_events.parquet", index=False)
    return out


def pooled_stats(results: list[dict]) -> dict:
    pooled = {}
    for spd in sorted({r["speed"] for r in results}):
        rs = [r for r in results if r["speed"] == spd]
        pooled[f"{spd:g}x"] = {
            "runs": [r["name"] for r in rs],
            "alarm_latency_ms": _pct(np.array([x for r in rs for x in r["alarm_latencies_ms"]], float)),
            "burst_processing_latency_ms": (
                _pct(np.array([x for r in rs for x in r["burst_latencies_ms"]], float))
                if all("burst_latencies_ms" in r for r in rs) else
                {"note": "raw per-burst latencies were not stored by these runs; per-run percentiles below",
                 "per_run": {r["name"]: r["burst_processing_latency_ms"] for r in rs},
                 "max_p99": max(r["burst_processing_latency_ms"]["p99"] for r in rs)}),
            "events": sum(r["events_replayed"] for r in rs), "faults": sum(r["faults"] for r in rs),
            "quiet_windows": sum(r["quiet_windows"] for r in rs),
            "alarms": sum(r["alarms"] for r in rs), "bursts": sum(r["bursts_received"] for r in rs),
            "bursts_expected": sum(r["bursts_expected"] for r in rs),
            "telemetry": sum(r["telemetry_received_unique"] for r in rs),
            "telemetry_expected": sum(r["telemetry_expected"] for r in rs),
            "gap_flagged_messages": sum(r["gap_flagged_messages"] for r in rs),
            "live_false_alarms": {k: sum(r["live_false_alarms"][k] for r in rs)
                                  for k in rs[0]["live_false_alarms"]},
            "live_batch_identical": all(r["live_batch"]["identical"] for r in rs),
            "wall_s": sum(r["wall_s"] for r in rs)}
    return pooled


def build_summary(plan: dict) -> int:
    """Live summary from the six week files and the 1x file (Step F after the
    27 Oct interruption: the weeks are not rerun)."""
    names = [f"week_{w['start']}" for w in plan["weeks"]] + ["one_x"]
    missing = [n for n in names if not (LIVE / f"{n}.json").exists()]
    if missing:
        print(f"missing run files: {missing}", file=sys.stderr)
        return 1
    results = [json.loads((LIVE / f"{n}.json").read_text(encoding="utf-8")) for n in names]
    ok = all(r["live_batch"]["identical"] and not r["timed_out"] and not r["subscriber_errors"] for r in results)
    out = {"plan": plan, "pooled": pooled_stats(results), "runs": results, "passed": ok,
           "note": "the six 900x weeks ran on 27 Oct 07:20-08:31; the machine shut down during the 1x window, "
                   "which was rerun alone, shortened to three steps (user instruction)"}
    (LIVE / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"passed": ok, "pooled": out["pooled"]}, indent=2, default=str))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "check", "one_x", "summary"])
    ap.add_argument("--start", help="check: window start (tuning-only mechanics check)")
    ap.add_argument("--end")
    ap.add_argument("--speed", type=float, default=900.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    warnings.simplefilter("ignore", FutureWarning)
    LIVE.mkdir(parents=True, exist_ok=True)
    plan = plan_weeks()
    if args.cmd == "summary":
        return build_summary(plan)
    if args.cmd == "plan":
        (LIVE / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return 0
    from src.analytics import evaluate as V
    from src.analytics import fault_engine as E
    from src.data_source import meter_server as ms
    params = E.load_params()
    labels = V.load_labels()
    docs_all = V.read_docs(OUT / "batch" / "bursts.jsonl.gz")
    z = np.load(RECORDS, allow_pickle=False)
    idx = json.loads(str(z["index"]))
    records = [{**e, "v": z[f"v{k}"], "i": z[f"i{k}"]} for k, e in enumerate(idx)]
    del ms
    results = []
    if args.cmd == "check":
        results.append(run_one("check", args.start, args.end, args.speed, params, labels, docs_all, records))
    elif args.cmd == "one_x":
        o = plan["one_x"]
        results.append(run_one("one_x", args.start or o["start"], args.end or o["end"], 1.0,
                               params, labels, docs_all, records))
    else:
        for w in plan["weeks"]:
            s = pd.Timestamp(w["start"])
            results.append(run_one(f"week_{w['start']}", str(s), str(s + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)),
                                   900.0, params, labels, docs_all, records))
            print(json.dumps({k: results[-1][k] for k in ("name", "wall_s", "bursts_received", "latency_ms")},
                             default=str))
            if not results[-1]["live_batch"]["identical"]:
                print("LIVE-BATCH DIFFERENCE: stop (rule 5)", file=sys.stderr)
                break
        else:
            o = plan["one_x"]
            results.append(run_one("one_x", o["start"], o["end"], 1.0, params, labels, docs_all, records))
    ok = all(r["live_batch"]["identical"] and not r["timed_out"] and not r["subscriber_errors"] for r in results)
    (LIVE / f"summary_{args.cmd}.json").write_text(json.dumps({"plan": plan, "pooled": pooled_stats(results),
                                                               "runs": results, "passed": ok},
                                                              indent=2, default=str), encoding="utf-8")
    print(json.dumps({"passed": ok, "runs": [{k: r[k] for k in ("name", "latency_ms", "live_batch")}
                                             for r in results]}, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
