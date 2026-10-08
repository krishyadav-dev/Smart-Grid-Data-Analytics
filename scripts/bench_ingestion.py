"""End-to-end ingestion benchmark (D6).

For each playback speed, runs the whole chain for one simulated day — real
meter server fed by real ``run_timeseries`` output, gateway, Mosquitto,
listener — and measures per-hop latency, throughput, loss, duplicates and
gaps.  No latency budget is asserted (``LATENCY_BUDGET_MS`` stays ``None``).

Writes ``results/bench_<timestamp>.json`` and ``results/bench_<timestamp>.csv``
(one row per validated message).

PowerShell::

    python scripts/bench_ingestion.py --speeds 900 1800 3600 7200
    python scripts/bench_ingestion.py --profiles sgcc     # SGCC-driven load profiles
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
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

from devstack import Recorder, Stack, find_mosquitto, stop, wait_for  # noqa: E402

STEP_S = 900.0
N_METERS = 8


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


def run_speed(speed: float, frame_csv: Path, steps: int, poll_interval: float) -> tuple[dict, list[dict]]:
    st = Stack()
    rec = None
    try:
        st.start_broker()
        rec = Recorder(st.mqtt_port)
        rec.connected.wait(10)
        st.start_listener(persistent=True, client_id=f"bench-listener-{int(speed)}")
        st.start_gateway(poll_interval=poll_interval, playback_speed=speed)
        st.start_meter(speed, frame_in=frame_csv, start_delay=2.0)
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
        tel = [m.json() for m in rec.snapshot(suffix="/telemetry")]
        val_rec = rec.snapshot(contains="/validated/")
        val = [m.json() for m in val_rec]
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
    result = {
        "playback_speed": speed,
        "wall_s_per_step": STEP_S / speed,
        "poll_interval_s": poll_interval,
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
    }
    return result, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speeds", type=float, nargs="+", default=[900, 1800, 3600, 7200])
    ap.add_argument("--poll-interval", type=float, default=0.25,
                    help="gateway poll_interval_s (default = config default 0.25)")
    ap.add_argument("--profiles", choices=["synthetic", "sgcc"], default="synthetic")
    ap.add_argument("--out", default=str(ROOT / "results"))
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
    steps = frame["timestamp"].nunique()

    results, all_rows = [], []
    for speed in args.speeds:
        print(f"speed {speed:g}x ({STEP_S / speed:.3f} s/step) ...", flush=True)
        res, rows = run_speed(speed, frame_csv, steps, args.poll_interval)
        results.append(res)
        all_rows.extend(rows)
        lt = res["latency_ms"]["total"]
        print(f"  delivered {res['unique_meter_steps']}/{res['expected_messages']} "
              f"loss={res['loss_steps']} gaps={res['gap_messages']} "
              f"total p50/p95/p99 = {lt['p50']}/{lt['p95']}/{lt['p99']} ms", flush=True)

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
    jpath = out / f"bench_{stamp}.json"
    jpath.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    cpath = out / f"bench_{stamp}.csv"
    with cpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()) if all_rows else ["speed"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {jpath}\nwrote {cpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
