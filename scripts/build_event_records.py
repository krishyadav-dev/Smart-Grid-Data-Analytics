"""Layer 3 Stage 1: the event records of all 400 CP4 events (Layer 1 data).

Every fault and every quiet window of the CP4 event set
(:func:`faults.make_event_set`, same seed) is rebuilt with its own pre-fault
state from the CP3 cache (:func:`faults.steps_from_cache`: DT loads and OLTC
tap of its step, as the CP5 week and the live burst test do), the CP4 relay
(:func:`faults.relay_settings` from the year's highest feeder-head current,
= config/relay.yaml) and the 60 ms breaker, and turned into 10 ms bursts at
all 23 meter points (:func:`faults.build_burst`).  A quiet window is the
1.5 s steady record CP4 classified (:func:`faults.run_event`).  Each record
starts 0.5 s before its event (the meter's pre-trigger) and is written to the
meters right after its 15-min step; if an earlier event already uses that
step, the record is written after the next free step (the meter keeps one
record, so two records written back to back would hide the first from the
gateway; t_start is unchanged).

Consistency with CP4 (exit 1 on failure): per event and meter, the number of
samples equals ``meter_classes.parquet``, and per fault the clearing time and
relay element equal ``fault_events.parquet``; the largest clearing-time
difference is reported (the pre-fault loads come from the cached DT meter
rows instead of the household population).  Writes
``results/faults_2024/event_records.npz`` (:func:`meter_server.save_event_file`).

PowerShell::

    python scripts/build_event_records.py        # ~2 min
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_source import faults as F  # noqa: E402
from src.data_source import meter_server as ms  # noqa: E402
from src.data_source.feeder_model import build_network  # noqa: E402

OUT = ROOT / "results" / "faults_2024"
STEP = pd.Timedelta(minutes=15)


def build(net, relay):
    events = F.make_event_set(net)
    cache = F.steps_from_cache(net, events.step)
    bursts, rows, used = [], [], set()
    for ev in events.sort_values(["step", "offset_s"]).itertuples():
        F.set_prefault(net, *cache[pd.Timestamp(ev.step)])
        sn = F.build_sequence_network(net)
        out = F.run_event(net, sn, ev, relay, keep_bursts=True)
        trig = pd.Timestamp(ev.step)
        while trig in used:
            trig += STEP
        used.add(trig)
        t_start = pd.Timestamp(ev.step) + pd.Timedelta(seconds=float(ev.offset_s)) - pd.Timedelta(seconds=F.PRE_S)
        for lid, (v, i, vb) in out["bursts"].items():
            bursts.append({"trigger": trig, "load_id": lid, "event_id": int(ev.event_id), "t_start": t_start,
                           "v": v, "i": i, "v_base_ln_v": vb, "phases": "abc"})
        rows.append({"event_id": int(ev.event_id), "kind": ev.kind, "trigger": trig,
                     "trigger_moved": trig != pd.Timestamp(ev.step),
                     "t_clear_s": out.get("t_clear_s", math.inf), "element": out.get("element", "none"),
                     **{f"n_{m['meter']}": m["n_samples"] for m in out["meters"]}})
    return bursts, pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    warnings.simplefilter("ignore", FutureWarning)
    t0 = time.perf_counter()
    net = build_network("B")
    relay = F.relay_settings(F.max_head_current_a())
    bursts, rec = build(net, relay)
    # -- consistency with CP4 --------------------------------------------------
    ev = pd.read_parquet(OUT / "fault_events.parquet").set_index("event_id")
    mc = pd.read_parquet(OUT / "meter_classes.parquet")
    rec = rec.set_index("event_id").sort_index()
    n_new = rec.filter(like="n_").rename(columns=lambda c: c[2:]).stack().rename("n_new")
    n_old = mc.set_index(["event_id", "meter"]).n_samples.rename("n_old")
    nn = pd.concat([n_new, n_old], axis=1)
    fl = rec[rec.kind == "fault"]
    dclr = (fl.t_clear_s - ev.loc[fl.index, "t_clear_s"]).abs()
    checks = {
        "events": int(len(rec)), "bursts": len(bursts),
        "n_samples_mismatches": int((nn.n_new != nn.n_old).sum()),
        "t_clear_max_abs_diff_s": float(dclr.max()),
        "element_mismatches": int((fl.element != ev.loc[fl.index, "element"]).sum()),
        "triggers_moved": rec[rec.trigger_moved].index.tolist(),
        "relay": relay.__dict__,
    }
    checks["passed"] = bool(checks["n_samples_mismatches"] == 0 and checks["element_mismatches"] == 0)
    ms.save_event_file(OUT / "event_records.npz", bursts)
    (OUT / "event_records.json").write_text(json.dumps({**checks, "run_time_s": round(time.perf_counter() - t0, 1)},
                                                       indent=2, default=str), encoding="utf-8")
    print(json.dumps(checks, indent=2, default=str))
    if not checks["passed"]:
        print("EVENT RECORDS DISAGREE WITH CP4", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
