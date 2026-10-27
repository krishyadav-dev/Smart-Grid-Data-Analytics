"""Checkpoint 1 decision and addendum (commit G): new analyses of the frozen
test-half outputs only.  The engine is not changed, re-tuned or re-run.

Inputs: results/layer3/fault/summary.json and summary_events.parquet (Step E,
frozen), the validated burst documents (batch/bursts.jsonl.gz), the year's
validated 15-min stream (batch/stream_2024.parquet) and the labels (read here
through src/analytics/evaluate.py only).

1. Unbalance at the DT meters (user decision 28 Oct): per physical meter, the
   2024 weekly alarms of the Step 5 function (parameter-free, same input as
   Step E; its totals are checked against Step E) against each meter's own
   V_N / V_P from meter_frame_true, and the P2 results.  Labelled as a
   consistency check against the power flow, not detection skill.
2. Detection delay of the flat +/-10 % baseline (same definition as the
   engine's: first alerting sample minus fault inception), and for engine and
   baseline how many test faults are detected before the breaker clears
   (delay < observed clearing time, inception to the first sample with every
   feeder-head phase current at 0 A), also for the 11 no-dip test faults.
3. Descriptive: the smallest feeder-head current rise rho_p among the no-dip
   test faults and the largest rho_p on the 100 test quiet windows (signal
   statistics of the bursts, the same formula as Step 2), around rho* = 1.70.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import fault_engine as E  # noqa: E402
from src.analytics import meters as M  # noqa: E402
from src.data_source import faults as F  # noqa: E402
from src.data_source.config import YEAR_CACHE_DIR  # noqa: E402
from src.data_source.unbalance_scenarios import vu_true  # noqa: E402

OUT = ROOT / "results" / "layer3" / "fault"
LABEL = ("consistency check against the power flow: VU from line-to-line magnitudes is an exact identity "
         "and meter error does not touch voltage, so agreement is expected; not detection skill")


def _q(x) -> dict | None:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return None if not len(x) else {"n": int(len(x)), "min": float(x.min()), "p50": float(np.median(x)),
                                    "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def unbalance_per_meter(summary: dict) -> dict:
    lid = {m.meter_id: m.load_id for m in M.physical_meters()}
    stream = E.stream_frame(pd.read_parquet(OUT / "batch" / "stream_2024.parquet"))
    eng = E.weekly_unbalance(stream)
    head = eng.meter_id == "m10"
    reproduces = (int(eng[head].alarm.sum()) == summary["stream_2024"]["engine_unbalance_alarms_feeder_head"]
                  and int(eng[~head].alarm.sum()) == summary["stream_2024"]["engine_unbalance_alarms_dt_meters"])
    ids = list(lid.values())
    f = pd.read_parquet(YEAR_CACHE_DIR / "meter_frame_true.parquet",
                        columns=["timestamp", "load_id", "phase", "voltage_v", "voltage_angle_deg"],
                        filters=[("load_id", "in", ids)])
    t = vu_true(f)
    t["week"] = t.timestamp.dt.normalize() - pd.to_timedelta(t.timestamp.dt.weekday, unit="D")
    g = t.groupby(["load_id", "week"]).vu_true_pct
    truth = g.agg(vu_max_pct="max").reset_index()
    truth["exceeds"] = g.apply(lambda x: float((x > E.VU_LIMIT_PCT).mean()) > E.WEEK_EXCEED_SHARE).to_numpy()
    eng = eng.assign(load_id=eng.meter_id.map(lid))
    m = eng.merge(truth, on=["load_id", "week"], how="outer", suffixes=("_engine", "_truth"),
                  validate="one_to_one")
    per = {}
    p2 = summary["p2"]["per_meter"]
    for (mid, l), gm in m.groupby(["meter_id", "load_id"]):
        a, x = gm.alarm.fillna(False).astype(bool), gm.exceeds.fillna(False).astype(bool)
        per[f"{mid} {l}"] = {
            "weeks": int(len(gm)), "engine_weekly_alarms": int(a.sum()), "truth_weeks_exceeding": int(x.sum()),
            "agreement_weeks": int((a == x).sum()),
            "vu_max_pct_engine": float(gm.vu_max_pct_engine.max()), "vu_max_pct_truth": float(gm.vu_max_pct_truth.max()),
            "weeks_alarmed": [str(w.date()) for w in gm.week[a]],
            "p2": p2.get(l)}
    return {"label": LABEL, "decision": "unbalance check runs at the DT meters as well as the feeder head "
                                        "(user decision at Checkpoint 1: IS 17036 supply-quality limit; the DT "
                                        "LV side is the supply point)",
            "engine_totals_reproduce_step_e": bool(reproduces), "per_meter": per}


def addendum(summary: dict) -> dict:
    split, labels = V.load_split(), V.load_labels()
    ft, qt = split["faults_test"], split["quiet_test"]
    ev = pd.read_parquet(OUT / "summary_events.parquet")
    first = V.first_alert(ev, ft)
    lab = labels.loc[first.index]
    delay_e = (first.onset_ms - lab.t_event_ms).astype(float)
    clear_obs = (first.onset_ms + first.clearing_ms - lab.t_event_ms).astype(float)       # inception -> breaker open
    docs = V.select_docs(V.read_docs(OUT / "batch" / "bursts.jsonl.gz"), ft + qt)
    bursts = [E.burst_from_doc(d) for d in docs]
    alerts = [t for b in bursts for t in E.flat_alerts_burst(b)]
    m = pd.Series(V.match_onsets(alerts, labels), dtype="object")
    fa = pd.DataFrame({"t": alerts, "e": m}).dropna()
    fa = fa[fa.e.isin(set(ft))]
    flat_first = fa.groupby("e").t.min()
    delay_f = (flat_first.reindex(first.index) - lab.t_event_ms).astype(float)
    # no-dip test faults: feeder-head burst never below 90 % (IS 17036 dip threshold) on any phase
    head = {}
    for d, b in zip(docs, bursts):
        if b.role == "feeder_head":
            head[int(d["event_id"])] = b
    nodip = sorted(e for e in ft if (100.0 * head[e].v / head[e].v_ref).min() >= F.DIP_PCT)

    def rho(b):
        pre = b.i[:F.N_PRE].mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(pre > 0, b.i.max(axis=0) / pre, np.nan)

    rho_nodip = {e: float(np.nanmax(rho(head[e]))) for e in nodip}
    quiet_head = [float(np.nanmax(rho(head[e]))) for e in qt]
    quiet_all = [float(np.nanmax(rho(b))) for d, b in zip(docs, bursts) if int(d["event_id"]) in set(qt)]

    def before(delay, sub):
        s = [e for e in sub if np.isfinite(delay[e])]
        return {"detected": len(s), "before_breaker_clears": int(sum(delay[e] < clear_obs[e] for e in s)),
                "at_or_after_clearing": int(sum(delay[e] >= clear_obs[e] for e in s))}
    rho_star = summary["params"]["rho_star"]
    return {
        "note": "new analyses of the frozen Step E outputs; the engine was not changed, re-tuned or re-run",
        "detection_delay_ms": {"engine": _q(delay_e), "flat_baseline": _q(delay_f),
                               "engine_no_dip": _q(delay_e[nodip]), "flat_baseline_no_dip": _q(delay_f[nodip])},
        "observed_clearing_ms_from_inception": _q(clear_obs),
        "detected_before_breaker_clears": {
            "engine": before(delay_e, list(first.index)), "flat_baseline": before(delay_f, list(first.index)),
            "engine_no_dip": before(delay_e, nodip), "flat_baseline_no_dip": before(delay_f, nodip)},
        "no_dip_test_faults": nodip,
        "rho_margin": {
            "rho_star": rho_star,
            "min_head_rho_no_dip_faults": min(rho_nodip.values()),
            "min_at_event": min(rho_nodip, key=rho_nodip.get),
            "head_rho_no_dip_faults": rho_nodip,
            "max_head_rho_quiet_test": max(quiet_head),
            "max_rho_any_meter_quiet_test": max(quiet_all),
            "margin_above_rho_star": min(rho_nodip.values()) - rho_star,
            "margin_below_rho_star_head": rho_star - max(quiet_head),
            "label": "descriptive only; rho* stays 1.70 (frozen, commit 70b0941)"},
    }


def main() -> int:
    path = OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["unbalance_per_meter_2024"] = unbalance_per_meter(summary)
    summary["addendum_g"] = addendum(summary)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"engine_totals_reproduce_step_e":
                      summary["unbalance_per_meter_2024"]["engine_totals_reproduce_step_e"],
                      **summary["addendum_g"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
