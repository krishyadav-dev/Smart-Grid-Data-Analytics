"""Checkpoint 3 addendum I: post-hoc, descriptive analysis after Checkpoint 3.

No re-tuning and no re-run of anything tuned: reads the step B reference
trajectories and the step D forecast trajectories (``table_e1``), and writes
``summary.json`` key ``addendum_i``, labelled "post-hoc, after Checkpoint 3".

For table_e1 at 98 degC, the 47 event DT-days (test origins) are split by
whether the reference theta_h was already >= 98 degC at 00:00 (issue time; the
state every forecast trajectory starts from).  On the DT-days that start below
98 degC: event count, precision, recall (Wilson) and lead time per model.  On
the DT-days that start at or above 98 degC: the forecast-only DT-days per model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.analytics import evaluate as V  # noqa: E402
from src.analytics import forecast as F  # noqa: E402
from src.analytics import thermal as T  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_layer3_thermal as R  # noqa: E402

LABEL = "post-hoc, after Checkpoint 3"
PARAM_SET = "table_e1"
THRESHOLD = 98.0


def _group(fc_first: np.ndarray, ref_first: np.ndarray) -> dict:
    """Counts, precision / recall (Wilson) and lead time on one group of DT-days."""
    ev, pred = ~np.isnan(ref_first), ~np.isnan(fc_first)
    det = ev & pred
    tp, fp = int(det.sum()), int((pred & ~ev).sum())
    lead = ref_first[det] / 60.0
    return {"dt_days": int(ev.size), "event_dt_days": int(ev.sum()), "detected": tp,
            "forecast_crossing_dt_days": int(pred.sum()), "forecast_only_dt_days": fp,
            "precision": V.wilson(tp, tp + fp), "recall": V.wilson(tp, int(ev.sum())),
            "lead_time_h": ({"mean": float(lead.mean()), "median": float(np.median(lead)),
                             "min": float(lead.min()), "max": float(lead.max())} if det.any() else None)}


def main() -> int:
    z = np.load(R.FORECASTS_TEST)
    origins = z["origins"]
    ref = R.load_reference(PARAM_SET)
    ref_first = T.day_summary(ref)["first_cross_min"][THRESHOLD][origins]               # [origin, DT]
    start = T.state_at(ref, origins * T.STEPS_PER_DAY - 1).theta_h                     # theta_h at 00:00
    above = start >= THRESHOLD
    tz = np.load(R.trajectories_path(PARAM_SET))
    out = {"label": LABEL, "parameter_set": PARAM_SET, "threshold_c": THRESHOLD,
           "definition": "start = reference theta_h at 00:00 of the origin (end of the previous day's last "
                         "sub-step), the state every forecast trajectory starts from; groups: start < 98 degC "
                         "and start >= 98 degC; P / R / lead time as in step E, within the group",
           "event_dt_days": {"all": int((~np.isnan(ref_first)).sum()),
                             "start_at_or_above_98": int((~np.isnan(ref_first) & above).sum()),
                             "start_below_98": int((~np.isnan(ref_first) & ~above).sum())},
           "dt_days_starting_at_or_above_98": int(above.sum()), "models": {}}
    for m in F.MODELS:
        fc_first = T.day_summary(R._flat(tz, m))["first_cross_min"][THRESHOLD]
        out["models"][m] = {"start_below_98": _group(fc_first[~above], ref_first[~above]),
                            "start_at_or_above_98": {
                                "forecast_only_dt_days": int((~np.isnan(fc_first) & np.isnan(ref_first) & above).sum()),
                                "detected": int((~np.isnan(fc_first) & ~np.isnan(ref_first) & above).sum())}}
    R._write_summary("addendum_i", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
