"""Validation pipeline (brief §6.2).

Ordered steps, each counted:

    parse -> schema_version -> model -> known_meter -> duplicate
          -> ordering_staleness -> plausibility -> energy

Rejects (``RejectReason``) go to the DLQ; flags are appended and the message
continues.  **Measurement values are never altered**: the validated document
is the parsed input dict plus an ``ingest`` block and merged flags.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import ValidationError

from ..config import ListenerSettings, MeterConfig
from ..schema import (
    SUPPORTED_SCHEMA_VERSIONS,
    QualityFlag,
    RejectReason,
    Telemetry,
)
from .. import topics

STEPS = (
    "parse", "schema_version", "model", "known_meter", "duplicate",
    "ordering_staleness", "plausibility", "energy", "accepted",
)

# Strictly non-negative magnitudes; anything below 0 is physically impossible.
_NON_NEGATIVE = ("voltage_v", "current_a")


@dataclass
class Accepted:
    doc: dict                  # validated document (input values untouched)
    model: Telemetry
    added_flags: list[str]


@dataclass
class Rejected:
    reason: RejectReason
    detail: str


@dataclass
class _MeterState:
    last_t_sim: Optional[datetime] = None
    last_run_id: Optional[str] = None
    last_seq: Optional[int] = None
    last_energy: Optional[float] = None
    seen_seq: "OrderedDict[tuple[str, int], None]" = field(default_factory=OrderedDict)
    seen_t_sim: "OrderedDict[datetime, None]" = field(default_factory=OrderedDict)


class Validator:
    def __init__(self, settings: ListenerSettings, meters: dict[str, MeterConfig]):
        self.s = settings
        self.meters = meters
        self.step_counts: dict[str, int] = {k: 0 for k in STEPS}
        self.reject_counts: dict[str, int] = {r.value: 0 for r in RejectReason}
        self.flag_counts: dict[str, int] = {f.value: 0 for f in QualityFlag}
        self._state: dict[str, _MeterState] = {}

    # ------------------------------------------------------------------ #
    def _reject(self, reason: RejectReason, detail: str) -> Rejected:
        self.reject_counts[reason.value] += 1
        return Rejected(reason, detail)

    @staticmethod
    def _remember(od: OrderedDict, key, limit: int) -> None:
        od[key] = None
        while len(od) > limit:
            od.popitem(last=False)

    # ------------------------------------------------------------------ #
    def process(self, topic: str, payload: bytes, t_recv: datetime) -> Accepted | Rejected:
        c = self.step_counts

        # 1. parse --------------------------------------------------------
        c["parse"] += 1
        try:
            doc = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._reject(RejectReason.MALFORMED_JSON, str(exc)[:200])
        if not isinstance(doc, dict):
            return self._reject(RejectReason.MALFORMED_JSON, "top level is not an object")

        # 2. schema version -----------------------------------------------
        c["schema_version"] += 1
        ver = doc.get("schema_version")
        if not isinstance(ver, str):
            return self._reject(RejectReason.SCHEMA_INVALID, "schema_version missing")
        if ver not in SUPPORTED_SCHEMA_VERSIONS:
            return self._reject(RejectReason.UNSUPPORTED_SCHEMA_VERSION, ver[:50])

        # 3. model (types, units, NaN/inf) ----------------------------------
        c["model"] += 1
        try:
            msg = Telemetry.model_validate(doc)
        except ValidationError as exc:
            errs = exc.errors(include_url=False)[:3]
            return self._reject(RejectReason.SCHEMA_INVALID,
                                "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errs))
        pt = topics.parse(topic)
        if pt is not None and pt.kind == "telemetry" and (
                pt.meter_id != msg.meter_id or pt.feeder_id != msg.feeder_id):
            return self._reject(RejectReason.SCHEMA_INVALID,
                                f"topic {topic} does not match payload {msg.feeder_id}/{msg.meter_id}")

        # 4. known meter ----------------------------------------------------
        c["known_meter"] += 1
        mc = self.meters.get(msg.meter_id)
        if mc is None or msg.feeder_id != self.s.feeder_id:
            return self._reject(RejectReason.UNKNOWN_METER, f"{msg.feeder_id}/{msg.meter_id}")
        if mc.unit_id != msg.unit_id or mc.load_id != msg.load_id:
            return self._reject(RejectReason.UNKNOWN_METER,
                                f"{msg.meter_id}: unit/load {msg.unit_id}/{msg.load_id} "
                                f"!= configured {mc.unit_id}/{mc.load_id}")

        st = self._state.setdefault(msg.meter_id, _MeterState())
        run_id = str(msg.run_id)

        # 5. duplicate -------------------------------------------------------
        c["duplicate"] += 1
        if (run_id, msg.seq) in st.seen_seq:
            return self._reject(RejectReason.DUPLICATE, f"{msg.meter_id} run {run_id[:8]} seq {msg.seq}")
        if msg.t_sim is not None and msg.t_sim in st.seen_t_sim:
            return self._reject(RejectReason.DUPLICATE, f"{msg.meter_id} t_sim {msg.t_sim.isoformat()}")

        added: list[str] = []
        gap_steps = msg.quality.gap_steps

        # 6. ordering & staleness ---------------------------------------------
        c["ordering_staleness"] += 1
        out_of_order = False
        if msg.t_sim is not None and st.last_t_sim is not None and msg.t_sim < st.last_t_sim:
            out_of_order = True
        if (st.last_run_id == run_id and st.last_seq is not None and msg.seq < st.last_seq):
            out_of_order = True
        if out_of_order:
            if self.s.out_of_order_policy == "reject":
                return self._reject(RejectReason.OUT_OF_ORDER,
                                    f"{msg.meter_id} t_sim {msg.t_sim} / seq {msg.seq}")
            added.append(QualityFlag.OUT_OF_ORDER.value)
        elif msg.t_sim is not None and st.last_t_sim is not None:
            steps = round((msg.t_sim - st.last_t_sim).total_seconds() / self.s.expected_step_s)
            if steps > 1:
                added.append(QualityFlag.GAP.value)
                gap_steps = max(gap_steps, steps - 1)
        age_s = (t_recv - msg.t_poll_utc).total_seconds()
        if age_s > self.s.stale_after_s:
            added.append(QualityFlag.STALE.value)

        # 7. plausibility (impossible values only — never clip) -----------------
        c["plausibility"] += 1
        m = msg.measurements
        bad = []
        for key in _NON_NEGATIVE:
            d = getattr(m, key)
            if d and any(v < 0 for v in d.values()):
                bad.append(key)
        if m.energy_import_kwh is not None and m.energy_import_kwh < 0:
            bad.append("energy_import_kwh")
        if m.frequency_hz is not None and m.frequency_hz <= 0:
            bad.append("frequency_hz")
        if bad:
            added.append(QualityFlag.SUSPECT_RANGE.value)

        # 8. energy decrease -----------------------------------------------------
        c["energy"] += 1
        e = m.energy_import_kwh
        if e is not None and st.last_energy is not None and e < st.last_energy and not out_of_order:
            added.append(QualityFlag.ENERGY_DECREASE.value)

        # ---- accept: update state --------------------------------------------
        c["accepted"] += 1
        self._remember(st.seen_seq, (run_id, msg.seq), self.s.dedup_window)
        if msg.t_sim is not None:
            self._remember(st.seen_t_sim, msg.t_sim, self.s.dedup_window)
        if not out_of_order:
            if msg.t_sim is not None:
                st.last_t_sim = msg.t_sim
            if e is not None:
                st.last_energy = e
        if st.last_run_id != run_id or st.last_seq is None or msg.seq > st.last_seq:
            st.last_run_id, st.last_seq = run_id, msg.seq

        quality = doc.setdefault("quality", {"flags": [], "gap_steps": 0})
        flags = list(quality.get("flags", []))
        for f in added:
            if f not in flags:
                flags.append(f)
        quality["flags"] = flags
        quality["gap_steps"] = gap_steps
        for f in flags:
            if f in self.flag_counts:
                self.flag_counts[f] += 1
        return Accepted(doc=doc, model=msg, added_flags=added)
