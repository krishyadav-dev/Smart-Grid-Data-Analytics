"""Layer 3 fault engine: rule-based, IS 17036 (docs/Layer3_plan.md §1).

Inputs are validated telemetry only: the fault bursts of
``validated_event/{meter}`` and the 15-min readings of ``validated/{meter}``
of the 14 physical meters (:mod:`meters`), plus configuration
(``config/relay.yaml``, ``config/zones.yaml``, ``config/layer3_params.yaml``).
Bursts are grouped by time only; the record header's identifier is never used
(run sheet rule 1), and no fault timing comes from anywhere but the samples.

Per burst (one meter):

- **Step 1, IS 17036.**  u_p(t) = 100 V_p(t) / V_ref with V_ref the meter's
  declared line-to-neutral voltage (``v_base_ln_v``, meters.yaml).  **F10
  (stated deviation from the IS 17036 guideline that three-wire systems use
  line-to-line voltages, PDF p. 10):** every meter, including the 11 kV feeder
  head, is classified on phase-to-neutral magnitudes, because the event record
  carries no line-to-line values or angles.  Classification by
  :func:`faults.classify` unchanged (Table 13 dips, Table 12 interruptions,
  PDF p. 9; polyphase rules: start = first sample with any phase < 90 %, end =
  first later sample with every phase >= 90 %, residual = lowest phase,
  interruption = all phases < 5 %; Cl. 3.19, 3.23, 3.24, PDF p. 5).  Swell:
  any phase > 110 % (Cl. 3.29, PDF p. 5).
- **Step 2, current rise.**  rho_p = max_t I_p(t) / mean_pre I_p, with the
  pre-event reference = the burst's first 50 samples (the meter's fixed
  0.5 s pre-trigger, :data:`faults.N_PRE`).  A phase with rho_p > rho* is an
  overcurrent phase; any overcurrent phase is a detection.
- **Fault samples** (dip depth for Step 7, fault currents for Steps 3-4): the
  samples after the pre-trigger, except those where every phase is below 5 %
  (the interruption after the feeder breaker opens).

Per event (decisions of 27 Oct 2026, recorded in config/layer3_params.yaml):

- **Step 6, merging ("open event absorbs").**  Detections are sorted by onset.
  An event stays open while any meter in it is still disturbed (below 90 %,
  interrupted, above 110 % or above rho* x its pre-event current) and for one
  coincidence window after that; a detection that starts while it is open
  joins it.  A detection after that opens a new event.  ``mode="strict"``
  (onsets within one window of the event's first onset) is kept for the
  split count only.
- **Step 3, type** from the feeder-head burst only (DT meters see the 11 kV
  phases through Dyn11): the overcurrent phases, or the dipping phases if no
  phase rose above rho*; 1 -> SLG, 3 -> 3PH, 2 -> LL if the two phases'
  fault currents differ by at most the phase-balance tolerance (relative to
  the larger), else LLG.  No head detection -> type undetermined (marked).
- **Step 4, IDMT.**  Expected phase-element operate time from the head's
  highest fault current: t = TMS k / ((I/Is)^alpha - 1) + breaker time (NPAG
  §9.4 Table 9.1, PDF p. 134; config/relay.yaml).  The earth-fault element
  needs 3 I0, which the event record does not carry (F1).  Observed clearing:
  the first sample at which every head phase current reads 0 A (breaker
  open), or, without a head detection, the first DT sample with every phase
  below 5 %.
- **Step 7, zone.**  For each DT meter in the event, the dip = pre-event minus
  lowest fault-sample voltage, worst phase, in percentage points of V_ref.
  DTs within delta of the deepest form the set; its most upstream common
  node on the feeder tree (:mod:`zones`) gives the zone.  Baseline: the zone
  of the single DT with the deepest dip.
- **Flat baseline:** an alert whenever any phase leaves 90-110 % of the
  declared voltage, on every burst sample and every 15-min sample, with no
  duration rule (one alert per contiguous excursion run per meter).
- **Step 5, unbalance** (15-min stream): VU from the line-to-line magnitudes
  (:func:`unbalance.vuf_pct_from_ll`); an alarm per meter and ISO week when
  VU > 2 % on more than 5 % of the week's received samples (IS 17036 Table 8;
  missing and GAP intervals are missing, never zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from src.data_source import faults as F
from src.data_source.config import PROJECT_ROOT

from . import meters as M
from . import zones as Z
from .unbalance import VU_LIMIT_PCT, vuf_pct_from_ll

RELAY_YAML = PROJECT_ROOT / "config" / "relay.yaml"
PARAMS_YAML = PROJECT_ROOT / "config" / "layer3_params.yaml"
SAMPLE_MS = F.SAMPLE_MS                 # 10 ms half-cycle RMS (IS 17036 Cl. 3.18)
N_PRE = F.N_PRE                         # 0.5 s pre-trigger of the meter's event record
DIP_PCT = F.DIP_PCT                     # IS 17036 Cl. 3.23: dip below 90 %
INTERRUPTION_PCT = F.INTERRUPTION_PCT   # IS 17036 Cl. 3.19: interruption below 5 %
SWELL_PCT = 110.0                       # IS 17036 Cl. 3.29, PDF p. 5: swell above 110 %
FLAT_LOW_PCT, FLAT_HIGH_PCT = 90.0, 110.0   # flat +/-10 % baseline (plan §1 "Baseline")
WEEK_EXCEED_SHARE = 0.05                # IS 17036 Table 8: the 2 % limit holds for 95 % of each week
TYPES = ("SLG", "LL", "LLG", "3PH")
# Live mode only bounds which earlier detections are re-merged on arrival; one
# event record is at most 0.5 s + 60 s + 1.0 s long (faults.N_CAP), so an hour
# of simulated time covers any event.  It never changes the final events.
LIVE_HORIZON_MS = 3_600_000


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Relay:
    is_a: float
    ef_is_a: float
    tms: float
    ef_tms: float
    breaker_s: float
    k: float
    alpha: float


def load_relay(path: str | Path = RELAY_YAML) -> Relay:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Relay(float(d["is_a"]), float(d["ef_is_a"]), float(d["tms"]), float(d["ef_tms"]),
                 float(d["breaker_s"]), float(d["curve"]["k"]), float(d["curve"]["alpha"]))


def idmt_time_s(i_a: float, is_a: float, tms: float, k: float = 0.14, alpha: float = 0.02) -> float:
    """IEC standard inverse (NPAG §9.4 Table 9.1): relay time, inf at or below pickup."""
    r = i_a / is_a
    return math.inf if r <= 1.0 else tms * k / (r ** alpha - 1.0)


@dataclass(frozen=True)
class Params:
    rho_star: float
    coincidence_samples: int
    balance_tol: float
    delta_pp: float


def load_params(path: str | Path = PARAMS_YAML) -> Params:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["frozen"]
    return Params(float(d["rho_star"]), int(d["coincidence_samples"]), float(d["balance_tol"]),
                  float(d["delta_pp"]))


# --------------------------------------------------------------------------
# Bursts
# --------------------------------------------------------------------------

def _ms(t: str | datetime) -> int:
    """Epoch milliseconds of an aware ISO timestamp."""
    return int(round(pd.Timestamp(t).value / 1e6))


@dataclass
class Burst:
    """One validated burst of a physical meter, as arrays."""
    meter_id: str
    role: str
    t0_ms: int                  # first sample
    v: np.ndarray               # (n, 3) V
    i: np.ndarray               # (n, 3) A
    v_ref: float                # declared line-to-neutral voltage (meters.yaml)
    t_poll_utc: Optional[str] = None


def burst_from_doc(doc: dict) -> Burst:
    """Validated ``EventBurst`` document -> :class:`Burst` (physical meters only)."""
    m = M.meter(doc["meter_id"])                 # raises for any non-physical meter
    s = doc["samples"]
    n = int(doc["n_samples"])
    v = np.column_stack([np.asarray(s["voltage_v"].get(p, [np.nan] * n), float) for p in "abc"])
    i = np.column_stack([np.asarray(s["current_a"].get(p, [np.nan] * n), float) for p in "abc"])
    return Burst(m.meter_id, str(m.role), _ms(doc["t_start"]), v, i, float(m.v_base_ln_v),
                 doc.get("t_poll_utc"))


@dataclass
class Analysis:
    """Per-burst result of Steps 1-2 and the quantities Steps 3, 4 and 7 use."""
    meter_id: str
    role: str
    t0_ms: int
    n: int
    detected: bool
    by: tuple                     # subset of ("is17036", "swell", "current_rise")
    onset_k: Optional[int]
    last_k: Optional[int]
    kind: str                     # faults.classify kind
    residual_pct: Optional[float]
    duration_ms: Optional[float]
    u_bin: Optional[str]
    t_bin: Optional[str]
    rho: np.ndarray               # (3,)
    oc_phases: str
    dip_phases: str
    i_fault_max: np.ndarray       # (3,) A over the samples after the pre-trigger
    u_pre: np.ndarray             # (3,) % of V_ref
    u_min_fault: np.ndarray       # (3,) % of V_ref over the fault samples
    dip_pp: float                 # worst-phase dip, percentage points of V_ref
    clear_k: Optional[int]
    t_poll_utc: Optional[str] = None

    @property
    def onset_ms(self) -> Optional[int]:
        return None if self.onset_k is None else self.t0_ms + self.onset_k * SAMPLE_MS

    @property
    def last_ms(self) -> Optional[int]:
        return None if self.last_k is None else self.t0_ms + self.last_k * SAMPLE_MS

    @property
    def clear_ms(self) -> Optional[int]:
        return None if self.clear_k is None else self.t0_ms + self.clear_k * SAMPLE_MS


def _run(mask: np.ndarray) -> tuple[Optional[int], Optional[int]]:
    idx = np.flatnonzero(mask)
    return (int(idx[0]), int(idx[-1])) if idx.size else (None, None)


def analyse(b: Burst, rho_star: float) -> Analysis:
    """Steps 1 and 2 on one burst."""
    u = 100.0 * b.v / b.v_ref
    n = len(u)
    pre = slice(0, min(N_PRE, n))
    # Step 1: IS 17036 (faults.classify unchanged; F10: phase-to-neutral)
    c = F.classify(b.v, b.v_ref)
    below = u.min(axis=1) < DIP_PCT
    on1 = last1 = None
    if c.kind != "no_dip":
        on1 = int(np.argmax(below))
        rec = np.flatnonzero(~below[on1:])
        last1 = (on1 + int(rec[0]) - 1) if rec.size else n - 1
    on_sw, last_sw = _run(u.max(axis=1) > SWELL_PCT)
    # Step 2: current rise against the pre-trigger mean
    i_pre = b.i[pre].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(i_pre > 0, b.i.max(axis=0) / i_pre, np.nan)
        rise = (b.i > rho_star * i_pre) & (i_pre > 0)
    on2, last2 = _run(rise.any(axis=1))
    oc = "".join(p for j, p in enumerate("abc") if np.isfinite(rho[j]) and rho[j] > rho_star)
    by, onsets, lasts = [], [], []
    for name, o, l in (("is17036", on1, last1), ("swell", on_sw, last_sw), ("current_rise", on2, last2)):
        if o is not None:
            by.append(name)
            onsets.append(o)
            lasts.append(l)
    # fault samples: after the pre-trigger, outside the all-phase interruption
    post = np.arange(n) >= N_PRE
    allint = (u < INTERRUPTION_PCT).all(axis=1)
    fs = post & ~allint
    if not fs.any():
        fs = post
    u_pre = u[pre].mean(axis=0)
    u_min = u[fs].min(axis=0) if fs.any() else u_pre.copy()
    i_fmax = b.i[post].max(axis=0) if post.any() else b.i.max(axis=0)
    dip_ph = "".join(p for j, p in enumerate("abc") if u_min[j] < DIP_PCT)
    # observed clearing: head current to 0 A (breaker open) / DT all phases < 5 %
    if b.role == "feeder_head":
        clr = post & (b.i == 0.0).all(axis=1)
    else:
        clr = post & allint
    clear_k, _ = _run(clr)
    return Analysis(b.meter_id, b.role, b.t0_ms, n, bool(by), tuple(by),
                    min(onsets) if onsets else None, max(lasts) if lasts else None,
                    c.kind, c.residual_pct, c.duration_ms if c.duration_ms is not None else c.interruption_ms,
                    c.u_bin, c.t_bin, rho, oc, dip_ph, i_fmax, u_pre, u_min,
                    float(np.max(u_pre - u_min)), clear_k, b.t_poll_utc)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

@dataclass
class Event:
    onset_ms: int
    close_ms: int
    members: list = field(default_factory=list)
    ftype: Optional[str] = None
    type_source: str = "undetermined"
    zone: Optional[str] = None
    zone_node: Optional[str] = None
    zone_set: tuple = ()
    baseline_zone: Optional[str] = None
    baseline_bus: Optional[str] = None
    residual_pct: Optional[float] = None
    duration_ms: Optional[float] = None
    clearing_ms: Optional[float] = None
    clearing_source: Optional[str] = None
    i_head_max_a: Optional[float] = None
    t_idmt_phase_s: Optional[float] = None
    t_expected_clear_s: Optional[float] = None
    head_detected: bool = False
    head_no_dip: bool = False
    head_rise_only: bool = False

    @property
    def meters(self) -> tuple:
        return tuple(sorted(a.meter_id for a in self.members))

    def record(self) -> dict:
        return {"onset_ms": self.onset_ms, "onset": str(pd.Timestamp(self.onset_ms, unit="ms", tz="Asia/Kolkata")),
                "close_ms": self.close_ms, "n_meters": len(self.members), "meters": ",".join(self.meters),
                "ftype": self.ftype, "type_source": self.type_source, "zone": self.zone,
                "zone_node": self.zone_node, "zone_set": ",".join(self.zone_set),
                "baseline_zone": self.baseline_zone, "baseline_bus": self.baseline_bus,
                "residual_pct": self.residual_pct, "duration_ms": self.duration_ms,
                "clearing_ms": self.clearing_ms, "clearing_source": self.clearing_source,
                "i_head_max_a": self.i_head_max_a, "t_idmt_phase_s": self.t_idmt_phase_s,
                "t_expected_clear_s": self.t_expected_clear_s, "head_detected": self.head_detected,
                "head_no_dip": self.head_no_dip, "head_rise_only": self.head_rise_only}


def merge(analyses: Iterable[Analysis], coincidence_samples: int, mode: str = "open") -> list[Event]:
    """Step 6: one event per group of detections (time only; module docstring)."""
    w = coincidence_samples * SAMPLE_MS
    det = sorted((a for a in analyses if a.detected), key=lambda a: (a.onset_ms, a.meter_id))
    events: list[Event] = []
    for a in det:
        cur = events[-1] if events else None
        if cur is not None and (a.onset_ms <= cur.close_ms if mode == "open"
                                else a.onset_ms <= cur.onset_ms + w):
            cur.members.append(a)
            cur.close_ms = max(cur.close_ms, a.last_ms + w)
        elif mode in ("open", "strict"):
            events.append(Event(a.onset_ms, a.last_ms + w, [a]))
        else:
            raise ValueError(mode)
    return events


def fault_type(head: Analysis, balance_tol: float) -> Optional[str]:
    """Step 3 on the feeder-head analysis (module docstring)."""
    phases = head.oc_phases or head.dip_phases
    if len(phases) == 1:
        return "SLG"
    if len(phases) == 3:
        return "3PH"
    if len(phases) == 2:
        i = [head.i_fault_max["abc".index(p)] for p in phases]
        return "LL" if abs(i[0] - i[1]) / max(i) <= balance_tol else "LLG"
    return None


def locate(dts: list[Analysis], delta_pp: float) -> dict:
    """Step 7 and the deepest-single-DT baseline."""
    if not dts:
        return {"zone": None, "node": None, "set": (), "baseline_zone": None, "baseline_bus": None}
    deepest = max(a.dip_pp for a in dts)
    sel = sorted(a.meter_id for a in dts if a.dip_pp >= deepest - delta_pp)
    node = Z.common_upstream_node([Z.bus_of_meter(m) for m in sel])
    best = max(dts, key=lambda a: (a.dip_pp, a.meter_id))
    bbus = Z.bus_of_meter(best.meter_id)
    return {"zone": Z.zone_of_bus(node), "node": node, "set": tuple(sel),
            "baseline_zone": Z.zone_of_bus(bbus), "baseline_bus": bbus}


def finalise(ev: Event, params: Params, relay: Relay) -> Event:
    """Steps 3, 4 and 7 and the event fields for a merged event."""
    head = next((a for a in ev.members if a.role == "feeder_head"), None)
    dts = [a for a in ev.members if a.role == "dt"]
    ev.head_detected = head is not None
    if head is not None:
        ev.ftype = fault_type(head, params.balance_tol)
        ev.type_source = "feeder_head" if ev.ftype else "undetermined"
        ev.head_no_dip = head.kind == "no_dip"
        ev.head_rise_only = head.by == ("current_rise",)
        ev.i_head_max_a = float(np.nanmax(head.i_fault_max))
        t = idmt_time_s(ev.i_head_max_a, relay.is_a, relay.tms, relay.k, relay.alpha)
        ev.t_idmt_phase_s = t
        ev.t_expected_clear_s = t + relay.breaker_s
        if head.clear_ms is not None:
            ev.clearing_ms = float(head.clear_ms - ev.onset_ms)
            ev.clearing_source = "feeder_head"
        ev.duration_ms = head.duration_ms if head.kind != "no_dip" else (
            float((head.last_k - head.onset_k + 1) * SAMPLE_MS))
    if ev.clearing_ms is None:
        clr = [a.clear_ms for a in dts if a.clear_ms is not None]
        if clr:
            ev.clearing_ms = float(min(clr) - ev.onset_ms)
            ev.clearing_source = "dt_meters"
    if ev.duration_ms is None:
        ev.duration_ms = ev.clearing_ms
    res = [a.residual_pct for a in ev.members if a.residual_pct is not None]
    ev.residual_pct = float(min(res)) if res else None
    loc = locate(dts, params.delta_pp)
    ev.zone, ev.zone_node, ev.zone_set = loc["zone"], loc["node"], loc["set"]
    ev.baseline_zone, ev.baseline_bus = loc["baseline_zone"], loc["baseline_bus"]
    return ev


def run_batch(docs: Iterable[dict], params: Params, relay: Relay | None = None,
              mode: str = "open") -> tuple[list[Event], list[Analysis]]:
    """Batch mode: every burst document -> analyses -> merged, finalised events."""
    relay = load_relay() if relay is None else relay
    an = [analyse(burst_from_doc(d), params.rho_star) for d in docs]
    return [finalise(e, params, relay) for e in merge(an, params.coincidence_samples, mode)], an


# --------------------------------------------------------------------------
# Live mode
# --------------------------------------------------------------------------

class LiveEngine:
    """Live wrapper: each burst is analysed on arrival; an alarm is raised at
    once when a detection does not fall inside an already-alarmed open event
    (latency = the burst's t_poll_utc -> alarm).  The final events are
    :func:`merge` of every analysis, the same function as batch mode."""

    def __init__(self, params: Params, relay: Relay | None = None, clock=None):
        self.params = params
        self.relay = load_relay() if relay is None else relay
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.analyses: list[Analysis] = []
        self.alarms: list[dict] = []

    def on_burst(self, doc: dict) -> Optional[dict]:
        t_in = self.clock()
        a = analyse(burst_from_doc(doc), self.params.rho_star)
        self.analyses.append(a)
        if not a.detected:
            return None
        # the detections already received around this one (all earlier ones are alarmed)
        near = [x for x in self.analyses if x.detected and abs(x.onset_ms - a.onset_ms) <= LIVE_HORIZON_MS]
        mine = next(e for e in merge(near, self.params.coincidence_samples) if any(x is a for x in e.members))
        if len(mine.members) > 1:
            return None                          # joins an event that already raised its alarm
        t_out = self.clock()
        alarm = {"meter_id": a.meter_id, "onset_ms": a.onset_ms, "t_poll_utc": a.t_poll_utc,
                 "t_received_utc": str(t_in), "t_alarm_utc": str(t_out),
                 "latency_ms": (t_out - pd.Timestamp(a.t_poll_utc)).total_seconds() * 1e3
                 if a.t_poll_utc else None}
        self.alarms.append(alarm)
        return alarm

    def events(self) -> list[Event]:
        return [finalise(e, self.params, self.relay)
                for e in merge(self.analyses, self.params.coincidence_samples)]


# --------------------------------------------------------------------------
# Flat +/-10 % baseline
# --------------------------------------------------------------------------

def flat_alerts_burst(b: Burst) -> list[int]:
    """Onset times (epoch ms) of the flat-baseline alerts in one burst."""
    u = 100.0 * b.v / b.v_ref
    out_ = ((u < FLAT_LOW_PCT) | (u > FLAT_HIGH_PCT)).any(axis=1)
    starts = np.flatnonzero(out_ & ~np.r_[False, out_[:-1]])
    return [b.t0_ms + int(k) * SAMPLE_MS for k in starts]


def flat_alerts_stream(stream: pd.DataFrame) -> pd.DataFrame:
    """Flat-baseline alerts on the 15-min stream (:func:`stream_frame` rows):
    one alert per contiguous run of out-of-band samples per meter."""
    rows = []
    for mid, g in stream.sort_values("t_sim").groupby("meter_id"):
        v_ref = float(M.meter(mid).v_base_ln_v)
        u = 100.0 * g[[f"voltage_v_{p}" for p in "abc"]].to_numpy(float) / v_ref
        bad = ((u < FLAT_LOW_PCT) | (u > FLAT_HIGH_PCT)).any(axis=1)
        prev = np.r_[False, bad[:-1]]
        for k in np.flatnonzero(bad & ~prev):
            rows.append({"meter_id": mid, "t_sim": g.t_sim.iloc[k]})
    return pd.DataFrame(rows, columns=["meter_id", "t_sim"])


# --------------------------------------------------------------------------
# Step 5: unbalance on the 15-min stream
# --------------------------------------------------------------------------

def stream_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Physical meters only; ``t_sim`` as naive IST wall time."""
    phys = {m.meter_id for m in M.physical_meters()}
    out = rows[rows.meter_id.isin(phys)].copy()
    out["t_sim"] = pd.to_datetime(out.t_sim.str.slice(0, 19))
    return out


def weekly_unbalance(stream: pd.DataFrame) -> pd.DataFrame:
    """Per meter and ISO week (Mon-Sun, IST): VU > 2 % share and the alarm."""
    s = stream.copy()
    s["vu_pct"] = vuf_pct_from_ll(s.voltage_ll_v_ab, s.voltage_ll_v_bc, s.voltage_ll_v_ca)
    s = s[np.isfinite(s.vu_pct)]
    s["week"] = s.t_sim.dt.normalize() - pd.to_timedelta(s.t_sim.dt.weekday, unit="D")
    g = s.groupby(["meter_id", "week"])
    out = g.vu_pct.agg(samples="size", vu_max_pct="max").reset_index()
    out["share_above_limit"] = g.vu_pct.apply(lambda x: float((x > VU_LIMIT_PCT).mean())).to_numpy()
    out["alarm"] = out.share_above_limit > WEEK_EXCEED_SHARE
    return out
