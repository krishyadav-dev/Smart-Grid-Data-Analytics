"""Counters, per-meter rates and latency percentiles for ``/api/v1/stats``."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional


def percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    """Nearest-rank percentile (p in 0..100) of an already sorted list."""
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, math.ceil(p / 100 * len(sorted_vals)) - 1))
    return sorted_vals[k]


class LatencyStats:
    HOPS = ("poll_to_pub", "pub_to_recv", "total")

    def __init__(self, sample_size: int = 10000):
        self._samples = {h: deque(maxlen=sample_size) for h in self.HOPS}

    def add(self, latency_ms: dict) -> None:
        for h in self.HOPS:
            self._samples[h].append(latency_ms[h])

    def summary(self) -> dict:
        out = {}
        for h, dq in self._samples.items():
            s = sorted(dq)
            out[h] = {"n": len(s), "p50": percentile(s, 50), "p95": percentile(s, 95),
                      "p99": percentile(s, 99), "max": s[-1] if s else None}
        return out


class RateTracker:
    """Messages per second per meter over a sliding wall-clock window."""

    def __init__(self, window_s: float = 60.0, clock=time.monotonic):
        self.window_s = window_s
        self._clock = clock
        self._times: dict[str, deque[float]] = {}
        self.totals: dict[str, int] = {}

    def add(self, meter_id: str) -> None:
        now = self._clock()
        dq = self._times.setdefault(meter_id, deque())
        dq.append(now)
        self.totals[meter_id] = self.totals.get(meter_id, 0) + 1
        self._trim(dq, now)

    def _trim(self, dq: deque[float], now: float) -> None:
        while dq and now - dq[0] > self.window_s:
            dq.popleft()

    def rates(self) -> dict[str, float]:
        now = self._clock()
        out = {}
        for m, dq in self._times.items():
            self._trim(dq, now)
            out[m] = len(dq) / self.window_s
        return out
