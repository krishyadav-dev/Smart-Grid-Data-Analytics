"""Exponential backoff with jitter (used by the gateway and the listener)."""

from __future__ import annotations

import random


class Backoff:
    def __init__(self, initial_s: float = 0.5, max_s: float = 15.0,
                 jitter: float = 0.3, rng: random.Random | None = None):
        self.initial_s = initial_s
        self.max_s = max_s
        self.jitter = jitter
        self._rng = rng or random.Random()
        self._attempt = 0

    def reset(self) -> None:
        self._attempt = 0

    def next_delay(self) -> float:
        base = min(self.max_s, self.initial_s * (2 ** self._attempt))
        self._attempt += 1
        if self.jitter:
            base *= 1 + self._rng.uniform(-self.jitter, self.jitter)
        return max(0.0, base)
