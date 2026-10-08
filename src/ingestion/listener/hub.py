"""In-process fan-out of validated telemetry (Layer 3 plugs in here).

``TelemetryHub.subscribe()`` returns an async iterator with its own bounded
queue.  ``publish`` never blocks: when a subscriber's queue is full its oldest
message is dropped and its ``dropped`` counter increments, so a slow consumer
can never stall ingestion.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator, Optional


class Subscription:
    def __init__(self, hub: "TelemetryHub", maxsize: int, name: str):
        self._hub = hub
        self.name = name
        self.maxsize = maxsize
        self._dq: deque[dict] = deque()
        self._event = asyncio.Event()
        self.dropped = 0
        self.delivered = 0
        self.closed = False

    def _offer(self, msg: dict) -> None:
        if len(self._dq) >= self.maxsize:
            self._dq.popleft()
            self.dropped += 1
        self._dq.append(msg)
        self._event.set()

    def qsize(self) -> int:
        return len(self._dq)

    def __aiter__(self) -> AsyncIterator[dict]:
        return self

    async def __anext__(self) -> dict:
        while not self._dq:
            if self.closed:
                raise StopAsyncIteration
            self._event.clear()
            await self._event.wait()
        self.delivered += 1
        return self._dq.popleft()

    def close(self) -> None:
        self.closed = True
        self._event.set()
        self._hub._unsubscribe(self)


class TelemetryHub:
    def __init__(self, default_maxsize: int = 1000):
        self.default_maxsize = default_maxsize
        self._subs: list[Subscription] = []
        self.published = 0

    def subscribe(self, maxsize: Optional[int] = None, name: str = "") -> Subscription:
        sub = Subscription(self, maxsize or self.default_maxsize,
                           name or f"sub{len(self._subs) + 1}")
        self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def publish(self, msg: dict) -> None:
        self.published += 1
        for sub in list(self._subs):
            sub._offer(msg)

    def close(self) -> None:
        for sub in list(self._subs):
            sub.close()

    def stats(self) -> dict:
        return {
            "published": self.published,
            "subscribers": [
                {"name": s.name, "queued": s.qsize(), "maxsize": s.maxsize,
                 "delivered": s.delivered, "dropped": s.dropped}
                for s in self._subs
            ],
        }
