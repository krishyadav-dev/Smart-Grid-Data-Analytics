"""MQTT subscriber for the listener (aiomqtt), with reconnect + backoff.

Subscribes to telemetry and status topics, feeds ``ListenerService`` and
publishes validated / DLQ messages through a separate outbound task so QoS 1
acknowledgements never block the receive loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiomqtt

from .. import topics
from ..backoff import Backoff
from ..config import ListenerSettings
from .service import ListenerService, Outbound

logger = logging.getLogger("sg.listener.mqtt")


class MqttIngest:
    def __init__(self, settings: ListenerSettings, service: ListenerService):
        self.s = settings
        self.service = service
        self._out: asyncio.Queue[Outbound] = asyncio.Queue(maxsize=100_000)
        self.out_dropped = 0
        self.connects = 0
        self._task: Optional[asyncio.Task] = None

    def _client(self) -> aiomqtt.Client:
        s = self.s
        return aiomqtt.Client(
            hostname=s.mqtt_host, port=s.mqtt_port, identifier=s.mqtt_client_id,
            keepalive=s.mqtt_keepalive_s,
            clean_session=not s.mqtt_persistent_session, timeout=10,
        )

    async def _publisher(self, client: aiomqtt.Client) -> None:
        while True:
            ob = await self._out.get()
            try:
                await client.publish(ob.topic, ob.payload, qos=ob.qos, retain=ob.retain)
            except aiomqtt.MqttError:
                # connection gone: keep the message for the next session
                self._requeue(ob)
                raise

    def _enqueue(self, ob: Outbound) -> None:
        try:
            self._out.put_nowait(ob)
        except asyncio.QueueFull:
            self.out_dropped += 1

    def _requeue(self, ob: Outbound) -> None:
        self._enqueue(ob)

    async def run(self) -> None:
        s = self.s
        backoff = Backoff(s.backoff_initial_s, s.backoff_max_s, s.backoff_jitter)
        while True:
            try:
                async with self._client() as client:
                    self.connects += 1
                    backoff.reset()
                    await client.subscribe(topics.telemetry_filter(s.feeder_id), qos=1)
                    await client.subscribe(topics.status_filter(s.feeder_id), qos=1)
                    self.service.broker_connected = True
                    logger.info("connected to %s:%d (persistent_session=%s)",
                                s.mqtt_host, s.mqtt_port, s.mqtt_persistent_session)
                    pub = asyncio.create_task(self._publisher(client))
                    try:
                        async for message in client.messages:
                            payload = message.payload
                            if isinstance(payload, str):
                                payload = payload.encode()
                            elif payload is None:
                                payload = b""
                            elif not isinstance(payload, (bytes, bytearray)):
                                payload = str(payload).encode()
                            for ob in self.service.handle(message.topic.value, bytes(payload)):
                                self._enqueue(ob)
                            if pub.done():
                                pub.result()   # surface publisher failure
                    finally:
                        pub.cancel()
                        await asyncio.gather(pub, return_exceptions=True)
            except aiomqtt.MqttError as exc:
                self.service.broker_connected = False
                delay = backoff.next_delay()
                logger.warning("broker unavailable (%s); retry in %.1fs", exc, delay)
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.service.broker_connected = False
                raise

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.run(), name="mqtt-ingest")
        return self._task

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
