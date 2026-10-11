"""``python -m src.ingestion.listener`` — run the listener with uvicorn.

Uses a selector event loop on Windows (aiomqtt cannot use the Proactor loop).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import uvicorn

from ..config import ListenerSettings
from .app import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ingestion.listener")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mqtt-port", type=int, default=None)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {}
    if args.mqtt_port:
        overrides["mqtt_port"] = args.mqtt_port
    app = create_app(ListenerSettings(**overrides))
    config = uvicorn.Config(app, host=args.host, port=args.port,
                            log_level=args.log_level, loop="asyncio")
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(server.serve())
    else:
        asyncio.run(server.serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
