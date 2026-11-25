"""python -m src.dashboard [--host 127.0.0.1] [--port 8100]"""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .settings import DashboardSettings


def main(argv=None) -> None:
    s = DashboardSettings()
    ap = argparse.ArgumentParser(description="Smart Grid dashboard (Layer 4)")
    ap.add_argument("--host", default=s.host, help="bind address (default 127.0.0.1: local only)")
    ap.add_argument("--port", type=int, default=s.port)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(s.model_copy(update={"host": args.host, "port": args.port})),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
