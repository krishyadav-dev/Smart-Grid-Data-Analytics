"""Run the Layer 1 meter server fed by real ``feeder_model.run_timeseries`` output.

Layer 1 is frozen; this script only *calls* it.  Used for Step 0, the live
tests and the benchmark.

PowerShell::

    python scripts/run_meter_server.py --port 5020 --speed 60
    python scripts/run_meter_server.py --profiles sgcc   # SGCC-derived load profiles
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

logger = logging.getLogger("run_meter_server")


def build_frame(start: str, end: str, profiles: str = "synthetic",
                sgcc_path: str | None = None) -> pd.DataFrame:
    """Return the ``run_timeseries`` frame (optionally driven by SGCC profiles)."""
    from src.data_source import feeder_model

    prof = None
    if profiles == "sgcc":
        from src.data_source import config as l1_config, dataset_loader
        if sgcc_path is None:
            # Local copy only (no download).  The small file is the default:
            # loading the full 163 MB file takes ~20 s and ~8 GB RAM
            # (pass --sgcc-path for it).
            for cand in (l1_config.SGCC_SMALL_FALLBACK_PATH, l1_config.SGCC_FALLBACK_PATH):
                if cand.exists():
                    sgcc_path = str(cand)
                    break
            else:
                raise FileNotFoundError(
                    f"No SGCC CSV at {l1_config.SGCC_SMALL_FALLBACK_PATH.parent}")
        sgcc = dataset_loader.load_sgcc(sgcc_path)
        prof = dataset_loader.sample_load_profiles(
            feeder_model.get_load_points(), sgcc, start=start, end=end)
    return feeder_model.run_timeseries(start=start, end=end, profiles=prof)


async def serve(frame: pd.DataFrame, host: str, port: int, speed: float,
                start_delay_s: float = 0.0) -> None:
    from src.data_source import meter_server as ms

    if start_delay_s <= 0:
        await ms.start_meter_server(host=host, port=port, timeseries_data=frame,
                                    playback_speed=speed)
        return
    # Same composition as start_meter_server, but the first frame is written
    # only after `start_delay_s`, so clients can connect before step 0.
    from pymodbus.server import ModbusTcpServer

    devices_list, _, meters = ms.build_server_context()
    server = ModbusTcpServer(context=devices_list, address=(host, port))

    async def _delayed() -> None:
        await asyncio.sleep(start_delay_s)
        logger.info("starting playback")
        await ms._update_loop(server.context, meters, frame, speed)
        logger.info("playback finished")

    asyncio.get_running_loop().create_task(_delayed())
    await server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5020)
    ap.add_argument("--speed", type=float, default=60.0,
                    help="playback speed (sim seconds per wall second)")
    ap.add_argument("--start", default="2024-01-01 00:00",
                    help="SGCC dates outside 2014-2016 map onto the SGCC calendar")
    ap.add_argument("--end", default=None,
                    help="default: 23:45 on the start day")
    ap.add_argument("--profiles", choices=["synthetic", "sgcc"], default="synthetic")
    ap.add_argument("--sgcc-path", default=None)
    ap.add_argument("--frame-out", default=None,
                    help="write the run_timeseries frame to this CSV (ground truth)")
    ap.add_argument("--frame-in", default=None,
                    help="serve a previously written frame instead of re-solving")
    ap.add_argument("--start-delay", type=float, default=0.0,
                    help="seconds to serve zeros before writing the first frame")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.end is None:
        args.end = str(pd.Timestamp(args.start).normalize() + pd.Timedelta(hours=23, minutes=45))
    if args.frame_in:
        frame = pd.read_csv(args.frame_in, parse_dates=["timestamp"])
    else:
        frame = build_frame(args.start, args.end, args.profiles, args.sgcc_path)
    if args.frame_out:
        frame.to_csv(args.frame_out, index=False)
    logger.info("Serving %d rows, %d timestamps on %s:%d at %.1fx",
                len(frame), frame["timestamp"].nunique(), args.host, args.port, args.speed)
    try:
        asyncio.run(serve(frame, args.host, args.port, args.speed, args.start_delay))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
