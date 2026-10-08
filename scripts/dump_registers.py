"""Dump every register of every unit on a running meter server (diagnostics).

Plain ``pymodbus`` client; no knowledge of the register map is assumed.  Each
address is read one register at a time so that unmapped / out-of-range
addresses show up as exceptions instead of failing a whole block.

PowerShell::

    python scripts/dump_registers.py --port 5020 --units 1-10 --max-addr 40
"""

from __future__ import annotations

import argparse
import asyncio
import json


def _parse_units(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


async def dump(host: str, port: int, units: list[int], max_addr: int) -> dict:
    from pymodbus.client import AsyncModbusTcpClient

    client = AsyncModbusTcpClient(host, port=port, timeout=2, retries=0)
    await client.connect()
    if not client.connected:
        raise SystemExit(f"cannot connect to {host}:{port}")
    result: dict = {}
    try:
        for unit in units:
            unit_res: dict = {}
            for fc, fn in ((3, client.read_holding_registers),
                           (4, client.read_input_registers)):
                regs: dict[int, int | str] = {}
                for addr in range(0, max_addr + 1):
                    try:
                        rr = await fn(addr, count=1, device_id=unit)
                        regs[addr] = ("exc" if rr.isError() else rr.registers[0])
                    except Exception as exc:  # timeout = unit absent
                        regs[addr] = f"err:{type(exc).__name__}"
                unit_res[f"fc{fc}"] = regs
            result[unit] = unit_res
    finally:
        client.close()
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5020)
    ap.add_argument("--units", default="1-8")
    ap.add_argument("--max-addr", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args(argv)
    res = asyncio.run(dump(args.host, args.port, _parse_units(args.units), args.max_addr))
    if args.json:
        print(json.dumps(res, indent=1))
        return
    for unit, fcs in res.items():
        for fc, regs in fcs.items():
            row = " ".join(f"{a}:{v}" for a, v in regs.items() if v != "exc")
            print(f"unit {unit:>2} {fc}: {row}")


if __name__ == "__main__":
    main()
