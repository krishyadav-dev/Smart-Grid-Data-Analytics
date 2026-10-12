"""Poll one meter of the Layer 1 Modbus server and print its readings (schema 2.0).

Decoding is table-driven from config/register_map.yaml, so this script follows
the register map instead of hard-coding addresses.

    python read_meter.py            # unit 1 on 127.0.0.1:5020
"""

import time

from pymodbus.client import ModbusTcpClient

from src.ingestion.register_map import RegisterMap

UNIT_ID = 1
rm = RegisterMap.load()

print("Connecting to Modbus Server at 127.0.0.1:5020...")
client = ModbusTcpClient("127.0.0.1", port=5020)
if not client.connect():
    raise SystemExit("Failed to connect to the meter server.")
print(f"Connected. Monitoring unit {UNIT_ID} (Ctrl+C to stop).\n")

try:
    for i in range(10):
        raw: dict[int, dict[int, int]] = {}
        for b in rm.blocks():
            rr = client.read_input_registers(address=rm.wire_address(b.start),
                                             count=b.count, device_id=UNIT_ID)
            if rr.isError():
                raise SystemExit(f"read error: {rr}")
            raw.setdefault(b.function_code, {}).update(
                {b.start + k: w for k, w in enumerate(rr.registers)})
        values = rm.decode_block_values(raw)
        meas = rm.build_measurements(values, ["a", "b", "c"])
        print("-" * 75)
        print(f"Poll {i + 1}/10  t_sim {rm.decode_time(values)}  "
              f"event counter {rm.decode_event_counter(values)}")
        for key, val in meas.items():
            print(f"  {key:20s}: {val}")
        time.sleep(2)
except KeyboardInterrupt:
    pass
finally:
    client.close()
