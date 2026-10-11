# Ingestion Layer — Step 0: Layer 1 register verification

**How verified:** `scripts/run_meter_server.py` started the Layer 1 meter server with the real
`feeder_model.run_timeseries("2024-01-01 00:00", "2024-01-01 23:45")` frame (96 steps), and
`scripts/dump_registers.py` read every address (0–32) of units 0–10 one register at a time with
a plain `pymodbus` client. Environment: macOS (Darwin 27, arm64), Python 3.14.6,
pymodbus 3.15.0. Layer 1 docs read: `docs/data_source_layer_brief.md` and the
`meter_server.py` docstring. The repository has no `docs/meter_register_map.md` or
`docs/feeder_model_contract.md`.

## History

1. **At commit 70c77e0, every register of every unit read 0 for the whole run.** pymodbus 3.15
   copies device memory when `ModbusTcpServer` is built, and the server kept writing to the
   discarded copies (CI-1).
2. Upstream commit **4fc7c50** fixed this by writing through `server.context.async_setValues`.
   It also moved the map to address 0 and added frequency, reactive power and total reactive
   power.
3. Layer 2 added the **simulated-time register** at fc4 28–29 (CI-2, approved by the project
   owner).

The dump below is from that final code.

## Item checklist (on-wire addresses, `address_base` = 0; every value is 2 registers, big-endian word order, u32)

| Item | Present | Address / function code | Notes |
|---|---|---|---|
| Voltage per phase | yes | fc4 0–1, 2–3, 4–5 | V×100 |
| Current per phase | yes, **but not the load's own current** | fc4 6–7, 8–9, 10–11 | mA. The largest current of any line connected to the bus (CI-4) |
| Active power per phase | yes | fc4 12–13, 14–15, 16–17 | W |
| Frequency | yes (placeholder) | fc4 18–19 | Hz×100, always 50.00 |
| Reactive power per phase | yes | fc4 20–21, 22–23, 24–25 | var, **unsigned** (negative would clamp to 0; CI-3) |
| Total reactive power | yes | fc4 26–27 | var, sum of the three phases |
| Apparent power | no | — | derivable downstream from P and Q |
| Simulated-time register | **added** | fc4 28–29 | Unix seconds of the naive `t_sim` read as UTC; 0 before the first frame |
| Energy import | yes | fc3 0–1 | Wh, saturating |
| Unit ids present | 1–8 | — | units 0 and 9+ do not respond |
| Phases present per meter | yes | — | absent phases are zero-filled on the wire; Layer 2 omits them using `meters.yaml` |
| Role | consumer only | — | no DT or feeder-head meter in Layer 1 |
| Base voltage per meter | derived | — | bus `vn_kv` from `build_ieee13_network()` ÷ √3 |
| Rated current per meter | no | — | `i_rated_a: null` |

## Live dump excerpt (t_sim = 2024-01-01 00:45)

```
unit 2 (load_645_b, phase b only)
  fc4: 2:9 3:14578 (V_b) 9:22957 (I_b) 14:1 15:2535 (P_b) 19:5000 (50.00 Hz) 23:50053 (Q_b) 27:50053 (Q_tot) 28:26002 29:2828 (t_sim)
  fc3: 0:1 1:2504 (energy)
unit 5 (load_671_abc)
  fc4: 0:9 1:10783 2:9 3:15838 4:9 5:11033 ... 19:5000 ... 26:4 27:2133 28:26002 29:2828
units 0 and 9: no response
```

`26002 << 16 | 2828` = 1704069900 s = 2024-01-01T00:45:00. Every non-zero word lies inside
`config/register_map.yaml` (asserted live by `test_c2_register_map_matches_live_server`).

## Meter list (matches `get_load_points()` row order)

| unit | meter_id | load_id | phases | v_base_ln_v |
|---|---|---|---|---|
| 1 | m01 | load_634_abc | a b c | 732.79 (XFM-1 secondary, 1.269 kV LL) |
| 2 | m02 | load_645_b | b | 6350.85 |
| 3 | m03 | load_646_b | b | 6350.85 |
| 4 | m04 | load_652_a | a | 6350.85 |
| 5 | m05 | load_671_abc | a b c | 6350.85 |
| 6 | m06 | load_675_abc | a b c | 6350.85 |
| 7 | m07 | load_692_c | c | 6350.85 |
| 8 | m08 | load_611_c | c | 6350.85 |

## Gaps that remain open

1. `current_a` is a line current, not the meter's own load current (CI-4).
2. Frequency is a constant placeholder. There is no rated current and no DT/feeder-head meter (CI-6).
3. Meters report *true* values; theft-manipulated values are not served (CI-7).
