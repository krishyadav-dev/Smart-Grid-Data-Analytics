# Telemetry schema v1.0

* Source of truth: `src/ingestion/schema.py` (pydantic v2).
* Exported JSON Schema: `schemas/telemetry_v1.json`. Regenerate with
  `python -m src.ingestion.schema`. `tests/ingestion/test_schema.py::test_exported_schema_is_up_to_date`
  fails if the file and the models differ.
* Encoding: compact JSON, UTF-8. `NaN` and `Infinity` are invalid.

## Topics (MQTT 3.1.1)

`feeder_id = ieee13_11kv`; `meter_id = "m" + zero-padded unit id` (`m05`).

| Topic | Publisher | QoS | Retained |
|---|---|---|---|
| `sg/v1/{feeder_id}/{meter_id}/telemetry` | gateway | 1 | no |
| `sg/v1/{feeder_id}/gateway/status` (Last Will = `offline`) | gateway | 1 | yes |
| `sg/v1/{feeder_id}/{meter_id}/status` | gateway | 1 | yes |
| `sg/v1/{feeder_id}/validated/{meter_id}` | listener | 1 | no |
| `sg/v1/{feeder_id}/dlq` | listener | 0 | no |

QoS 1 delivers at least once, so the listener removes duplicates.

## Telemetry fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string `1.x` | `1.0` for this version |
| `feeder_id` | string | feeder identifier |
| `meter_id` | string `m\d{2,}` | must match the topic |
| `unit_id` | int 1–247 | Modbus unit id |
| `load_id` | string | Layer 1 load id (`load_<bus>_<phases>`) |
| `meter_role` | `consumer` \| `dt` \| `feeder_head` | Layer 1 only has `consumer` |
| `run_id` | UUID | new for every gateway start |
| `seq` | int ≥ 0 | per meter, starts at 0 for each `run_id` |
| `t_sim` | naive ISO 8601 or `null` | simulated time from the meter's time register; `null` + `NO_SIM_TIME` if unavailable |
| `t_poll_utc` | aware ISO 8601 | wall-clock UTC when the meter's read sequence completed |
| `t_pub_utc` | aware ISO 8601 | wall-clock UTC when the message was handed to MQTT; includes any buffering while the broker was down |
| `measurements` | object | see below |
| `quality.flags` | list of flags | see below |
| `quality.gap_steps` | int ≥ 0 | simulated steps missed before this one |

The gateway and the listener run on the same host, so wall clocks are comparable and
per-hop latencies are meaningful.

### `measurements`: canonical units, already scaled

A key is present only if the register map serves it and the read succeeded. Phases a meter
does not have are omitted, never zero-filled. `total` is the sum of the present phases and is
computed only when every present phase was read and the map has no total register.

| Key | Unit | Shape | Register (fc4 unless noted) |
|---|---|---|---|
| `voltage_v` | V (line-to-neutral) | `{a,b,c}` | 0–5, u32 ×0.01 |
| `current_a` | A | `{a,b,c}` | 6–11, u32 ×0.001 (Layer 1 serves a line current; see contract issue CI-4) |
| `active_power_kw` | kW | `{a,b,c,total}` | 12–17, u32 ×0.001; `total` = sum of present phases |
| `reactive_power_kvar` | kvar | `{a,b,c,total}` | 20–25, u32 ×0.001; `total` from register 26–27 |
| `energy_import_kwh` | kWh, cumulative | number | fc3 0–1, u32 ×0.001 |
| `frequency_hz` | Hz | number | 18–19, u32 ×0.01 (Layer 1 writes a constant 50 Hz) |
| (`t_sim`) | — | — | 28–29, u32 Unix seconds |

Values are rounded to the register quantum (for example 0.01 V), so float artefacts never
reach the payload.

## Quality flags

| Flag | Set by | Meaning |
|---|---|---|
| `GAP` | gateway, listener | `t_sim` advanced by more than one `expected_step_s` (default 900 s); `gap_steps` = missed steps |
| `NO_SIM_TIME` | gateway | no time register, or its read failed; `t_sim = null` |
| `OUT_OF_ORDER` | listener | `t_sim` (or `seq` within a run) went backwards. With `out_of_order_policy=reject` the message is rejected instead |
| `STALE` | listener | `t_recv_utc − t_poll_utc > stale_after_s` (default 30 s) |
| `ENERGY_DECREASE` | listener | `energy_import_kwh` lower than the meter's previous value |
| `SUSPECT_RANGE` | listener | physically impossible value: negative voltage, current or energy, or non-positive frequency. **Sags, overcurrent and imbalance are never flagged or altered.** |
| `PARTIAL_READ` | gateway | at least one register block failed; its measurements are omitted |
| `TORN_READ` | gateway | the time register differed across the read sequence after `torn_read_retries` retries |

## Reject codes (go to the DLQ, not forwarded)

`MALFORMED_JSON`, `SCHEMA_INVALID` (includes NaN/inf and topic/payload mismatch),
`UNSUPPORTED_SCHEMA_VERSION`, `UNKNOWN_METER` (unknown id, or unit/load mismatch),
`DUPLICATE` (same `(meter_id, run_id, seq)` or same `(meter_id, t_sim)` within the window),
and `OUT_OF_ORDER` (only when the policy is `reject`).

## Example: gateway telemetry (real message from a live run)

```json
{"schema_version":"1.0","feeder_id":"ieee13_11kv","meter_id":"m05","unit_id":5,"load_id":"load_671_abc","meter_role":"consumer","run_id":"6e3d0e9f-baf8-44ea-9ce2-d9362d67586e","seq":3,"t_sim":"2024-01-01T00:45:00","t_poll_utc":"2026-10-08T06:40:19.445666Z","t_pub_utc":"2026-10-08T06:40:19.447051Z","measurements":{"voltage_v":{"a":6006.07,"b":6056.62,"c":6008.57},"current_a":{"a":69.597,"b":34.211,"c":64.86},"active_power_kw":{"a":154.162,"b":154.162,"c":154.162,"total":462.486},"reactive_power_kvar":{"a":88.092,"b":88.092,"c":88.092,"total":264.277},"energy_import_kwh":462.275,"frequency_hz":50.0},"quality":{"flags":[],"gap_steps":0}}
```

## Validated message

The same document with `quality.flags` merged with the listener's flags, plus the block below
(values are illustrative; measured latencies are in `docs/build_report_D.md`):

```json
"ingest": {"t_recv_utc": "2026-10-08T06:00:33.881900Z", "ingest_seq": 1234,
           "latency_ms": {"poll_to_pub": 1.172, "pub_to_recv": 1.518, "total": 2.69}}
```

## Status and DLQ payloads

* Gateway status: `{"state": "online|offline|degraded", "run_id": "...", "t_utc": "..."}`,
  plus the additive fields `buffer_dropped` and `unreachable_meters`. It is republished whenever
  the state or the set of unreachable meters changes. `degraded` means at least one meter is
  unreachable or the publish buffer has dropped messages.
* Meter status: `{"state": "reachable|unreachable", "since_utc": "...", "last_error": "..."}`,
  published only on change.
* DLQ: `{"reason": "<code>", "detail": "...", "topic": "...", "t_recv_utc": "...", "payload_head": "<first 2 KB>"}`.

## Versioning policy

* **Additive** changes, such as a new optional measurement or a new optional status field, keep
  `1.x` and bump the minor version. The listener accepts only versions in
  `SUPPORTED_SCHEMA_VERSIONS`, and models forbid unknown fields. Roll out a `1.x` change by
  updating the listener before the gateway.
* **Breaking** changes (renamed or removed fields, unit or meaning changes) bump to `2.0` and
  new topic prefixes (`sg/v2/...`).
