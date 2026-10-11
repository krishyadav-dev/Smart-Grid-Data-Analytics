# Ingestion & Broker Layer (Layer 2): Agent Build Brief

**How to use:** save as `docs/ingestion_layer_brief.md`. Give it to two agents, one per part:
**Part C** (broker, register-map decoding, gateway) and **Part D** (FastAPI listener,
validation, hand-off, instrumentation). Each agent reads the whole file but works only
on its own scope. Use Plan mode first, and review the Plan Artifact against §7 before
letting the agent execute. Part C can run first; Part D can start as soon as Part C has
delivered milestone C0 (the schema module), because §4 already fixes the contract.

## 1. Goal

Build roadmap Layer 2. Data flows one way:

```
Layer 1 meter server (Modbus TCP, 8+ units)
   -> [Part C] gateway: poll, decode, publish
   -> Mosquitto MQTT broker
   -> [Part D] FastAPI listener: validate, enrich, hand off
   -> validated stream: MQTT "validated" topic + in-process hub + REST (latest/history/stats)
```

The output is a clean, timestamped, quality-flagged telemetry stream that the analytics
layer (Layer 3) can consume later. Layer 2 does **not** analyse the grid.

## 2. Ground rules (both parts)

- Layer 1 is **frozen**. Do not edit `feeder_model.py`, `meter_server.py`,
  `dataset_loader.py`, `theft_injector.py` or their tests. If something in Layer 1 blocks
  you, write it in `docs/contract_issues.md` and say so in your report. Never patch
  around it silently.
- Plan first; wait for approval. Tick a criterion in §7 only after running the test that
  proves it, and paste the test name in your report.
- Items marked **live** must run against the real meter server fed by the real
  `feeder_model.run_timeseries` output. An injected or mocked frame does not count.
- If this brief conflicts with what Layer 1 really does, stop and report. Do not invent
  registers, fields, units or values. Do not fabricate numbers in reports.
- Pin every dependency version actually used in `requirements.txt` (one shared file;
  append, do not reformat). Record Python version, OS and the Mosquitto image/version in
  the docs.
- Windows development machine: use `pathlib`, no POSIX-only commands, and document run
  commands for PowerShell.
- Anomalous-but-physical data (sags, overcurrent, imbalance) is the *signal* of this
  project. Layer 2 must **never** drop, clip, smooth or "repair" it. Layer 2 rejects only
  broken data (malformed, impossible, duplicate) and flags suspect data.
- Do not set a latency budget. Measure and report; the project owner will set the budget
  from your numbers.

## 3. Step 0: read Layer 1 and verify (both parts, before planning)

Read `docs/data_source_layer_brief.md`, `docs/feeder_model_contract.md`,
`docs/meter_register_map.md` (if it exists; otherwise the `meter_server.py` docstring),
and `docs/contract_issues.md`. Start the meter server with real `run_timeseries`
output and dump every register of every unit with a plain `pymodbus` client
(Part C writes `scripts/dump_registers.py`; Part D may use it once it exists).

Write `docs/ingestion_step0.md` with a table: item, present in the register map (yes/no),
address/function code if yes. Items to check:

| Item | Needed for |
|---|---|
| Voltage per phase, current per phase (the load's own current), active power, energy import | core telemetry |
| Reactive power and/or apparent power | transformer thermal pillar (kVA loading) |
| Frequency | placeholder only (constant 50 Hz is acceptable) |
| Simulated-time register(s) | `t_sim`; timing, latency and ground-truth alignment |
| Address base, word order, signedness, scale, unit, per register | table-driven decoding |
| Meter list: unit id, load id, phases present, role (consumer / DT / feeder head) | `meters.yaml` |
| Rated current and base voltage per meter | later fault rules (fields may be null) |

For any missing item: say so, continue with the documented fallback (§4.5), and put it
at the top of your report. A missing simulated-time register is the most important gap.

## 4. Shared contract (both parts build to this)

### 4.1 Components and ownership

| Path | Owner |
|---|---|
| `config/mosquitto/mosquitto.conf`, `docker-compose.yml` | C |
| `src/ingestion/schema.py` (pydantic v2 models, enums, `SCHEMA_VERSION = "1.0"`), `schemas/telemetry_v1.json` (exported) | C (milestone C0) |
| `src/ingestion/topics.py`, `register_map.py`, `gateway.py`, `config/register_map.yaml`, `config/meters.yaml`, `scripts/dump_registers.py`, `docs/telemetry_schema.md` | C |
| `src/ingestion/listener/*` (app, mqtt client, validation, hub, api, stats), `scripts/bench_ingestion.py`, `docs/ingestion_ops.md` | D |
| `src/ingestion/config.py` | C creates; D appends |
| `tests/ingestion/test_*` | each owner for its modules |

### 4.2 Topics (MQTT 3.1.1, UTF-8, lowercase)

`feeder_id` = `ieee13_11kv`; `meter_id` = `m` + zero-padded unit id (`m05`).

| Topic | Publisher | QoS | Retained |
|---|---|---|---|
| `sg/v1/{feeder_id}/{meter_id}/telemetry` | gateway | 1 | no |
| `sg/v1/{feeder_id}/gateway/status` (LWT = offline) | gateway | 1 | yes |
| `sg/v1/{feeder_id}/{meter_id}/status` | gateway | 1 | yes |
| `sg/v1/{feeder_id}/validated/{meter_id}` | listener | 1 | no |
| `sg/v1/{feeder_id}/dlq` | listener | 0 | no |

QoS 1 means duplicates are possible, so the listener must deduplicate (§6).

### 4.3 Telemetry payload (JSON, compact, UTF-8)

Field names and units are canonical engineering units, not raw register units. Convert
using `register_map.yaml` scales. Optional measurement keys are present only if the
register map serves them. Phases not present on a meter are omitted, never zero-filled.

```json
{
  "schema_version": "1.0",
  "feeder_id": "ieee13_11kv",
  "meter_id": "m05",
  "unit_id": 5,
  "load_id": "load_671_abc",
  "meter_role": "consumer",
  "run_id": "<uuid4 of this gateway run>",
  "seq": 0,
  "t_sim": "2024-01-01T00:15:00",
  "t_poll_utc": "2026-10-08T10:15:00.123456+00:00",
  "t_pub_utc": "2026-10-08T10:15:00.125000+00:00",
  "measurements": {
    "voltage_v": {"a": 6006.19, "b": 6056.69, "c": 6008.68},
    "current_a": {"a": 29.5, "b": 28.9, "c": 29.1},
    "active_power_kw": {"a": 154.04, "b": 154.04, "c": 154.04, "total": 462.12},
    "reactive_power_kvar": {"a": 88.02, "b": 88.02, "c": 88.02, "total": 264.06},
    "energy_import_kwh": 1234.567,
    "frequency_hz": 50.0
  },
  "quality": {"flags": [], "gap_steps": 0}
}
```

(The numbers are illustrative only. Real values come from the meters.)

The **validated** message is the same document plus:

```json
"ingest": {"t_recv_utc": "...", "ingest_seq": 1234,
           "latency_ms": {"poll_to_pub": 1.9, "pub_to_recv": 3.4, "total": 5.3}}
```

and `quality.flags` merged with any flags the listener adds.

**Status payloads:** gateway `{"state": "online|offline|degraded", "run_id": "...",
"t_utc": "..."}`; meter `{"state": "reachable|unreachable", "since_utc": "...",
"last_error": "..."}`.
**DLQ payload:** `{"reason": "<code>", "detail": "...", "topic": "...",
"t_recv_utc": "...", "payload_head": "<first 2 KB>"}`.

### 4.4 Reject codes and quality flags

Reject (to DLQ, not forwarded): `MALFORMED_JSON`, `SCHEMA_INVALID` (includes NaN/inf),
`UNSUPPORTED_SCHEMA_VERSION`, `UNKNOWN_METER`, `DUPLICATE`.

Flags (forwarded with the flag set): `GAP` (t_sim jumped by more than one expected step;
`gap_steps` says how many), `NO_SIM_TIME`, `OUT_OF_ORDER`, `STALE` (wall-clock age above
`stale_after_s`), `ENERGY_DECREASE`, `SUSPECT_RANGE` (physically impossible values only,
for example negative voltage or current), `PARTIAL_READ` (some registers failed),
`TORN_READ` (blocks disagreed on `t_sim` after retries).

Policies are configuration, not code: `out_of_order_policy = flag|reject` (default
`flag`), `stale_after_s`, `expected_step_s` (default 900), dedup window size.

### 4.5 Time semantics

- `t_sim`: simulated time from the meter's time register, timezone-naive ISO 8601
  (Layer 1 convention). The gateway never invents it. **Fallback if no time register
  exists:** `t_sim = null` plus flag `NO_SIM_TIME`; report it as the top unresolved item.
- `t_poll_utc`, `t_pub_utc`, `t_recv_utc`: wall-clock UTC, timezone-aware. Gateway and
  listener run on the same machine, so the clocks are comparable. State this assumption
  in the docs.
- `seq` is a per-meter counter starting at 0 for each `run_id`. A gateway restart creates
  a new `run_id`.
- **Consistency rule (torn reads):** the meter server updates registers while the gateway
  reads. Read the simulated-time register at the start and end of each unit's read
  sequence; if they differ, retry (default 3), then publish with flag `TORN_READ`.

### 4.6 Config files

`config/meters.yaml`: per meter `meter_id, unit_id, load_id, role, phases_present,
v_base_ln_v (nullable), i_rated_a (nullable)`. A test must check it against
`feeder_model.get_load_points()` (load ids and phases) and against the live server.

`config/register_map.yaml`: per register `name, function_code, address, width_registers,
word_order, signed, scale, unit, phase, obis (conceptual only)`, plus a global
`address_base`. The decoder reads this file; **no address, scale or unit is hard-coded**
in Python. A test must read every mapped address from a live server and fail on any
non-zero register that is not in the map.

## 5. Part C: broker, decoding, gateway

C0 (first, small): `schema.py`, exported `schemas/telemetry_v1.json`, `topics.py`,
and a skeleton `docs/telemetry_schema.md`, so Part D can start.

1. **Broker.** `docker-compose.yml` with a pinned Eclipse Mosquitto 2.x image and an
   explicit `mosquitto.conf` (Mosquitto 2.x refuses remote and anonymous connections
   unless configured). Bind the published port to `127.0.0.1` only; enable persistence
   so retained status survives a restart. Dev-only security: no TLS, no authentication.
   State that limitation in the docs. Also document the native Windows Mosquitto
   alternative.
2. **Libraries (check the installed versions' real APIs; do not rely on remembered
   snippets).** Async Modbus client from the pinned `pymodbus` (argument names such as
   `slave` vs `device_id` differ between 3.x versions). MQTT: `aiomqtt` or
   `paho-mqtt` 2.x (2.x requires an explicit callback API version). On Windows,
   `aiomqtt` does not work with the default Proactor event loop; handle it (selector loop
   policy or a threaded client) and document the choice.
3. **Decode.** Table-driven from `register_map.yaml`. Read contiguous blocks per unit and
   function code, respecting the 125-register limit. Use the map's address base. Apply
   word order, signedness and scale. Convert to canonical units (§4.3).
4. **Poll loop.** `poll_interval_s` configurable (default 0.25). Publish only when
   `t_sim` has advanced (change detection); count skipped duplicate polls. If `t_sim`
   jumps by more than `expected_step_s`, set `GAP` with `gap_steps`. Log a startup
   warning if `poll_interval_s` is too large to catch every step at the configured
   playback speed.
5. **Publish.** Topics and QoS per §4.2, `retain=False` for telemetry, `run_id` and `seq`
   per §4.5.
6. **Status.** Gateway status with a Last Will (`offline`); per-meter status
   `reachable/unreachable` published on change only.
7. **Resilience.** Reconnect to Modbus and to the broker with exponential backoff and
   jitter (configurable). A failing unit must not stop the other units. While the broker
   is down, keep a bounded in-memory buffer (drop-oldest, with a drop counter in logs
   and status). Graceful shutdown on Ctrl+C.
8. **CLI.** `python -m src.ingestion.gateway --config config/gateway.yaml`;
   `scripts/dump_registers.py` for diagnostics.
9. **Docs.** `docs/telemetry_schema.md`: every field, unit, flag, example message,
   versioning policy (additive changes keep `1.x`; breaking changes bump to `2.0`).

## 6. Part D: listener, validation, hand-off

1. **App.** FastAPI app factory with lifespan-managed MQTT subscriber (reconnect with
   backoff). Subscribe to the telemetry and status topics. Settings via
   `pydantic-settings` with env vars; commit a `.env.example`.
2. **Validation pipeline** (ordered, each step increments a counter): parse JSON ->
   check schema version -> validate against the pydantic model -> known meter ->
   duplicate check (key `(meter_id, run_id, seq)` and `(meter_id, t_sim)`, bounded
   window) -> ordering and staleness -> plausibility -> energy-decrease check. Rejects go
   to the DLQ topic with a reason code; flags are added and the message continues. Never
   alter measurement values.
3. **Enrichment.** Add the `ingest` block (§4.3): `t_recv_utc`, a global `ingest_seq`,
   and per-hop latencies computed from the gateway timestamps.
4. **Hand-off (no database).** Publish each accepted message to the `validated` topic;
   keep a per-meter ring buffer (configurable length, default one simulated day);
   provide an in-process `TelemetryHub` with `subscribe()` returning an async iterator,
   bounded queues, a drop-oldest policy and a drop counter, so a slow consumer never
   blocks ingestion. This hub is what Layer 3 will plug into.
5. **REST** (response models in pydantic): `GET /health` (broker connected, last
   message age, gateway state), `GET /api/v1/meters`,
   `GET /api/v1/meters/{meter_id}/latest`,
   `GET /api/v1/meters/{meter_id}/history?limit=`, `GET /api/v1/stats` (counters by
   reject reason and flag, per-meter message rate, latency p50/p95/p99).
6. **Benchmark.** `scripts/bench_ingestion.py` runs the whole chain (real meter server
   with real `run_timeseries`, gateway, broker, listener) for one simulated day at
   several playback speeds (include one speed that causes gaps, to find the limit). It
   writes `results/bench_<timestamp>.json` and a per-message CSV, with p50/p95/p99 per
   hop, throughput, loss, duplicates, gaps, and metadata (versions, OS, speeds).
   `LATENCY_BUDGET_MS` exists in config as `None`; do not fill it.
7. **Docs.** `docs/ingestion_ops.md`: start order, ports, env vars, PowerShell commands,
   troubleshooting (broker down, no data, gaps, duplicates).

## 7. Acceptance criteria (verify yourself before declaring done)

**Part C**
- [ ] **C1 (live).** For one simulated day, decoded gateway values equal the
  `run_timeseries` frame for every unit and mapped register, within register quantisation.
- [ ] **C2 (live).** `register_map.yaml` matches the live server; any unmapped non-zero
  register fails the test. `meters.yaml` matches `get_load_points()`.
- [ ] **C3.** Golden-message tests validate against `schemas/telemetry_v1.json`; invalid
  samples fail; the exported schema file is regenerated and compared in a test.
- [ ] **C4 (live).** Kill and restart the meter server mid-run: the meter turns
  `unreachable` on its retained status topic, the gateway does not crash, and publishing
  resumes without a gateway restart. Same for a broker restart.
- [ ] **C5.** Killing the gateway abruptly makes the broker publish `offline` on the
  retained gateway status within the keepalive window (state the measured time).
- [ ] **C6.** Repeated polls of the same `t_sim` publish once; a skipped step publishes
  with `GAP` and the right `gap_steps`.
- [ ] **C7.** `seq` is monotonic per meter per `run_id`; a restart creates a new `run_id`.
- [ ] **C8.** A torn read (forced in a test) is retried, then flagged `TORN_READ`.
- [ ] **C9.** The mosquitto config does not enable anonymous remote access; the published
  port is bound to localhost.
- [ ] **C10.** The gateway imports nothing from `feeder_model` or `pandapower` at
  runtime; run commands work in PowerShell.

**Part D**
- [ ] **D1.** A validation matrix test triggers every reject code and every flag with
  crafted messages. A test shows a **0.6 pu voltage sag and a 3x overcurrent pass
  through unmodified** (no rejection, no clipping).
- [ ] **D2.** QoS 1 redelivery of the same message is counted once (`DUPLICATE`).
- [ ] **D3.** Accepted messages appear on the `validated` topic and through the hub;
  two hub consumers both receive them; a deliberately slow consumer does not block
  ingestion and its drop counter increments.
- [ ] **D4.** REST tests: shapes, 404 for an unknown meter, `/health` false when the
  broker is down; `/stats` counters match a known injected mix (for example 100 sent:
  90 valid, 5 duplicates, 5 malformed).
- [ ] **D5 (live).** End to end, one simulated day: accepted = published - known rejects;
  per-meter order preserved; zero unexplained loss.
- [ ] **D6.** Benchmark results file exists with the required metrics and metadata. No
  budget is asserted.
- [ ] **D7.** Listener restart with a persistent MQTT session: report (measured) whether
  missed QoS 1 messages are delivered afterwards.
- [ ] **D8.** No database, analytics or Kafka code anywhere in Layer 2.

## 8. Non-goals

TimescaleDB/PostgreSQL writes, any analytics (fault detection, theft detection,
forecasting), the dashboard, Telegram alerts, Kafka, TLS/authentication (documented
limitation), the DLMS/COSEM wire protocol, and any IS 15959 compliance claim.

## 9. Suggested layout

```
src/ingestion/{__init__,config,schema,topics,register_map,gateway}.py
src/ingestion/listener/{__init__,app,mqtt_client,validation,hub,api,stats}.py
config/{meters.yaml,register_map.yaml,gateway.yaml,mosquitto/mosquitto.conf}
schemas/telemetry_v1.json
scripts/{dump_registers,bench_ingestion}.py
tests/ingestion/
docs/{ingestion_layer_brief,ingestion_step0,telemetry_schema,ingestion_ops}.md
docker-compose.yml  .env.example
results/            (benchmark outputs)
```

## 10. Reports

Each part writes `docs/build_report_<C|D>.md` with: what was built; the result of every
criterion with its test name; deviations from this brief and why; open issues; for
Part C, the Step 0 gaps; for Part D, the benchmark summary table. Then a final chat
report under 15 lines.

## References

1. Eclipse Mosquitto documentation (configuration, persistence, listeners).
2. MQTT 3.1.1 specification (QoS, retained messages, Last Will).
3. `pymodbus` and `aiomqtt`/`paho-mqtt` documentation for the pinned versions.
4. Layer 1: `docs/data_source_layer_brief.md`, `docs/feeder_model_contract.md`,
   `docs/meter_register_map.md`.
