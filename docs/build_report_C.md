# Build report — Part C (broker, register-map decoding, gateway)

Environment: macOS (Darwin 27.0, arm64), Python 3.14.6, pymodbus 3.15.0, aiomqtt 2.5.1,
paho-mqtt 2.1.0, Mosquitto 2.1.2 (native, Homebrew). **Not executed on Windows or Docker**;
see Deviations.

## Step 0 gaps (top of report)

1. **Blocking: the Layer 1 meter server served only zeros** (as of 70c77e0). pymodbus 3.15
   copies register memory when the server is built. This was fixed upstream in `4fc7c50`
   (`async_setValues`), and Layer 2 is built on that version (CI-1).
2. **No simulated-time register** (most important gap per the brief). Added at fc4 28–29,
   u32 Unix seconds, written together with the measurements after energy (CI-2). The
   `NO_SIM_TIME` fallback still exists in the gateway for maps without it.
3. **Reactive power and frequency:** served by upstream `4fc7c50` (frequency is a constant
   50 Hz; reactive is u32, CI-3).
4. **Open:** `current_a` is a connected-line current, not the load's own current (CI-4).
   Absent phases are zero-filled on the wire (CI-5; Layer 2 omits them). There is no rated
   current and no DT/feeder-head meter (CI-6).

Full table: `docs/ingestion_step0.md`. Issues: `docs/contract_issues.md`.

## What was built

| Path | Purpose |
|---|---|
| `src/ingestion/schema.py`, `schemas/telemetry_v1.json` | pydantic v2 contract (`SCHEMA_VERSION = "1.0"`), enums, exported JSON Schema |
| `src/ingestion/topics.py` | topic builders, filters and parser |
| `src/ingestion/config.py` | YAML loaders (gateway, meters); listener settings appended by Part D |
| `src/ingestion/register_map.py` + `config/register_map.yaml` | table-driven decoding: blocks of ≤125 registers, address base, word order, signedness, scale, quantum rounding |
| `src/ingestion/gateway.py` | per-meter poll tasks, change detection, `GAP`, torn-read retry, retained status with LWT, backoff with jitter, drop-oldest buffer, graceful shutdown (Ctrl+C / SIGTERM) |
| `src/ingestion/backoff.py` | shared exponential backoff with jitter |
| `config/meters.yaml`, `config/gateway.yaml` | meter inventory and gateway settings |
| `docker-compose.yml`, `config/mosquitto/mosquitto.conf`, `mosquitto.native.conf` | broker: pinned image, persistence, localhost-only exposure |
| `scripts/dump_registers.py` | raw register dump (diagnostics, Step 0, C2) |
| `scripts/run_meter_server.py` | runs Layer 1 `start_meter_server` with real `run_timeseries` output (`--profiles sgcc` uses the local SGCC CSV; `--start-delay` for tests) |
| `scripts/devstack.py` | process harness for live tests and the benchmark |
| `docs/telemetry_schema.md`, `docs/ingestion_step0.md`, `docs/contract_issues.md` | docs |

## Acceptance criteria

| # | Result | Evidence (test) | Measured |
|---|---|---|---|
| C1 live | **PASS** | `tests/ingestion/test_live_day.py::test_c1_values_match_frame`, `::test_c1_every_step_published_once_in_order` | 5,376 phase values (8 meters × 96 steps × present phases × 4 quantities), plus energy. Worst absolute errors: V 0.004999, A 0.000500, kW 0.000499, kvar 0.000500, all within ½ quantum. Every step published exactly once, in order. |
| C2 live | **PASS** | `test_live_day.py::test_c2_register_map_matches_live_server`, `::test_c2_meters_yaml_matches_live_phases`, `tests/ingestion/test_config_contract.py` | No non-zero word outside the map on units 0–10, addresses 0–40. Units 1–8 present, 0 and 9–10 absent. Address base 0. Live voltage phases equal `phases_present`. `meters.yaml` equals `get_load_points()`. |
| C3 | **PASS** | `tests/ingestion/test_schema.py` (golden files in `tests/ingestion/golden/`, 15 invalid mutations, NaN/inf, schema regenerate-and-compare) | — |
| C4 live | **PASS** | `test_live_resilience.py::test_c4_meter_server_restart`, `::test_c4_broker_restart` | Over two runs: meter server killed → `m01` retained status `unreachable` after 1.25 s. Publishing resumed 1.20–1.21 s after the server restarted, same `run_id`. Broker down 4 s: telemetry flowed 0.28–0.60 s after the broker restart, no `seq` lost (buffer flushed), retained `online` re-asserted. |
| C5 | **PASS** | `test_live_resilience.py::test_c5_lwt_on_abrupt_gateway_kill` | Retained `offline` (Last Will) observed **4–5 ms** after SIGKILL (two runs). The TCP close is seen at once on localhost; the keepalive window is 10 s × 1.5 = 15 s. |
| C6 | **PASS** | `test_gateway_unit.py::test_c6_repeated_polls_publish_once`, `::test_c6_skipped_step_sets_gap` | 5 polls of one `t_sim` → 1 publish, 4 counted skips. Skipping 2 steps → `GAP`, `gap_steps=2`. |
| C7 | **PASS** | `test_gateway_unit.py::test_c7_seq_monotonic_and_new_run_id_on_restart`; live `test_c1_every_step_published_once_in_order` | — |
| C8 | **PASS** | `test_gateway_unit.py::test_c8_torn_read_retried_then_flagged`, `::test_torn_read_recovers_on_retry` | 3 retries, then `TORN_READ`; recovers on the first retry when the tear stops. |
| C9 | **PASS** | `tests/ingestion/test_static.py::test_c9_*` | Compose publishes `127.0.0.1:1883` only. The native config listens on 127.0.0.1. Persistence is on. |
| C10 | **PASS** (import part) | `test_gateway_unit.py::test_c10_gateway_does_not_import_layer1` | PowerShell commands documented in `docs/ingestion_ops.md` but **not executed** (no Windows machine). |

Additional tests: `test_gateway_unit.py::test_one_failing_unit_does_not_stop_others`,
`::test_partial_read_flag_and_missing_energy`, `::test_buffer_drop_oldest`, and
`tests/test_meter_server.py::TestSimTimeRegister`.

## Deviations from the brief and why

1. **Layer 1 was edited** (frozen per the brief; both changes approved by the project owner):
   `meter_server.py` gained the simulated-time register at fc4 28–29 on top of upstream
   `4fc7c50`, with no other address moved. `dataset_loader.py` got the SGCC profile fixes
   (CI-8 to CI-10). Every Layer 1 test still passes.
2. **Docker was not run.** The build machine has no Docker. `docker-compose.yml` pins
   `eclipse-mosquitto:2.0.20`, but all live tests used native Mosquitto 2.1.2 with the
   equivalent native config (localhost listener, persistence).
3. **Windows/PowerShell not executed.** The code uses `pathlib`, a `SelectorEventLoop` on
   Windows, and no POSIX-only commands. SIGTERM handling is POSIX-only; Ctrl+C works everywhere.
4. **Frequency is forwarded as served** (Layer 1 writes a constant 50 Hz placeholder; the brief
   allows this).
5. **Gateway status has two extra fields**, `buffer_dropped` and `unreachable_meters`
   (an additive 1.x change), to expose the drop counter in status as §5.7 asks. Status is
   republished whenever the set of unreachable meters changes.
6. **Python 3.14.6** is used; `StrEnum` and `asyncio.timeout` need 3.11 or newer.

## Open issues

* CI-4 (line current vs load current), CI-5, CI-6 and CI-7 in `docs/contract_issues.md`.
* Layer 1 uses the pymodbus 3.15 `SimDevice`/`SimData` API, so `pymodbus==3.15.0` is pinned in
  the Layer 2 section of `requirements.txt`.
* A meter-server restart replays from the first timestamp. The gateway republishes those steps,
  and the listener rejects them as `DUPLICATE` or flags them `OUT_OF_ORDER`/`ENERGY_DECREASE`.
  This is correct behaviour, but Layer 3 should know about it.
* `/docs` is listed in the upstream `.gitignore`, so these docs won't be committed unless that
  line changes.
