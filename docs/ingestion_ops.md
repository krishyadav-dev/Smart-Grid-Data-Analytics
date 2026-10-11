# Ingestion layer — operations

## Environment used to build and test

| Item | Version |
|---|---|
| OS | macOS (Darwin 27.0, arm64). **Not run on Windows** — the PowerShell commands below were written for Windows but not executed there |
| Python | 3.14.6 |
| Broker used in tests | Eclipse Mosquitto 2.1.2 (Homebrew, native) |
| Broker in `docker-compose.yml` | `eclipse-mosquitto:2.0.20` (pinned; **not run** — Docker isn't installed on the build machine) |
| Python packages | pinned in `requirements.txt` (Layer 2 section) |

## Ports

| Component | Default | Setting |
|---|---|---|
| Modbus meter server (Layer 1) | 5020 | `--port` / `config.MODBUS_PORT` |
| MQTT broker | 1883 (bound to 127.0.0.1) | `docker-compose.yml`, `config/gateway.yaml: mqtt.port`, `SG_MQTT_PORT` |
| Listener HTTP | 8000 | `--port` |

## Start order (PowerShell, from the project root)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Broker — Docker
docker compose up -d mosquitto
#    ...or native Windows Mosquitto (https://mosquitto.org/download/):
New-Item -ItemType Directory -Force results\mosquitto-data | Out-Null
& "C:\Program Files\mosquitto\mosquitto.exe" -c config\mosquitto\mosquitto.native.conf -v

# 2. Listener (subscribes first, so nothing is missed)
Copy-Item .env.example .env      # optional
python -m src.ingestion.listener --port 8000

# 3. Gateway
python -m src.ingestion.gateway --config config\gateway.yaml

# 4. Layer 1 meter server fed by real run_timeseries output
python scripts\run_meter_server.py --port 5020 --speed 10
#    SGCC-driven load profiles (default: datasetsmall.csv; fast, light):
python scripts\run_meter_server.py --port 5020 --speed 10 --profiles sgcc
#    full SGCC calendar (~20 s and ~8 GB RAM to load):
python scripts\run_meter_server.py --port 5020 --speed 10 --profiles sgcc --sgcc-path "data\fallback\SGCC_theft_detecton_data\data set.csv"
```

The gateway and listener may start in any order; both retry with backoff. Starting them
before the meter server means the first simulated step is not missed.

Check it:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/stats
Invoke-RestMethod http://127.0.0.1:8000/api/v1/meters/m05/latest
& "C:\Program Files\mosquitto\mosquitto_sub.exe" -t "sg/v1/#" -v
```

### Windows event-loop note

aiomqtt (paho) needs `loop.add_reader`, which the default Windows Proactor loop doesn't
support. Both entry points run on `asyncio.SelectorEventLoop` on Windows:
`gateway.run_async()` and `python -m src.ingestion.listener`. Use these entry points rather
than bare `uvicorn ... :create_app` on Windows.

## Gateway settings (`config/gateway.yaml`)

| Key | Default | Notes |
|---|---|---|
| `poll_interval_s` | 0.25 | must be below half a simulated step in wall time, or the gateway logs a startup warning and produces `GAP` |
| `expected_step_s` | 900 | simulated seconds per Layer 1 step |
| `playback_speed` | 10 | informational, used only for the startup warning |
| `torn_read_retries` | 3 | |
| `backoff.initial_s / max_s / jitter` | 0.5 / 15 / 0.3 | Modbus (per meter) and MQTT reconnect |
| `buffer_max_messages` | 10000 | drop-oldest buffer while the broker is down |
| `mqtt.keepalive_s` | 10 | Last Will fires within about 1.5× this after a silent death |

CLI overrides: `--modbus-port`, `--mqtt-port`, `--poll-interval`, `--playback-speed`, `--log-level`.

## Listener settings (env vars, prefix `SG_`; see `.env.example`)

`SG_MQTT_HOST`, `SG_MQTT_PORT`, `SG_MQTT_CLIENT_ID`, `SG_MQTT_PERSISTENT_SESSION` (default
true), `SG_OUT_OF_ORDER_POLICY` (`flag`|`reject`), `SG_STALE_AFTER_S`, `SG_EXPECTED_STEP_S`,
`SG_DEDUP_WINDOW`, `SG_HISTORY_LEN` (96 = one simulated day), `SG_HUB_QUEUE_SIZE`,
`SG_LATENCY_SAMPLE_SIZE`, `SG_RATE_WINDOW_S`. `SG_LATENCY_BUDGET_MS` is intentionally unset.

## REST

`GET /health`, `GET /api/v1/meters`, `GET /api/v1/meters/{id}/latest`,
`GET /api/v1/meters/{id}/history?limit=N`, `GET /api/v1/stats`. Interactive docs: `/docs`.

## Layer 3 hand-off

* MQTT: subscribe to `sg/v1/ieee13_11kv/validated/+` (QoS 1).
* In-process: `app.state.service.hub.subscribe(maxsize=...)` returns an async iterator.
  A slow consumer only drops its own oldest messages; check `/api/v1/stats` → `hub.subscribers[].dropped`.

## Tests

```powershell
pytest tests -m "not live"            # unit tests, no broker needed
$env:SG_MOSQUITTO_BIN = "C:\Program Files\mosquitto\mosquitto.exe"
pytest tests/ingestion -m live -s      # real meter server + broker + gateway + listener
python scripts\bench_ingestion.py --speeds 900 1800 3600 7200
```

Live tests are skipped automatically when no `mosquitto` binary is found.

## Security (dev-only limitation)

No TLS and no authentication. Exposure is limited by binding only: Docker publishes
`127.0.0.1:1883`, and the native config listens on `127.0.0.1`. Don't expose port 1883 or 5020
on an untrusted network.

## Troubleshooting

| Symptom | Check |
|---|---|
| **Broker down** | `/health` shows `broker_connected: false`. The gateway logs `MQTT unavailable ... buffered=N dropped=M` and buffers up to `buffer_max_messages` (drop-oldest), then flushes on reconnect. Start the broker: `docker compose up -d` or the native command. |
| **No data** | `mosquitto_sub -t "sg/v1/#" -v`. If there are no `*/status` messages, the gateway isn't connected to the broker. If meters show `unreachable`, the meter server isn't running or the port is wrong. If they're reachable but no telemetry arrives, the meter server hasn't written its first frame (sim time 0) or playback has finished. Run `python scripts/dump_registers.py --port 5020` to see the raw registers; all zeros means no frame has been written yet. |
| **Gaps (`GAP` flags)** | `poll_interval_s` is too slow for the playback speed (look for the startup warning), or the meter was unreachable. Rule of thumb: `poll_interval_s < 900 / speed / 2`. |
| **Duplicates** | `DUPLICATE` rejects are expected after QoS 1 redelivery, gateway restarts (same `t_sim` re-read) and meter-server restarts (playback replays from the start). They go to the DLQ topic and are counted in `/api/v1/stats`. |
| **`OUT_OF_ORDER` / `ENERGY_DECREASE`** | Usually a meter-server restart replaying from 00:00 with a reset energy register. |
| **`STALE`** | The broker was down longer than `stale_after_s` and the buffer was flushed late. |
| **Listener restart lost data** | Keep `SG_MQTT_PERSISTENT_SESSION=true` and a fixed `SG_MQTT_CLIENT_ID`: the broker queues QoS 1 messages while the listener is down (measured in D7). |
