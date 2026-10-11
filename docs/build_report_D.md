# Build report — Part D (FastAPI listener, validation, hand-off, instrumentation)

Environment: macOS (Darwin 27.0, arm64), Python 3.14.6, FastAPI 0.142.4, uvicorn 0.54.0,
pydantic 2.13.5, pydantic-settings 2.15.0, aiomqtt 2.5.1, Mosquitto 2.1.2 (native).
Not run on Windows.

## What was built

| Path | Purpose |
|---|---|
| `src/ingestion/listener/app.py` | app factory `create_app(settings)`; the lifespan starts and stops the MQTT subscriber |
| `src/ingestion/listener/__main__.py` | `python -m src.ingestion.listener` (uvicorn; selector loop on Windows) |
| `src/ingestion/listener/mqtt_client.py` | aiomqtt subscriber: reconnect with backoff and jitter, persistent session (fixed client id, `clean_session=False`), separate outbound publisher task |
| `src/ingestion/listener/validation.py` | ordered, counted pipeline: parse → version → model → known meter → duplicate (bounded per-meter window on `(run_id, seq)` and `t_sim`) → ordering/staleness → plausibility → energy. Never alters values |
| `src/ingestion/listener/service.py` | enrichment (`ingest` block, global `ingest_seq`, per-hop latency), per-meter ring buffer, DLQ, status tracking |
| `src/ingestion/listener/hub.py` | `TelemetryHub.subscribe()` returns an async iterator; per-subscriber bounded drop-oldest queue with a drop counter. Layer 3 plugs in here |
| `src/ingestion/listener/stats.py` | counters, sliding-window per-meter rate, p50/p95/p99 |
| `src/ingestion/listener/api.py` | `/health`, `/api/v1/meters`, `/latest`, `/history?limit=`, `/api/v1/stats` (pydantic response models) |
| `src/ingestion/config.py` (appended) | `ListenerSettings` (env prefix `SG_`), `LATENCY_BUDGET_MS = None` |
| `.env.example`, `scripts/bench_ingestion.py`, `docs/ingestion_ops.md` | config, benchmark, operations |

## Acceptance criteria

| # | Result | Evidence (test) | Notes / measured |
|---|---|---|---|
| D1 | **PASS** | `tests/ingestion/test_validation.py` (every reject code and every flag; `test_every_flag_and_reject_covered` guards completeness), `::test_d1_sag_and_overcurrent_pass_unmodified` | A 0.6 pu sag (3810.51 V) and 3× overcurrent (208.5 A) are accepted with no flags and identical measurements on the validated output. Negative active power is not flagged. |
| D2 | **PASS** | `test_validation.py::test_d2_qos1_redelivery_counted_once`, `::test_duplicate_by_t_sim_across_runs` | 1 accepted, 3 `DUPLICATE`. |
| D3 | **PASS** | `tests/ingestion/test_hub_api.py::test_d3_two_consumers_both_receive_and_slow_one_drops`; live `test_live_day.py::test_d3_validated_topic_receives_accepted` | Both fast consumers received 100/100. The slow consumer (queue 5) dropped 95 while `publish` never awaited. |
| D4 | **PASS** | `test_hub_api.py::test_rest_shapes_and_404`, `::test_health_false_when_broker_down`, `::test_d4_stats_match_known_mix` | 100 sent = 90 accepted + 5 `DUPLICATE` + 5 `MALFORMED_JSON`; DLQ 10. |
| D5 live | **PASS** | `test_live_day.py::test_d5_end_to_end_accounting`, `::test_d5_per_meter_order_preserved` | One simulated day: 768 published = 768 received = 768 accepted (0 rejects). Every `(meter, t_sim)` delivered exactly once, per-meter order preserved. |
| D6 | **PASS** | `results/bench_20261008T063448Z.json` + `.csv` (synthetic profiles), `results/bench_20261008T063820Z.json` + `.csv` (SGCC profiles, 3600×) | No budget asserted; `LATENCY_BUDGET_MS = null`. |
| D7 | **MEASURED** | `tests/ingestion/test_live_resilience.py::test_d7_listener_restart[persistent]` and `[clean]` | Listener stopped for 5 s while 40 QoS 1 messages were published. **Persistent session: 40/40 delivered after restart. Clean session: 0/40.** |
| D8 | **PASS** | `tests/ingestion/test_static.py::test_d8_no_database_analytics_or_kafka_code` | No DB, analytics or Kafka imports under `src/ingestion`. |

## Benchmark summary (synthetic profiles, gateway `poll_interval_s = 0.25` default)

One simulated day = 96 steps × 8 meters = 768 messages per speed. Same host for every
component. Latencies in ms:

| Speed | s/step (wall) | Delivered | Loss | GAP msgs (steps) | Dups | Throughput msg/s | poll→pub p50/p95/p99 | pub→recv p50/p95/p99 | total p50/p95/p99 (max) |
|---|---|---|---|---|---|---|---|---|---|
| 900× | 1.000 | 768/768 | 0 | 0 | 0 | 7.92 | 0.244 / 0.865 / 1.127 | 0.843 / 1.311 / 1.557 | 1.144 / 1.920 / 2.188 (2.362) |
| 1800× | 0.500 | 768/768 | 0 | 0 | 0 | 15.55 | 0.246 / 1.082 / 1.267 | 0.773 / 1.316 / 1.527 | 1.149 / 1.939 / 2.351 (2.753) |
| 3600× | 0.250 | 768/768 | 0 | 0 | 0 | 30.10 | 0.284 / 0.777 / 0.968 | 0.819 / 1.361 / 1.571 | 1.228 / 1.729 / 2.145 (4.421) |
| **7200×** | 0.125 | **441/768** | **327** | 322 (322) | 0 | 31.82 | 0.217 / 0.459 / 0.536 | 0.702 / 1.195 / 1.362 | 0.929 / 1.537 / 1.646 (1.729) |

* **Limit:** with the default 0.25 s poll, the chain is lossless up to 3600× (one step per
  0.25 s) and samples about every other step at 7200×. Of the 327 missed steps at 7200×, 322
  carry `GAP`/`gap_steps`. The other 5 are the very first step (00:00) of m01, m03, m04, m06
  and m07: no
  earlier reading exists, so no gap can be detected. Raise the poll rate
  (`poll_interval_s < 900 / speed / 2`) for faster playback.
* Validated-topic subscriber hop (listener receive → a second MQTT subscriber): p50 0.52–0.56 ms,
  p99 0.97–1.20 ms across speeds.
* SGCC-driven run (fixed loader, `datasetsmall.csv`, 2024-01-01, 3600×,
  `results/bench_20261008T063820Z.*`): 768/768, 0 loss, no flags or rejects, total
  p50/p95/p99 = 1.511/2.169/2.570 ms.
* Metadata in the JSON: Python, OS, package and Mosquitto versions, speeds, poll interval,
  `run_timeseries` solve time (2.9 s for one day).

## Deviations from the brief and why

1. **`service.py` added** (not in the suggested layout) to keep validation, enrichment and
   hand-off transport-agnostic, so the REST and validation tests run without a broker.
2. **Listener-side `GAP`** is detected in addition to the gateway's flag, so messages lost
   between gateway and listener also surface. `gap_steps` takes the maximum of the two.
3. **`OUT_OF_ORDER` added to `RejectReason`**, used only when `out_of_order_policy=reject`
   (§4.4 makes the policy configurable but lists no reject code for it).
4. **Timestamps serialise with a `Z` suffix** (pydantic default) rather than `+00:00`. Both are
   ISO 8601 / RFC 3339, and both are accepted on input.
5. **Uvicorn's `--factory` path on Windows:** use `python -m src.ingestion.listener`, which
   selects the selector loop that aiomqtt needs.
6. Not executed on Windows or Docker (see `build_report_C.md`).

## Open issues

* In-memory state (dedup window, ring buffers, `ingest_seq`) resets when the listener restarts.
  With a persistent session, messages queued during downtime are still delivered and validated.
  A redelivered message whose original was accepted before the restart would be accepted again
  (the dedup window doesn't persist; no database, by design).
* The rate in `/stats` is a sliding wall-clock window (`SG_RATE_WINDOW_S`, 60 s), so it reads
  0 once traffic stops.
* `/health.ok` reflects broker connectivity only. Gateway state and last-message age are
  reported for the caller to judge.
