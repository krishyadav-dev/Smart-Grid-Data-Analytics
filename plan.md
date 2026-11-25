# Dashboard and Map Plan (Layer 4)

Status: **P0 report mode built** (backend `src/dashboard/`, web app `web/`); P0.5 roads waits for a Mapbox token; P1-P5 not started.  Branch `claude/eager-davinci-utukxv`.

Built so far:

- Report API (§4.2) plus detail views over the git-ignored files (`src/dashboard/detail.py`): `/api/results/faults/bursts/{event_id}`, `/api/results/theft/inspection`, `/api/results/grid/daily`, `/api/results/grid/day/{day}`, `/api/results/thermal/daily`, `/api/results/thermal/day/{day}`, `/api/results/unbalance/daily`.  Each one reproduces a committed summary in `tests/dashboard/test_detail_api.py` (zone accuracy 0.85, true theft per DT-day, Jokar fold tp/fp, per-meter VU max, thermal peaks).
- Theft pages use injection 2 (`results/layer3/ntl_inj2`, Stage 4) when present, else injection 1; Zheng uses `sgcc_full` when present.
- Web app: grid map (day slider, DT metric), faults, energy and theft, transformer thermal, power quality, system; light and dark themes; the map falls back to an SVG drawing of the same layers when no Mapbox token is set.

Decisions so far:

| Decision | Choice |
|---|---|
| Map provider | **Mapbox** (Mapbox GL JS v3 + Mapbox Directions / Map Matching / Geocoding APIs) |
| Location | **Delhi**; substation (bus 650) anchored at the ERA5 grid point already used for air temperature, **28.6 N, 77.2 E** (`scripts/download_era5.py`) |
| Frontend | React + Vite + TypeScript, `react-map-gl` (Mapbox adapter), ECharts for charts |
| Backend | New FastAPI service `src/dashboard/` (FastAPI, uvicorn, aiomqtt, pydantic are already pinned in `requirements.txt`) |
| Live LSTM forecast | **Open** (see §12).  Default: report-only in P0-P2, model export in P3 |

---

## 1. Goal

A web dashboard for the simulated Delhi 11 kV feeder with two modes in one app:

- **Report mode**: reads the frozen Layer 3 results.  Works from a fresh clone, with no running stack.  For demos, reviews and the viva.
- **Live mode**: connects to the running stack (`scripts/devstack.py`) and shows the feeder as it is replayed at 1x to 3600x: meter readings, fault alarms located on the map, unbalance and transformer temperatures.

The dashboard adds **presentation and wiring**, not new analytics.  Every number it shows comes from code that already exists in `src/analytics/` or from a results file that code wrote.  The one optional new model (distance to fault, P5) must be scored before it is shown.

---

## 2. What already exists

| Layer | Code | What it gives the dashboard |
|---|---|---|
| L1 Simulation | `src/data_source/` | IEEE 13-bus feeder converted to Indian 11 kV (Stage B: lengths x (11/4.16)^2, 50 Hz, three-phase laterals, 13 distribution transformers (DTs), OLTC).  23 Modbus meters: m01-m09 `bus_virtual`, m10 feeder head, m11-m23 DT meters (`config/meters.yaml`).  Fault bursts (half-cycle RMS), 8,020 households from SGCC, theft injection. |
| L2 Ingestion | `src/ingestion/` | Gateway (Modbus -> MQTT), Mosquitto, listener (validates against `schemas/telemetry_v2.json`).  REST: `/health`, `/api/v1/meters`, `/api/v1/meters/{id}/latest`, `/api/v1/meters/{id}/history` (last 96 only), `/api/v1/stats`.  MQTT: `sg/v2/{feeder}/validated/{meter}`, `sg/v2/{feeder}/validated_event/{meter}`, gateway and meter status topics. |
| L3 Faults | `src/analytics/fault_engine.py` | Rule-based IS 17036 engine with a live mode (`LiveEngine`): detection, type (SLG/LL/LLG/3PH), zone A-G, residual voltage, duration, clearing vs IDMT expected time. |
| L3 Unbalance | `src/analytics/unbalance.py` | Voltage unbalance factor from line-to-line voltages. |
| L3 Theft | `src/analytics/ntl_engine.py`, `zheng_cnn.py` | Energy balance (head -> billed + technical losses + theft + offline + meter error), Jokar SVM consumer detector, Zheng Wide & Deep CNN on real SGCC labels. |
| L3 Thermal | `src/analytics/thermal.py`, `forecast.py` | IS 2026-7 transformer thermal model (hot spot, top oil, loss of life), day-ahead load forecast (LSTM vs persistence vs seasonal naive). |

**Gaps the dashboard has to close:**

1. No push channel to a browser (no WebSocket or SSE).
2. No storage: the listener keeps 96 readings per meter in memory.
3. Fault alarms and events are not exposed by any API; `LiveEngine` only runs inside `scripts/run_layer3_live.py`.
4. Fault bursts cannot be fetched after the fact.
5. No coordinates: the IEEE 13-bus feeder is a textbook network with line lengths only.
6. No saved model weights (see §6).

---

## 3. Data: sources, flow, what is in git

### 3.1 Inputs

| Input | Source | In git? | How to get it |
|---|---|---|---|
| Household consumption + theft labels | SGCC dataset (real smart-meter data) | No (`data/` is ignored) | Downloaded on first run, or manual fallback (README "Dataset Preparation") |
| Delhi load shape | Delhi SLDC 2024-25 | **Yes** `data/delhi_sldc_2024_25_reference.csv` | - |
| Air temperature | ERA5, 28.6 N 77.2 E | No | `python scripts/download_era5.py` (free CDS key) |
| Feeder | IEEE 13-bus, Stage B | **Yes** (code) | - |

### 3.2 Regenerated outputs (git-ignored, regenerable from the master seed)

| Output | Script | Used by the dashboard for |
|---|---|---|
| `results/year_2024/` meter frames, DT loading, households, AMI export | `scripts/run_year.py` (~1.4 h) | Live replay, time slider, map colours by step |
| `results/faults_2024*/fault_events.parquet` | `scripts/run_faults.py` | True fault bus (report: hit/miss on map) |
| `results/layer3/fault/summary_events.parquet` (committed) and `live/*_events.parquet` | `scripts/run_layer3_fault.py`, `run_layer3_live.py` | Predicted zone per event |
| `results/layer3/ntl/dt_day.parquet`, `feeder_day.parquet`, `jokar_test_flags.npz` | `scripts/run_layer3_ntl.py` | Daily theft per DT, household inspection ranking |
| `results/layer3/thermal/**/*.npz` (reference trajectories, test forecasts) | `scripts/run_layer3_thermal.py` | Hot-spot curves, forecast vs reference |

### 3.3 Committed summaries (report mode works from these alone)

`results/layer3/fault/summary.json`, `fault/live/summary.json`, `ntl/summary.json`, `ntl/inputs.json`, `thermal/summary.json`, `sgcc/summary.json`, `unbalance_scenarios/summary.json`, `results/bench_*.json`.

### 3.4 Flow

```
SGCC + SLDC + ERA5 ─► run_year.py (power flow every 15 min, theft injected) ─► results/year_2024/
                                         │
               ┌─────────────────────────┴───────────────────────────┐
               ▼ live path                                           ▼ report path
meter_server (Modbus replay 1x-3600x)                     Layer 3 scripts ─► results/layer3/**
  ─► gateway ─► Mosquitto ─► listener (validate)                     │
                     │ validated/+ , validated_event/+               │
                     ▼                                               ▼
             src/dashboard/ (subscribe, run engines, store)  ◄── reads JSON / parquet / npz
                     │ REST + WebSocket
                     ▼
             web/ (React + Mapbox GL + ECharts)
```

The live path sees only what a control room would see (validated meter readings), never the simulation's ground truth.  Ground truth is used only on report pages, and is labelled as such.

---

## 4. Architecture

### 4.1 New files

```
config/
  geo_anchor.yaml            # anchor lat/lon, bearing per segment, DT offsets (hand-edited)
  geo_layout.geojson         # generated by scripts/build_geo_layout.py (committed)
scripts/
  build_geo_layout.py        # builds geo_layout.geojson (schematic or Mapbox-routed)
  export_lstm.py             # P3: retrain the frozen LSTM config and torch.save it
src/dashboard/
  __main__.py                # python -m src.dashboard --port 8100
  app.py                     # FastAPI app, static web build, CORS for dev
  settings.py                # pydantic-settings: SG_DASH_*, MAPBOX_TOKEN, MQTT, listener URL
  geo.py                     # loads + validates geo_layout.geojson
  results.py                 # loaders that shape results JSON into chart-ready views
  live/
    subscriber.py            # aiomqtt: validated/+, validated_event/+, status topics
    engines.py               # LiveEngine, unbalance, thermal stepping
    state.py                 # latest per meter, per DT, open events, alarms
    store.py                 # SQLite: readings, bursts, events, alarms
    ws.py                    # WebSocket fan-out
web/
  package.json, vite.config.ts, src/{api,map,pages,components}/
tests/dashboard/
  test_geo_layout.py, test_results_api.py, test_live_engines.py, test_ws.py
```

**Why a separate service** and not routes inside the listener: L2 stays unchanged (its contract tests and latency benchmarks still hold), and this is the same pattern `scripts/run_layer3_live.py` already uses.  It talks to MQTT, not to the listener's process.

### 4.2 REST API

| Endpoint | Mode | Returns |
|---|---|---|
| `GET /api/config` | both | Mapbox public token, feeder id, mode (live available?) |
| `GET /api/geo/layout` | both | `geo_layout.geojson` |
| `GET /api/meta` | both | Meters, zones, feeder tree, DT ratings (from `config/*.yaml`) |
| `GET /api/results/faults` | report | Detection, type confusion, zone accuracy (overall / per zone / per type / per Rf), live-replay latency |
| `GET /api/results/faults/events` | report | Per event: true bus (if fault set present), predicted zone, type, residual, clearing |
| `GET /api/results/theft` | report | Energy identity, Sankey values, per-DT daily estimates, lambda sweep, Jokar and Zheng metrics |
| `GET /api/results/thermal` | report | Per-DT peaks, events at 98/120/140 C, loss of life, forecast RMSE table, bootstrap |
| `GET /api/results/unbalance` | report | Per-meter VU max per scenario severity |
| `GET /api/results/ingestion` | report | Benchmarks from `results/bench_*.json` |
| `GET /api/live/meters`, `/api/live/meters/{id}/history?from&to` | live | From the SQLite store (not limited to 96) |
| `GET /api/live/events`, `/api/live/events/{id}`, `/api/live/bursts/{event_id}` | live | Engine events and the bursts behind them |
| `GET /api/live/health` | live | Proxies listener `/health` and `/api/v1/stats` |
| `WS /ws` | live | Messages below |

Each results loader returns `{"available": false, "how": "<script to run>"}` when a git-ignored file is missing, so pages degrade instead of failing.

### 4.3 WebSocket messages

```json
{"type": "reading",   "meter_id": "m15", "t_sim": "...+05:30", "v": {...}, "i": {...}, "p_kw": {...}, "flags": []}
{"type": "alarm",     "meter_id": "m18", "onset": "...", "latency_ms": 19.4}
{"type": "event",     "id": 12, "ftype": "LLG", "zone": "F", "zone_set": ["dt_675_1", ...], "residual_pct": 41.2, "clearing_ms": 310, "t_expected_clear_s": 0.29}
{"type": "unbalance", "dt": "dt_611_1", "t_sim": "...", "vuf_pct": 2.3}
{"type": "thermal",   "dt": "dt_634_1", "t_sim": "...", "theta_h_c": 87.1, "theta_o_c": 61.0, "aging": 0.4}
{"type": "link",      "meter_id": "m21", "state": "unreachable"}
{"type": "gateway",   "state": "degraded"}
```

The map updates Mapbox feature-state by id, so the geometry is sent once and only values stream.

---

## 5. Map (Mapbox)

### 5.1 Placing the feeder in Delhi

The feeder has lengths but no coordinates, so it is given an illustrative Delhi placement.  The map says so in its legend: "Illustrative placement of the IEEE 13-bus test feeder; not a real DISCOM network."

Segment lengths come from the Stage B network (`feeder_model.build_network("B")`, `net.line.length_km`), never typed by hand.  Expected values (IEEE ft x 0.3048 x 6.992):

| Segment | m | Segment | m |
|---|---|---|---|
| 650-632 | ~4,262 | 671-684 | ~639 |
| 632-633 | ~1,066 | 671-680 | ~2,131 |
| 632-645 | ~1,066 | 684-652 | ~1,705 |
| 645-646 | ~639 | 684-611 | ~639 |
| 632-670 | ~1,421 | 692-675 | ~1,066 |
| 670-671 | ~2,841 | 633-634 | transformer (0) |
| | | 671-692 | switch (0) |

Longest path 650-632-671-684-652: about 10.9 km, a normal size for an urban 11 kV feeder.

`scripts/build_geo_layout.py` has two modes:

1. **`--mode schematic`** (no token, no network): starts at the anchor, walks the feeder tree in `config/zones.yaml` using the bearings in `config/geo_anchor.yaml` and the model lengths.  Straight lines at true length.  This version is committed first so the map works for everyone.
2. **`--mode mapbox`** (needs `MAPBOX_TOKEN`, run on your machine): for each segment, the schematic end points become waypoints for the **Mapbox Directions API** (walking/driving profile).  The returned road geometry replaces the straight line; the end bus is moved along the route so the drawn length equals the model length.  Hand-drawn paths can instead be snapped with the **Map Matching API**.  The anchor can be set by name with the **Geocoding API** (`--anchor "Connaught Place, New Delhi"`).

Note: `api.mapbox.com` is blocked from the cloud session where this plan was written, so mode 2 runs locally.  Mode 1 needs no network.

Each feature carries ids that match the rest of the repo: `bus` (`"632"`), `load_id` (`"dt_675_2"`), `meter_id` (`"m19"`), `zone` (`"F"`), `segment` (`"692-675"`).

**Test** (`tests/dashboard/test_geo_layout.py`):

- Every bus in `feeder_tree` and every DT in `meters.yaml` has exactly one feature.
- Every drawn segment length is within 10% of the Stage B model length.
- Zone membership matches `config/zones.yaml`.

### 5.2 Other features

- **Zones A-G**: line and bus buffers from `zones.yaml`.
- **DT service areas**: one polygon per DT around its marker, area scaled by household count (`results/layer3/ntl/inputs.json`: e.g. 926 on `dt_634_1`, 296 on `dt_652_1`).
- **Households** (optional, theft page): synthetic points scattered inside the service area, seeded, one per household id.

### 5.3 Layers

| Layer | Drawn as | Data |
|---|---|---|
| Base | Mapbox `streets-v12` / `satellite-streets-v12` / `dark-v11` toggle | Mapbox |
| Feeder lines | Width by current, colour by loading % of ampacity | Live readings or year cache |
| Substation + buses | Symbols; head meter KPIs in a popup | m10 |
| DTs | Circle sized by kVA, colour by the selected metric: voltage %, loading % (`i_rated_a`), VUF %, hot-spot C | Live or report |
| Faults (live) | Pulsing zone buffer, rings on DTs scaled by dip depth, deepest-dip set outlined, popup with type, residual, duration, clearing vs IDMT | `event` messages |
| Faults (report) | Predicted zone vs true bus, green hit / red miss, heatmap of all test faults, filters by type and Rf class | `summary_events.parquet` + fault set |
| Theft | Service areas shaded by estimated theft kWh/day; household dots by Jokar rank | `dt_day.parquet`, `jokar_test_flags.npz` |
| Thermal | DT colour by hot-spot C; alerts at 98 / 120 / 140 C | Thermal engine or reference npz |
| Meter health | Unreachable meters greyed, quality-flag badges | Status topics, quality flags |
| Weather chip | ERA5 ambient C at the current step | ERA5 cache |
| Time slider | Scrub 2024 (report mode) | Year cache, if present |

The single-line diagram (SVG from `feeder_tree`) stays as a second view; selecting an element in one selects it in the other.

### 5.4 Fault location: what the marker means

- The engine outputs a **zone**, not a point.  The map highlights the whole zone, never a pin.
- Zone E (671, 680, 692) cannot be split: 671-692 is a zero-impedance switch.  The map shows all three.
- Zone accuracy from the frozen test half: 85% overall vs 70% for the deepest-dip baseline; per zone A 2/4, B 4/4, C 21/24, D 4/6, E 20/26, F 9/9, G 25/27.  The report page shows these next to the map.
- A point estimate ("1.8 km from 671") needs a new impedance-based distance-to-fault model (P5).  It is not shown until it is scored on the test half like the zone result.

### 5.5 Token handling

- `.env` (already git-ignored): `MAPBOX_TOKEN=pk....` (public token, URL-restricted to `localhost` and the demo host).  Add `MAPBOX_TOKEN=` to `.env.example`.
- The browser gets it from `GET /api/config`.  No secret (`sk.`) token is needed.
- Without a token the app falls back to a blank style with the feeder layers only and a notice, so tests and CI still work.

---

## 6. Model integration

No trained model weights are saved anywhere in the repo: the LSTM, the Zheng CNN and the Jokar SVMs are trained in memory, scored once, and only metrics and outputs are written.  The fault engine, unbalance, thermal model and energy balance need no weights.

| Model | Type | Weights saved? | How the dashboard uses it | Work needed | Phase |
|---|---|---|---|---|---|
| Fault engine | Rules (IS 17036), frozen params `config/layer3_params.yaml` | n/a | `LiveEngine.on_burst(doc)` per `validated_event` message -> alarm; `LiveEngine.events()` -> finalised event with zone | None; reuse the wiring in `scripts/run_layer3_live.py` | P2 |
| Flat +/-10 % baseline | Rules | n/a | Shown beside engine alarms for comparison | None | P2 |
| Unbalance | Formula | n/a | `vuf_pct_from_ll` on each 15-min DT reading | None | P3 |
| Thermal (IS 2026-7) | Physics | n/a | `thermal.simulate` stepped per 15 min with live DT currents and ERA5 ambient; state kept per DT | Keep `State` between steps; initial state = steady state at first reading | P3 |
| Energy balance | Formula + `runpp_3ph` per step for technical losses | n/a | Daily roll-up per DT and feeder | Run `ltech` in a background worker, or read `dt_day.parquet` in report mode | P3 |
| Load forecast LSTM | Residual LSTM 1x64, 3 seeds, frozen in `config/layer3_thermal_params.yaml` | **No** | Day-ahead K_p forecast -> forecast hot-spot for the next 24 h | `scripts/export_lstm.py`: retrain the frozen config on the same training days and seeds, `torch.save` to `results/layer3/thermal/models/` (git-ignored), check RMSE against `summary.json`, then CPU inference once per simulated day | P3 (if chosen) |
| Jokar SVM | Per-household SVMs | **No**; per-household flags saved in `jokar_test_flags.npz` | Inspection ranking per DT area (Jul-Dec) | Read the saved flags; no live retraining | P0 (report) |
| Zheng CNN | Wide & Deep CNN on real SGCC labels | **No** | Validation panel only | None.  It needs 1,034 days per consumer and the simulated year has 366, so it cannot score feeder households | P0 (report) |

**Honesty rules on screen:**

- Jokar: ungated macro precision about 2% (h1-h4).  Shown as "inspect first", never as "thief".
- LSTM: hot-spot RMSE is not better than persistence (bootstrap CI includes 0).  Shown with persistence beside it.
- Zheng: test AUC 0.74 (3 seeds, 0.72-0.75) vs 0.78 in the paper.
- Every report number shows its window (tuning vs test half, Jan-Jun vs Jul-Dec) and is marked "post-hoc" where the summary says so.

---

## 7. Pages

1. **Grid overview (map)**: map + KPI row (head kW / kvar / PF, simulated time IST, playback speed, active alarms, gateway state) + alarm ticker.
2. **Meter explorer**: per-phase V, I, P, Q, PF, energy for any meter; quality flags; DT vs feeder head comparison.
3. **Fault events**: live alarm feed and event table; event drawer with half-cycle RMS burst plots per meter and phase, zone on the map.  Report tab: detection (100/100, 0 quiet-window alarms), 4-class confusion matrix (LLG 68% is the weak spot), zone accuracy breakdowns, live latency (alarm p95 about 35 ms at 900x).
4. **Power quality**: VUF heatmap (meter x time) with the IS 17036 limit; unbalance scenario results.
5. **Energy and theft**: Sankey of the energy identity (closes to -0.08 kWh), per-DT estimated vs true daily theft, lambda sensitivity, Jokar inspection list, Zheng panel.
6. **Transformer thermal**: hot-spot and top-oil per DT, events at 98/120/140 C, loss of life, forecast RMSE at 1/6/24 h (LSTM vs persistence vs seasonal naive), Annex E acceptance badge.
7. **System health**: broker / gateway / meter link state, message rates, latency p50/p95/p99 per hop, rejects, DLQ, flags, hub subscriber drops, benchmark history.

---

## 8. Phases

| Phase | Deliverables | Done when |
|---|---|---|
| **P0 Report mode + map** | `src/dashboard/` skeleton, `/api/config`, `/api/geo/layout`, `/api/meta`, `/api/results/*`; `scripts/build_geo_layout.py` (schematic mode) and committed `config/geo_layout.geojson`; web shell with Mapbox map and pages 3-6 report tabs | `python -m src.dashboard` + `npm run build` serve a working report from a fresh clone; geo layout test passes; works without a token (fallback style) |
| **P0.5 Roads** | `--mode mapbox` run locally with your token; commit the routed GeoJSON | Length test still passes with road geometry |
| **P1 Live feed** | MQTT subscriber, SQLite store, WebSocket; pages 1, 2, 7 live | With `devstack.py` at 900x the map and meter charts update every step; reconnects after a broker restart |
| **P2 Live faults** | `LiveEngine` in the service; alarm and event messages; burst API and viewer; zone highlight | Replaying `week_2024-02-05` gives the same events as `results/layer3/fault/live/week_2024-02-05.json` (same rule as `run_layer3_live.py`: live == batch) |
| **P3 All engines live** | Unbalance and thermal stepping; daily theft roll-up; time slider; optional LSTM export + live forecast | Thermal stepping on the replayed year matches the batch reference trajectory (`results/layer3/thermal/**` npz) to within 0.1 K at every step (same code, same inputs; tolerance to confirm in P3) |
| **P4 Hardening** | Bind to 127.0.0.1 by default, playback controls through devstack, contract tests against `schemas/telemetry_v2.json`, Playwright smoke test of every page | CI green; smoke test screenshots for each page |
| **P5 Optional** | Impedance-based distance-to-fault estimate from head fault current (`faults.py` sequence-impedance code) | Scored on the test half; error distribution shown before any point marker appears |

---

## 9. Testing

- `tests/dashboard/test_geo_layout.py`: ids, zone membership, lengths within 10%.
- `tests/dashboard/test_results_api.py`: every `/api/results/*` returns numbers that match the source JSON (e.g. zone accuracy 0.85, energy identity residual), and `available: false` when a git-ignored file is missing.
- `tests/dashboard/test_live_engines.py`: feed recorded burst documents through the service path and compare events with `run_batch` (live == batch).
- `tests/dashboard/test_ws.py`: message schemas, slow-client drop behaviour (same rule as `TelemetryHub`: drop oldest, never block ingestion).
- Web: TypeScript strict, ESLint, Playwright smoke test per page (P4).
- The existing suite (`pytest tests/`) must stay green; L2 code is not modified.

---

## 10. Setup the dashboard will document

```bash
pip install -r requirements.txt
```

```bash
cd web && npm install && npm run build
```

```bash
python -m src.dashboard --port 8100
```

Live mode additionally:

```bash
python scripts/devstack.py
```

Full data (once, for the time slider and live replay):

```bash
python scripts/download_era5.py
```

```bash
python scripts/run_year.py
```

```bash
python scripts/run_faults.py
```

---

## 11. Risks and limits

- **Placement is illustrative.**  The feeder is a test network placed in Delhi; the legend says so.
- **Mapbox costs.**  Map loads and API calls are free up to Mapbox's free-tier limits; routing runs once (P0.5), not per page load.
- **Regeneration time.**  `run_year.py` takes about 1.4 h; live mode and the time slider need it.  Report mode does not.
- **torch in the dashboard.**  Only needed if the live LSTM is chosen (P3); otherwise the service has no torch dependency.
- **Technical-loss power flows** for the live theft roll-up cost one `runpp_3ph` per step; run them off the event loop.

---

## 12. Open decisions

1. **Live LSTM forecast**: add `scripts/export_lstm.py` and run the forecast live (adds torch to the service), or show forecasts from the saved test results only?
2. **Anchor point**: keep 28.6 N 77.2 E (matches ERA5), or name a locality for the Geocoding API?
3. **Demo host**: local only, or deployed somewhere (affects token URL restrictions and binding)?

---

## 13. Repo housekeeping noticed while planning

The README is out of date: it says 79 tests, documents MQTT topics as `sg/v1/...` (code uses `sg/v2`, `src/ingestion/topics.py`), and points to `docs/`, which is git-ignored.  Worth a separate small fix.
