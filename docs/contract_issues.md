# Contract issues (Layer 1 ↔ Layer 2)

## CI-1 — Meter server served only zeros (pymodbus 3.15) — FIXED UPSTREAM

* **Symptom (as of commit 70c77e0):** a live dump of every register on every unit returned 0
  throughout a full `run_timeseries` playback.
* **Cause:** in pymodbus 3.15, `ModbusTcpServer(context=...)` builds `SimCore`, and
  `SimRuntime` copies each device's register lists. Writes to the original
  `SimData.values` lists were invisible to clients. The old client test wrote values *before*
  constructing the server, so it didn't catch this.
* **Resolution:** upstream commit `4fc7c50` (Krish) rewrote `meter_server.py` to build
  `SimDevice`/`SimData` directly and write through `server.context.async_setValues(...)`.
  The interim Layer 2 workaround (`bind_live_registers`) was dropped in favour of it.
  Registers now start at address 0. Live tests C1/C2 confirm the served values.

## CI-2 — No simulated-time register — FIXED (register added on top of upstream)

`t_sim` is needed for change detection, `GAP`, de-duplication, torn-read detection and
ground-truth alignment. Added `REG_SIM_TIME` at fc4 28–29 (inside the existing 30-register
block; no other address moved): u32 Unix seconds of the naive timestamp read as UTC.
`update_registers` now writes the energy holding register first and then the input block with
the time, so a new time always comes with its energy. It is 0 before the first frame.
Test: `tests/test_meter_server.py::TestSimTimeRegister`.

## CI-3 — Reactive power and frequency — FIXED UPSTREAM (note on signedness)

Upstream `4fc7c50` serves reactive power per phase (fc4 20–25), total reactive power (26–27)
and frequency (18–19, a constant 50.00 Hz placeholder from `_update_loop`). Reactive power is
encoded as **u32**, so a negative (capacitive) value would clamp to 0. Every load in the
IEEE 13-bus model has positive Q, so this doesn't happen today. Make it i32 if capacitive or
generating meters are added.

## CI-4 — `current_a` is not the load's current — OPEN

`run_timeseries` sets `current_a` to the maximum current of any line connected to the load bus.
For m01 (bus 634, behind XFM-1), at 2024-01-01 00:00 phase a reads 294.7 A. The load's own P/V is only about 92 A (64.0 kW at 692.8 V line-to-neutral).
Layer 2 passes the value through untouched, and the schema documents it. Fault and overcurrent
rules in Layer 3 should not treat it as the meter's own load current until Layer 1 changes.

## CI-5 — Absent phases are zero-filled on the wire — OPEN (handled)

Single-phase meters serve 0 on the phases they do not have. Layer 2 uses
`meters.yaml:phases_present` and omits those phases (never forwards the zeros).

## CI-6 — No rated current, no DT/feeder-head meters; frequency is a constant — OPEN

`i_rated_a` is null and every meter's role is `consumer`. `frequency_hz` is forwarded, but
Layer 1 always writes 50.00 Hz.

## CI-7 — Meters report true, not theft-manipulated, values — OPEN (Layer 1 open question)

The SGCC theft labels and `theft_injector` are not wired into the served registers.

## CI-8 / CI-9 / CI-10 — SGCC-driven load profiles — FIXED (owner-approved Layer 1 change)

Three bugs in `dataset_loader.sample_load_profiles`, all fixed in one change:

* **CI-8, magnitude:** the old factor `daily_kwh / (nominal_kW × 24)` gave about 0.001 (a
  household's few kWh/day against a load point's MWh/day). Measured before the fix:
  `load_675_abc` at 0.002 kW against 485 kW nominal. Each household is now normalised to its
  own median day, so a typical day averages the load point's nominal demand. The daily ratio is
  capped at `config.SGCC_MAX_DAILY_RATIO = 3.0` so one metering outlier can't push a load to
  many times its rating.
* **CI-9, off-by-one:** the profile index used `inclusive="left"` while `run_timeseries`
  includes `end`, so the last step had no factor and jumped to nominal (for example
  0.012 → 170 kW). The index now includes `end`.
* **CI-10, dates and missing days:** SGCC shapes applied only on 2014–2016 dates, and a missing
  day gave a flat 1.0 with no intraday shape. Simulation dates are now mapped to the same
  month/day of an SGCC year (`first_year + (year − first_year) mod n_years`; 2024-01-01 →
  2015-01-01 with the full set). Missing days keep the intraday shape with a daily ratio of 1.0.
  Households with readings on every mapped day are preferred when sampling.

The intraday shape is now scaled to a daily mean of 1.0, independent of the window length. The
old helper normalised over whatever timestamps it was given.
Regression tests: `tests/test_dataset_loader.py::TestProfileRescaling` (5 tests).

After the fix, 2024-01-01 with the full SGCC file: `load_611_c` (170 kW nominal) ranges
153–268 kW, mean 201 kW, and every load keeps its shape at 23:45.

## CI-11 — Theft rows in `assign_theft_labels` ignore SGCC dates — OPEN (Layer 1)

`assign_theft_labels` matches SGCC theft readings by the raw simulation date. Outside 2014–2016
it always takes the "no data → 60 % of true" branch, and when it does match, `reported_kwh`
reduces to a constant `0.6 × nominal × interval`. This doesn't affect the served meter registers
(Layer 1 serves true values; see CI-7), but the ground-truth table's theft rows aren't
SGCC-shaped.

## CI-12 — Loading the full SGCC file needs about 8 GB RAM — NOTE

`load_sgcc` melts the 42,372 × 1,035 wide table to long format. Measured: about 20 s and
7.9 GB peak RSS. `run_meter_server.py --profiles sgcc` therefore defaults to `datasetsmall.csv`
(Jan 1–26 2014; other dates keep the shape with ratio 1.0). Pass
`--sgcc-path "data/fallback/SGCC_theft_detecton_data/data set.csv"` for the full calendar.
