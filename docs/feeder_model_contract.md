# Feeder Model Contract — IEEE 13-Bus at 11 kV

> **Status:** Final (M5)
> **Author:** Part A agent
> **Last updated:** 2026-10-11 (Phase 1 topology fixes: L1-1, L1-7, L1-8, L1-18)

This document is the handoff from Part A (feeder model & fault engine) to
Part B (meter server, dataset loader, theft injector).  It defines the
interface contract that Part B depends on.

---

## 1. Function Signatures & Return Schemas

### `get_load_points(net=None) → DataFrame`

Returns one row per load point with columns:

| Column | Dtype | Unit | Description |
|--------|-------|------|-------------|
| `load_id` | `str` | — | Stable unique identifier, format `load_<bus>_<phase>` |
| `bus` | `str` | — | IEEE 13-bus node name (e.g. `"671"`) |
| `phase` | `str` | — | `'a'`, `'b'`, `'c'`, `'ab'`, `'bc'`, or `'abc'` |
| `nominal_kw` | `float64` | kW | Sum of per-phase active power at nominal |
| `nominal_kvar` | `float64` | kvar | Sum of per-phase reactive power at nominal |

### `run_snapshot(net=None, load_scaling=None) → dict`

- **`load_scaling`**: `{load_id: float}`.  Every load is reset to 1.0 before
  the dict is applied, so missing ids run at 1.0.  Unknown ids raise
  `KeyError`.
- Returns:
  - `bus_voltages` — `DataFrame` indexed by bus, columns from `res_bus_3ph`
    (includes `vm_a_pu`, `va_a_degree`, `vm_b_pu`, …).
  - `line_currents` — `DataFrame` indexed by line, columns from `res_line_3ph`
    (includes `i_a_from_ka`, `i_b_from_ka`, …).
  - `line_powers` — `DataFrame` of power columns from `res_line_3ph`.
  - `net` — the solved `pandapowerNet` object.

### `run_timeseries(start, end, freq, net=None, profiles=None) → DataFrame`

- **`profiles`**: optional `DataFrame[timestamp × load_id]` of multiplicative
  scaling factors.  If `None`, a built-in synthetic daily shape is used.
  A missing timestamp or load_id column raises `KeyError`; a failed power
  flow raises `RuntimeError` (no step is skipped or run at nominal); NaN
  output raises `RuntimeError`.
  A missing timestamp or load_id column raises `KeyError`; a failed power
  flow raises `RuntimeError` (no step is skipped or run at nominal); NaN
  output raises `RuntimeError`.
- Returns a **tidy** DataFrame with one row per `(timestamp, load_id, phase)`:

| Column | Dtype | Unit | Description |
|--------|-------|------|-------------|
| `timestamp` | `datetime64[ns]` | — | **Timezone-naive** |
| `load_id` | `str` | — | Matches `get_load_points().load_id` |
| `bus` | `str` | — | Bus node name |
| `phase` | `str` | — | `'a'`, `'b'`, or `'c'` |
| `voltage_v` | `float64` | V | Line-to-neutral voltage at the load bus |
| `current_a` | `float64` | A | Max branch current at bus (connected lines and transformer windings) |
| `active_power_kw` | `float64` | kW | Scheduled active power × scaling |
| `reactive_power_kvar` | `float64` | kvar | Scheduled reactive power × scaling |

### `inject_fault(bus, fault_type, timestamp=None, net=None) → dict`

- **`fault_type`**: `"SLG"` | `"LL"` | `"3PH"`
- Returns:

| Key | Type | Unit | Description |
|-----|------|------|-------------|
| `timestamp` | `str`/`datetime`/`None` | — | Echo of input |
| `bus` | `str` | — | Bus node name |
| `fault_type` | `str` | — | Echo of input |
| `fault_current_ka` | `float` | kA | Initial symmetrical SC current (IEC 60909) |
| `affected_phases` | `list[str]` | — | `['a']`, `['a','b']`, or `['a','b','c']` |
| `prefault_current_ka` | `float`/`None` | kA | Max pre-fault line current if timestamp given |

---

## 2. Load Point Table

| load_id | bus | phase | nominal_kw | nominal_kvar |
|---------|-----|-------|------------|--------------|
| `load_634_abc` | 634 | abc | 400.0 | 290.0 |
| `load_645_b` | 645 | b | 170.0 | 125.0 |
| `load_646_b` | 646 | b | 230.0 | 132.0 |
| `load_652_a` | 652 | a | 128.0 | 86.0 |
| `load_671_abc` | 671 | abc | 1155.0 | 660.0 |
| `load_675_abc` | 675 | abc | 843.0 | 462.0 |
| `load_692_c` | 692 | c | 170.0 | 151.0 |
| `load_611_c` | 611 | c | 170.0 | 80.0 |
| `load_670_abc` | 670 | abc | 200.0 | 116.0 |

**Total connected load: 3.466 MW / 2.102 Mvar** (IEEE spot + distributed
load, checked at build time to ±0.5 %).  `load_670_abc` is the 632-671
distributed load and is appended last (unit 9 / `m09`); units 1-8 are
unchanged.

---

## 3. Sample Output

### `run_timeseries` — head (actual output)

```
  timestamp             load_id        bus  phase   voltage_v  current_a  active_power_kw  reactive_power_kvar
  2024-01-01 00:00:00   load_634_abc   634  a       760.87      102.10        64.02            44.01
  2024-01-01 00:00:00   load_634_abc   634  b       765.15       78.44        48.01            36.01
  2024-01-01 00:00:00   load_634_abc   634  c       767.40       78.21        48.01            36.01
  2024-01-01 00:00:00   load_645_b     645  b      6737.09       20.60        68.02            50.01
  2024-01-01 00:00:00   load_646_b     646  b      6736.42        9.11        92.02            52.81
  2024-01-01 00:00:00   load_652_a     652  a      6688.42        9.22        51.21            34.41
  2024-01-01 00:00:00   load_671_abc   671  a      6691.26       62.18       154.04            88.02
  2024-01-01 00:00:00   load_671_abc   671  b      6749.97       34.98       154.04            88.02
  2024-01-01 00:00:00   load_671_abc   671  c      6675.83       58.65       154.04            88.02
  2024-01-01 00:00:00   load_675_abc   675  a      6689.06       36.29       194.05            76.02
```

(Phase 1 network, `results/frame_phase1.csv`.  Before Phase 1 the 634 rows
read ~294 A: circulating current through the 633-634 loop, audit L1-1.)

### `inject_fault` — sample result (actual output)

```python
{
    "timestamp": "2024-01-01",
    "bus": "671",
    "fault_type": "3PH",
    "fault_current_ka": 2.619,       # IEC 60909 ikss_ka
    "affected_phases": ["a", "b", "c"],
    "prefault_current_ka": 0.173,    # from pre-fault power flow
    # Ratio: 2.619 / 0.173 ≈ 15.2× (well above the 2× minimum)
}
```

---

## 4. Modelling Judgement Calls

1. **Source and substation transformer** — The external grid (`vm_pu=1.0`)
   sits on a 115 kV bus `sourcebus` (as in the OpenDSS IEEE13 model).  The
   substation transformer connects `sourcebus` → `650`, 115 kV / 11 kV,
   Dyn, 5 MVA, replacing the original 115 / 4.16 kV ratio.  Bus 650 is
   11 kV and feeds 632 through a 2000 ft config-601 line (Kersting 1991).
   Impedance parameters (vk=8%, vkr=1%) have no source (audit §5).

2. **XFM-1 (633→634)** — The original 4.16 / 0.48 kV transformer is scaled
   proportionally to 11 / 1.269 kV so per-unit voltages downstream of XFM-1
   are preserved.  Impedance: vk=4.5%, vkr=1.1% (unsourced; audit L1-12).
   There is no line between 633 and 634: XFM-1 is the only connection
   (the former 1 m `Line_633_634` closed an 11 kV ↔ LV loop, audit L1-1).

3. **Voltage regulator 650→632: fixed-tap equivalent of the IEEE
   regulator** — pandapower's `runpp_3ph` has no regulator model.  The
   regulator is replaced by a fixed off-load tap on the LV side of the
   substation transformer: 0.625 %/step, ±16 steps (the IEEE regulator's
   32-step ±10 % range), position `SUB_TAP_POS = 10` = +6.25 %.  10 is the
   median of Kersting's published regulator taps (10/8/11) and puts bus 632
   at 1.012–1.034 pu at nominal load.  The external grid stays at 1.0 pu.
   A fixed tap does not regulate: at light load the voltage rises and at
   peak it falls (no tap changes during a time series).

4. **Distributed load (632→671)** — Lumped at a new bus 670 one third of
   the way along the segment, as in the OpenDSS IEEE13 model: 632-670
   (667 ft) and 670-671 (1333 ft), both config 601; load 17/66/117 kW and
   10/38/68 kvar on phases A/B/C, wye, constant PQ (`load_670_abc`).

5. **671→692 switch** — Modelled as a very short line (0.001 km) with
   configuration 606 impedance.  This preserves the connectivity without
   requiring a dedicated switch model.

6. **Capacitor banks** — Bus 675 (600 kVAR, 3-phase) and bus 611 (100 kVAR,
   single-phase C) are modelled as shunts at their original kVAR values.
   These are voltage-independent in pandapower's shunt model.

7. **Line impedances** — Original IEEE 13-bus line impedances are in
   ohms/mile with lengths in feet.  All values are converted to ohms/km
   and km.  The per-unit base changes with the 11 kV voltage, but the
   absolute impedances stay the same.

8. **Zero-sequence parameters** — Required by `calc_sc` for SLG faults.
   Zero-sequence impedances on lines use R0 ≈ 3×R1 and X0 ≈ 3×X1 ratios
   from the IEEE 13-bus configuration data.  External grid has
   `x0x_max=1.0`, `r0x0_max=0.1`.  Transformers have `vk0_percent`
   equal to their `vk_percent`.

9. **Short-circuit data** — The external grid has `s_sc_max_mva=1000`,
   `rx_max=0.1` (representative for a strong grid behind a 5 MVA
   transformer).

---

## 5. Running the Tests

```bash
# From project root, with venv activated:
python -m pytest tests/test_feeder_model.py -v
```

### Known Issues

- pandapower versions before 2.14 may not support all `runpp_3ph` features.
  Use pandapower ≥ 2.14.0.
- The `calc_sc` 1ph (SLG) fault requires zero-sequence parameters on ext_grid,
  transformers, and lines.  If any are missing, the SLG result may be zero or
  incorrect.
- The convergence test escalates warnings to errors (library
  `FutureWarning`/`DeprecationWarning` excepted) and asserts
  `net.converged`; it no longer retries.
- Physical-sanity tests (energy balance, losses, voltage band, loadings,
  radial topology, load totals) live in `tests/test_feeder_physics.py`.
- XFM-1 runs at ~120 % (worst phase) at nominal load: the IEEE load on 634
  is 194 kVA on phase A against 167 kVA per phase.  This is IEEE data and
  is reported, not asserted.

---

## 6. Requests for Part B

1. **`dataset_loader.py`** should produce a `profiles` DataFrame with columns
   matching `get_load_points().load_id` and a `DatetimeIndex` of timestamps.
   Values are multiplicative scaling factors (1.0 = nominal).  Pass this to
   `run_timeseries(profiles=...)`.

2. **`meter_server.py`** should call `get_load_points()` at startup to discover
   load points and their nominal demands.  Each meter maps to one `load_id`.

3. **`theft_injector.py`** operates on consumption series (the
   `active_power_kw` column from `run_timeseries` output) and is independent
   of the feeder model.
