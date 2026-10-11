# Data Source Layer — Agent Build Brief

**How to use this document:** hand this file to the Antigravity agent as its task
(paste it into the Editor or Manager surface, or drop it in the repo as
`docs/data_source_layer_brief.md` and tell the agent to implement it). Use Plan mode
first — review the agent's Plan Artifact against §7 (Acceptance Criteria) before
letting it execute. This brief describes *behavior*, not literal code — the agent
should design the implementation; the acceptance criteria in §7 are what "done"
means.

## 1. Goal

Build **only** the Data Source Layer (roadmap Layer 1): an unbalanced power-flow
simulation of a real benchmark feeder, a set of simulated smart meters exposing
their readings over Modbus TCP, and a loader that blends in a real public
theft-labeled dataset for both realistic load shapes and theft ground truth.

**Do not** build the MQTT gateway, FastAPI listener, Kafka, TimescaleDB/PostgreSQL
writes, or any dashboard work — those are Layer 2–4, covered by separate briefs.
Layer 1 stops at meters serving Modbus TCP registers and files on disk.

## 2. Environment & dependencies

- Python 3.11+, `venv` + `pip` (or `uv` — agent's choice), versions pinned in
  `requirements.txt`.
- Required libraries: `pandapower`, `pymodbus` (v3.x), `pandas`, `numpy`.
- Treat every external data path (feeder data, dataset) as a config constant in
  `config.py`, never a hardcoded absolute path.

### 2.1 Required external data (not built into pandapower)

**IEEE 13-bus feeder data.** pandapower does **not** ship the IEEE 13-bus test
feeder — its built-in `networks` module only has balanced MATPOWER-style cases and
one unrelated European LV asymmetric case. Get the real bus/line/transformer/load
tables from the primary source: `https://site.ieee.org/pes-testfeeders/resources/`
(IEEE PES Distribution System Analysis Subcommittee, originally Kersting 1991). If
the agent has browser access, point it at this page directly and have it parse the
published tables. Otherwise, pre-download the feeder data and place it under
`data/ieee13/` before starting the task. Either way, have the agent sanity-check its
constructed network's total connected load against the feeder's published summary
figures (~3.5 MW) as a build-time assertion — this catches a mistyped or
misattributed parameter before it silently propagates into every downstream module.

**SGCC theft dataset.** Prefer the **original release from the paper that published
it** over an arbitrary Kaggle mirror — it's the more defensible citation and needs
no API credentials:
- Primary: `github.com/henryRDlab/ElectricityTheftDetection` (Zheng, Yang, Niu, Dai,
  Zhou, "Wide and Deep Convolutional Neural Networks for Electricity-Theft
  Detection to Secure Smart Grids," *IEEE Trans. Industrial Informatics*, vol. 14,
  no. 4, pp. 1606–1615, 2018). Data ships as `data.zip` + `data.z01` + `data.z02`,
  extracted together — no account needed.
- Fallback (if GitHub's LFS/large-file hosting is unreachable from the sandbox):
  the same dataset mirrored on Kaggle, e.g. `kaggle.com/datasets/bensalem14/sgcc-dataset`
  — this path needs a Kaggle API token (`kaggle.json`). downloading it
  yourself once and pointing the agent at the local CSV is the more conservative
  option and avoids the auth step entirely. Downloaded version has been provided to you at data\fallback\SGCC_theft_detecton_data.

## 3. Module — `feeder_model.py`

- Build the IEEE 13-bus test feeder (Kersting, 1991) in pandapower, adapted so the
  primary feeder operates at **11 kV line-to-line** instead of the original 4.16 kV
  — scale voltage-dependent parameters accordingly; line impedances stay in
  ohms/km from the original data, but per-unit bases change with the new voltage.
- Model it as a genuinely **unbalanced** network: represent the original single-phase
  and three-phase loads explicitly per phase, and solve with
  `pandapower.pf.runpp_3ph` — never the default balanced `runpp()`. Note this
  explicitly in a code comment; it's easy to call the wrong one by habit.
- `run_snapshot(load_scaling: dict) -> dict` — takes a per-load scaling factor
  (time-of-day variation) and returns per-phase bus voltages, per-phase line
  currents, and per-phase power flows for that snapshot.
- `run_timeseries(start, end, freq) -> pandas.DataFrame` — steps through a time
  range at the given frequency, applying load profiles (from Module 5) at each
  step, returns a tidy DataFrame of all measured quantities.
- `inject_fault(bus, fault_type, timestamp) -> dict` — wraps
  `pandapower.shortcircuit.calc_sc` (IEC 60909) for single-line-to-ground,
  line-to-line, and three-phase faults. Output must include fault current
  magnitude and affected phase(s) — this feeds the Evaluation Plan's fault-engine
  ground truth directly, so its output schema matters.

## 4. Module — `meter_server.py`

- One Modbus TCP server per modeled consumer/load point, or one server with
  multiple unit IDs (agent's choice — document which, and why, in the module
  docstring).
- Each meter exposes at minimum: voltage (V), current (A), active power (kW), and
  a monotonically increasing energy accumulator (kWh), per phase where applicable.
  Put the exact register map in a table at the top of the file — this is the
  "protocol layer" decision and needs to be defensible on its own.
- A background task updates each meter's registers from `feeder_model`'s
  `run_timeseries` output on a configurable interval (default: every simulated 15
  minutes; make playback speed — real-time vs. accelerated — configurable).
- **DLMS/COSEM scope:** data-model alignment only. Name and structure the register
  map so it mirrors DLMS/COSEM OBIS object conventions (e.g. document which OBIS
  code each register conceptually maps to, such as `1.8.0` for total active energy
  import) in comments/docstrings. Do **not** implement the DLMS/COSEM wire
  protocol, HDLC framing, or security association layer. State this scoping
  decision explicitly in the module docstring.

## 5. Module — `dataset_loader.py`

Load the SGCC theft dataset and produce two outputs:

1. **Realistic load shapes.** For each simulated load point, sample one real
   labeled-normal SGCC household series and rescale its magnitude to that load
   point's expected demand from the feeder model — replacing an arbitrary flat
   profile with a real daily/weekly consumption shape.
2. **Theft ground truth.** For a configurable subset of simulated consumers,
   sample a labeled-theft SGCC series instead, and record its label (and the
   reported-vs-actual gap where SGCC provides it) as ground truth metadata
   alongside the simulation output.

Output one tidy table:
`consumer_id, timestamp, reported_kwh, feeder_true_kwh, is_theft_label, source`
— this is what the Evaluation Plan's real-ground-truth test set is built from
directly, so keep the schema stable.

## 6. Module — `theft_injector.py`

- Implement the attack functions from Jokar, Arianpoo & Leung (2016): constant
  partial reporting, random reduction, time-of-day-dependent reporting, and
  reverse/zero reporting, at minimum — each as a named function taking a true
  consumption series and returning a manipulated "reported" series plus a
  ground-truth label and severity value.
- Keep this module independent of `dataset_loader.py` — it should operate on any
  consumption series (real or simulated), so the evaluation harness can call it on
  raw pandapower output as well as on SGCC series not already labeled theft.

## 7. Acceptance criteria (agent verifies these itself before declaring done)

- [ ] `run_snapshot` and `run_timeseries` both converge without pandapower warnings
      on the default IEEE-13-bus-at-11kV network.
- [ ] The network is genuinely unbalanced: per-phase currents differ by more than a
      trivial tolerance on at least one known-asymmetric load point (a balanced
      solve would fail this check by construction — this is the test that catches
      an accidental `runpp()` call).
- [ ] `inject_fault` returns a fault current at least 2× the pre-fault current for
      at least one tested bus/fault-type combination.
- [ ] Starting one meter server and reading a register with a plain `pymodbus`
      client (or `mbpoll`) returns a value matching the feeder model's output for
      that timestamp.
- [ ] `dataset_loader` output has no null `feeder_true_kwh` values and at least one
      row per simulated consumer with `is_theft_label = True` — confirms the theft
      class actually survived sampling, not just the normal class.
- [ ] `theft_injector` functions are unit-tested against a synthetic constant
      series where the expected manipulated output can be computed by hand.

## 8. Explicit non-goals for this task

- No MQTT publishing, no FastAPI, no Kafka, no TimescaleDB/PostgreSQL writes.
- No DLMS/COSEM wire protocol implementation.
- No frontend/dashboard work.

## 9. Suggested file layout

```
src/
  data_source/
    feeder_model.py
    meter_server.py
    dataset_loader.py
    theft_injector.py
    config.py
tests/
  test_feeder_model.py
  test_meter_server.py
  test_dataset_loader.py
  test_theft_injector.py
requirements.txt
docs/
  data_source_layer_brief.md   <- this file
```

## References

1. Kersting, W.H., "Radial Distribution Test Feeders," *IEEE Trans. Power Systems*,
   vol. 6, no. 3, pp. 975–985, 1991.
2. Jokar, P., Arianpoo, N., Leung, V.C.M., "Electricity Theft Detection in AMI Using
   Customers' Consumption Patterns," *IEEE Trans. Smart Grid*, vol. 7, no. 1,
   pp. 216–226, 2016.
3. pandapower documentation — `runpp_3ph` (unbalanced three-phase power flow) and
   `shortcircuit.calc_sc` (IEC 60909 short-circuit calculation).
4. `pymodbus` documentation — synchronous/asynchronous Modbus TCP server API.
5. Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., Zhou, Y., "Wide and Deep Convolutional
   Neural Networks for Electricity-Theft Detection to Secure Smart Grids," *IEEE
   Trans. Industrial Informatics*, vol. 14, no. 4, pp. 1606–1615, 2018 — source of
   the SGCC dataset.
6. IEEE PES Distribution System Analysis Subcommittee, Test Feeder Resources —
   `site.ieee.org/pes-testfeeders/resources` — source of the IEEE 13-bus feeder data.
