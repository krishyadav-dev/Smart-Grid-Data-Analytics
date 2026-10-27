"""Layer 3 leakage-guard list (plan §0 rule 1 and §4; Stage 1 run sheet rules 1-2;
Stage 2 run sheet §0 leakage guard).

Ground truth that only ``src/analytics/evaluate.py`` may read.  The guard
tests (``tests/analytics/test_leakage.py``) fail if any engine module
references one of these names.
"""

# Label files (by file name, wherever they live)
LABEL_FILES = (
    "state.parquet",                    # results/year_2024: power-flow state, true losses, loading
    "meter_frame_true.parquet",         # results/year_2024: meter frame before meter error
    "households.parquet",               # results/year_2024: thief, attack, onset, severity
    "household_daily_labels.parquet",   # results/year_2024: would-have-reported, theft, offline (P3, N5)
    "fault_events.parquet",             # results/faults_2024: fault type, bus, Rf, clearing
    "meter_classes.parquet",            # results/faults_2024: CP4 classification per event and meter
    "summary.json",                     # results/faults_2024 etc.: the engine reads config/relay.yaml instead
    "event_records.npz",                # results/faults_2024: meter-side records; the engine reads telemetry
    "unbalance_labels.parquet",         # results/unbalance_scenarios: V_N / V_P per step and meter (P2)
    "unbalance_weekly_labels.parquet",  # results/unbalance_scenarios: weekly IS 17036 flags (P2)
    "baseline_true_frame.parquet",      # results/unbalance_scenarios: s = 0 true frame (P2)
    "states.parquet",                   # results/unbalance_scenarios: power-flow states (P2)
)

# Label columns (by column name)
LABEL_COLUMNS = (
    "true_kwh",                         # household_daily / households
    "reported_kwh",                     # household_daily / households (before meter error)
    "would_have_reported_kwh",          # P3 (N5) offline-day label
    "theft_kwh",                        # P3 (N5) theft truth
    "offline_kwh",                      # P3 (N5) offline category
    "attack_active",                    # household_daily
    "fault_time",                       # fault_events: inception (run sheet rule 2)
    "t_clear_s",                        # fault_events: clearing time (rule 2)
    "offset_s",                         # fault_events: offset inside the step (rule 2)
    "rf_ohm",                           # fault_events: fault resistance
    "rf_model",                         # fault_events: fixed / arc
    "vu_true_pct",                      # P2 labels: true V_N / V_P
    "exceeds_limit",                    # P2 labels: weekly IS 17036 flag
    "meter_gain",                       # dts / households: each meter's own error (N6), unknown to the utility
)

# Layer 1 modules that read the true frame or rebuild the labelled population
# (Stage 2 run sheet §0: scenario generators live in Layer 1 and write
# metered-format inputs).  Engines read their outputs, never import them.
LABEL_MODULES = ("simulation", "metering", "ntl_inputs", "households")

# The consumer index the theft engine reads (step A; user decision 27 Oct:
# consumer-to-DT indexing only, no LV phase)
CONSUMER_INDEX_COLUMNS = ("household", "dt", "bus")

# The event record's event_id is a label (run sheet rule 1): engines must not
# use it to group, merge or match bursts; only evaluate.py matches on it.
EVENT_ID_FIELD = "event_id"
