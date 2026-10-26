"""Layer 3 leakage-guard list (plan §0 rule 1 and §4).

Ground truth that only ``src/analytics/evaluate.py`` may read.  The guard
test (plan §4) fails if any engine module references one of these names.
"""

# Label files (by file name, wherever they live)
LABEL_FILES = (
    "state.parquet",                    # results/year_2024: power-flow state, true losses, loading
    "meter_frame_true.parquet",         # results/year_2024: meter frame before meter error
    "households.parquet",               # results/year_2024: thief, attack, onset, severity
    "household_daily_labels.parquet",   # results/year_2024: would-have-reported, theft, offline (P3, N5)
    "fault_events.parquet",             # results/faults_2024: fault type, bus, Rf, clearing
)

# Label columns (by column name)
LABEL_COLUMNS = (
    "true_kwh",                         # household_daily / households
    "reported_kwh",                     # household_daily / households (before meter error)
    "would_have_reported_kwh",          # P3 (N5) offline-day label
    "theft_kwh",                        # P3 (N5) theft truth
    "offline_kwh",                      # P3 (N5) offline category
    "attack_active",                    # household_daily
)

# The P2 unbalance-scenario labels are added here when Stage 1c writes them.
