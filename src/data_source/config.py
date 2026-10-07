"""
Centralised configuration for the Data Source Layer.

Every external data path and tuneable parameter is declared here so that no
module contains a hard-coded absolute path.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/data_source/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# IEEE 13-bus feeder — voltage scaling
# ---------------------------------------------------------------------------
FEEDER_VOLTAGE_KV = 11.0          # Target primary-feeder voltage (line-to-line)
ORIGINAL_VOLTAGE_KV = 4.16        # Original IEEE 13-bus voltage
VOLTAGE_SCALE_FACTOR = FEEDER_VOLTAGE_KV / ORIGINAL_VOLTAGE_KV  # ≈ 2.644

# Build-time sanity check: total connected load must be within ±15 % of this
TOTAL_LOAD_MW_ASSERTION = 3.5     # MW — published IEEE 13-bus summary figure

# ---------------------------------------------------------------------------
# SGCC theft dataset — primary (GitHub) then fallback (local)
# ---------------------------------------------------------------------------
SGCC_GITHUB_REPO = "henryRDlab/ElectricityTheftDetection"
SGCC_GITHUB_BASE_URL = (
    f"https://github.com/{SGCC_GITHUB_REPO}/raw/master/"
)
SGCC_GITHUB_FILES = ["data.zip", "data.z01", "data.z02"]

# Where downloaded / extracted SGCC data lives
SGCC_DATA_DIR = PROJECT_ROOT / "data" / "sgcc"

# Offline fallback (pre-downloaded dataset)
SGCC_FALLBACK_PATH = (
    PROJECT_ROOT / "data" / "fallback"
    / "SGCC_theft_detecton_data" / "data set.csv"
)
SGCC_SMALL_FALLBACK_PATH = (
    PROJECT_ROOT / "data" / "fallback"
    / "SGCC_theft_detecton_data" / "datasetsmall.csv"
)

# ---------------------------------------------------------------------------
# Modbus TCP meter server
# ---------------------------------------------------------------------------
MODBUS_HOST = "0.0.0.0"
MODBUS_PORT = 5020

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIMULATION_FREQ = "15min"         # Time-series step interval
PLAYBACK_SPEED = 10.0             # 10× accelerated by default
RANDOM_SEED = 42                  # Reproducible random number generation

# ---------------------------------------------------------------------------
# Feeder — build-time validation
# ---------------------------------------------------------------------------
LOAD_TOLERANCE_FRACTION = 0.15    # Total load must be within ±15 % of published

# ---------------------------------------------------------------------------
# Theft injection
# ---------------------------------------------------------------------------
THEFT_FRACTION = 0.2              # 20 % of simulated consumers get theft labels
