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

# Build-time sanity check: IEEE 13-bus spot + distributed load (Kersting 1991)
TOTAL_LOAD_MW_ASSERTION = 3.466   # MW
TOTAL_LOAD_MVAR_ASSERTION = 2.102 # Mvar

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
MODBUS_HOST = "127.0.0.1"         # loopback by default (audit MB-6)
MODBUS_PORT = 5020

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIMULATION_FREQ = "15min"         # Time-series step interval
PLAYBACK_SPEED = 10.0             # 10× accelerated by default
RANDOM_SEED = 42                  # Reproducible random number generation
MASTER_SEED = RANDOM_SEED         # Phase 2X: every random stream derives from it
SIMULATION_YEAR = 2024            # brief v2 X.2 (leap year, IST)
YEAR_CACHE_DIR = PROJECT_ROOT / "results" / "year_2024"

# Prayas eMARC household load data (brief v2 X.3.3 item 1; git-ignored, never committed)
EMARC_DIR = PROJECT_ROOT / "data" / "emarc"
# SLDC Delhi 2024-25 reference (load-duration curve + monthly peaks; the one versioned data file)
SLDC_REFERENCE_CSV = PROJECT_ROOT / "data" / "delhi_sldc_2024_25_reference.csv"


def rng(name: str):
    """Named random stream derived from ``MASTER_SEED``.

    ``SeedSequence(MASTER_SEED, spawn_key=(crc32(name),))``: the same name
    always gives the same stream, and streams with different names are
    independent.  Every call is logged with its name.
    """
    import logging
    import zlib

    import numpy as np

    key = zlib.crc32(name.encode("utf-8"))
    logging.getLogger(__name__).info("random stream %r (master seed %d, key %d)",
                                     name, MASTER_SEED, key)
    return np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=(key,)))

# ---------------------------------------------------------------------------
# Feeder — build-time validation
# ---------------------------------------------------------------------------
LOAD_TOLERANCE_FRACTION = 0.005   # Total P and Q must be within ±0.5 % of published

# ---------------------------------------------------------------------------
# SGCC load profiles (Phase 1 path, dataset_loader.sample_load_profiles)
# ---------------------------------------------------------------------------
# Theft (Part 2C): thief share p = FLAG = 1 share of the SGCC file, computed in
# sgcc.SGCCMatrix.theft_prevalence (the fixed THEFT_FRACTION was removed).

# SGCC profile rescaling: a household's daily kWh is divided by its own median
# day, so 1.0 = a typical day for that household = the load point's nominal.
# Ratios are capped so a single metering outlier cannot drive a load point to
# many times its rating (the feeder is sized for nominal demand).
SGCC_MAX_DAILY_RATIO = 3.0
