"""
SGCC Theft Dataset Loader & Profile Sampler (Phase 1 path).

Loads the SGCC dataset (Zheng et al., 2018 — "Wide and Deep Convolutional
Neural Networks for Electricity-Theft Detection to Secure Smart Grids") in
long format and produces **load profiles**: a ``profiles`` DataFrame with one
column per ``load_id`` and a DatetimeIndex, containing multiplicative scaling
factors that can be passed directly to ``run_timeseries(profiles=...)``
(one SGCC household per load point; used by ``scripts/run_meter_server.py
--profiles sgcc``).

The Part 2C metering structure (households behind DTs, Jokar theft, ground
truth) lives in :mod:`sgcc`, :mod:`households`, :mod:`metering` and
:mod:`simulation`.  The old ``assign_theft_labels`` / ``build_ground_truth_table``
(fixed 60 % reporting, audit L1-5) were removed in Phase 2X checkpoint 3.

Resolution bridging (SGCC daily → feeder 15-minute)
-----------------------------------------------------
The SGCC dataset holds **daily** kWh readings per consumer (one value per
calendar day, many missing).  The feeder simulation steps at **15 minutes**.
Resolution is bridged as follows:

  - We compute a **normalized intra-day shape** from the feeder model's own
    synthetic daily profile (the double-hump morning/evening shape defined in
    ``feeder_model._synthetic_daily_profile``).  This gives 96 values per day
    (15-min steps) that sum to 1.0.

  - Each sampled household is first normalised to **its own median daily
    kWh**, so the SGCC magnitude (a few kWh/day) is rescaled to the load
    point's nominal demand (hundreds of kW).  For each day d and step t:
      ``factor[t] = min(daily_kwh[d] / median_daily_kwh, SGCC_MAX_DAILY_RATIO)
                    × rel_shape[t]``
    where ``rel_shape`` is the double-hump intra-day shape scaled to a daily
    mean of 1.0.  A typical day therefore averages the nominal load, and the
    household's real day-to-day variation is preserved.

  - Simulation dates outside the SGCC range (2014-01-01..2016-10-31) are
    mapped to the same month/day of an SGCC year
    (``year = first_year + (sim_year - first_year) mod n_years``), so e.g.
    2024-01-01 uses 2015-01-01.

  - If a mapped day has no reading, the daily ratio is 1.0 but the intra-day
    shape is still applied.  Households with readings on every mapped day of
    the window are preferred when sampling.

  - The profile index includes ``end`` (same convention as
    ``run_timeseries``), so every simulated step gets a factor.

This approach preserves daily energy totals from SGCC while injecting a
realistic intra-day shape from the power system model.  The deviation from a
real household's 15-min shape is accepted as a limitation; a more accurate
approach would require SGCC at sub-daily resolution, which is not available.

Data sourcing (two-tier)
------------------------
  1. **Primary** — download ``data.zip + data.z01 + data.z02`` from
     ``github.com/henryRDlab/ElectricityTheftDetection`` (original paper
     release, no credentials needed).
  2. **Fallback** — pre-downloaded CSV at the path configured in
     ``config.SGCC_FALLBACK_PATH`` (or ``config.SGCC_SMALL_FALLBACK_PATH``).
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)


# ===================================================================== #
#                       SGCC DATA ACQUISITION                            #
# ===================================================================== #

def download_sgcc() -> Path:
    """Download the SGCC dataset from GitHub primary source.

    Downloads the multi-part zip archive (``data.zip``, ``data.z01``,
    ``data.z02``) from the original paper repository, combines and extracts
    them.  Skips download if the extracted CSV already exists locally.

    Returns
    -------
    Path
        Path to the extracted CSV file.  Falls back to the local pre-
        downloaded dataset if the GitHub download fails.
    """
    data_dir = config.SGCC_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    extracted_csvs = list(data_dir.glob("*.csv"))
    if extracted_csvs:
        logger.info("SGCC data already present at %s", extracted_csvs[0])
        return extracted_csvs[0]

    # Attempt GitHub download
    try:
        logger.info("Downloading SGCC dataset from GitHub (%s) ...",
                     config.SGCC_GITHUB_REPO)
        downloaded_parts = {}
        for filename in config.SGCC_GITHUB_FILES:
            url = config.SGCC_GITHUB_BASE_URL + filename
            logger.info("  Fetching %s ...", url)
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            dest = data_dir / filename
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded_parts[filename] = dest
            logger.info("  Saved %s (%.1f MB)",
                         filename, dest.stat().st_size / 1e6)

        # Combine multi-part zip: concatenate z01 + z02 + zip
        combined_path = data_dir / "data_combined.zip"
        with open(combined_path, "wb") as out:
            for part_name in ["data.z01", "data.z02", "data.zip"]:
                part_path = downloaded_parts.get(part_name)
                if part_path and part_path.exists():
                    with open(part_path, "rb") as part:
                        shutil.copyfileobj(part, out)

        # Extract
        try:
            with zipfile.ZipFile(combined_path, "r") as zf:
                zf.extractall(data_dir)
                logger.info("Extracted SGCC data to %s", data_dir)
        except zipfile.BadZipFile:
            logger.warning("Combined zip failed, trying data.zip alone ...")
            main_zip = downloaded_parts.get("data.zip")
            if main_zip and main_zip.exists():
                with zipfile.ZipFile(main_zip, "r") as zf:
                    zf.extractall(data_dir)

        # Cleanup temp files
        for f in downloaded_parts.values():
            if f.exists():
                f.unlink()
        if combined_path.exists():
            combined_path.unlink()

        # Find extracted CSV
        extracted_csvs = list(data_dir.glob("**/*.csv"))
        if extracted_csvs:
            logger.info("SGCC dataset ready: %s", extracted_csvs[0])
            return extracted_csvs[0]

        raise FileNotFoundError("No CSV found after extraction")

    except Exception as exc:
        logger.warning(
            "GitHub download failed (%s). Falling back to local dataset.",
            exc,
        )
        return _fallback_path()


def _fallback_path() -> Path:
    """Return the path to the local fallback SGCC dataset."""
    if config.SGCC_FALLBACK_PATH.exists():
        logger.info("Using fallback dataset: %s", config.SGCC_FALLBACK_PATH)
        return config.SGCC_FALLBACK_PATH
    if config.SGCC_SMALL_FALLBACK_PATH.exists():
        logger.info("Using small fallback: %s", config.SGCC_SMALL_FALLBACK_PATH)
        return config.SGCC_SMALL_FALLBACK_PATH
    raise FileNotFoundError(
        "No SGCC dataset found.  Place the CSV at "
        f"{config.SGCC_FALLBACK_PATH} or download from GitHub."
    )


# ===================================================================== #
#                         DATASET LOADING                                #
# ===================================================================== #

def load_sgcc(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load and clean the SGCC theft dataset into long format.

    Parameters
    ----------
    path : Path or str, optional
        Explicit CSV path.  If *None*, calls ``download_sgcc()`` to
        acquire the data automatically.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``consumer_id, date, kwh, is_theft`` (FLAG mapped to bool).
    """
    if path is None:
        path = download_sgcc()
    path = Path(path)
    logger.info("Loading SGCC data from %s ...", path)

    df = pd.read_csv(path, low_memory=False)

    # Identify date columns vs. metadata columns.
    # The dataset has date-string columns (e.g. "01/01/2014") plus
    # CONS_NO and FLAG as the last two columns.
    meta_cols = []
    date_cols = []
    cons_col = None
    flag_col = None

    for col in df.columns:
        col_upper = str(col).strip().upper()
        if col_upper == "CONS_NO":
            cons_col = col
            meta_cols.append(col)
        elif col_upper == "FLAG":
            flag_col = col
            meta_cols.append(col)
        else:
            date_cols.append(col)

    if cons_col is None or flag_col is None:
        raise ValueError(
            "Expected CONS_NO and FLAG columns in SGCC CSV.  "
            f"Found columns: {list(df.columns[:5])} ... {list(df.columns[-5:])}"
        )

    # Melt wide → long
    long = df.melt(
        id_vars=[cons_col, flag_col],
        value_vars=date_cols,
        var_name="date",
        value_name="kwh",
    )
    long = long.rename(columns={cons_col: "consumer_id", flag_col: "is_theft"})
    long["is_theft"] = long["is_theft"].astype(bool)

    # Parse dates (mixed formats)
    long["date"] = pd.to_datetime(long["date"], format="mixed", dayfirst=False)

    # Drop nulls and zeros; keep only valid readings
    long = long.dropna(subset=["kwh"])
    long["kwh"] = pd.to_numeric(long["kwh"], errors="coerce")
    long = long.dropna(subset=["kwh"])
    long = long[long["kwh"] > 0].copy()

    long = long.sort_values(["consumer_id", "date"]).reset_index(drop=True)
    logger.info(
        "SGCC loaded: %d rows, %d consumers, %d theft-labeled",
        len(long),
        long["consumer_id"].nunique(),
        long[long["is_theft"]]["consumer_id"].nunique(),
    )
    return long


# ===================================================================== #
#                       PROFILE SAMPLING                                 #
# ===================================================================== #

def _relative_intraday_shape(timestamps: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Double-hump intra-day shape at each timestamp, scaled so that its mean
    over one full day (at ``freq``) is 1.0."""
    def raw(hour: np.ndarray) -> np.ndarray:
        return (
            0.4
            + 0.3 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
            + 0.3 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
        )
    day = pd.date_range("2000-01-01", periods=int(pd.Timedelta("1D") / pd.Timedelta(freq)),
                        freq=freq)
    day_mean = raw(np.asarray(day.hour + day.minute / 60.0, dtype=float)).mean()
    hour = np.asarray(timestamps.hour + timestamps.minute / 60.0, dtype=float)
    return raw(hour) / day_mean


def _map_to_sgcc_calendar(days: pd.DatetimeIndex, lo: pd.Timestamp,
                          hi: pd.Timestamp) -> pd.DatetimeIndex:
    """Map simulation days onto SGCC dates (same month/day, cycled years)."""
    n_years = hi.year - lo.year + 1
    out = []
    for d in days:
        if lo <= d <= hi:
            out.append(d)
            continue
        year = lo.year + (d.year - lo.year) % n_years
        try:
            out.append(d.replace(year=year))
        except ValueError:          # 29 Feb into a non-leap year
            out.append(d.replace(year=year, day=28))
    return pd.DatetimeIndex(out)


def sample_load_profiles(
    load_points: pd.DataFrame,
    sgcc_df: pd.DataFrame,
    start: str = "2024-01-01",
    end: str = "2024-01-02",
    freq: str = "15min",
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample one normal SGCC household per load point and build a profiles DataFrame.

    For each load point, samples a normal (is_theft=False) SGCC consumer,
    normalises its daily kWh to its own median day and applies the intra-day
    shape, so a typical day averages the load point's nominal demand (see the
    module docstring for the formula and the calendar mapping).

    Parameters
    ----------
    load_points : DataFrame
        Output of ``get_load_points()`` — must have a ``load_id`` column.
    sgcc_df : DataFrame
        Output of ``load_sgcc()``.
    start, end : str
        Simulation window bounds, inclusive (same as ``run_timeseries``).
    freq : str
        Time-step frequency string (default ``"15min"``).
    seed : int or None
        Random seed for reproducibility.  Defaults to ``config.RANDOM_SEED``.

    Returns
    -------
    pd.DataFrame
        Profiles DataFrame: DatetimeIndex, one column per ``load_id``,
        values are multiplicative scaling factors vs. nominal.
        Ready to pass to ``run_timeseries(profiles=...)``.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    rng = np.random.default_rng(seed)
    normal = sgcc_df[~sgcc_df["is_theft"]]
    if normal.empty:
        raise ValueError("No normal consumers in SGCC data")

    timestamps = pd.date_range(start=start, end=end, freq=freq)
    shape = _relative_intraday_shape(timestamps, freq)

    sim_days = timestamps.normalize()
    lo, hi = normal["date"].min().normalize(), normal["date"].max().normalize()
    mapped = _map_to_sgcc_calendar(sim_days, lo, hi)
    window_days = set(mapped.unique())

    # Daily kWh per consumer (wide: consumer x date) and each consumer's median.
    daily = normal.assign(date=normal["date"].dt.normalize()).pivot_table(
        index="consumer_id", columns="date", values="kwh", aggfunc="first")
    medians = daily.median(axis=1)
    usable = medians[medians > 0].index
    in_window = [d for d in window_days if d in daily.columns]
    if len(in_window) == len(window_days):
        complete = daily.loc[usable, in_window].notna().all(axis=1)
        candidates = np.asarray(sorted(complete[complete].index.astype(str)))
    else:
        candidates = np.asarray([])
    if len(candidates) == 0:
        candidates = np.asarray(sorted(usable.astype(str)))
    if len(candidates) == 0:
        raise ValueError("No normal SGCC consumer with positive readings")
    daily.index = daily.index.astype(str)
    medians.index = medians.index.astype(str)

    cap = config.SGCC_MAX_DAILY_RATIO
    profiles: dict[str, np.ndarray] = {}
    for _, lp_row in load_points.iterrows():
        chosen = str(rng.choice(candidates))
        row = daily.loc[chosen]
        med = float(medians[chosen])
        ratio_by_day = {}
        for d in window_days:
            v = row.get(d, np.nan)
            ratio_by_day[d] = min(float(v) / med, cap) if pd.notna(v) and v > 0 else 1.0
        ratios = np.array([ratio_by_day[d] for d in mapped], dtype=float)
        profiles[lp_row["load_id"]] = ratios * shape

    return pd.DataFrame(profiles, index=timestamps)
