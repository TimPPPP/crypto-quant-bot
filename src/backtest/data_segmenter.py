from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config_backtest as cfg

logger = logging.getLogger("backtest.segmenter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


@dataclass
class DataQualityReport:
    """Report on data quality metrics (Problem #8 fix)."""
    total_rows: int
    total_gaps: int
    max_gap_minutes: float
    gaps_over_warn_threshold: int
    gaps_over_max_threshold: int
    gap_distribution: Dict[str, int]  # e.g., {"1-5min": 10, "5-10min": 3, ...}
    interpolated_gaps: int
    missing_pct: float  # percentage of expected rows that are missing


def analyze_data_gaps(df: pd.DataFrame, freq_minutes: float = 1.0) -> DataQualityReport:
    """
    Analyze data gaps and return a quality report.

    Problem #8 Fix: Provides detailed gap analysis to understand data quality
    issues that can affect Kalman filter assumptions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    freq_minutes : float
        Expected frequency in minutes (default 1.0 for 1-min bars)

    Returns
    -------
    DataQualityReport
        Detailed gap analysis
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return DataQualityReport(
            total_rows=0,
            total_gaps=0,
            max_gap_minutes=0.0,
            gaps_over_warn_threshold=0,
            gaps_over_max_threshold=0,
            gap_distribution={},
            interpolated_gaps=0,
            missing_pct=0.0,
        )

    time_diffs = df.index.to_series().diff().dropna()
    time_diffs_mins = time_diffs.dt.total_seconds() / 60.0

    # Expected vs actual
    expected_freq = pd.Timedelta(minutes=freq_minutes)
    total_span = (df.index.max() - df.index.min()).total_seconds() / 60.0
    expected_rows = int(total_span / freq_minutes) + 1
    actual_rows = len(df)
    missing_pct = 100.0 * (1.0 - actual_rows / expected_rows) if expected_rows > 0 else 0.0

    # Identify gaps (anything > expected frequency)
    gap_threshold = freq_minutes * 1.5  # Allow 50% tolerance
    gaps = time_diffs_mins[time_diffs_mins > gap_threshold]

    # Categorize gaps
    warn_threshold = getattr(cfg, 'WARN_DATA_GAP_MINS', 10)
    max_threshold = getattr(cfg, 'MAX_DATA_GAP_MINS', 60)

    gaps_over_warn = int((gaps > warn_threshold).sum())
    gaps_over_max = int((gaps > max_threshold).sum())

    # Gap distribution
    gap_dist = {
        "1-5min": int(((gaps > 1) & (gaps <= 5)).sum()),
        "5-10min": int(((gaps > 5) & (gaps <= 10)).sum()),
        "10-30min": int(((gaps > 10) & (gaps <= 30)).sum()),
        "30-60min": int(((gaps > 30) & (gaps <= 60)).sum()),
        "60min+": int((gaps > 60).sum()),
    }

    return DataQualityReport(
        total_rows=actual_rows,
        total_gaps=len(gaps),
        max_gap_minutes=float(time_diffs_mins.max()) if len(time_diffs_mins) > 0 else 0.0,
        gaps_over_warn_threshold=gaps_over_warn,
        gaps_over_max_threshold=gaps_over_max,
        gap_distribution=gap_dist,
        interpolated_gaps=0,  # Set by interpolation function
        missing_pct=missing_pct,
    )


def validate_data_continuity(
    df: pd.DataFrame,
    strict: bool = True,
) -> DataQualityReport:
    """
    Validate data continuity and return quality report.

    Problem #8 Fix: Enhanced validation with detailed reporting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    strict : bool
        If True, raise error on gaps exceeding MAX_DATA_GAP_MINS.
        If False, only warn.

    Returns
    -------
    DataQualityReport
        Detailed gap analysis
    """
    if df.empty:
        logger.warning("Data continuity check skipped: empty DataFrame.")
        return DataQualityReport(
            total_rows=0, total_gaps=0, max_gap_minutes=0.0,
            gaps_over_warn_threshold=0, gaps_over_max_threshold=0,
            gap_distribution={}, interpolated_gaps=0, missing_pct=0.0,
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for continuity checks.")

    # Get detailed gap analysis
    report = analyze_data_gaps(df)

    # Log summary
    logger.info(
        "Data quality: %d rows, %.1f%% missing, %d gaps total, max gap %.1f min",
        report.total_rows, report.missing_pct, report.total_gaps, report.max_gap_minutes,
    )

    if report.gap_distribution:
        dist_str = ", ".join(f"{k}:{v}" for k, v in report.gap_distribution.items() if v > 0)
        if dist_str:
            logger.info("Gap distribution: %s", dist_str)

    # Check thresholds
    warn_threshold = getattr(cfg, 'WARN_DATA_GAP_MINS', 10)
    max_threshold = getattr(cfg, 'MAX_DATA_GAP_MINS', 60)

    if report.gaps_over_warn_threshold > 0:
        logger.warning(
            "⚠️ Found %d gaps exceeding %d minutes (warn threshold)",
            report.gaps_over_warn_threshold, warn_threshold,
        )

    if report.gaps_over_max_threshold > 0:
        msg = (
            f"🛑 FATAL DATA ERROR: Found {report.gaps_over_max_threshold} gaps exceeding "
            f"{max_threshold} minutes (max threshold). Max gap: {report.max_gap_minutes:.1f} min."
        )
        if strict:
            logger.critical(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg + " (strict=False, continuing anyway)")

    if report.max_gap_minutes <= warn_threshold:
        logger.info("✅ Data continuity check passed (no critical gaps).")

    return report


def interpolate_small_gaps(
    df: pd.DataFrame,
    max_interpolate_mins: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Interpolate small gaps in price data.

    Problem #8 Fix: For small gaps (e.g., < 5 minutes), interpolation is safer
    than having holes in the data for Kalman filter.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with DatetimeIndex
    max_interpolate_mins : int, optional
        Maximum gap size to interpolate. Default from config.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with gaps filled
    n_interpolated : int
        Number of gaps that were interpolated
    """
    if df.empty:
        return df, 0

    if max_interpolate_mins is None:
        max_interpolate_mins = getattr(cfg, 'MAX_INTERPOLATE_MINS', 5)

    # Identify gaps
    time_diffs = df.index.to_series().diff()
    expected_freq = pd.Timedelta(minutes=1)  # Assume 1-min bars
    max_interpolate = pd.Timedelta(minutes=max_interpolate_mins)

    # Find small gaps (> expected but <= max_interpolate)
    small_gaps = time_diffs[(time_diffs > expected_freq * 1.5) & (time_diffs <= max_interpolate)]

    if small_gaps.empty:
        return df, 0

    # Resample to fill gaps, then interpolate
    # This creates rows for missing timestamps
    df_resampled = df.resample('1min').asfreq()

    # Count new rows (interpolated)
    n_new = len(df_resampled) - len(df)

    # Interpolate numeric columns
    numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
    df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method='linear')

    # Forward fill any remaining NaN (at edges)
    df_resampled = df_resampled.ffill().bfill()

    logger.info("Interpolated %d small gaps (%d new rows)", len(small_gaps), n_new)

    return df_resampled, n_new


def resample_to_timeframe(
    df: pd.DataFrame,
    target_tf: str = "15min",
    source_tf: str = "1min",
) -> pd.DataFrame:
    """
    Resample price data from source timeframe to target timeframe.

    Problem #1 Fix: Aggregate 1-min bars to higher timeframes where
    mean-reversion edge exceeds trading friction.

    For price data (close prices), we take the last value in each period.
    This is appropriate for pairs trading signals where we care about
    the price at each signal evaluation point.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with DatetimeIndex. Columns are asset prices.
    target_tf : str
        Target timeframe pandas offset alias: "5min", "15min", "30min", "1h", "4h"
    source_tf : str
        Source timeframe (for logging). Default "1min".

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame at target timeframe
    """
    supported = getattr(cfg, "SUPPORTED_TIMEFRAMES", ("1min", "5min", "15min", "30min", "1h", "4h"))
    if target_tf not in supported:
        raise ValueError(f"Unsupported timeframe: {target_tf}. Supported: {supported}")

    if target_tf == source_tf:
        logger.info("Target timeframe == source timeframe (%s), no resampling needed.", target_tf)
        return df

    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex for resampling.")

    # For price data, take last price in each bar (close)
    # This is appropriate for signal generation on close prices
    df_resampled = df.resample(target_tf).last()

    # Drop any rows that are all NaN (can happen at edges)
    df_resampled = df_resampled.dropna(how='all')

    # Calculate aggregation ratio
    source_rows = len(df)
    target_rows = len(df_resampled)
    ratio = source_rows / target_rows if target_rows > 0 else 0

    logger.info(
        "Resampled %s -> %s: %d rows -> %d rows (%.1fx aggregation)",
        source_tf, target_tf, source_rows, target_rows, ratio
    )

    return df_resampled


def get_timeframe_minutes(tf: str) -> int:
    """
    Convert timeframe string to minutes.

    Parameters
    ----------
    tf : str
        Timeframe string like "1min", "5min", "15min", "1h", "4h"

    Returns
    -------
    int
        Number of minutes
    """
    tf_map = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1h": 60,
        "4h": 240,
    }
    if tf not in tf_map:
        raise ValueError(f"Unknown timeframe: {tf}")
    return tf_map[tf]


def _assert_index_sane(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Parquet data must be indexed by a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex is not monotonic increasing after sort.")
    if df.index.has_duplicates:
        # Backtest assumes one row per bar; duplicates cause silent errors in loops.
        raise ValueError("DatetimeIndex contains duplicate timestamps. Deduplicate upstream.")


def load_and_split(
    parquet_path: Optional[Union[str, Path]] = None,
    signal_timeframe: Optional[str] = None,
    strict_validation: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price data and split into train/test sets.

    Parameters
    ----------
    parquet_path : Path, optional
        Path to parquet file. Default from config.
    signal_timeframe : str, optional
        Target timeframe for signal generation (Problem #1 fix).
        If None, uses cfg.SIGNAL_TIMEFRAME.
        Set to "1min" to disable resampling.
    strict_validation : bool
        If True (default), raise error on data gaps exceeding MAX_DATA_GAP_MINS.
        If False, warn but continue (useful for testing with imperfect data).

    Returns
    -------
    train_df, test_df : tuple of DataFrame
        Train and test sets at the specified timeframe.
    """
    path: Path = Path(parquet_path) if parquet_path is not None else cfg.PATH_RAW_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    logger.info("1️⃣ Loading data from %s ...", path)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("pd.read_parquet failed (%s). Falling back to pyarrow manual read.", exc)
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            names = table.schema.names
            if "timestamp" in names:
                # Use to_pylist() instead of to_pandas() for better compatibility
                ts = table.column("timestamp").to_pylist()
                data = {}
                for name in names:
                    if name == "timestamp":
                        continue
                    # Use to_pylist() for reliable conversion
                    data[name] = table.column(name).to_pylist()
                df = pd.DataFrame(data)
                df.index = pd.to_datetime(ts)
                df.index.name = "timestamp"
            else:
                data = {name: table.column(name).to_pylist() for name in names}
                df = pd.DataFrame(data)
        except Exception:
            raise

    # Ensure sorted by time (critical for both continuity and split)
    df = df.sort_index()
    _assert_index_sane(df)

    # --- Phase 0.5: Interpolate small gaps (Problem #8 fix) ---
    df, n_interp = interpolate_small_gaps(df)
    if n_interp > 0:
        logger.info("✅ Interpolated %d missing rows before continuity check.", n_interp)

    # --- Phase 1: Safety check on FULL DF (at source timeframe) ---
    validate_data_continuity(df, strict=strict_validation)

    # --- Phase 1.5: Resample to signal timeframe (Problem #1 fix) ---
    target_tf = signal_timeframe if signal_timeframe is not None else getattr(cfg, "SIGNAL_TIMEFRAME", "1min")
    if target_tf != "1min":
        logger.info("2️⃣ Resampling to signal timeframe: %s", target_tf)
        df = resample_to_timeframe(df, target_tf=target_tf, source_tf="1min")
        _assert_index_sane(df)  # Re-validate after resampling

    # --- Phase 2: Split ---
    n_rows = len(df)
    if n_rows == 0:
        logger.warning("Loaded DataFrame is empty; returning empty train/test.")
        return df.copy(), df.copy()

    if getattr(cfg, "SPLIT_MODE", "ratio") == "days":
        # Split by day counts (recommended for your 45/15 plan)
        # Assumes minute bars but uses timestamps (robust to missing rows if continuity passed).
        train_end = df.index.min() + pd.Timedelta(days=cfg.TRAIN_DAYS)
        test_end = train_end + pd.Timedelta(days=cfg.TEST_DAYS)

        train_df = df[df.index < train_end]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Day-based split produced empty train or test set. "
                "Check TRAIN_DAYS/TEST_DAYS vs dataset span."
            )
    else:
        # Default: ratio split
        split_idx = int(n_rows * (1.0 - cfg.TEST_RATIO))
        split_idx = max(1, min(split_idx, n_rows - 1))  # ensure both sets non-empty
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    # --- Phase 3: Look-ahead prevention (strict) ---
    train_max = train_df.index.max()
    test_min = test_df.index.min()
    if train_max >= test_min:
        msg = (
            f"❌ TIME LEAK: Train max timestamp ({train_max}) "
            f">= test min timestamp ({test_min}). "
            "Dataset is not strictly temporally separated."
        )
        logger.critical(msg)
        raise AssertionError(msg)

    logger.info(
        "   ✂️ Split: %s train rows | %s test rows",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
    )
    return train_df, test_df


# NOTE: For data export from QuestDB with coverage filtering, use:
#   research/pipeline/step0_export_data_full_year.py
# The exported parquet files are then used by the backtest via cfg.PATH_RAW_PARQUET.


def load_and_resample(
    parquet_path: Optional[Union[str, Path]] = None,
    signal_timeframe: Optional[str] = None,
    strict_validation: bool = True,
) -> pd.DataFrame:
    """
    Load raw parquet, interpolate small gaps, validate continuity, and resample to signal timeframe.
    """
    path: Path = Path(parquet_path) if parquet_path is not None else cfg.PATH_RAW_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    logger.info("1️⃣ Loading data from %s ...", path)
    df = pd.read_parquet(path)
    df = df.sort_index()
    _assert_index_sane(df)

    df, n_interp = interpolate_small_gaps(df)
    if n_interp > 0:
        logger.info("✅ Interpolated %d missing rows before continuity check.", n_interp)

    validate_data_continuity(df, strict=strict_validation)

    target_tf = signal_timeframe if signal_timeframe is not None else getattr(cfg, "SIGNAL_TIMEFRAME", "1min")
    if target_tf != "1min":
        logger.info("2️⃣ Resampling to signal timeframe: %s", target_tf)
        df = resample_to_timeframe(df, target_tf=target_tf, source_tf="1min")
        _assert_index_sane(df)

    return df
