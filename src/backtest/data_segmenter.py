from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from . import config_backtest as cfg

logger = logging.getLogger("backtest.segmenter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def validate_data_continuity(df: pd.DataFrame) -> None:
    if df.empty:
        logger.warning("Data continuity check skipped: empty DataFrame.")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for continuity checks.")

    time_diffs = df.index.to_series().diff().dropna()
    if time_diffs.empty:
        logger.info("✅ Data continuity check passed (single row or no diffs).")
        return

    max_gap = time_diffs.max()
    limit = pd.Timedelta(minutes=cfg.MAX_DATA_GAP_MINS)

    if max_gap > limit:
        msg = (
            f"🛑 FATAL DATA ERROR: Detected gap of {max_gap} between rows; "
            f"limit is {limit}. Fix data or increase MAX_DATA_GAP_MINS."
        )
        logger.critical(msg)
        raise ValueError(msg)

    logger.info("✅ Data continuity check passed (no critical gaps).")


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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # --- Phase 1: Safety check on FULL DF ---
    validate_data_continuity(df)

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
