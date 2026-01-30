from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when
# running this file directly (e.g. `poetry run python research/pipeline/step0_export_data.py`).
# This mirrors running as a package or having the project installed in the environment.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import config_backtest as cfg

# Multi-source merger (optional, for merged exports)
try:
    from src.collectors.merge_sources import export_merged_data, print_coverage_report
    MERGE_AVAILABLE = True
except ImportError:
    MERGE_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

QUESTDB_HOST: str = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PORT: int = int(os.getenv("QUESTDB_PORT", "9000"))
QUESTDB_EXPORT_URL: str = f"http://{QUESTDB_HOST}:{QUESTDB_PORT}/exp"

TABLE_NAME: str = "candles_1m"
FUNDING_TABLE_NAME: str = "funding_history"
LOOKBACK_DAYS: int = int(os.getenv("BACKTEST_LOOKBACK_DAYS", cfg.BACKTEST_LOOKBACK_DAYS))
MIN_DATA_THRESHOLD: float = float(os.getenv("MIN_DATA_THRESHOLD", cfg.MIN_DATA_COVERAGE))

logger = logging.getLogger("backtest.step0_export")


# =============================================================================
# Core Function
# =============================================================================

def export_funding_data(
    session: Optional[requests.Session] = None,
) -> Path:
    """
    Export funding rate data from QuestDB into a Parquet file.

    - Pulls data via QuestDB's /exp endpoint.
    - Pivots to a (timestamp x symbol) funding rate matrix.
    - Writes to data/raw_downloads/funding_rates.parquet.

    Returns
    -------
    output_path : Path
        The path to the written Parquet file.
    """
    output_path = cfg.RAW_DATA_DIR / "funding_rates.parquet"
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📤 Exporting funding rates from last %d days...", LOOKBACK_DAYS)

    base_query = f"""
    SELECT timestamp, symbol, funding_rate
    FROM {FUNDING_TABLE_NAME}
    WHERE timestamp >= dateadd('d', -{LOOKBACK_DAYS}, now())
    """
    query = base_query + """
      AND (data_source = 'hyperliquid' OR data_source IS NULL)
    ORDER BY timestamp ASC;
    """
    fallback_query = base_query + """
    ORDER BY timestamp ASC;
    """

    sess = session or requests.Session()

    try:
        try:
            resp = sess.get(
                QUESTDB_EXPORT_URL,
                params={"query": query},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "Funding export with data_source filter failed (%s). Retrying without filter.",
                exc,
            )
            resp = sess.get(
                QUESTDB_EXPORT_URL,
                params={"query": fallback_query},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()
        logger.info("   ↳ Streaming funding data...")

        df = pd.read_csv(
            resp.raw,
            parse_dates=["timestamp"],
            dtype={
                "funding_rate": "float32",
                "symbol": "category",
            },
        )

        if df.empty:
            logger.warning("⚠️ No funding data found, creating empty file")
            df = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

        logger.info("   ↳ Loaded %s funding records", f"{len(df):,}")

        # Deduplicate
        df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")

        # Pivot to (Time x Coins) funding rate matrix
        funding_matrix = df.pivot(index="timestamp", columns="symbol", values="funding_rate")
        funding_matrix = funding_matrix.sort_index()

        # Forward-fill missing values (funding rates persist until next update)
        funding_matrix = funding_matrix.ffill()

        # Fix categorical column index issue
        funding_matrix.columns = pd.Index(funding_matrix.columns.astype(str))

        logger.info(
            "   ↳ Funding matrix shape: %s (time) x %s (coins)",
            f"{funding_matrix.shape[0]:,}",
            f"{funding_matrix.shape[1]:,}",
        )

        # Write to parquet
        funding_matrix.to_parquet(output_path, engine='pyarrow', index=True)
        logger.info("✅ Funding data saved to: %s", output_path)
        return output_path

    except Exception as exc:
        logger.error("❌ Funding export failed: %s", exc)
        raise


def export_csv_stream_to_parquet(
    session: Optional[requests.Session] = None,
) -> Path:
    """
    Export 1-minute candle data from QuestDB into a Parquet price matrix.

    - Pulls data via QuestDB's /exp endpoint.
    - Pivots to a (timestamp x symbol) close-price matrix.
    - Applies basic quality filters (drops sparse coins).
    - Writes atomically to cfg.PATH_RAW_PARQUET.

    Parameters
    ----------
    session :
        Optional pre-configured requests.Session for connection pooling.

    Returns
    -------
    output_path : Path
        The path to the written Parquet file.

    Raises
    ------
    RuntimeError
        On HTTP / parsing / processing failures.
    """
    output_path: Path = cfg.PATH_RAW_PARQUET
    output_dir: Path = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📤 Exporting last %d days from QuestDB table '%s'...", LOOKBACK_DAYS, TABLE_NAME)

    # Database-side filtering to minimize transfer size
    query = f"""
    SELECT timestamp, symbol, close
    FROM {TABLE_NAME}
    WHERE timestamp >= dateadd('d', -{LOOKBACK_DAYS}, now())
    ORDER BY timestamp ASC;
    """

    sess = session or requests.Session()
    temp_path: Optional[Path] = None

    try:
        resp = sess.get(
            QUESTDB_EXPORT_URL,
            params={"query": query},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
        logger.info("   ↳ Connection established. Streaming CSV into pandas...")

        # NOTE: QuestDB /exp returns CSV. We stream via resp.raw.
        df = pd.read_csv(
            resp.raw,
            parse_dates=["timestamp"],
            dtype={
                "close": "float32",
                "symbol": "category",
            },
        )

        if df.empty:
            msg = f"No data returned from QuestDB table '{TABLE_NAME}'."
            logger.warning("⚠️ %s", msg)
            raise RuntimeError(msg)

        logger.info("   ↳ Loaded %s rows. Cleaning data...", f"{len(df):,}")

        # ---------------------------------------------------------------------
        # 1. Deduplicate (important if backfills overlap)
        # ---------------------------------------------------------------------
        df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")

        # ---------------------------------------------------------------------
        # 2. Pivot to (Time x Coins) price matrix
        # ---------------------------------------------------------------------
        price_matrix = df.pivot(index="timestamp", columns="symbol", values="close")

        # Ensure sorted index
        price_matrix = price_matrix.sort_index()

        # ---------------------------------------------------------------------
        # 3. Quality control: remove "ghost coins"
        # ---------------------------------------------------------------------
        valid_data_pct = 1.0 - price_matrix.isna().mean()
        drop_coins = valid_data_pct[valid_data_pct < MIN_DATA_THRESHOLD].index.tolist()

        if drop_coins:
            logger.info(
                "   🗑️ Dropping %d coins with < %.1f%% data coverage.",
                len(drop_coins),
                MIN_DATA_THRESHOLD * 100.0,
            )
            price_matrix = price_matrix.drop(columns=drop_coins)

        # ---------------------------------------------------------------------
        # 4. Forward-fill & drop all-NaN timestamps
        # ---------------------------------------------------------------------
        price_matrix = price_matrix.ffill()
        price_matrix = price_matrix.dropna(axis=0, how="all")

        logger.info(
            "   ↳ Final matrix shape: %s (time) x %s (coins)",
            f"{price_matrix.shape[0]:,}",
            f"{price_matrix.shape[1]:,}",
        )

        if price_matrix.empty:
            msg = "Final price matrix is empty after cleaning."
            logger.error(msg)
            raise RuntimeError(msg)

        # ---------------------------------------------------------------------
        # 5. Ensure parquet-friendly dtypes and atomic write
        # ---------------------------------------------------------------------
        # Convert any categorical columns to strings and make sure index/columns
        # are primitive types (pyarrow may choke on pandas categorical/index metadata).
        try:
            # Convert categorical columns to string
            for col in price_matrix.select_dtypes(include=["category"]).columns:
                price_matrix[col] = price_matrix[col].astype(str)

            # Ensure index is datetime (CRITICAL for backtest)
            if not isinstance(price_matrix.index, pd.DatetimeIndex):
                price_matrix.index = pd.to_datetime(price_matrix.index)

            # Ensure column names are primitive types
            if any(not isinstance(c, (str, int, float)) for c in price_matrix.columns):
                price_matrix.columns = [str(c) for c in price_matrix.columns]

            # Force columns to plain Index (not CategoricalIndex)
            price_matrix.columns = pd.Index(price_matrix.columns.astype(str))

            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            price_matrix.to_parquet(temp_path, engine='pyarrow', index=True)
            shutil.move(str(temp_path), output_path)

            logger.info("✅ Success! Data saved to: %s", output_path)
            return output_path
        except Exception as exc:
            logger.warning("Parquet write failed, attempting coercion: %s", exc)
            # Fallback: coerce everything conservatively
            pm = price_matrix.copy()
            pm.columns = pd.Index([str(c) for c in pm.columns])
            # Keep index as datetime, don't convert to string
            if not isinstance(pm.index, pd.DatetimeIndex):
                pm.index = pd.to_datetime(pm.index)
            for col in pm.columns:
                try:
                    pm[col] = pd.to_numeric(pm[col], errors="ignore")
                except Exception:
                    pm[col] = pm[col].astype(str)

            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            pm.to_parquet(temp_path, engine='pyarrow', index=True)
            shutil.move(str(temp_path), output_path)
            logger.info("✅ Success after coercion! Data saved to: %s", output_path)
            return output_path

    except requests.RequestException as exc:
        logger.error("❌ HTTP error while exporting from QuestDB: %s", exc)
        raise RuntimeError(f"QuestDB export failed: {exc}") from exc

    except Exception as exc:
        logger.error("❌ Export failed: %s", exc)
        raise

    finally:
        # Clean up temp file if it exists
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                logger.warning("Failed to remove temporary file: %s", temp_path)


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main(use_merged: bool = False, min_coverage: float = None) -> None:
    """
    Export data from QuestDB to parquet files.

    Args:
        use_merged: If True, use multi-source merger (Hyperliquid + Binance + Coinbase)
        min_coverage: Minimum coverage threshold (0-1). Default from config.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    coverage = min_coverage if min_coverage is not None else MIN_DATA_THRESHOLD

    try:
        if use_merged and MERGE_AVAILABLE:
            logger.info("=" * 60)
            logger.info("MULTI-SOURCE EXPORT MODE")
            logger.info("=" * 60)

            # Show coverage report first
            print_coverage_report(LOOKBACK_DAYS)

            # Export merged candle data
            export_merged_data(
                lookback_days=LOOKBACK_DAYS,
                min_coverage=coverage,
            )
        else:
            if use_merged and not MERGE_AVAILABLE:
                logger.warning("Merge module not available, falling back to single-source export")

            # Standard single-source export
            export_csv_stream_to_parquet()

        # Export funding rates (always from Hyperliquid)
        export_funding_data()

        logger.info("✅ All exports complete!")

    except Exception as exc:
        logger.error("Step0 export failed: %s", exc)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export data from QuestDB to parquet")
    parser.add_argument(
        "--merged", "-m",
        action="store_true",
        help="Use multi-source merger (Hyperliquid + Binance + Coinbase)"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help=f"Minimum data coverage threshold (0-1). Default: {MIN_DATA_THRESHOLD}"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Lookback days. Default: {LOOKBACK_DAYS}"
    )
    args = parser.parse_args()

    if args.days:
        LOOKBACK_DAYS = args.days

    main(use_merged=args.merged, min_coverage=args.min_coverage)
