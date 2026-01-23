from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.step0_export_full_year")


def _parse_date(value: str, end_of_day: bool = False) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if end_of_day and ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        ts = ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


def export_full_year_prices(
    *,
    start_date: str,
    end_date: str,
    min_coverage: float,
    output_path: Path,
    coins_path: Path,
    table_name: str,
    bar_minutes: int,
    session: Optional[requests.Session] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coins_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = _parse_date(start_date, end_of_day=False)
    end_ts = _parse_date(end_date, end_of_day=True)

    bar_delta = pd.Timedelta(minutes=bar_minutes)
    expected_bars = int(((end_ts - start_ts) / bar_delta) + 1)
    logger.info(
        "Exporting prices %s -> %s (%d expected bars, %d min bars, table=%s)",
        start_ts,
        end_ts,
        expected_bars,
        bar_minutes,
        table_name,
    )

    query = f"""
    SELECT timestamp, symbol, close
    FROM {table_name}
    WHERE timestamp >= '{start_ts.isoformat()}'
      AND timestamp <= '{end_ts.isoformat()}'
    ORDER BY timestamp ASC;
    """

    sess = session or requests.Session()
    resp = sess.get(
        f"http://{os.getenv('QUESTDB_HOST', 'localhost')}:{int(os.getenv('QUESTDB_PORT', '9000'))}/exp",
        params={"query": query},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    df = pd.read_csv(
        resp.raw,
        parse_dates=["timestamp"],
        dtype={"close": "float32", "symbol": "category"},
    )
    if df.empty:
        raise RuntimeError("No candle data returned for the requested range.")

    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    price_matrix = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    price_matrix.columns = pd.Index(price_matrix.columns.astype(str))

    # Reindex to full bar grid to measure coverage against the full range.
    full_index = pd.date_range(start=start_ts, end=end_ts, freq=bar_delta, tz="UTC")
    price_matrix = price_matrix.reindex(full_index)

    coverage = 1.0 - price_matrix.isna().mean()
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    logger.info("Coins meeting coverage >= %.2f: %d", min_coverage, len(keep_cols))

    keep_cols = sorted(keep_cols)
    price_matrix = price_matrix[keep_cols]

    # Forward-fill after filtering to avoid bias from dropped coins.
    price_matrix = price_matrix.ffill()

    # Save coin list and coverage stats
    stats = {
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "expected_bars": expected_bars,
        "min_coverage": min_coverage,
        "coins": keep_cols,
        "coverage": {c: float(coverage[c]) for c in keep_cols},
    }
    coins_path.write_text(json.dumps(stats, indent=2))
    logger.info("Saved coin coverage list: %s", coins_path)

    price_matrix.to_parquet(output_path, engine="pyarrow", index=True)
    logger.info("Saved price matrix: %s", output_path)
    return output_path


def export_full_year_funding(
    *,
    start_date: str,
    end_date: str,
    output_path: Path,
    session: Optional[requests.Session] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = _parse_date(start_date, end_of_day=False)
    end_ts = _parse_date(end_date, end_of_day=True)

    query = f"""
    SELECT timestamp, symbol, funding_rate
    FROM funding_history
    WHERE timestamp >= '{start_ts.isoformat()}'
      AND timestamp <= '{end_ts.isoformat()}'
      AND (data_source = 'hyperliquid' OR data_source IS NULL)
    ORDER BY timestamp ASC;
    """

    sess = session or requests.Session()
    resp = sess.get(
        f"http://{os.getenv('QUESTDB_HOST', 'localhost')}:{int(os.getenv('QUESTDB_PORT', '9000'))}/exp",
        params={"query": query},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    df = pd.read_csv(
        resp.raw,
        parse_dates=["timestamp"],
        dtype={"funding_rate": "float32", "symbol": "category"},
    )

    if df.empty:
        logger.warning("No funding data returned for the requested range.")
        df = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    funding_matrix = df.pivot(index="timestamp", columns="symbol", values="funding_rate").sort_index()
    funding_matrix.columns = pd.Index(funding_matrix.columns.astype(str))
    funding_matrix = funding_matrix.ffill()
    funding_matrix.columns = pd.Index(funding_matrix.columns.astype(str))

    funding_matrix.to_parquet(output_path, engine="pyarrow", index=True)
    logger.info("Saved funding matrix: %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    start_date = os.getenv("EXPORT_START_DATE", "2025-01-01")
    end_date = os.getenv("EXPORT_END_DATE", "2025-12-31")
    min_coverage = float(os.getenv("EXPORT_MIN_COVERAGE", "0.90"))
    table_name = os.getenv("EXPORT_TABLE", "candles_1m")
    bar_minutes = int(os.getenv("EXPORT_BAR_MINUTES", "1"))
    coverage_tag = f"{int(min_coverage * 100):02d}p"

    price_path = Path(
        os.getenv(
            "EXPORT_PRICE_PARQUET",
            str(cfg.RAW_DATA_DIR / f"crypto_prices_1m_2025_full_year_{coverage_tag}.parquet"),
        )
    )
    funding_path = Path(
        os.getenv(
            "EXPORT_FUNDING_PARQUET",
            str(cfg.RAW_DATA_DIR / "funding_rates_2025_full_year.parquet"),
        )
    )
    coins_path = Path(
        os.getenv(
            "EXPORT_COINS_JSON",
            str(cfg.RAW_DATA_DIR / f"full_year_coins_2025_{coverage_tag}.json"),
        )
    )

    export_full_year_prices(
        start_date=start_date,
        end_date=end_date,
        min_coverage=min_coverage,
        output_path=price_path,
        coins_path=coins_path,
        table_name=table_name,
        bar_minutes=bar_minutes,
    )
    export_full_year_funding(
        start_date=start_date,
        end_date=end_date,
        output_path=funding_path,
    )

    logger.info("✅ Full-year export complete.")


if __name__ == "__main__":
    main()
