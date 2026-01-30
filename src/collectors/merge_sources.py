"""
Multi-Source Data Merger.

Combines candle data from multiple exchanges (Hyperliquid, Binance, Coinbase)
with intelligent gap filling and priority-based conflict resolution.

Data Priority:
1. Hyperliquid (native, highest priority)
2. Binance (best liquidity)
3. Coinbase (fallback)
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

from src.collectors.symbol_mapping import (
    get_binance_symbol,
    get_coinbase_symbol,
    get_price_multiplier,
    get_available_sources,
    get_source_priority,
    HYPERLIQUID_ONLY,
)
from src.backtest import config_backtest as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MergeSource")

# QuestDB connection
QUESTDB_HOST = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PORT = int(os.getenv("QUESTDB_PORT", "9000"))
QUESTDB_EXPORT_URL = f"http://{QUESTDB_HOST}:{QUESTDB_PORT}/exp"


def query_questdb(query: str, timeout: int = 120) -> pd.DataFrame:
    """Execute a query against QuestDB and return a DataFrame."""
    try:
        resp = requests.get(
            QUESTDB_EXPORT_URL,
            params={"query": query},
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        return df
    except Exception as e:
        logger.error(f"QuestDB query failed: {e}")
        return pd.DataFrame()


def fetch_candles_by_source(
    lookback_days: int = 365,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch candle data from QuestDB, optionally filtered by source.

    Returns DataFrame with columns: timestamp, symbol, close, data_source
    """
    source_filter = ""
    if source:
        source_filter = f"AND data_source = '{source}'"

    query = f"""
    SELECT timestamp, symbol, close, data_source
    FROM candles_1m
    WHERE timestamp >= dateadd('d', -{lookback_days}, now())
    {source_filter}
    ORDER BY timestamp ASC;
    """

    df = query_questdb(query)

    if df.empty:
        return df

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_coverage_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate coverage statistics per symbol and source.

    Returns dict: {symbol: {source: coverage_pct, ...}, ...}
    """
    if df.empty:
        return {}

    stats = {}
    total_timestamps = df['timestamp'].nunique()

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        stats[symbol] = {}

        for source in symbol_df['data_source'].unique():
            source_df = symbol_df[symbol_df['data_source'] == source]
            coverage = len(source_df) / total_timestamps * 100
            stats[symbol][source] = {
                'coverage': coverage,
                'records': len(source_df),
                'first': source_df['timestamp'].min(),
                'last': source_df['timestamp'].max(),
            }

    return stats


def merge_candle_data(
    lookback_days: int = 365,
    min_coverage: float = 0.85,
    fill_gaps: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge candle data from all sources into a unified price matrix.

    Strategy:
    1. Use Hyperliquid as primary source
    2. Fill gaps with Binance data
    3. Fill remaining gaps with Coinbase
    4. Forward-fill any small remaining gaps

    Args:
        lookback_days: How many days of data to fetch
        min_coverage: Minimum coverage (0-1) to include a symbol
        fill_gaps: Whether to fill gaps from secondary sources

    Returns:
        Tuple of (price_matrix DataFrame, merge_stats dict)
    """
    logger.info(f"Fetching candle data for last {lookback_days} days...")

    # Fetch all data
    all_data = fetch_candles_by_source(lookback_days)

    if all_data.empty:
        logger.warning("No data fetched from QuestDB")
        return pd.DataFrame(), {}

    logger.info(f"Fetched {len(all_data):,} records for {all_data['symbol'].nunique()} symbols")

    # Get unique symbols and timestamps
    symbols = sorted(all_data['symbol'].unique())
    all_timestamps = all_data['timestamp'].sort_values().unique()

    logger.info(f"Date range: {all_timestamps[0]} to {all_timestamps[-1]}")

    # Create expected timestamp index (1-min frequency)
    ts_start = pd.Timestamp(all_timestamps[0]).floor('T')
    ts_end = pd.Timestamp(all_timestamps[-1]).ceil('T')
    expected_index = pd.date_range(start=ts_start, end=ts_end, freq='1min', tz='UTC')

    # Initialize result matrix
    price_matrix = pd.DataFrame(index=expected_index, columns=symbols, dtype=float)

    merge_stats = {
        'total_symbols': len(symbols),
        'lookback_days': lookback_days,
        'expected_bars': len(expected_index),
        'symbol_stats': {},
    }

    # Priority order for sources
    # Note: Hyperliquid only has ~3.5 days of data, so OKX/Coinbase are primary for historical
    # Hyperliquid data is still highest priority WHEN AVAILABLE (for recent bars)
    source_priority = ['hyperliquid', 'okx', 'coinbase', 'binance']

    # Process each symbol
    for symbol in symbols:
        symbol_data = all_data[all_data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('timestamp')

        # Track which sources provided data
        sources_used = []
        gaps_filled = 0

        # Get data by source priority
        for source in source_priority:
            source_data = symbol_data[symbol_data['data_source'] == source]
            if source_data.empty:
                continue

            # Apply price multiplier if needed
            multiplier = get_price_multiplier(symbol, source)

            for _, row in source_data.iterrows():
                ts = row['timestamp']
                if ts in price_matrix.index:
                    # Only fill if not already set (priority)
                    if pd.isna(price_matrix.loc[ts, symbol]):
                        price_matrix.loc[ts, symbol] = row['close'] * multiplier
                        if source != 'hyperliquid':
                            gaps_filled += 1

            if source not in sources_used:
                sources_used.append(source)

        # Calculate coverage
        non_nan = price_matrix[symbol].notna().sum()
        coverage = non_nan / len(expected_index)

        merge_stats['symbol_stats'][symbol] = {
            'coverage': coverage * 100,
            'sources': sources_used,
            'gaps_filled': gaps_filled,
            'primary_source': sources_used[0] if sources_used else None,
        }

    # Forward-fill small gaps (up to 5 minutes)
    if fill_gaps:
        logger.info("Forward-filling small gaps...")
        price_matrix = price_matrix.ffill(limit=5)

    # Filter symbols by coverage
    valid_data_pct = price_matrix.notna().mean()
    drop_symbols = valid_data_pct[valid_data_pct < min_coverage].index.tolist()

    if drop_symbols:
        logger.info(f"Dropping {len(drop_symbols)} symbols with < {min_coverage*100:.0f}% coverage")
        price_matrix = price_matrix.drop(columns=drop_symbols)

        for sym in drop_symbols:
            if sym in merge_stats['symbol_stats']:
                merge_stats['symbol_stats'][sym]['dropped'] = True

    merge_stats['final_symbols'] = len(price_matrix.columns)
    merge_stats['dropped_symbols'] = len(drop_symbols)

    # Final forward-fill
    price_matrix = price_matrix.ffill()

    logger.info(f"Merged data: {len(price_matrix.columns)} symbols, {len(price_matrix)} bars")

    return price_matrix, merge_stats


def export_merged_data(
    output_path: Optional[Path] = None,
    lookback_days: int = 365,
    min_coverage: float = 0.85,
) -> Path:
    """
    Export merged candle data to parquet file.

    Args:
        output_path: Path to save parquet file (default: cfg.PATH_RAW_PARQUET)
        lookback_days: How many days of data to fetch
        min_coverage: Minimum coverage to include a symbol

    Returns:
        Path to the exported file
    """
    output_path = output_path or cfg.PATH_RAW_PARQUET
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MULTI-SOURCE DATA EXPORT")
    logger.info("=" * 60)

    # Merge data from all sources
    price_matrix, stats = merge_candle_data(
        lookback_days=lookback_days,
        min_coverage=min_coverage,
    )

    if price_matrix.empty:
        logger.error("No data to export!")
        return output_path

    # Write atomically (temp file then rename)
    temp_path = output_path.with_suffix('.tmp')
    price_matrix.to_parquet(temp_path, engine='pyarrow')
    temp_path.rename(output_path)

    logger.info(f"Exported to: {output_path}")
    logger.info(f"  Symbols: {len(price_matrix.columns)}")
    logger.info(f"  Bars: {len(price_matrix):,}")
    logger.info(f"  Date range: {price_matrix.index.min()} to {price_matrix.index.max()}")

    # Log source breakdown
    source_counts = {'hyperliquid': 0, 'binance': 0, 'coinbase': 0}
    for sym, sym_stats in stats['symbol_stats'].items():
        if sym_stats.get('dropped'):
            continue
        primary = sym_stats.get('primary_source')
        if primary:
            source_counts[primary] = source_counts.get(primary, 0) + 1

    logger.info("  Primary sources: " + ", ".join(f"{k}: {v}" for k, v in source_counts.items()))

    return output_path


def print_coverage_report(lookback_days: int = 365):
    """Print a detailed coverage report for all symbols and sources."""
    all_data = fetch_candles_by_source(lookback_days)

    if all_data.empty:
        print("No data available")
        return

    # Calculate expected bars
    ts_range = all_data['timestamp'].max() - all_data['timestamp'].min()
    expected_bars = int(ts_range.total_seconds() / 60) + 1

    print("\n" + "=" * 80)
    print("MULTI-SOURCE COVERAGE REPORT")
    print("=" * 80)
    print(f"Date range: {all_data['timestamp'].min()} to {all_data['timestamp'].max()}")
    print(f"Expected bars: {expected_bars:,}")
    print()

    # Group by symbol
    symbols = sorted(all_data['symbol'].unique())

    print(f"{'Symbol':<12} {'Hyperliquid':>12} {'Binance':>12} {'Coinbase':>12} {'Combined':>12}")
    print("-" * 64)

    for symbol in symbols:
        sym_data = all_data[all_data['symbol'] == symbol]

        hl_count = len(sym_data[sym_data['data_source'] == 'hyperliquid'])
        bn_count = len(sym_data[sym_data['data_source'] == 'binance'])
        cb_count = len(sym_data[sym_data['data_source'] == 'coinbase'])

        # Combined = unique timestamps
        combined = sym_data['timestamp'].nunique()

        hl_pct = hl_count / expected_bars * 100
        bn_pct = bn_count / expected_bars * 100
        cb_pct = cb_count / expected_bars * 100
        combined_pct = combined / expected_bars * 100

        print(f"{symbol:<12} {hl_pct:>11.1f}% {bn_pct:>11.1f}% {cb_pct:>11.1f}% {combined_pct:>11.1f}%")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge multi-source candle data")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--min-coverage", type=float, default=0.85, help="Minimum coverage (0-1)")
    parser.add_argument("--report", action="store_true", help="Print coverage report only")
    parser.add_argument("--export", action="store_true", help="Export merged data")
    args = parser.parse_args()

    if args.report:
        print_coverage_report(args.days)
    elif args.export:
        export_merged_data(
            lookback_days=args.days,
            min_coverage=args.min_coverage,
        )
    else:
        # Default: show report
        print_coverage_report(args.days)
