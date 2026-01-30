#!/usr/bin/env python3
"""
Master Ingestion Script for Multi-Source Data Collection.

Orchestrates data ingestion from multiple exchanges:
1. Hyperliquid (primary - candles + funding)
2. Binance (secondary - gap filling)
3. Coinbase (tertiary - additional coverage)

Usage:
    python scripts/ingest_all.py                    # Run all ingesters
    python scripts/ingest_all.py --hyperliquid      # Only Hyperliquid
    python scripts/ingest_all.py --binance          # Only Binance
    python scripts/ingest_all.py --funding          # Only funding rates
    python scripts/ingest_all.py --days 180         # Custom lookback
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IngestAll")


def run_hyperliquid_candles(days: int, top_n: int, resume: bool):
    """Run Hyperliquid candle ingestion."""
    logger.info("=" * 60)
    logger.info("HYPERLIQUID CANDLE INGESTION")
    logger.info("=" * 60)

    os.environ['INGEST_LOOKBACK_DAYS'] = str(days)
    os.environ['INGEST_TOP_N'] = str(top_n)

    from src.collectors.ingest_hyperliquid import HyperliquidCandleIngester
    ingester = HyperliquidCandleIngester()
    ingester.run(resume=resume)


def run_okx_candles(days: int, top_n: int, resume: bool):
    """Run OKX candle ingestion."""
    logger.info("=" * 60)
    logger.info("OKX CANDLE INGESTION")
    logger.info("=" * 60)

    os.environ['INGEST_LOOKBACK_DAYS'] = str(days)
    os.environ['INGEST_TOP_N'] = str(top_n)

    from src.collectors.ingest_okx import OKXCandleIngester
    ingester = OKXCandleIngester()
    ingester.run(resume=resume)


def run_binance_candles(days: int, top_n: int, resume: bool):
    """Run Binance candle ingestion (if not geo-restricted)."""
    logger.info("=" * 60)
    logger.info("BINANCE CANDLE INGESTION")
    logger.info("=" * 60)

    os.environ['INGEST_LOOKBACK_DAYS'] = str(days)
    os.environ['INGEST_TOP_N'] = str(top_n)

    from src.collectors.ingest_binance import BinanceCandleIngester
    ingester = BinanceCandleIngester()
    ingester.run(resume=resume)


def run_coinbase_candles(days: int, top_n: int, resume: bool):
    """Run Coinbase candle ingestion."""
    logger.info("=" * 60)
    logger.info("COINBASE CANDLE INGESTION")
    logger.info("=" * 60)

    os.environ['INGEST_LOOKBACK_DAYS'] = str(days)
    os.environ['INGEST_TOP_N'] = str(top_n)

    from src.collectors.ingest_coinbase import CoinbaseIngester
    ingester = CoinbaseIngester()
    ingester.run(resume=resume)


def run_funding(days: int, top_n: int, resume: bool):
    """Run Hyperliquid funding rate ingestion."""
    logger.info("=" * 60)
    logger.info("FUNDING RATE INGESTION")
    logger.info("=" * 60)

    os.environ['INGEST_LOOKBACK_DAYS'] = str(days)
    os.environ['INGEST_TOP_N'] = str(top_n)

    from src.collectors.ingest_funding import FundingIngester
    ingester = FundingIngester()
    ingester.run(resume=resume)


def run_export(merged: bool, days: int, min_coverage: float):
    """Export data to parquet files."""
    logger.info("=" * 60)
    logger.info("DATA EXPORT")
    logger.info("=" * 60)

    os.environ['BACKTEST_LOOKBACK_DAYS'] = str(days)

    from research.pipeline.step0_export_data import main as export_main
    export_main(use_merged=merged, min_coverage=min_coverage)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-source data ingestion orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/ingest_all.py                     # All sources, 365 days
    python scripts/ingest_all.py --days 180          # 6 months
    python scripts/ingest_all.py --hyperliquid       # Only Hyperliquid candles
    python scripts/ingest_all.py --binance           # Only Binance candles
    python scripts/ingest_all.py --export --merged   # Export with merging
    python scripts/ingest_all.py --no-resume         # Fresh start (clear progress)
"""
    )

    # What to run
    parser.add_argument("--hyperliquid", "-hl", action="store_true",
                        help="Run Hyperliquid candle ingestion")
    parser.add_argument("--okx", "-ox", action="store_true",
                        help="Run OKX candle ingestion (recommended)")
    parser.add_argument("--binance", "-bn", action="store_true",
                        help="Run Binance candle ingestion (may be geo-restricted)")
    parser.add_argument("--coinbase", "-cb", action="store_true",
                        help="Run Coinbase candle ingestion")
    parser.add_argument("--funding", "-f", action="store_true",
                        help="Run funding rate ingestion")
    parser.add_argument("--export", "-e", action="store_true",
                        help="Export data to parquet after ingestion")

    # Options
    parser.add_argument("--days", "-d", type=int, default=365,
                        help="Lookback days (default: 365)")
    parser.add_argument("--top-n", "-n", type=int, default=100,
                        help="Top N coins to ingest (default: 100)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from progress")
    parser.add_argument("--merged", "-m", action="store_true",
                        help="Use merged export (multi-source)")
    parser.add_argument("--min-coverage", type=float, default=0.85,
                        help="Minimum data coverage (default: 0.85)")

    args = parser.parse_args()

    # If no specific source selected, run recommended sources
    # NOTE: Hyperliquid API only returns ~3.5 days of data, not historical!
    # Use OKX + Coinbase for historical data, Hyperliquid only for native tokens
    run_all = not any([args.hyperliquid, args.okx, args.binance, args.coinbase, args.funding, args.export])

    resume = not args.no_resume

    try:
        # OKX first - best historical data source (works in most regions)
        if run_all or args.okx:
            run_okx_candles(args.days, args.top_n, resume)

        # Coinbase second - good coverage for major coins
        if run_all or args.coinbase:
            run_coinbase_candles(args.days, args.top_n, resume)

        # Hyperliquid ONLY if explicitly requested (only provides ~3.5 days of data!)
        # Useful only for Hyperliquid-native tokens (HYPE, PURR) not on other exchanges
        if args.hyperliquid:
            logger.warning("NOTE: Hyperliquid API only provides ~3.5 days of recent data!")
            run_hyperliquid_candles(args.days, args.top_n, resume)

        # Binance only if explicitly requested (often geo-restricted)
        if args.binance:
            run_binance_candles(args.days, args.top_n, resume)

        if run_all or args.funding:
            run_funding(args.days, args.top_n, resume)

        if run_all or args.export:
            run_export(args.merged or run_all, args.days, args.min_coverage)

        logger.info("=" * 60)
        logger.info("ALL INGESTION COMPLETE!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
