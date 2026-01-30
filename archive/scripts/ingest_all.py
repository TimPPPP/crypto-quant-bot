#!/usr/bin/env python
"""
Parallel data ingestion script.

Runs ingest_1m (candles) and ingest_funding (funding rates) concurrently
to hydrate the QuestDB database with all necessary market data.

Usage:python3 tests/test_safety.py
  # From project root:
  python scripts/ingest_all.py

  # Or in Docker:
  docker-compose exec worker python scripts/ingest_all.py
"""

import sys
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IngestAll")


def run_ingest_1m():
    """Run 1-minute candle ingestion."""
    logger.info("Starting 1m candle ingestion...")
    from src.collectors.ingest_1m import run_ingest
    try:
        run_ingest()
        logger.info("✅ 1m candle ingestion completed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ 1m candle ingestion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_ingest_funding():
    """Run funding history ingestion."""
    logger.info("Starting funding history ingestion...")
    from src.collectors.ingest_funding import run_funding_ingest
    try:
        run_funding_ingest(resume=True)
        logger.info("✅ Funding history ingestion completed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Funding history ingestion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run both ingestions in parallel."""
    logger.info("=" * 80)
    logger.info("STARTING PARALLEL DATA INGESTION")
    logger.info("=" * 80)
    logger.info("This will fetch and ingest:")
    logger.info("  1. 1-minute candle data (OHLCV)")
    logger.info("  2. Funding rate history")
    logger.info("=" * 80)

    results = {}
    
    # Run both ingestions in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_1m = executor.submit(run_ingest_1m)
        future_funding = executor.submit(run_ingest_funding)

        # Collect results as they complete
        for future in as_completed([future_1m, future_funding]):
            try:
                result = future.result()
                if future == future_1m:
                    results['1m'] = result
                else:
                    results['funding'] = result
            except Exception as e:
                logger.error(f"Task failed with exception: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # Summary
    logger.info("=" * 80)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"1m Candles:       {'✅ SUCCESS' if results.get('1m') else '❌ FAILED'}")
    logger.info(f"Funding History:  {'✅ SUCCESS' if results.get('funding') else '❌ FAILED'}")
    logger.info("=" * 80)

    # Exit with appropriate code
    if all(results.values()):
        logger.info("All ingestions completed successfully!")
        return 0
    else:
        logger.warning("One or more ingestions failed. See logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
