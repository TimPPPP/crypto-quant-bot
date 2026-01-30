"""
Data Availability Validator.

Filters coins based on actual historical data availability in QuestDB.
Used to ensure all coins in universe have sufficient data for backtesting.
"""

import logging
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataValidator")

QUESTDB_HTTP_URL = "http://localhost:9000"


def check_data_availability(
    symbols: List[str],
    table_name: str = "candles_1m",
    min_days: int = 180,
    min_coverage: float = 0.95
) -> Dict[str, Dict]:
    """
    Check data availability for a list of symbols.

    Args:
        symbols: List of symbol strings to check
        table_name: QuestDB table to query
        min_days: Minimum number of days of data required
        min_coverage: Minimum coverage ratio (0-1)

    Returns:
        Dictionary mapping symbol to availability info:
        {
            'BTC': {
                'available': True,
                'days': 180,
                'coverage': 0.99,
                'first_date': '2025-06-16',
                'last_date': '2025-12-13'
            },
            ...
        }
    """
    results = {}

    for symbol in symbols:
        try:
            query = f"""
                SELECT
                    min(timestamp) as first_ts,
                    max(timestamp) as last_ts,
                    count(*) as records
                FROM {table_name}
                WHERE symbol = '{symbol}'
            """

            resp = requests.get(
                f"{QUESTDB_HTTP_URL}/exec",
                params={'query': query},
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                dataset = data.get('dataset', [])

                if dataset and dataset[0][0] is not None:
                    first_ts = dataset[0][0]
                    last_ts = dataset[0][1]
                    records = dataset[0][2]

                    # Parse timestamps
                    first_date = datetime.fromisoformat(first_ts.replace('Z', '+00:00'))
                    last_date = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))

                    # Calculate metrics
                    days = (last_date - first_date).days
                    expected_records = days * 24 * 60  # 1-minute candles
                    coverage = records / expected_records if expected_records > 0 else 0

                    results[symbol] = {
                        'available': days >= min_days and coverage >= min_coverage,
                        'days': days,
                        'coverage': round(coverage, 3),
                        'records': records,
                        'first_date': first_date.strftime('%Y-%m-%d'),
                        'last_date': last_date.strftime('%Y-%m-%d'),
                    }
                else:
                    results[symbol] = {
                        'available': False,
                        'days': 0,
                        'coverage': 0,
                        'records': 0,
                        'first_date': None,
                        'last_date': None,
                    }
        except Exception as e:
            logger.warning(f"Error checking {symbol}: {e}")
            results[symbol] = {
                'available': False,
                'error': str(e)
            }

    return results


def filter_by_availability(
    symbols: List[str],
    table_name: str = "candles_1m",
    min_days: int = 180,
    min_coverage: float = 0.95
) -> List[str]:
    """
    Filter symbols to only those with sufficient data.

    Args:
        symbols: List of symbol strings to filter
        table_name: QuestDB table to query
        min_days: Minimum number of days of data required
        min_coverage: Minimum coverage ratio (0-1)

    Returns:
        List of symbols that meet the availability criteria
    """
    availability = check_data_availability(symbols, table_name, min_days, min_coverage)

    available = [
        symbol for symbol, info in availability.items()
        if info.get('available', False)
    ]

    unavailable = [
        symbol for symbol, info in availability.items()
        if not info.get('available', False)
    ]

    logger.info(f"Data availability check:")
    logger.info(f"  Available: {len(available)}/{len(symbols)} coins")
    logger.info(f"  Unavailable: {unavailable}")

    return available


def get_validated_universe(
    top_n: int = 60,
    min_days: int = 180,
    min_coverage: float = 0.95
) -> List[str]:
    """
    Get universe with data availability validation.

    Fetches universe from Hyperliquid, then filters to only coins
    with sufficient historical data in QuestDB.

    Args:
        top_n: Target number of coins
        min_days: Minimum days of data required
        min_coverage: Minimum data coverage ratio

    Returns:
        List of validated symbols
    """
    from src.utils.universe import get_liquid_universe

    # Fetch extra coins to account for filtering
    fetch_count = int(top_n * 1.5)
    raw_universe = get_liquid_universe(top_n=fetch_count, use_buffer=True)

    logger.info(f"Raw universe: {len(raw_universe)} coins")

    # Filter by data availability
    validated = filter_by_availability(
        raw_universe,
        table_name="candles_1m",
        min_days=min_days,
        min_coverage=min_coverage
    )

    # Return top_n after filtering
    return validated[:top_n]


if __name__ == "__main__":
    # Test validation
    test_symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']

    print("Testing data availability...")
    availability = check_data_availability(test_symbols, min_days=180)

    for symbol, info in availability.items():
        status = "✓" if info.get('available') else "✗"
        print(f"{status} {symbol}: {info.get('days', 0)} days, {info.get('coverage', 0):.1%} coverage")
