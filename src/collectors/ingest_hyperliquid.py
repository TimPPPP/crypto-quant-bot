"""
1-Minute Candle Data Ingester for Hyperliquid.

Fetches historical OHLCV data and stores in QuestDB.
Primary data source for Hyperliquid-native coins.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError
from questdb.ingress import Sender, TimestampNanos

from src.collectors.base_ingester import BaseIngester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_Hyperliquid")


class HyperliquidCandleIngester(BaseIngester):
    """
    Ingester for 1-minute candle data from Hyperliquid.
    """

    TIMEFRAME = '1m'
    DATA_SOURCE = 'hyperliquid'
    REQUEST_DELAY = 0.2  # 200ms between requests

    def __init__(
        self,
        top_n_coins: int = None,
        lookback_days: int = None,
        progress_file: str = None,
        symbols: Optional[List[str]] = None,
    ):
        top_n_coins = top_n_coins or int(os.getenv('INGEST_TOP_N', 100))
        lookback_days = lookback_days or int(os.getenv('INGEST_LOOKBACK_DAYS', 365))
        progress_file = progress_file or os.getenv(
            'HYPERLIQUID_PROGRESS_FILE',
            'data/state/ingest_hyperliquid_progress.json'
        )

        super().__init__(
            table_name='candles_1m',
            progress_file=progress_file,
            top_n_coins=top_n_coins,
            lookback_days=lookback_days
        )

        self.symbols = symbols
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)

    def get_ddl(self) -> str:
        """Return the CREATE TABLE DDL with data_source column."""
        return """
        CREATE TABLE IF NOT EXISTS candles_1m (
            timestamp TIMESTAMP,
            symbol SYMBOL capacity 256 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            turnover DOUBLE,
            data_source SYMBOL
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch candle data with exponential backoff retry."""
        max_retries = 5
        delay = 1.0

        for attempt in range(max_retries):
            try:
                data = self.info.candles_snapshot(coin, self.TIMEFRAME, start_time, end_time)
                return data if data else []
            except ClientError as e:
                error_code = e.args[0] if e.args else None
                if error_code == 429:
                    self.logger.warning(f"Rate limit (429) for {coin}. Backoff {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)  # Cap at 60s
                else:
                    self.logger.warning(f"Client error for {coin}: {e}")
                    return []
            except Exception as e:
                self.logger.warning(f"Error fetching {coin}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return []

        return []

    def process_record(self, sender: Sender, coin: str, record: Dict):
        """Process a single candle record with data_source tracking."""
        ts_utc = datetime.fromtimestamp(record['t'] / 1000, tz=timezone.utc)

        sender.row(
            self.table_name,
            symbols={
                'symbol': coin,
                'data_source': self.DATA_SOURCE,
            },
            columns={
                'open': float(record['o']),
                'high': float(record['h']),
                'low': float(record['l']),
                'close': float(record['c']),
                'volume': float(record['v']),
                'turnover': float(record['v']) * float(record['c'])
            },
            at=TimestampNanos.from_datetime(ts_utc)
        )

    def get_last_timestamp(self, data: List[Dict]) -> int:
        """Get the last timestamp from candle data."""
        if not data:
            return 0
        return data[-1]['t']


def run_ingest(resume: bool = True, symbols: Optional[List[str]] = None):
    """
    Run Hyperliquid candle data ingestion.

    Args:
        resume: If True, resume from last saved progress
        symbols: Optional list of specific symbols to ingest
    """
    ingester = HyperliquidCandleIngester(symbols=symbols)
    ingester.run(resume=resume)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Hyperliquid candle data")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--top-n", type=int, default=100, help="Top N coins")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to ingest")
    args = parser.parse_args()

    os.environ['INGEST_LOOKBACK_DAYS'] = str(args.days)
    os.environ['INGEST_TOP_N'] = str(args.top_n)

    run_ingest(resume=not args.no_resume, symbols=args.symbols)
