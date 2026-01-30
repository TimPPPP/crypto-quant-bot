"""
1-Minute Candle Data Ingester for Hyperliquid.

Fetches historical OHLCV data and stores in QuestDB.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError
from questdb.ingress import Sender, TimestampNanos

from src.collectors.base_ingester import BaseIngester

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_1m")


class CandleIngester(BaseIngester):
    """
    Ingester for 1-minute candle data from Hyperliquid.
    """

    TIMEFRAME = '1m'

    def __init__(
        self,
        top_n_coins: int = None,
        lookback_days: int = None,
        progress_file: str = None
    ):
        top_n_coins = top_n_coins or int(os.getenv('INGEST_TOP_N', 50))
        lookback_days = lookback_days or int(os.getenv('INGEST_LOOKBACK_DAYS', 180))
        progress_file = progress_file or os.getenv(
            'INGEST_PROGRESS_FILE',
            'data/state/ingest_1m_progress.json'
        )

        super().__init__(
            table_name=f'candles_{self.TIMEFRAME}',
            progress_file=progress_file,
            top_n_coins=top_n_coins,
            lookback_days=lookback_days
        )

        # Initialize Hyperliquid client
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)

    def get_ddl(self) -> str:
        """Return the CREATE TABLE DDL."""
        return f"""
        CREATE TABLE IF NOT EXISTS candles_{self.TIMEFRAME} (
            timestamp TIMESTAMP,
            symbol SYMBOL capacity 256 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            turnover DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch candle data with retry logic."""
        retries = 3
        delay = 2

        for _ in range(retries):
            try:
                return self.info.candles_snapshot(coin, self.TIMEFRAME, start_time, end_time)
            except ClientError as e:
                if e.args[0] == 429:
                    self.logger.warning(f"Rate limit (429). Sleeping {delay}s...")
                    import time
                    time.sleep(delay)
                    delay *= 2
                else:
                    return []
            except Exception:
                return []

        return []

    def process_record(self, sender: Sender, coin: str, record: Dict):
        """Process a single candle record."""
        ts_utc = datetime.fromtimestamp(record['t'] / 1000, tz=timezone.utc)

        sender.row(
            self.table_name,
            symbols={'symbol': coin},
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


def run_ingest(resume: bool = True):
    """
    Run candle data ingestion.

    Args:
        resume: If True, resume from last saved progress
    """
    ingester = CandleIngester()
    ingester.run(resume=resume)


if __name__ == "__main__":
    run_ingest()
