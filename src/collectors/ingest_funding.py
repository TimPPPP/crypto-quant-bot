"""
Funding Rate Data Ingester for Hyperliquid.

Fetches historical funding rates and stores in QuestDB.
"""

import os
import time
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
logger = logging.getLogger("Ingest_Funding")


class FundingIngester(BaseIngester):
    """
    Ingester for funding rate data from Hyperliquid.
    """

    # Funding data has longer delays between requests
    REQUEST_DELAY = 0.3

    def __init__(
        self,
        top_n_coins: int = None,
        lookback_days: int = None,
        progress_file: str = None
    ):
        top_n_coins = top_n_coins or int(os.getenv('INGEST_TOP_N', 50))
        lookback_days = lookback_days or int(os.getenv('INGEST_LOOKBACK_DAYS', 180))
        progress_file = progress_file or os.getenv(
            'INGEST_FUNDING_PROGRESS',
            'data/state/ingest_funding_progress.json'
        )

        super().__init__(
            table_name='funding_history',
            progress_file=progress_file,
            top_n_coins=top_n_coins,
            lookback_days=lookback_days
        )

        # Initialize Hyperliquid client
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)

    def get_ddl(self) -> str:
        """Return the CREATE TABLE DDL."""
        return """
        CREATE TABLE IF NOT EXISTS funding_history (
            timestamp TIMESTAMP,
            symbol SYMBOL,
            funding_rate DOUBLE,
            premium DOUBLE,
            open_interest DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch funding data with backoff."""
        max_retries = 5
        delay = 2.0

        for attempt in range(max_retries):
            try:
                return self.info.funding_history(coin, start_time)
            except ClientError as e:
                if e.args[0] == 429:
                    self.logger.warning(f"Rate limit (429). Cooling down {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
            except Exception as e:
                self.logger.warning(f"Network error: {e}")
                time.sleep(1)

        return []

    def process_record(self, sender: Sender, coin: str, record: Dict):
        """Process a single funding record."""
        premium_val = float(record['premium']) if record['premium'] else 0.0
        funding_val = float(record['fundingRate'])

        ts_utc = datetime.fromtimestamp(record['time'] / 1000, tz=timezone.utc)

        sender.row(
            self.table_name,
            symbols={'symbol': coin},
            columns={
                'funding_rate': funding_val,
                'premium': premium_val,
                'open_interest': 0.0  # Not available in this API
            },
            at=TimestampNanos.from_datetime(ts_utc)
        )

    def get_last_timestamp(self, data: List[Dict]) -> int:
        """Get the last timestamp from funding data."""
        if not data:
            return 0
        return data[-1]['time']

    def _run_ingestion(self, sender: Sender, coins: List[str], progress: Dict):
        """
        Override to handle funding-specific logic.

        Funding API returns all data from start_time, so we check for
        reaching near-current time differently.
        """
        end_time_ms = int(time.time() * 1000)
        start_time_ms = end_time_ms - (self.lookback_days * 24 * 60 * 60 * 1000)

        for coin in coins:
            # Check if already completed
            coin_progress = progress.get(coin, {})
            if coin_progress.get('completed', False):
                self.logger.info(f"Skipping {coin} (already completed)")
                continue

            self.logger.info(f"Fetching History for {coin}...")

            # Resume from last position
            current_pointer = coin_progress.get('last_timestamp', start_time_ms)
            total_records = coin_progress.get('total_records', 0)
            consecutive_errors = 0

            while True:
                try:
                    data = self.fetch_data_batch(coin, current_pointer, end_time_ms)
                except Exception as e:
                    self.logger.warning(f"Fetch error for {coin}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        self.logger.warning(f"Too many errors for {coin}, moving to next")
                        break
                    time.sleep(1)
                    continue

                if not data:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        break
                    time.sleep(1)
                    continue

                consecutive_errors = 0

                for record in data:
                    self.process_record(sender, coin, record)

                sender.flush()
                total_records += len(data)

                last_time = self.get_last_timestamp(data)

                # Save progress
                progress[coin] = {
                    'last_timestamp': last_time,
                    'total_records': total_records,
                    'completed': False
                }
                self.save_progress(progress)

                # Stop if close to NOW (within 2 hours)
                if last_time > (time.time() * 1000) - (2 * 3600 * 1000):
                    break

                current_pointer = last_time + 1
                time.sleep(self.REQUEST_DELAY)

            # Mark as completed
            progress[coin] = {
                'last_timestamp': int(time.time() * 1000),
                'total_records': total_records,
                'completed': True
            }
            self.save_progress(progress)
            self.logger.info(f"Ingested {total_records} records for {coin}")


def run_funding_ingest(resume: bool = True):
    """
    Run funding rate data ingestion.

    Args:
        resume: If True, resume from last saved progress
    """
    ingester = FundingIngester()
    ingester.run(resume=resume)


if __name__ == "__main__":
    run_funding_ingest()
