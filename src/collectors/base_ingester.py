"""
Base class for data ingesters.

Provides common functionality for:
- Progress tracking and resumption
- QuestDB connection with retry logic
- Rate limiting and error handling
"""

import os
import json
import time
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from datetime import datetime, date, time as dt_time, timezone

import requests
from questdb.ingress import Sender, TimestampNanos

# --- CONFIGURATION ---
QUESTDB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
QUESTDB_PORT = 9009
QUESTDB_HTTP = f"http://{QUESTDB_HOST}:9000/exec"

logger = logging.getLogger("BaseIngester")

def _parse_iso_datetime(value: str, end_of_day: bool = False) -> datetime:
    """
    Parse an ISO date/datetime string into UTC.

    If a date-only string is provided and end_of_day is True, use 23:59:59.999.
    """
    raw = value.strip()
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        d = date.fromisoformat(raw)
        if end_of_day:
            dt = datetime.combine(d, dt_time(23, 59, 59, 999000))
        else:
            dt = datetime.combine(d, dt_time(0, 0, 0))
    else:
        if end_of_day and len(raw) <= 10:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class BaseIngester(ABC):
    """
    Abstract base class for data ingesters.

    Subclasses must implement:
    - table_name: Name of the QuestDB table
    - get_ddl(): Returns the CREATE TABLE DDL
    - fetch_data(): Fetches data from the source
    - process_record(): Processes a single record for insertion
    """

    # Configuration (can be overridden by subclasses)
    MAX_CONSECUTIVE_ERRORS = 10
    RETRY_DELAY_BASE = 2.0
    REQUEST_DELAY = 0.1  # Delay between requests in seconds

    def __init__(
        self,
        table_name: str,
        progress_file: str,
        top_n_coins: int = 50,
        lookback_days: int = 180
    ):
        """
        Initialize the ingester.

        Args:
            table_name: QuestDB table name
            progress_file: Path to progress tracking file
            top_n_coins: Number of top coins to ingest
            lookback_days: Days of historical data to fetch
        """
        self.table_name = table_name
        self.progress_file = progress_file
        self.top_n_coins = top_n_coins
        self.lookback_days = lookback_days
        self.start_time_ms, self.end_time_ms = self._resolve_time_range()

        # Setup logger for this instance
        self.logger = logging.getLogger(f"Ingester.{table_name}")

    def _resolve_time_range(self) -> tuple[Optional[int], Optional[int]]:
        """Resolve an explicit ingest time range from env vars, if provided."""
        start_ts = os.getenv("INGEST_START_TS")
        end_ts = os.getenv("INGEST_END_TS")
        start_date = os.getenv("INGEST_START_DATE")
        end_date = os.getenv("INGEST_END_DATE")

        if start_ts or end_ts:
            if not (start_ts and end_ts):
                raise ValueError("Both INGEST_START_TS and INGEST_END_TS must be set.")
            return int(start_ts), int(end_ts)

        if start_date or end_date:
            if not (start_date and end_date):
                raise ValueError("Both INGEST_START_DATE and INGEST_END_DATE must be set.")
            start_dt = _parse_iso_datetime(start_date, end_of_day=False)
            end_dt = _parse_iso_datetime(end_date, end_of_day=True)
            return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

        return None, None

    def _get_time_range(self) -> tuple[int, int]:
        """Return the effective start/end timestamps for ingestion in ms."""
        if self.start_time_ms is not None and self.end_time_ms is not None:
            return self.start_time_ms, self.end_time_ms

        end_time_ms = int(time.time() * 1000)
        start_time_ms = end_time_ms - (self.lookback_days * 24 * 60 * 60 * 1000)
        return start_time_ms, end_time_ms

    def load_progress(self) -> Dict:
        """Load progress from file to resume interrupted ingestion."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
        return {}

    def save_progress(self, progress: Dict):
        """Save progress to file."""
        try:
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            self.logger.warning(f"Could not save progress: {e}")

    def ensure_table_schema(self):
        """Ensure the QuestDB table exists with proper schema."""
        self.logger.info(f"Verifying schema for '{self.table_name}'...")

        ddl = self.get_ddl()
        try:
            r = requests.get(QUESTDB_HTTP, params={'query': ddl}, timeout=30)
            if r.status_code == 200:
                self.logger.info(f"Table '{self.table_name}' ready.")
            else:
                self.logger.warning(f"Schema Warning: {r.text}")
        except Exception as e:
            self.logger.error(f"DB Connection Failed: {e}")

    @abstractmethod
    def get_ddl(self) -> str:
        """Return the CREATE TABLE DDL for this ingester."""
        pass

    @abstractmethod
    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """
        Fetch a batch of data for a coin.

        Args:
            coin: Coin symbol
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of data records
        """
        pass

    @abstractmethod
    def process_record(self, sender: Sender, coin: str, record: Dict):
        """
        Process a single record and send to QuestDB.

        Args:
            sender: QuestDB Sender instance
            coin: Coin symbol
            record: Data record to process
        """
        pass

    def get_coins(self) -> List[str]:
        """Get list of coins to ingest."""
        from src.utils.universe import get_liquid_universe
        return get_liquid_universe(top_n=self.top_n_coins, use_buffer=True)

    def run(self, resume: bool = True, max_retries: int = 3):
        """
        Run the ingestion process.

        Args:
            resume: If True, resume from last saved progress
            max_retries: Maximum connection retries for QuestDB
        """
        self.ensure_table_schema()

        start_time_ms, end_time_ms = self._get_time_range()
        if self.start_time_ms is not None and self.end_time_ms is not None:
            start_ts = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc)
            end_ts = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc)
            self.logger.info(
                "Starting ingestion: %s, range %s -> %s (UTC).",
                self.table_name,
                start_ts.isoformat(),
                end_ts.isoformat(),
            )
        else:
            self.logger.info(f"Starting ingestion: {self.table_name}, Last {self.lookback_days} days.")

        coins = self.get_coins()
        progress = self.load_progress() if resume else {}

        # Retry loop for QuestDB connection
        for attempt in range(max_retries):
            try:
                with Sender(QUESTDB_HOST, QUESTDB_PORT) as sender:
                    self._run_ingestion(sender, coins, progress)
                    sender.flush()
                    self.logger.info("All data flushed to QuestDB.")
                return  # Success
            except ConnectionError as e:
                self.logger.error(f"QuestDB connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.RETRY_DELAY_BASE * (2 ** attempt))
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Fatal error during ingestion: {e}")
                self.logger.error(traceback.format_exc())
                self.save_progress(progress)
                raise

    def _run_ingestion(self, sender: Sender, coins: List[str], progress: Dict):
        """
        Internal ingestion loop.

        Args:
            sender: QuestDB Sender instance
            coins: List of coins to process
            progress: Progress tracking dictionary
        """
        start_time_ms, end_time_ms = self._get_time_range()

        for coin in coins:
            # Check if already completed
            coin_progress = progress.get(coin, {})
            if coin_progress.get('completed', False):
                self.logger.info(f"Skipping {coin} (already completed)")
                continue

            self.logger.info(f"Processing {coin}...")

            # Resume from last position if available
            current_pointer = coin_progress.get('last_timestamp', start_time_ms)
            total_records = coin_progress.get('total_records', 0)
            consecutive_errors = 0

            while current_pointer < end_time_ms:
                try:
                    data = self.fetch_data_batch(coin, current_pointer, end_time_ms)
                except Exception as e:
                    self.logger.warning(f"Fetch error for {coin}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > self.MAX_CONSECUTIVE_ERRORS:
                        self.logger.warning(f"Too many errors for {coin}, moving to next")
                        break
                    time.sleep(self.RETRY_DELAY_BASE)
                    continue

                if not data:
                    consecutive_errors += 1
                    if consecutive_errors > self.MAX_CONSECUTIVE_ERRORS:
                        self.logger.warning(f"No data/too many gaps for {coin}, moving to next")
                        break
                    # Skip forward on gaps
                    current_pointer += (4 * 60 * 60 * 1000)  # 4 hours
                    continue

                consecutive_errors = 0

                for record in data:
                    self.process_record(sender, coin, record)

                sender.flush()
                total_records += len(data)

                # Get last timestamp from data
                last_time = self.get_last_timestamp(data)

                # Save progress
                progress[coin] = {
                    'last_timestamp': last_time,
                    'total_records': total_records,
                    'completed': False
                }
                self.save_progress(progress)

                # Check if we've reached the end
                if last_time >= end_time_ms - (60 * 1000):
                    break

                # Move pointer forward
                if last_time <= current_pointer:
                    current_pointer += (60 * 1000)
                else:
                    current_pointer = last_time + 1

                time.sleep(self.REQUEST_DELAY)

            # Mark coin as completed
            progress[coin] = {
                'last_timestamp': end_time_ms,
                'total_records': total_records,
                'completed': True
            }
            self.save_progress(progress)
            self.logger.info(f"Ingested {total_records} records for {coin}.")

    def get_last_timestamp(self, data: List[Dict]) -> int:
        """
        Get the last timestamp from a batch of data.

        Override in subclass if timestamp field differs.
        """
        if not data:
            return 0
        # Default assumes 't' or 'time' field
        last = data[-1]
        return last.get('t', last.get('time', 0))
