import time
import os
import json
import requests
import logging
from datetime import datetime, timezone
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError
from questdb.ingress import Sender, TimestampNanos

# --- CONFIGURATION ---
QUESTDB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
QUESTDB_PORT = 9009
QUESTDB_HTTP = f"http://{QUESTDB_HOST}:9000/exec"
TOP_N_COINS = int(os.getenv('INGEST_TOP_N', 50))
LOOKBACK_DAYS = int(os.getenv('INGEST_LOOKBACK_DAYS', 180))
PROGRESS_FILE = os.getenv('INGEST_FUNDING_PROGRESS', 'data/state/ingest_funding_progress.json')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_Funding")

# Import Universe Utils
from src.utils.universe import get_liquid_universe


def load_progress():
    """Load progress from file to resume interrupted ingestion."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return {}


def save_progress(progress):
    """Save progress to file."""
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")

def ensure_table_schema():
    """
    FIX: Enforce DEDUP on the funding table.
    """
    logger.info("🛠️ Verifying QuestDB Funding Schema...")
    ddl = """
    CREATE TABLE IF NOT EXISTS funding_history (
        timestamp TIMESTAMP,
        symbol SYMBOL,
        funding_rate DOUBLE,
        premium DOUBLE,
        open_interest DOUBLE
    ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
    DEDUP UPSERT KEYS(timestamp, symbol);
    """
    try:
        r = requests.get(QUESTDB_HTTP, params={'query': ddl})
        if r.status_code == 200:
            logger.info("   ✅ Table 'funding_history' configured with DEDUP.")
        else:
            logger.warning(f"   ⚠️ Schema Warning: {r.text}")
    except Exception as e:
        logger.error(f"   ❌ DB Connection Failed: {e}")

def fetch_with_backoff(info_client, coin, start_time, max_retries=5):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return info_client.funding_history(coin, start_time)
        except ClientError as e:
            if e.args[0] == 429:
                logger.warning(f"    ⚠️  Rate Limit (429). Cooling down {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
        except Exception as e:
            logger.warning(f"    ❌ Network error: {e}")
            time.sleep(1)
    return []

def run_funding_ingest(resume: bool = True):
    """
    Run funding rate data ingestion.

    Args:
        resume: If True, resume from last saved progress
    """
    ensure_table_schema()

    logger.info(f"Starting Deep Funding Ingestion ({LOOKBACK_DAYS} days)...")

    coins = get_liquid_universe(top_n=TOP_N_COINS, use_buffer=True)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    # Load progress for resumption
    progress = load_progress() if resume else {}

    try:
        with Sender(QUESTDB_HOST, QUESTDB_PORT) as sender:
            for coin in coins:
                # Check if we should skip this coin
                coin_progress = progress.get(coin, {})
                if coin_progress.get('completed', False):
                    logger.info(f"Skipping {coin} (already completed)")
                    continue

                logger.info(f"Fetching History for {coin}...")

                # Start from saved position or from lookback period
                default_start = int((time.time() - (LOOKBACK_DAYS * 24 * 3600)) * 1000)
                start_time = coin_progress.get('last_timestamp', default_start)
                records_ingested = coin_progress.get('total_records', 0)
                consecutive_errors = 0

                while True:
                    funding_data = fetch_with_backoff(info, coin, start_time)

                    if not funding_data:
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            logger.warning(f"Too many errors for {coin}, moving to next")
                            break
                        time.sleep(1)
                        continue

                    consecutive_errors = 0

                    for row in funding_data:
                        premium_val = float(row['premium']) if row['premium'] else 0.0
                        funding_val = float(row['fundingRate'])

                        ts_utc = datetime.fromtimestamp(row['time'] / 1000, tz=timezone.utc)

                        sender.row(
                            'funding_history',
                            symbols={'symbol': coin},
                            columns={
                                'funding_rate': funding_val,
                                'premium': premium_val,
                                'open_interest': 0.0
                            },
                            at=TimestampNanos.from_datetime(ts_utc)
                        )

                    sender.flush()
                    records_ingested += len(funding_data)

                    last_time = funding_data[-1]['time']

                    # Save progress
                    progress[coin] = {
                        'last_timestamp': last_time,
                        'total_records': records_ingested,
                        'completed': False
                    }
                    save_progress(progress)

                    # Stop if close to NOW
                    if last_time > (time.time() * 1000) - (2 * 3600 * 1000):
                        break

                    start_time = last_time + 1
                    time.sleep(0.3)

                # Mark as completed
                progress[coin] = {
                    'last_timestamp': int(time.time() * 1000),
                    'total_records': records_ingested,
                    'completed': True
                }
                save_progress(progress)
                logger.info(f"   Ingested {records_ingested} records for {coin}")

            sender.flush()
            logger.info("All funding data flushed to QuestDB.")

    except Exception as e:
        logger.error(f"Fatal error during funding ingestion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        save_progress(progress)
        raise

if __name__ == "__main__":
    run_funding_ingest()