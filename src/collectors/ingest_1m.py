import time
import os
import json
import requests
import logging
from datetime import datetime, timezone, timedelta
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError
from questdb.ingress import Sender, TimestampNanos

# --- CONFIGURATION ---
QUESTDB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
QUESTDB_PORT = 9009
QUESTDB_HTTP = f"http://{QUESTDB_HOST}:9000/exec"

TIMEFRAME = '1m'
LOOKBACK_DAYS = int(os.getenv('INGEST_LOOKBACK_DAYS', 180))
TOP_N_COINS = int(os.getenv('INGEST_TOP_N', 50))
PROGRESS_FILE = os.getenv('INGEST_PROGRESS_FILE', 'data/state/ingest_1m_progress.json')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_1m")

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
    Ensures the QuestDB table exists with DEDUP enabled.
    """
    logger.info(f"🛠️ Verifying schema for '{TIMEFRAME}'...")
    
    # PARTITION BY MONTH is safer for 1m data to avoid "Too many open files" errors
    ddl = f"""
    CREATE TABLE IF NOT EXISTS candles_{TIMEFRAME} (
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
    try:
        r = requests.get(QUESTDB_HTTP, params={'query': ddl})
        if r.status_code == 200:
            logger.info(f"   ✅ Table 'candles_{TIMEFRAME}' ready.")
        else:
            logger.warning(f"   ⚠️ Schema Warning: {r.text}")
    except Exception as e:
        logger.error(f"   ❌ DB Connection Failed: {e}")

def fetch_candles_with_retry(info, coin, timeframe, start_time, end_time, retries=3):
    delay = 2
    for _ in range(retries):
        try:
            return info.candles_snapshot(coin, timeframe, start_time, end_time)
        except ClientError as e:
            if e.args[0] == 429:
                logger.warning(f"     ⚠️ 429 Rate Limit. Sleeping {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return []
        except Exception:
            return []
    return []

def run_ingest(resume: bool = True):
    """
    Run candle data ingestion.

    Args:
        resume: If True, resume from last saved progress
    """
    ensure_table_schema()

    logger.info(f"Starting Ingest: {TIMEFRAME} candles, Last {LOOKBACK_DAYS} days.")

    coins = get_liquid_universe(top_n=TOP_N_COINS, use_buffer=True)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    # Load progress for resumption
    progress = load_progress() if resume else {}

    try:
        with Sender(QUESTDB_HOST, QUESTDB_PORT) as sender:
            for coin in coins:
                # Check if we should skip this coin (already completed)
                coin_progress = progress.get(coin, {})
                if coin_progress.get('completed', False):
                    logger.info(f"Skipping {coin} (already completed)")
                    continue

                logger.info(f"Processing {coin}...")

                # Time Window (UTC)
                end_time_ms = int(time.time() * 1000)
                start_time_ms = end_time_ms - (LOOKBACK_DAYS * 24 * 60 * 60 * 1000)

                # Resume from last position if available
                current_pointer = coin_progress.get('last_timestamp', start_time_ms)
                total_candles = coin_progress.get('total_candles', 0)
                consecutive_gaps = 0

                while current_pointer < end_time_ms:
                    candles = fetch_candles_with_retry(info, coin, TIMEFRAME, current_pointer, end_time_ms)

                    # Gap Handling
                    if not candles:
                        consecutive_gaps += 1
                        if consecutive_gaps > 10:
                            logger.warning(f"Too many gaps for {coin}, moving to next coin")
                            break
                        logger.warning(f"Data gap for {coin}. Skipping forward 4 hours...")
                        current_pointer += (4 * 60 * 60 * 1000)
                        continue

                    consecutive_gaps = 0

                    for c in candles:
                        ts_utc = datetime.fromtimestamp(c['t'] / 1000, tz=timezone.utc)

                        sender.row(
                            f'candles_{TIMEFRAME}',
                            symbols={'symbol': coin},
                            columns={
                                'open': float(c['o']),
                                'high': float(c['h']),
                                'low': float(c['l']),
                                'close': float(c['c']),
                                'volume': float(c['v']),
                                'turnover': float(c['v']) * float(c['c'])
                            },
                            at=TimestampNanos.from_datetime(ts_utc)
                        )

                    sender.flush()
                    total_candles += len(candles)

                    last_candle_time = candles[-1]['t']

                    # Save progress periodically
                    progress[coin] = {
                        'last_timestamp': last_candle_time,
                        'total_candles': total_candles,
                        'completed': False
                    }
                    save_progress(progress)

                    if last_candle_time >= end_time_ms - 60000:
                        break

                    if last_candle_time <= current_pointer:
                        current_pointer += (60 * 1000)
                    else:
                        current_pointer = last_candle_time + 1

                    time.sleep(0.1)

                # Mark coin as completed
                progress[coin] = {
                    'last_timestamp': end_time_ms,
                    'total_candles': total_candles,
                    'completed': True
                }
                save_progress(progress)
                logger.info(f"   Ingested {total_candles} candles for {coin}.")

            sender.flush()
            logger.info("All data flushed to QuestDB.")

    except Exception as e:
        logger.error(f"Fatal error during ingestion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Save progress before raising
        save_progress(progress)
        raise

if __name__ == "__main__":
    run_ingest()