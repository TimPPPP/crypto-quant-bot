"""
1-Minute Candle Data Ingester for OKX.

Fetches historical OHLCV data from OKX public API and stores in QuestDB.
Good alternative when Binance is unavailable due to geo-restrictions.

OKX API limits:
- Rate limit: 20 requests/2 seconds
- 100 candles per request max
- No authentication required for market data
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from questdb.ingress import Sender, TimestampNanos

from src.collectors.base_ingester import BaseIngester, QUESTDB_HOST, QUESTDB_PORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_OKX")

# OKX API endpoint
OKX_API_URL = "https://www.okx.com"


# Symbol mapping: Hyperliquid symbol -> OKX instrument ID
SYMBOL_MAP = {
    # Standard mappings
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT",
    "DOGE": "DOGE-USDT",
    "ADA": "ADA-USDT",
    "AVAX": "AVAX-USDT",
    "LINK": "LINK-USDT",
    "DOT": "DOT-USDT",
    "UNI": "UNI-USDT",
    "LTC": "LTC-USDT",
    "BCH": "BCH-USDT",
    "ATOM": "ATOM-USDT",
    "XLM": "XLM-USDT",
    "NEAR": "NEAR-USDT",
    "FIL": "FIL-USDT",
    "ARB": "ARB-USDT",
    "OP": "OP-USDT",
    "SUI": "SUI-USDT",
    "SEI": "SEI-USDT",
    "INJ": "INJ-USDT",
    "TIA": "TIA-USDT",
    "APT": "APT-USDT",
    "AAVE": "AAVE-USDT",
    "CRV": "CRV-USDT",
    "FET": "FET-USDT",
    "RENDER": "RENDER-USDT",
    "TAO": "TAO-USDT",
    "WIF": "WIF-USDT",
    "ONDO": "ONDO-USDT",
    "ICP": "ICP-USDT",
    "STX": "STX-USDT",
    "ENA": "ENA-USDT",
    "PENDLE": "PENDLE-USDT",
    "ZRO": "ZRO-USDT",
    "EIGEN": "EIGEN-USDT",
    "WLD": "WLD-USDT",
    "STRK": "STRK-USDT",
    "ZK": "ZK-USDT",
    "COMP": "COMP-USDT",
    "SNX": "SNX-USDT",
    "LDO": "LDO-USDT",
    "GMT": "GMT-USDT",
    "MINA": "MINA-USDT",
    "ETC": "ETC-USDT",
    "XMR": "XMR-USDT",
    "ZEC": "ZEC-USDT",
    "POPCAT": "POPCAT-USDT",
    "TURBO": "TURBO-USDT",
    "PNUT": "PNUT-USDT",
    "PENGU": "PENGU-USDT",
    "HBAR": "HBAR-USDT",
    "ALGO": "ALGO-USDT",
    # 1000x tokens - OKX uses same format as we do
    "kPEPE": "PEPE-USDT",
    "kBONK": "BONK-USDT",
    "kFLOKI": "FLOKI-USDT",
    "kSHIB": "SHIB-USDT",
}

# Coins not available on OKX
OKX_UNAVAILABLE = {
    "HYPE",       # Hyperliquid native
    "PURR",       # Hyperliquid native
    "FARTCOIN",   # Not on OKX
    "TRUMP",      # Not on OKX
    "KAITO",      # Not on OKX
    "BERA",       # Not on OKX
    "IP",         # Not on OKX
    "PROMPT",     # Not on OKX
    "ME",         # Not on OKX
    "AERO",       # Not on OKX
    "XPL",        # Not on OKX
    "ASTER",      # Not on OKX
    "FOGO",       # Not on OKX
    "WLFI",       # Not on OKX
    "MON",        # Not on OKX
    "PUMP",       # Not on OKX
}


def get_okx_symbol(hl_symbol: str) -> Optional[str]:
    """Convert Hyperliquid symbol to OKX instrument ID."""
    if hl_symbol in OKX_UNAVAILABLE:
        return None
    if hl_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[hl_symbol]
    # Default: append -USDT
    return f"{hl_symbol}-USDT"


class OKXCandleIngester(BaseIngester):
    """
    Ingester for 1-minute candle data from OKX.
    Uses spot market for best data availability.

    Note: OKX API paginates BACKWARDS (newest first), so we override
    _run_ingestion to handle this correctly.
    """

    TIMEFRAME = '1m'
    DATA_SOURCE = 'okx'
    MAX_CANDLES_PER_REQUEST = 100  # OKX limit
    REQUEST_DELAY = 0.15  # Stay under rate limit

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
            'OKX_PROGRESS_FILE',
            'data/state/ingest_okx_progress.json'
        )

        super().__init__(
            table_name='candles_1m',
            progress_file=progress_file,
            top_n_coins=top_n_coins,
            lookback_days=lookback_days
        )

        self.symbols = symbols
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'crypto-quant-bot/1.0',
            'Accept': 'application/json',
        })

    def get_ddl(self) -> str:
        """Return the CREATE TABLE DDL."""
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

    def get_coins(self) -> List[str]:
        """Get list of coins to ingest, filtering unavailable ones."""
        from src.utils.universe import get_liquid_universe

        if self.symbols:
            coins = self.symbols
        else:
            coins = get_liquid_universe(top_n=self.top_n_coins)

        # Filter out coins not available on OKX
        available = []
        for coin in coins:
            if coin not in OKX_UNAVAILABLE and get_okx_symbol(coin):
                available.append(coin)
            else:
                self.logger.info(f"Skipping {coin} (not available on OKX)")

        return available

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch candle data from OKX with retry logic."""
        okx_symbol = get_okx_symbol(coin)
        if not okx_symbol:
            return []

        endpoint = f"{OKX_API_URL}/api/v5/market/history-candles"

        max_retries = 5
        delay = 1.0

        for attempt in range(max_retries):
            try:
                # OKX uses 'after' for pagination (exclusive, older data)
                # and 'before' for newer data
                params = {
                    'instId': okx_symbol,
                    'bar': '1m',
                    'after': str(end_time),  # Get data before this time
                    'limit': str(self.MAX_CANDLES_PER_REQUEST),
                }

                resp = self.session.get(endpoint, params=params, timeout=30)

                if resp.status_code == 429:
                    self.logger.warning(f"Rate limit for {coin}. Sleeping {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if data.get('code') != '0':
                    error_msg = data.get('msg', 'Unknown error')
                    if 'instrument' in error_msg.lower() or 'instId' in error_msg.lower():
                        self.logger.warning(f"Symbol {okx_symbol} not found on OKX")
                        return []
                    self.logger.warning(f"OKX error for {coin}: {error_msg}")
                    return []

                candles = data.get('data', [])

                # OKX format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                # Returns newest first, we need to reverse for oldest-first
                records = []
                for candle in reversed(candles):
                    ts = int(candle[0])
                    if ts >= start_time:
                        records.append({
                            't': ts,
                            'o': candle[1],
                            'h': candle[2],
                            'l': candle[3],
                            'c': candle[4],
                            'v': candle[5],
                            'q': candle[7] if len(candle) > 7 else '0',  # Quote volume
                        })

                return records

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request error for {coin}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
            except Exception as e:
                self.logger.warning(f"Error fetching {coin}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        return []

    def process_record(self, sender: Sender, coin: str, record: Dict):
        """Process a single candle record."""
        ts_utc = datetime.fromtimestamp(record['t'] / 1000, tz=timezone.utc)

        # Handle 1000x tokens - OKX uses actual price, we need to convert to k-units
        price_multiplier = 1.0
        vol_multiplier = 1.0
        if coin.startswith('k') and coin[1:] in ['PEPE', 'BONK', 'FLOKI', 'SHIB', 'LUNC']:
            price_multiplier = 1000.0  # Convert to k-units (multiply price)
            vol_multiplier = 1.0 / 1000.0  # Adjust volume inversely

        sender.row(
            self.table_name,
            symbols={
                'symbol': coin,
                'data_source': self.DATA_SOURCE,
            },
            columns={
                'open': float(record['o']) * price_multiplier,
                'high': float(record['h']) * price_multiplier,
                'low': float(record['l']) * price_multiplier,
                'close': float(record['c']) * price_multiplier,
                'volume': float(record['v']) * vol_multiplier,
                'turnover': float(record['q']) if record['q'] else 0.0,
            },
            at=TimestampNanos.from_datetime(ts_utc)
        )

    def get_last_timestamp(self, data: List[Dict]) -> int:
        """Get the last timestamp from candle data."""
        if not data:
            return 0
        return data[-1]['t']

    def get_first_timestamp(self, data: List[Dict]) -> int:
        """Get the first (oldest) timestamp from candle data."""
        if not data:
            return 0
        return data[0]['t']

    def _run_ingestion(self, sender, coins: List[str], progress: Dict):
        """
        Override to handle OKX backwards pagination.

        OKX's 'after' parameter means "get data BEFORE this timestamp",
        so we paginate from end_time backwards to start_time.
        """
        from questdb.ingress import Sender
        start_time_ms, end_time_ms = self._get_time_range()

        for coin in coins:
            # Check if already completed
            coin_progress = progress.get(coin, {})
            if coin_progress.get('completed', False):
                self.logger.info(f"Skipping {coin} (already completed)")
                continue

            self.logger.info(f"Processing {coin}...")

            # For OKX backwards pagination, we use 'cursor' as the next 'after' value
            # Start at end_time and work backwards
            cursor = end_time_ms
            total_records = coin_progress.get('total_records', 0)
            oldest_fetched = coin_progress.get('oldest_timestamp', end_time_ms)
            consecutive_errors = 0
            batch_count = 0

            # Check if we have partial progress - resume from where we left off
            if 'oldest_timestamp' in coin_progress and not coin_progress.get('completed', False):
                cursor = coin_progress['oldest_timestamp']
                self.logger.info(f"  Resuming from {datetime.fromtimestamp(cursor/1000, tz=timezone.utc)}")

            while cursor > start_time_ms:
                try:
                    # OKX: 'after' gets data BEFORE cursor (backwards pagination)
                    data = self.fetch_data_batch(coin, start_time_ms, cursor)
                except Exception as e:
                    self.logger.warning(f"Fetch error for {coin}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > self.MAX_CONSECUTIVE_ERRORS:
                        self.logger.warning(f"Too many errors for {coin}, moving to next")
                        break
                    time.sleep(self.RETRY_DELAY_BASE)
                    continue

                if not data:
                    # No more data available for this coin
                    if batch_count == 0:
                        self.logger.warning(f"No data found for {coin} on OKX")
                    break

                consecutive_errors = 0
                batch_count += 1

                for record in data:
                    self.process_record(sender, coin, record)

                sender.flush()
                total_records += len(data)

                # Get oldest timestamp from this batch for next pagination
                oldest_in_batch = self.get_first_timestamp(data)
                newest_in_batch = self.get_last_timestamp(data)

                # Log progress periodically
                if batch_count % 100 == 0:
                    oldest_dt = datetime.fromtimestamp(oldest_in_batch/1000, tz=timezone.utc)
                    self.logger.info(f"  {coin}: {total_records:,} records, oldest: {oldest_dt.strftime('%Y-%m-%d %H:%M')}")

                # Save progress with oldest timestamp for resume
                progress[coin] = {
                    'oldest_timestamp': oldest_in_batch,
                    'newest_timestamp': newest_in_batch,
                    'total_records': total_records,
                    'completed': False
                }
                self.save_progress(progress)

                # Check if we've reached the start
                if oldest_in_batch <= start_time_ms:
                    break

                # Move cursor backwards for next request
                # Subtract 1 to avoid duplicates
                cursor = oldest_in_batch - 1

                time.sleep(self.REQUEST_DELAY)

            # Mark coin as completed
            progress[coin] = {
                'oldest_timestamp': start_time_ms,
                'newest_timestamp': end_time_ms,
                'total_records': total_records,
                'completed': True
            }
            self.save_progress(progress)
            self.logger.info(f"Completed {coin}: {total_records:,} records")


def run_ingest(resume: bool = True, symbols: Optional[List[str]] = None):
    """
    Run OKX candle data ingestion.

    Args:
        resume: If True, resume from last saved progress
        symbols: Optional list of specific symbols to ingest
    """
    ingester = OKXCandleIngester(symbols=symbols)
    ingester.run(resume=resume)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest OKX candle data")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--top-n", type=int, default=100, help="Top N coins")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to ingest")
    args = parser.parse_args()

    os.environ['INGEST_LOOKBACK_DAYS'] = str(args.days)
    os.environ['INGEST_TOP_N'] = str(args.top_n)

    run_ingest(resume=not args.no_resume, symbols=args.symbols)
