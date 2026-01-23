"""
Coinbase 1-Minute Candle Data Ingester.

Fetches historical OHLCV data from Coinbase Exchange API (public, no auth needed).
Stores in QuestDB for backtesting.

Coinbase API limits:
- 300 candles per request
- Rate limit: 10 requests/second (public)
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from questdb.ingress import Sender, TimestampNanos

from src.collectors.base_ingester import BaseIngester, QUESTDB_HOST, QUESTDB_PORT
from src.utils.universe import get_liquid_universe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_Coinbase")

# Coinbase Exchange API (public, no auth required for market data)
COINBASE_API_URL = "https://api.exchange.coinbase.com"


def get_coinbase_product_id(symbol: str) -> str:
    """Convert a symbol to Coinbase product ID format."""
    # Handle special cases where Coinbase uses different symbols
    symbol_map = {
        "kPEPE": "PEPE",
        "kBONK": "BONK",
        "kFLOKI": "FLOKI",
        "kLUNC": "LUNC",
    }
    mapped = symbol_map.get(symbol, symbol)
    return f"{mapped}-USD"


class CoinbaseIngester(BaseIngester):
    """
    Ingester for 1-minute candle data from Coinbase Exchange.
    """

    TIMEFRAME = '1m'
    GRANULARITY = 60  # seconds
    MAX_CANDLES_PER_REQUEST = 300
    REQUEST_DELAY = 0.15  # 150ms between requests to stay under rate limit

    def __init__(
        self,
        top_n_coins: int = None,
        lookback_days: int = None,
        progress_file: str = None,
        symbols: List[str] = None,
    ):
        top_n_coins = top_n_coins or int(os.getenv('INGEST_TOP_N', 60))
        lookback_days = lookback_days or int(os.getenv('INGEST_LOOKBACK_DAYS', 180))
        progress_file = progress_file or os.getenv(
            'COINBASE_PROGRESS_FILE',
            'data/state/ingest_coinbase_progress.json'
        )

        super().__init__(
            table_name='candles_1m',  # Same table as Hyperliquid
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
            turnover DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """

    def get_coins(self) -> List[str]:
        """Get list of coins to ingest using universe selector.

        Automatically fetches enough coins from Hyperliquid to meet the target
        after filtering for Coinbase availability (~25% of coins are unavailable).
        """
        if self.symbols:
            return self.symbols

        # Get available Coinbase products first
        available = self._get_available_products()
        self.logger.info(f"Coinbase has {len(available)} USD products")

        # Fetch enough coins to meet target after Coinbase filtering
        # Strategy: fetch 1.5x the target to account for ~25% unavailable on Coinbase
        fetch_count = int(self.top_n_coins * 1.5)
        universe = get_liquid_universe(top_n=fetch_count, use_buffer=True)
        self.logger.info(f"Universe has {len(universe)} coins (fetched {fetch_count} to ensure {self.top_n_coins} target)")

        # Filter to coins available on Coinbase
        coins = []
        for symbol in universe:
            product_id = get_coinbase_product_id(symbol)
            if product_id in available:
                coins.append(symbol)
            else:
                self.logger.debug(f"Skipping {symbol} - not on Coinbase ({product_id})")

        self.logger.info(f"Found {len(coins)} coins available on both platforms (target: {self.top_n_coins})")

        # Warn if we didn't meet the target
        if len(coins) < self.top_n_coins:
            self.logger.warning(f"Only {len(coins)} coins available, below target of {self.top_n_coins}")

        return coins

    def _get_available_products(self) -> set:
        """Get set of available product IDs from Coinbase."""
        try:
            resp = self.session.get(f"{COINBASE_API_URL}/products", timeout=30)
            resp.raise_for_status()
            products = resp.json()
            return {p['id'] for p in products if p.get('status') == 'online'}
        except Exception as e:
            self.logger.warning(f"Failed to fetch products: {e}")
            # Return common USD pairs as fallback
            return {"BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"}

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """
        Fetch candle data from Coinbase.

        Args:
            coin: Our symbol (e.g., "BTC")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of candle dicts with keys: t, o, h, l, c, v
        """
        product_id = get_coinbase_product_id(coin)

        # Convert ms to ISO format
        start_dt = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)

        # Coinbase returns max 300 candles, so limit the window
        max_window = timedelta(minutes=self.MAX_CANDLES_PER_REQUEST)
        if end_dt - start_dt > max_window:
            end_dt = start_dt + max_window

        try:
            resp = self.session.get(
                f"{COINBASE_API_URL}/products/{product_id}/candles",
                params={
                    'start': start_dt.isoformat(),
                    'end': end_dt.isoformat(),
                    'granularity': self.GRANULARITY,
                },
                timeout=30
            )

            if resp.status_code == 429:
                self.logger.warning("Rate limited, sleeping...")
                time.sleep(2)
                return []

            resp.raise_for_status()
            raw_candles = resp.json()

            if not raw_candles:
                return []

            # Coinbase format: [timestamp, low, high, open, close, volume]
            # Convert to our format: {t, o, h, l, c, v}
            candles = []
            for c in raw_candles:
                candles.append({
                    't': c[0] * 1000,  # Convert seconds to milliseconds
                    'o': float(c[3]),
                    'h': float(c[2]),
                    'l': float(c[1]),
                    'c': float(c[4]),
                    'v': float(c[5]),
                })

            # Coinbase returns newest first, reverse to oldest first
            candles.sort(key=lambda x: x['t'])

            return candles

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.warning(f"Product {product_id} not found")
            else:
                self.logger.warning(f"HTTP error for {coin}: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"Fetch error for {coin}: {e}")
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


def run_ingest(
    resume: bool = True,
    lookback_days: int = 180,
    symbols: List[str] = None,
    top_n: int = 60,
):
    """
    Run Coinbase candle data ingestion.

    Args:
        resume: If True, resume from last saved progress
        lookback_days: Number of days to fetch
        symbols: Specific symbols to fetch (optional)
        top_n: Target number of coins to ingest (will fetch ~1.5x from Hyperliquid to meet target)
    """
    ingester = CoinbaseIngester(
        top_n_coins=top_n,
        lookback_days=lookback_days,
        symbols=symbols,
    )
    ingester.run(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Coinbase historical candle data")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    parser.add_argument("--top-n", type=int, default=60, help="Target number of coins (auto-fetches 1.5x from Hyperliquid)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., BTC,ETH,SOL)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore progress)")

    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None

    run_ingest(
        resume=not args.fresh,
        lookback_days=args.days,
        symbols=symbols,
        top_n=args.top_n,
    )
