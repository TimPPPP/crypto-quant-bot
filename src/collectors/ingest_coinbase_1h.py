"""
Coinbase 1-Hour Candle Data Ingester.

Fetches historical 1H OHLCV data from Coinbase Exchange API.
Much faster than 1-minute ingestion (~5 mins vs 4-5 hours for full year).
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from questdb.ingress import Sender, TimestampNanos

from src.collectors.base_ingester import BaseIngester, QUESTDB_HOST, QUESTDB_PORT, QUESTDB_HTTP
from src.utils.universe import get_liquid_universe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ingest_Coinbase_1H")

COINBASE_API_URL = "https://api.exchange.coinbase.com"


def get_coinbase_product_id(symbol: str) -> str:
    """Convert a symbol to Coinbase product ID format."""
    symbol_map = {
        "kPEPE": "PEPE",
        "kBONK": "BONK",
        "kFLOKI": "FLOKI",
        "kLUNC": "LUNC",
    }
    mapped = symbol_map.get(symbol, symbol)
    return f"{mapped}-USD"


class CoinbaseIngester1H(BaseIngester):
    """
    Ingester for 1-hour candle data from Coinbase Exchange.
    """

    TIMEFRAME = '1h'
    GRANULARITY = 3600  # 1 hour in seconds
    MAX_CANDLES_PER_REQUEST = 300  # ~12.5 days of hourly data
    REQUEST_DELAY = 0.15

    def __init__(
        self,
        top_n_coins: int = None,
        lookback_days: int = None,
        progress_file: str = None,
        symbols: List[str] = None,
    ):
        top_n_coins = top_n_coins or int(os.getenv('INGEST_TOP_N', 60))
        lookback_days = lookback_days or int(os.getenv('INGEST_LOOKBACK_DAYS', 365))
        progress_file = progress_file or os.getenv(
            'COINBASE_1H_PROGRESS_FILE',
            'data/state/ingest_coinbase_1h_progress.json'
        )

        super().__init__(
            table_name='candles_1h',
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
        CREATE TABLE IF NOT EXISTS candles_1h (
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

    def ensure_table_schema(self):
        super().ensure_table_schema()
        try:
            resp = self.session.get(
                QUESTDB_HTTP,
                params={"query": "SELECT dedup FROM tables() WHERE table_name = 'candles_1h';"},
                timeout=30,
            )
            resp.raise_for_status()
            df = resp.text.strip().splitlines()
            if len(df) >= 2:
                dedup_val = df[1].split(",")[-1].strip().lower()
                if dedup_val not in ("true", "t", "1"):
                    raise RuntimeError("candles_1h table is not DEDUP-enabled; rebuild or use candles_1h_dedup.")
        except Exception as exc:
            raise RuntimeError(f"Failed to validate candles_1h schema: {exc}") from exc

    def get_coins(self) -> List[str]:
        """Get list of coins to ingest."""
        if self.symbols:
            return self.symbols

        available = self._get_available_products()
        self.logger.info(f"Coinbase has {len(available)} USD products")

        fetch_count = int(self.top_n_coins * 1.5)
        universe = get_liquid_universe(top_n=fetch_count, use_buffer=True)
        self.logger.info(f"Universe has {len(universe)} coins")

        coins = []
        for symbol in universe:
            product_id = get_coinbase_product_id(symbol)
            if product_id in available:
                coins.append(symbol)

        self.logger.info(f"Found {len(coins)} coins available on Coinbase")
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
            return {"BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"}

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch 1H candle data from Coinbase."""
        product_id = get_coinbase_product_id(coin)

        start_dt = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)

        # Limit window to max candles (300 hours = 12.5 days)
        max_window = timedelta(hours=self.MAX_CANDLES_PER_REQUEST)
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
    lookback_days: int = 365,
    symbols: List[str] = None,
    top_n: int = 60,
):
    """Run Coinbase 1H candle data ingestion."""
    ingester = CoinbaseIngester1H(
        top_n_coins=top_n,
        lookback_days=lookback_days,
        symbols=symbols,
    )
    ingester.run(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Coinbase 1H historical candle data")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--top-n", type=int, default=60, help="Target number of coins")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore progress)")

    args = parser.parse_args()
    symbols = args.symbols.split(",") if args.symbols else None

    run_ingest(
        resume=not args.fresh,
        lookback_days=args.days,
        symbols=symbols,
        top_n=args.top_n,
    )
