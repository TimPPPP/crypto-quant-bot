"""
1-Minute Candle Data Ingester for Binance.

Fetches historical OHLCV data from Binance public API and stores in QuestDB.
Best coverage for major cryptocurrencies with reliable data quality.

Binance API limits:
- 1200 requests/minute (public)
- 1000 candles per request max
- No authentication required for historical data
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
logger = logging.getLogger("Ingest_Binance")

# Binance API endpoints
BINANCE_API_URL = "https://api.binance.com"
BINANCE_FUTURES_API_URL = "https://fapi.binance.com"


# Symbol mapping: Hyperliquid symbol -> Binance symbol
SYMBOL_MAP = {
    # Standard mappings (most are same)
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "DOT": "DOTUSDT",
    "MATIC": "MATICUSDT",
    "UNI": "UNIUSDT",
    "LTC": "LTCUSDT",
    "BCH": "BCHUSDT",
    "ATOM": "ATOMUSDT",
    "XLM": "XLMUSDT",
    "NEAR": "NEARUSDT",
    "FIL": "FILUSDT",
    "ARB": "ARBUSDT",
    "OP": "OPUSDT",
    "SUI": "SUIUSDT",
    "SEI": "SEIUSDT",
    "INJ": "INJUSDT",
    "TIA": "TIAUSDT",
    "APT": "APTUSDT",
    "HBAR": "HBARUSDT",
    "ALGO": "ALGOUSDT",
    "AAVE": "AAVEUSDT",
    "CRV": "CRVUSDT",
    "FET": "FETUSDT",
    "RENDER": "RENDERUSDT",
    "TAO": "TAOUSDT",
    "WIF": "WIFUSDT",
    "ONDO": "ONDOUSDT",
    "ICP": "ICPUSDT",
    "STX": "STXUSDT",
    "ENA": "ENAUSDT",
    "PENDLE": "PENDLEUSDT",
    "ZRO": "ZROUSDT",
    "EIGEN": "EIGENUSDT",
    "WLD": "WLDUSDT",
    "STRK": "STRKUSDT",
    "ZK": "ZKUSDT",
    "BLAST": "BLASTUSDT",
    "IO": "IOUSDT",
    "ZEN": "ZENUSDT",
    "COMP": "COMPUSDT",
    "SNX": "SNXUSDT",
    "LDO": "LDOUSDT",
    "GMT": "GMTUSDT",
    "MINA": "MINAUSDT",
    "TNSR": "TNSRUSDT",
    # Special Hyperliquid naming
    "kPEPE": "1000PEPEUSDT",
    "kBONK": "1000BONKUSDT",
    "kFLOKI": "1000FLOKIUSDT",
    "kSHIB": "1000SHIBUSDT",
    "kLUNC": "1000LUNCUSDT",
    # Meme coins
    "POPCAT": "POPCATUSDT",
    "TURBO": "TURBOUSDT",
    "MOODENG": "MOODENGUSDT",
    "PNUT": "PNUTUSDT",
    # POL (formerly MATIC)
    "POL": "POLUSDT",
}

# Coins not available on Binance (Hyperliquid-native or other exchanges)
BINANCE_UNAVAILABLE = {
    "HYPE",      # Hyperliquid native
    "PURR",      # Hyperliquid native
    "FARTCOIN",  # Not on Binance
    "TRUMP",     # Not on Binance
    "KAITO",     # Not on Binance
    "BERA",      # Not on Binance yet
    "IP",        # Story Protocol - not on Binance
    "PROMPT",    # Not on Binance
    "PAXG",      # Gold token - limited availability
    "ME",        # Not on Binance
    "AERO",      # Aerodrome - not on Binance
    "XPL",       # Not on Binance
    "ASTER",     # Not on Binance
    "FOGO",      # Not on Binance
    "WLFI",      # Not on Binance
    "MON",       # Not on Binance
    "LIT",       # Limited availability
}


def get_binance_symbol(hl_symbol: str) -> Optional[str]:
    """Convert Hyperliquid symbol to Binance symbol."""
    if hl_symbol in BINANCE_UNAVAILABLE:
        return None
    if hl_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[hl_symbol]
    # Default: append USDT
    return f"{hl_symbol}USDT"


class BinanceCandleIngester(BaseIngester):
    """
    Ingester for 1-minute candle data from Binance Futures.
    Uses futures API for better liquidity and data quality.
    """

    TIMEFRAME = '1m'
    DATA_SOURCE = 'binance'
    MAX_CANDLES_PER_REQUEST = 1000
    REQUEST_DELAY = 0.1  # 100ms between requests

    def __init__(
        self,
        top_n_coins: int = None,
        lookback_days: int = None,
        progress_file: str = None,
        symbols: Optional[List[str]] = None,
        use_futures: bool = True,
    ):
        top_n_coins = top_n_coins or int(os.getenv('INGEST_TOP_N', 100))
        lookback_days = lookback_days or int(os.getenv('INGEST_LOOKBACK_DAYS', 365))
        progress_file = progress_file or os.getenv(
            'BINANCE_PROGRESS_FILE',
            'data/state/ingest_binance_progress.json'
        )

        super().__init__(
            table_name='candles_1m',
            progress_file=progress_file,
            top_n_coins=top_n_coins,
            lookback_days=lookback_days
        )

        self.symbols = symbols
        self.use_futures = use_futures
        self.base_url = BINANCE_FUTURES_API_URL if use_futures else BINANCE_API_URL
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

        # Filter out coins not available on Binance
        available = []
        for coin in coins:
            if coin not in BINANCE_UNAVAILABLE and get_binance_symbol(coin):
                available.append(coin)
            else:
                self.logger.info(f"Skipping {coin} (not available on Binance)")

        return available

    def fetch_data_batch(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch candle data from Binance with retry logic."""
        binance_symbol = get_binance_symbol(coin)
        if not binance_symbol:
            return []

        endpoint = f"{self.base_url}/fapi/v1/klines" if self.use_futures else f"{self.base_url}/api/v3/klines"

        max_retries = 5
        delay = 1.0

        for attempt in range(max_retries):
            try:
                params = {
                    'symbol': binance_symbol,
                    'interval': '1m',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': self.MAX_CANDLES_PER_REQUEST,
                }

                resp = self.session.get(endpoint, params=params, timeout=30)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', delay))
                    self.logger.warning(f"Rate limit for {coin}. Sleeping {retry_after}s...")
                    time.sleep(retry_after)
                    delay = min(delay * 2, 120)
                    continue

                if resp.status_code == 400:
                    # Symbol might not exist
                    error_msg = resp.json().get('msg', '')
                    if 'Invalid symbol' in error_msg:
                        self.logger.warning(f"Symbol {binance_symbol} not found on Binance")
                        return []

                resp.raise_for_status()

                # Binance returns: [open_time, open, high, low, close, volume, close_time, quote_volume, ...]
                data = resp.json()

                # Convert to our format
                records = []
                for candle in data:
                    records.append({
                        't': candle[0],  # open_time in ms
                        'o': candle[1],  # open
                        'h': candle[2],  # high
                        'l': candle[3],  # low
                        'c': candle[4],  # close
                        'v': candle[5],  # volume
                        'q': candle[7],  # quote volume (turnover)
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

        # Handle 1000x tokens (kPEPE, kBONK, etc.)
        # Binance uses 1000PEPEUSDT, we store as kPEPE with adjusted price
        price_divisor = 1.0
        if coin.startswith('k') and coin[1:] in ['PEPE', 'BONK', 'FLOKI', 'SHIB', 'LUNC']:
            price_divisor = 1000.0

        sender.row(
            self.table_name,
            symbols={
                'symbol': coin,
                'data_source': self.DATA_SOURCE,
            },
            columns={
                'open': float(record['o']) / price_divisor,
                'high': float(record['h']) / price_divisor,
                'low': float(record['l']) / price_divisor,
                'close': float(record['c']) / price_divisor,
                'volume': float(record['v']) * price_divisor,  # Adjust volume inversely
                'turnover': float(record['q']),  # Quote volume is already in USD
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
    Run Binance candle data ingestion.

    Args:
        resume: If True, resume from last saved progress
        symbols: Optional list of specific symbols to ingest
    """
    ingester = BinanceCandleIngester(symbols=symbols)
    ingester.run(resume=resume)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Binance candle data")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--top-n", type=int, default=100, help="Top N coins")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to ingest")
    parser.add_argument("--spot", action="store_true", help="Use spot API instead of futures")
    args = parser.parse_args()

    os.environ['INGEST_LOOKBACK_DAYS'] = str(args.days)
    os.environ['INGEST_TOP_N'] = str(args.top_n)

    ingester = BinanceCandleIngester(
        symbols=args.symbols,
        use_futures=not args.spot,
    )
    ingester.run(resume=not args.no_resume)
