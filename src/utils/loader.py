import pandas as pd
import requests
import logging
import os
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLoader")

# --- CONFIGURATION ---
DB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
QUESTDB_QUERY_URL = f"http://{DB_HOST}:9000/exec"
DB_TIMEOUT = int(os.getenv('DB_TIMEOUT', 30))


def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class DataLoader:
    """
    Data loader for fetching market data from QuestDB.

    Features:
    - Connection pooling with retries
    - UTC timezone handling
    - Proper fill logic for missing data
    - Health check for connection validation
    """

    def __init__(self, universe):
        self.universe = list(universe) if not isinstance(universe, list) else universe
        self.session = create_session()

    def health_check(self) -> bool:
        """
        Check if QuestDB connection is healthy.

        Returns:
            True if connection is healthy, False otherwise.
        """
        try:
            r = self.session.get(
                QUESTDB_QUERY_URL,
                params={'query': 'SELECT 1'},
                timeout=5
            )
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def _query_questdb(self, query):
        """Execute a query against QuestDB with connection pooling."""
        try:
            r = self.session.get(
                QUESTDB_QUERY_URL,
                params={'query': query.strip()},
                timeout=DB_TIMEOUT
            )

            if r.status_code != 200:
                logger.error(f"DB ERROR {r.status_code}: {r.text[:200]}")
                return pd.DataFrame()

            data = r.json()

            if 'dataset' not in data or not data['dataset']:
                return pd.DataFrame()

            col_names = [c['name'] for c in data['columns']]
            df = pd.DataFrame(data['dataset'], columns=col_names)

            # Force UTC timezone
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Ensure numerical columns are floats
            for col in ['close', 'turnover', 'funding_rate', 'open', 'high', 'low', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except requests.Timeout:
            logger.error("DB query timeout")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()

    def fetch_data(self, lookback_days=35):
        """
        Fetch market data for all coins in universe.

        Args:
            lookback_days: Number of days of historical data

        Returns:
            DataFrame with MultiIndex columns (symbol, metric)
        """
        logger.info(f"Loading data for {len(self.universe)} coins (Lookback: {lookback_days} days)...")

        # Create master timeline in UTC
        end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=lookback_days)

        master_timeline = pd.date_range(
            start=start_date, end=end_date, freq='h', tz='UTC', name='timestamp'
        )

        cutoff_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000000Z')

        all_dataframes = []

        for symbol in self.universe:
            # Query candles
            candle_query = f"""
            SELECT timestamp, last(close) as close, sum(turnover) as turnover
            FROM candles_1h
            WHERE symbol = '{symbol}'
            AND timestamp >= '{cutoff_str}'
            SAMPLE BY 1h FILL(PREV) ALIGN TO CALENDAR;
            """
            df_candles = self._query_questdb(candle_query)

            # Query funding
            funding_query = f"""
            SELECT timestamp, last(funding_rate) as funding_rate
            FROM funding_history
            WHERE symbol = '{symbol}'
            AND timestamp >= '{cutoff_str}'
            SAMPLE BY 1h FILL(PREV) ALIGN TO CALENDAR;
            """
            df_funding = self._query_questdb(funding_query)

            if df_candles.empty:
                logger.warning(f"No candle data for {symbol}, skipping.")
                continue

            # Merge data
            df_candles = df_candles.set_index('timestamp')

            if not df_funding.empty:
                df_funding = df_funding.set_index('timestamp')
                df_combined = df_candles.join(df_funding, how='left')
            else:
                df_combined = df_candles
                df_combined['funding_rate'] = 0.0

            # Deduplicate
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

            # Reindex to master timeline
            df_combined = df_combined.reindex(master_timeline)

            # Structure for MultiIndex
            df_combined.columns = pd.MultiIndex.from_product([[symbol], df_combined.columns])
            all_dataframes.append(df_combined)

        if not all_dataframes:
            logger.error("No data loaded!")
            return pd.DataFrame()

        logger.info("Aligning and cleaning dataframes...")
        master_df = pd.concat(all_dataframes, axis=1)

        # Data cleaning
        idx = pd.IndexSlice

        # Fill missing funding/turnover with 0
        try:
            master_df.loc[:, idx[:, 'funding_rate']] = master_df.loc[:, idx[:, 'funding_rate']].fillna(0)
            master_df.loc[:, idx[:, 'turnover']] = master_df.loc[:, idx[:, 'turnover']].fillna(0)
        except KeyError:
            pass

        # Forward fill prices first (carry last known price)
        master_df = master_df.ffill()

        # Avoid backward fill to prevent look-ahead bias for newly listed coins.
        # Drop rows where any close price is still missing.
        try:
            close_cols = master_df.loc[:, idx[:, 'close']].columns
            master_df = master_df.dropna(subset=close_cols, how='any')
        except KeyError:
            pass

        # Drop fully empty rows
        master_df = master_df.dropna(how='all')

        # Validate we have data
        if 'BTC' in self.universe and 'BTC' in master_df.columns.get_level_values(0):
            try:
                btc_price = master_df['BTC']['close'].iloc[0]
                logger.info(f"BTC Start Price: {btc_price}")
            except Exception:
                pass

        logger.info(f"Loaded matrix with shape: {master_df.shape}")
        return master_df

if __name__ == "__main__":
    # Test
    loader = DataLoader(['BTC', 'ETH', 'SOL'])
    df = loader.fetch_data(7)
    if not df.empty:
        print(df.head())
