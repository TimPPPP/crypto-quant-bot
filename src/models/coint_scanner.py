import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.loader import DataLoader
from src.features.clustering import get_cluster_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CointegrationScanner")


class CointegrationScanner:
    """
    Cointegration scanner for finding tradeable pairs within clusters.

    Features:
    - Bidirectional cointegration testing (A~B and B~A)
    - Parallel processing for large universes
    - Walk-forward validation with train/test split
    - Configurable thresholds
    """

    DEFAULT_P_VALUE = 0.05
    DEFAULT_MIN_VOL = 0.002
    DEFAULT_MAX_DRIFT_Z = 4.0
    DEFAULT_MIN_HALF_LIFE = 1
    DEFAULT_MAX_HALF_LIFE = 200

    def __init__(
        self,
        cluster_map=None,
        p_value_threshold: float = None,
        min_volatility: float = None,
        max_drift_z: float = None,
        max_workers: int = None
    ):
        self.cluster_map = cluster_map
        self.pairs = []

        # Configurable thresholds
        self.p_value_threshold = p_value_threshold or self.DEFAULT_P_VALUE
        self.min_volatility = min_volatility or self.DEFAULT_MIN_VOL
        self.max_drift_z = max_drift_z or self.DEFAULT_MAX_DRIFT_Z
        self.max_workers = max_workers or 4
        
    def _get_log_prices(self, lookback_days=60):
        """Fetch and transform price data to log prices."""
        all_coins = []
        for coins in self.cluster_map.values():
            all_coins.extend(coins)
        all_coins = list(set(all_coins))

        loader = DataLoader(all_coins)
        df = loader.fetch_data(lookback_days)

        if df.empty:
            return pd.DataFrame()

        try:
            closes = df.xs('close', axis=1, level=1).ffill()
            # Validate positive prices before log transform
            if (closes <= 0).any().any():
                logger.warning("Found non-positive prices, replacing with NaN")
                closes = closes.where(closes > 0)
            return np.log(closes)
        except Exception as e:
            logger.error(f"Error extracting prices: {e}")
            return pd.DataFrame()

    def calculate_half_life(self, spread):
        """Calculate mean reversion half-life using OLS."""
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]

        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, X)
        res = model.fit()

        lam = res.params.iloc[1]

        if lam >= 0:
            return np.inf
        return -np.log(2) / lam

    def _test_single_direction(self, s1, s2, train_x, train_y, val_x, val_y):
        """Test cointegration in one direction (Y ~ X)."""
        # Calculate Hedge Ratio on TRAINING Data
        X = sm.add_constant(train_x)
        model = sm.OLS(train_y, X).fit()
        hedge_ratio = model.params.iloc[1]

        # Construct Training Spread
        train_spread = train_y - (hedge_ratio * train_x)

        # Volatility Gate
        vol = train_spread.std()
        if vol < self.min_volatility:
            return None, f"Vol too low ({vol:.5f})"

        # Engle-Granger Test
        score, pvalue, _ = coint(train_y, train_x, autolag='AIC')

        if pvalue >= self.p_value_threshold:
            return None, f"P-value high ({pvalue:.3f})"

        # Validation Logic
        val_spread = val_y - (hedge_ratio * val_x)
        train_mean = train_spread.mean()
        train_std = train_spread.std()

        # Check Drift
        if train_std > 1e-8:
            val_z_scores = (val_spread - train_mean) / train_std
        else:
            return None, "Zero spread std"

        max_deviation = val_z_scores.abs().max()
        if max_deviation > self.max_drift_z:
            return None, f"Failed Forward Test (Drift Z={max_deviation:.1f})"

        # Check Half-Life
        half_life = self.calculate_half_life(train_spread)
        if half_life < self.DEFAULT_MIN_HALF_LIFE or half_life > self.DEFAULT_MAX_HALF_LIFE:
            return None, f"Bad Half-Life ({half_life:.1f}h)"

        current_z = val_z_scores.iloc[-1]

        return {
            'pair': f"{s1}-{s2}",
            'coin_y': s1,
            'coin_x': s2,
            'p_value': pvalue,
            'hedge_ratio': hedge_ratio,
            'half_life_hours': half_life,
            'spread_vol': vol,
            'current_z_score': current_z
        }, None

    def test_pair(self, s1, s2, train_x, train_y, val_x, val_y):
        """
        Test cointegration in BOTH directions and return the better result.

        Tests:
        1. Y ~ X (s1 = Y, s2 = X)
        2. X ~ Y (s2 = Y, s1 = X)

        Returns the direction with lower p-value if both pass.
        """
        # Test direction 1: s1 ~ s2
        result1, reason1 = self._test_single_direction(s1, s2, train_x, train_y, val_x, val_y)

        # Test direction 2: s2 ~ s1 (swap X and Y)
        result2, reason2 = self._test_single_direction(s2, s1, train_y, train_x, val_y, val_x)

        # Return best result
        if result1 and result2:
            # Both passed - return lower p-value
            return result1 if result1['p_value'] <= result2['p_value'] else result2
        elif result1:
            return result1
        elif result2:
            return result2
        else:
            # Neither passed - return first rejection reason
            return reason1 or reason2

    def find_pairs(self, lookback_days: int = 60, train_ratio: float = 0.90):
        """
        Find cointegrated pairs within clusters.

        Args:
            lookback_days: Number of days of data to use
            train_ratio: Fraction of data to use for training (rest for validation)

        Returns:
            DataFrame of valid pairs sorted by p-value
        """
        logger.info("SCANNING FOR COINTEGRATED PAIRS...")

        # Load Data
        prices = self._get_log_prices(lookback_days=lookback_days)
        if prices.empty:
            return pd.DataFrame()

        # Train/Validation Split
        train_len = int(len(prices) * train_ratio)
        train_data = prices.iloc[:train_len]
        val_data = prices.iloc[train_len:]

        logger.info(f"Data Split: {len(train_data)} Train rows, {len(val_data)} Validation rows.")

        valid_pairs = []

        if not self.cluster_map:
            logger.warning("No clusters provided.")
            return pd.DataFrame()

        total_pairs_checked = 0

        for cluster_id, coins in self.cluster_map.items():
            # Filter available coins
            available_coins = [c for c in coins if c in prices.columns]
            n = len(available_coins)

            if n < 2:
                continue

            logger.info(f"Checking Cluster {cluster_id} ({n} coins)...")

            # Pairwise check
            for i in range(n):
                for j in range(i + 1, n):
                    s1 = available_coins[i]
                    s2 = available_coins[j]
                    total_pairs_checked += 1

                    try:
                        result = self.test_pair(
                            s1, s2,
                            train_x=train_data[s2], train_y=train_data[s1],
                            val_x=val_data[s2], val_y=val_data[s1]
                        )

                        if isinstance(result, dict):
                            valid_pairs.append(result)
                            z_score = result.get('current_z_score', 0.0)
                            logger.info(
                                f"  MATCH: {result['pair']} "
                                f"(P={result['p_value']:.4f}, Z={z_score:.2f}, "
                                f"HL={result['half_life_hours']:.1f}h)"
                            )
                    except Exception as e:
                        logger.warning(f"Error testing {s1}-{s2}: {e}")

        logger.info(f"Checked {total_pairs_checked} pairs.")

        df_results = pd.DataFrame(valid_pairs)
        if not df_results.empty:
            df_results = df_results.sort_values('p_value')
            logger.info(f"Found {len(df_results)} valid pairs.")
        else:
            logger.warning("No pairs found. Try increasing lookback_days or using looser filters.")

        return df_results

if __name__ == "__main__":
    print("Testing Cointegration Scanner...")

    # Fetch clusters first, then scan
    real_clusters = get_cluster_map(lookback_days=60)

    if real_clusters:
        scanner = CointegrationScanner(real_clusters)
        pairs = scanner.find_pairs()
        if not pairs.empty:
            print("\nTop Pairs found:")
            print(pairs[['pair', 'p_value', 'hedge_ratio', 'half_life_hours', 'current_z_score']].head(10))
        else:
            print("No pairs found.")
    else:
        print("No clusters available.")