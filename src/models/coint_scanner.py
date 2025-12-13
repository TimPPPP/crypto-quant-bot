import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from src.features.clustering import get_cluster_map
from src.utils.loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CointegrationScanner")


class CointegrationScanner:
    """
    Cointegration scanner for finding tradeable pairs within clusters.

    Features:
    - Bidirectional cointegration testing (A~B and B~A)
    - Walk-forward validation (train/validation split)
    - Configurable thresholds
    - NEW: Can scan either
        (a) from the DataLoader (legacy research mode), OR
        (b) from an already-prepared price matrix (Phase 5 backtest mode)
    """

    DEFAULT_P_VALUE = 0.05
    DEFAULT_MIN_VOL = 0.002
    DEFAULT_MAX_DRIFT_Z = 4.0
    DEFAULT_MIN_HALF_LIFE = 1
    DEFAULT_MAX_HALF_LIFE = 200

    def __init__(self, cluster_map: Optional[Dict[Union[str, int], List[str]]] = None,
                 p_value_threshold: float = None,
                 min_volatility: float = None,
                 max_drift_z: float = None):
        """Initialize scanner thresholds and performance knobs."""
        self.cluster_map = cluster_map or {}
        self.pairs: List[str] = []

        # Configurable thresholds
        self.p_value_threshold = p_value_threshold or self.DEFAULT_P_VALUE
        self.min_volatility = min_volatility or self.DEFAULT_MIN_VOL
        self.max_drift_z = max_drift_z or self.DEFAULT_MAX_DRIFT_Z

        # Performance tuning (safe defaults)
        # Only test pairs with absolute correlation >= this threshold
        self.corr_threshold = 0.8
        # Limit autolag search to this many lags to reduce SVD work
        self.max_autolag = 5

    # -------------------------------------------------------------------------
    # Data access / transforms
    # -------------------------------------------------------------------------

    def _get_log_prices(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Legacy research path:
        Fetch and transform price data to log prices from DataLoader.
        Expects DataLoader to return a MultiIndex columns frame with ('symbol', 'field')
        and includes 'close' at level=1.
        """
        if not self.cluster_map:
            logger.warning("No clusters provided. Returning empty.")
            return pd.DataFrame()

        all_coins: List[str] = []
        for coins in self.cluster_map.values():
            all_coins.extend(coins)
        all_coins = sorted(set(all_coins))

        loader = DataLoader(all_coins)
        df = loader.fetch_data(lookback_days)

        if df.empty:
            return pd.DataFrame()

        try:
            closes = df.xs("close", axis=1, level=1).ffill()
            closes = closes.where(closes > 0)  # guard log
            return np.log(closes)
        except Exception as e:
            logger.error(f"Error extracting prices: {e}")
            return pd.DataFrame()

    @staticmethod
    def _to_log_price_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a wide price matrix (index=time, columns=symbol) into log prices.
        Assumes columns are the symbols you trade (e.g., 'ETH', 'BTC').

        - ffill then bfill to handle missing values at start/end
        - replace non-positive prices with NaN before log
        - drop rows that still have NaN after filling
        """
        if price_matrix.empty:
            return pd.DataFrame()

        mat = price_matrix.copy()

        # Get numeric columns only (all columns should be numeric in price matrix)
        # Exclude any non-numeric columns that might exist
        numeric_cols = mat.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            # Try to convert all columns to numeric
            for col in mat.columns:
                mat[col] = pd.to_numeric(mat[col], errors="coerce")
            numeric_cols = mat.columns.tolist()

        # Forward fill then backward fill to handle NaN at start/end
        mat = mat.ffill().bfill()

        # Only drop rows where ALL values are NaN (keep rows with some data)
        mat = mat.dropna(how='all')

        if mat.empty:
            logger.warning("All rows dropped after cleaning NaN values")
            return pd.DataFrame()

        # For remaining NaN values, fill with column median
        for col in numeric_cols:
            if mat[col].isnull().any():
                mat[col] = mat[col].fillna(mat[col].median())

        # Guard log() - replace non-positive with small positive value
        for col in numeric_cols:
            if (mat[col] <= 0).any():
                min_positive = mat[col][mat[col] > 0].min()
                if pd.isna(min_positive):
                    min_positive = 1e-10
                mat[col] = mat[col].clip(lower=min_positive)

        # Apply log to all numeric columns
        mat[numeric_cols] = np.log(mat[numeric_cols])

        return mat

    # -------------------------------------------------------------------------
    # Core stats / tests
    # -------------------------------------------------------------------------

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life (in bars) using OLS.

        Returns:
            Half-life in bars, or np.inf if no mean reversion detected.
        """
        # Need at least 3 data points for meaningful regression
        if len(spread) < 3:
            return np.inf

        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag

        # Drop NaN values from the shift
        valid_mask = ~(spread_lag.isna() | spread_ret.isna())
        spread_lag = spread_lag[valid_mask]
        spread_ret = spread_ret[valid_mask]

        if len(spread_lag) < 2:
            return np.inf

        X = sm.add_constant(spread_lag)
        try:
            res = sm.OLS(spread_ret, X).fit()
            lam = res.params.iloc[1]
        except Exception:
            return np.inf

        # No mean reversion if lambda >= 0 or too close to zero
        if lam >= -1e-10:
            return np.inf

        half_life = -np.log(2) / lam

        # Sanity check: half-life should be positive and finite
        if not np.isfinite(half_life) or half_life <= 0:
            return np.inf

        return float(half_life)

    def _test_single_direction(
        self,
        s1: str,
        s2: str,
        train_x: pd.Series,
        train_y: pd.Series,
        val_x: pd.Series,
        val_y: pd.Series,
    ):
        """Test cointegration in one direction (Y ~ X)."""
        # Hedge ratio on training window
        X = sm.add_constant(train_x)
        model = sm.OLS(train_y, X).fit()
        hedge_ratio = float(model.params.iloc[1])

        # Training spread
        train_spread = train_y - (hedge_ratio * train_x)

        # Volatility gate
        vol = float(train_spread.std())
        if vol < self.min_volatility:
            return None, f"Vol too low ({vol:.5f})"

        # Engle-Granger (protected)
        try:
            _score, pvalue, _ = coint(train_y, train_x, autolag="AIC", maxlag=self.max_autolag)
        except Exception as e:
            logger.warning(f"coint test failed for {s1}-{s2}: {e}")
            return None, "coint failed"
        pvalue = float(pvalue)

        if pvalue >= self.p_value_threshold:
            return None, f"P-value high ({pvalue:.3f})"

        # Validation drift check
        val_spread = val_y - (hedge_ratio * val_x)
        train_mean = float(train_spread.mean())
        train_std = float(train_spread.std())

        if train_std <= 1e-8:
            return None, "Zero spread std"

        val_z_scores = (val_spread - train_mean) / train_std
        max_deviation = float(val_z_scores.abs().max())
        if max_deviation > self.max_drift_z:
            return None, f"Failed Forward Test (Drift Z={max_deviation:.1f})"

        # Half-life gate (in bars)
        half_life = self.calculate_half_life(train_spread)
        if half_life < self.DEFAULT_MIN_HALF_LIFE or half_life > self.DEFAULT_MAX_HALF_LIFE:
            return None, f"Bad Half-Life ({half_life:.1f} bars)"

        current_z = float(val_z_scores.iloc[-1])

        return (
            {
                "pair": f"{s1}-{s2}",
                "coin_y": s1,
                "coin_x": s2,
                "p_value": pvalue,
                "hedge_ratio": hedge_ratio,
                "half_life_bars": half_life,
                "spread_vol": vol,
                "current_z_score": current_z,
            },
            None,
        )

    def test_pair(
        self,
        s1: str,
        s2: str,
        train_x: pd.Series,
        train_y: pd.Series,
        val_x: pd.Series,
        val_y: pd.Series,
    ):
        """
        Test cointegration in BOTH directions and return the better result
        (lower p-value if both pass).
        """
        # Direction 1: s1 ~ s2
        result1, reason1 = self._test_single_direction(s1, s2, train_x, train_y, val_x, val_y)

        # Direction 2: s2 ~ s1 (swap)
        result2, reason2 = self._test_single_direction(s2, s1, train_y, train_x, val_y, val_x)

        if result1 and result2:
            return result1 if result1["p_value"] <= result2["p_value"] else result2
        if result1:
            return result1
        if result2:
            return result2
        return reason1 or reason2

    # -------------------------------------------------------------------------
    # Scanning entrypoints
    # -------------------------------------------------------------------------

    def _scan_with_log_prices(self, prices: pd.DataFrame, train_ratio: float) -> pd.DataFrame:
        """
        Shared scanning logic once we already have log prices.

        prices: log(price_matrix) with columns as symbols
        """
        if prices.empty:
            return pd.DataFrame()

        if not self.cluster_map:
            logger.warning("No clusters provided.")
            return pd.DataFrame()

        # Train/validation split *inside the provided window*
        train_len = int(len(prices) * train_ratio)
        train_len = max(2, min(train_len, len(prices) - 2))
        train_data = prices.iloc[:train_len]
        val_data = prices.iloc[train_len:]

        logger.info(f"Data Split: {len(train_data)} Train rows, {len(val_data)} Validation rows.")

        valid_pairs = []
        total_pairs_checked = 0

        for cluster_id, coins in self.cluster_map.items():
            available_coins = [c for c in coins if c in prices.columns]
            n = len(available_coins)
            if n < 2:
                continue

            logger.info(f"Checking Cluster {cluster_id} ({n} coins)...")
            # Correlation prefilter: compute absolute correlation matrix for this cluster
            cluster_prices = prices[available_coins]
            corr = cluster_prices.corr().abs()

            for i in range(n):
                for j in range(i + 1, n):
                    s1 = available_coins[i]
                    s2 = available_coins[j]
                    total_pairs_checked += 1

                    # Quick prefilter: skip if correlation below threshold
                    try:
                        if corr.loc[s1, s2] < self.corr_threshold:
                            continue
                    except Exception:
                        # missing value in corr -> skip
                        continue

                    try:
                        result = self.test_pair(
                            s1,
                            s2,
                            train_x=train_data[s2],
                            train_y=train_data[s1],
                            val_x=val_data[s2],
                            val_y=val_data[s1],
                        )
                        if isinstance(result, dict):
                            valid_pairs.append(result)
                            logger.info(
                                f"  MATCH: {result['pair']} "
                                f"(P={result['p_value']:.4f}, Z={result['current_z_score']:.2f}, "
                                f"HL={result['half_life_bars']:.1f} bars)"
                            )
                    except Exception as e:
                        logger.warning(f"Error testing {s1}-{s2}: {e}")

        logger.info(f"Checked {total_pairs_checked} pairs.")

        df_results = pd.DataFrame(valid_pairs)
        if not df_results.empty:
            df_results = df_results.sort_values("p_value")
            logger.info(f"Found {len(df_results)} valid pairs.")
        else:
            logger.warning("No pairs found. Try increasing lookback or using looser filters.")
        return df_results

    def find_pairs(self, lookback_days: int = 60, train_ratio: float = 0.90) -> pd.DataFrame:
        """
        Legacy research entrypoint:
        loads data using DataLoader and scans for pairs.
        """
        logger.info("SCANNING FOR COINTEGRATED PAIRS (DataLoader mode)...")
        prices = self._get_log_prices(lookback_days=lookback_days)
        return self._scan_with_log_prices(prices, train_ratio=train_ratio)

    def find_pairs_from_matrix(self, price_matrix: pd.DataFrame, train_ratio: float = 0.90) -> pd.DataFrame:
        """
        Phase 5 backtest entrypoint:
        scans for cointegrated pairs using an already-prepared price matrix (train_df).

        price_matrix must be wide:
            index = timestamps
            columns = symbols
            values = prices (close)

        This prevents data source drift and ensures you scan ONLY on backtest train window.
        """
        logger.info("SCANNING FOR COINTEGRATED PAIRS (matrix mode)...")
        log_prices = self._to_log_price_matrix(price_matrix)
        return self._scan_with_log_prices(log_prices, train_ratio=train_ratio)


if __name__ == "__main__":
    print("Testing Cointegration Scanner...")

    # Fetch clusters first, then scan (research mode)
    real_clusters = get_cluster_map(lookback_days=60)

    if real_clusters:
        scanner = CointegrationScanner(real_clusters)
        pairs = scanner.find_pairs(lookback_days=60, train_ratio=0.90)
        if not pairs.empty:
            print("\nTop Pairs found:")
            print(pairs[["pair", "p_value", "hedge_ratio", "half_life_bars", "current_z_score"]].head(10))
        else:
            print("No pairs found.")
    else:
        print("No clusters available.")
