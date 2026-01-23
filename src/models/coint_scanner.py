import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from joblib import Parallel, delayed

from src.features.clustering import get_cluster_map
from src.utils.loader import DataLoader

if TYPE_CHECKING:
    from src.adaptive.market_regime import MarketRegime

# Import half-life threshold from backtest config
try:
    from src.backtest.config_backtest import MIN_HALF_LIFE_BARS
except ImportError:
    MIN_HALF_LIFE_BARS = 40  # Default fallback


# =============================================================================
# FDR Control (Issue #1 fix: Multiple Testing Correction)
# =============================================================================

def apply_fdr_correction(
    pairs_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "benjamini_hochberg",
) -> pd.DataFrame:
    """
    Apply False Discovery Rate (FDR) correction to cointegration p-values.

    Issue #1 Fix: When scanning many pairs, p-value < 0.05 will produce many
    false positives. FDR control adjusts the threshold based on the number
    of tests to maintain the expected false discovery rate.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with 'p_value' column from cointegration tests.
    alpha : float
        Target FDR level (default 0.05 = expect 5% of significant results to be false).
    method : str
        "benjamini_hochberg" (default) or "bonferroni".

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - 'fdr_rank': rank of p-value (1 = smallest)
        - 'fdr_threshold': adjusted threshold for this rank
        - 'fdr_significant': bool, whether this pair passes FDR-adjusted test
    """
    if pairs_df.empty or 'p_value' not in pairs_df.columns:
        return pairs_df.copy()

    df = pairs_df.copy()
    n = len(df)
    p_values = df['p_value'].values

    if method == "bonferroni":
        # Bonferroni: simple but conservative
        # Adjusted threshold = alpha / n
        adjusted_alpha = alpha / n
        df['fdr_rank'] = df['p_value'].rank(method='first').astype(int)
        df['fdr_threshold'] = adjusted_alpha
        df['fdr_significant'] = df['p_value'] <= adjusted_alpha

    else:  # benjamini_hochberg (default)
        # BH procedure:
        # 1. Sort p-values ascending
        # 2. For rank k, threshold = (k/n) * alpha
        # 3. Find largest k where p_k <= threshold_k
        # 4. All p-values with rank <= k are significant

        # Sort by p-value
        sorted_idx = np.argsort(p_values)
        ranks = np.arange(1, n + 1)
        bh_thresholds = (ranks / n) * alpha

        # Find which pass under their own threshold
        sorted_pvals = p_values[sorted_idx]
        passes_own = sorted_pvals <= bh_thresholds

        # Find the largest k where p_k <= threshold_k
        if passes_own.any():
            max_k = np.where(passes_own)[0][-1]
            # All ranks <= max_k+1 are significant
            significant_mask = np.zeros(n, dtype=bool)
            significant_mask[sorted_idx[:max_k + 1]] = True
        else:
            significant_mask = np.zeros(n, dtype=bool)

        # Add columns to dataframe
        df['fdr_rank'] = 0
        df.loc[df.index[sorted_idx], 'fdr_rank'] = ranks

        df['fdr_threshold'] = 0.0
        for i, orig_idx in enumerate(sorted_idx):
            df.loc[df.index[orig_idx], 'fdr_threshold'] = bh_thresholds[i]

        df['fdr_significant'] = significant_mask

    n_significant = df['fdr_significant'].sum()
    logging.getLogger("CointegrationScanner").info(
        f"FDR correction ({method}): {n_significant}/{n} pairs significant at alpha={alpha}"
    )

    return df

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

    DEFAULT_P_VALUE = 0.03  # Cointegration requirement (moderately tightened from 0.05)
    DEFAULT_MIN_VOL = 0.002  # Minimum spread volatility
    DEFAULT_MAX_DRIFT_Z = 2.0  # Max |mean/std| of spread (moderately tightened from 3.0)
    DEFAULT_MIN_HALF_LIFE = 80  # Min half-life bars (moderately raised from 60)
    DEFAULT_MAX_HALF_LIFE = 720  # Max half-life bars (moderately lowered from 2000)

    # Rolling cointegration diagnostics (Problem #3 fix)
    DEFAULT_MAX_COINT_FAILURE_RATE = 0.3  # Reject if >30% of rolling windows fail ADF
    DEFAULT_MAX_BETA_DRIFT = 0.5  # Reject if beta drifts by >50% of initial value

    def __init__(self, cluster_map: Optional[Dict[Union[str, int], List[str]]] = None,
                 p_value_threshold: float = None,
                 min_volatility: float = None,
                 max_drift_z: float = None,
                 min_half_life: float = None,
                 max_half_life: float = None):
        """Initialize scanner thresholds and performance knobs."""
        self.cluster_map = cluster_map or {}
        self.pairs: List[str] = []

        # Configurable thresholds
        self.p_value_threshold = p_value_threshold or self.DEFAULT_P_VALUE
        self.min_volatility = min_volatility or self.DEFAULT_MIN_VOL
        self.max_drift_z = max_drift_z or self.DEFAULT_MAX_DRIFT_Z
        self.min_half_life = min_half_life or self.DEFAULT_MIN_HALF_LIFE
        self.max_half_life = max_half_life or self.DEFAULT_MAX_HALF_LIFE

        # Performance tuning (safe defaults)
        # Only test pairs with absolute correlation >= this threshold
        # Moderately tightened (was 0.5)
        self.corr_threshold = 0.6
        # Limit autolag search to this many lags to reduce SVD work
        self.max_autolag = 5

    def set_half_life_range(self, min_hl: int, max_hl: int) -> None:
        """
        Update half-life range for pair scanning.

        Used by regime-based adaptation to adjust search range
        based on current market conditions.

        Args:
            min_hl: Minimum half-life in bars
            max_hl: Maximum half-life in bars
        """
        self.min_half_life = min_hl
        self.max_half_life = max_hl
        logger.info(f"Half-life range updated: [{min_hl}, {max_hl}] bars")

    def set_half_life_from_regime(self, regime: "MarketRegime") -> None:
        """
        Update half-life range based on detected market regime.

        Args:
            regime: MarketRegime object from market_regime.py
        """
        self.set_half_life_range(
            regime.recommended_min_half_life,
            regime.recommended_max_half_life
        )
        logger.info(
            f"Half-life range set from regime: vol={regime.volatility.value}, "
            f"trend={regime.trend.value} -> [{self.min_half_life}, {self.max_half_life}] bars"
        )

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

        - ffill only (no backward fill to avoid look-ahead)
        - replace non-positive prices with NaN before log
        - drop rows with any NaN after filling
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

        # Forward fill only to avoid look-ahead
        mat = mat.ffill()

        # Drop rows with any missing values after fill
        mat = mat.dropna(how='any')

        if mat.empty:
            logger.warning("All rows dropped after cleaning NaN values")
            return pd.DataFrame()

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

    def compute_weighted_cointegration(
        self,
        y: pd.Series,
        x: pd.Series,
        half_life_days: float = 30.0,
        bars_per_day: int = 96,  # Depends on timeframe (96 for 15-min bars)
    ) -> Dict:
        """
        Compute cointegration with exponential decay weighting.

        Issue #2 Fix: Long training windows can dilute the "current relationship".
        Exponential weighting gives more importance to recent observations.

        Parameters
        ----------
        y : pd.Series
            Dependent variable (log price of Y)
        x : pd.Series
            Independent variable (log price of X)
        half_life_days : float
            Weighting half-life in days. Observations from half_life_days ago
            have half the weight of the most recent observation.
        bars_per_day : int
            Number of bars per day (depends on timeframe).

        Returns
        -------
        dict with:
            - p_value: approximate p-value (using weighted ADF)
            - hedge_ratio: exponentially weighted hedge ratio
            - half_life_bars: half-life of the weighted spread
            - valid: whether the weighted relationship is valid
        """
        n = len(y)
        if n < 50:
            return {
                'p_value': 1.0,
                'hedge_ratio': np.nan,
                'half_life_bars': np.inf,
                'valid': False,
            }

        # Drop NaN
        valid_mask = y.notna() & x.notna()
        y_valid = y[valid_mask]
        x_valid = x[valid_mask]
        n_valid = len(y_valid)

        if n_valid < 50:
            return {
                'p_value': 1.0,
                'hedge_ratio': np.nan,
                'half_life_bars': np.inf,
                'valid': False,
            }

        # Compute exponential weights
        # Weight = exp(-decay_rate * age), where age = n - 1 - t
        decay_rate = np.log(2) / (half_life_days * bars_per_day)
        ages = np.arange(n_valid)[::-1]  # Most recent = 0, oldest = n-1
        weights = np.exp(-decay_rate * ages)
        weights = weights / weights.sum()  # Normalize

        # Weighted OLS for hedge ratio
        try:
            X = sm.add_constant(x_valid.values)
            model = sm.WLS(y_valid.values, X, weights=weights).fit()
            hedge_ratio = float(model.params[1])
        except Exception:
            return {
                'p_value': 1.0,
                'hedge_ratio': np.nan,
                'half_life_bars': np.inf,
                'valid': False,
            }

        # Compute weighted spread
        spread = y_valid - hedge_ratio * x_valid

        # Weighted ADF approximation: use the more recent portion
        # (weights effectively downsample old data)
        recent_n = min(n_valid, int(half_life_days * bars_per_day * 3))
        recent_spread = spread.iloc[-recent_n:]

        try:
            adf_result = adfuller(recent_spread.dropna(), maxlag=self.max_autolag)
            p_value = float(adf_result[1])
        except Exception:
            p_value = 1.0

        # Half-life of the weighted spread
        half_life = self.calculate_half_life(spread)

        valid = (
            p_value < self.p_value_threshold and
            self.min_half_life <= half_life <= self.max_half_life
        )

        return {
            'p_value': p_value,
            'hedge_ratio': hedge_ratio,
            'half_life_bars': half_life,
            'valid': valid,
        }

    def validate_kalman_spread_stationarity(
        self,
        y: pd.Series,
        x: pd.Series,
        ols_beta: float,
        kalman_delta: float = 1e-6,
        kalman_R: float = 1e-2,
        adf_threshold: float = 0.05,
        max_beta_divergence: float = 0.3,
    ) -> Dict:
        """
        Run Kalman filter on training data and verify the resulting spread is stationary.

        Issue #3 Fix: Scanner uses OLS beta for cointegration test, but trading uses
        Kalman beta. This function validates that the Kalman-implied spread is also
        stationary and that the betas don't diverge too much.

        Parameters
        ----------
        y : pd.Series
            Dependent variable (log price of Y)
        x : pd.Series
            Independent variable (log price of X)
        ols_beta : float
            OLS hedge ratio from cointegration test
        kalman_delta : float
            Kalman filter transition noise parameter
        kalman_R : float
            Kalman filter observation noise parameter
        adf_threshold : float
            P-value threshold for ADF test (default 0.05)
        max_beta_divergence : float
            Maximum allowed |kalman_beta - ols_beta| / |ols_beta|

        Returns
        -------
        dict with:
            - kalman_spread_adf_pvalue: ADF p-value on Kalman-implied spread
            - beta_divergence: relative divergence between Kalman and OLS beta
            - kalman_beta_final: final Kalman beta estimate
            - kalman_spread_half_life: half-life of Kalman spread
            - valid: whether Kalman spread passes stationarity test
            - reason: rejection reason if invalid
        """
        n = len(y)
        if n < 100:
            return {
                'kalman_spread_adf_pvalue': 1.0,
                'beta_divergence': np.inf,
                'kalman_beta_final': np.nan,
                'kalman_spread_half_life': np.inf,
                'valid': False,
                'reason': 'Insufficient data',
            }

        # Simple Kalman filter for beta estimation
        # State: [beta, alpha] where spread = y - beta*x - alpha
        # We use a simplified version without importing the full Kalman class

        # Initialize state
        beta = ols_beta
        alpha = 0.0

        # Covariance matrix (2x2)
        P = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Transition noise
        Q = np.array([[kalman_delta, 0.0], [0.0, kalman_delta]])

        # Observation noise
        R = kalman_R

        kalman_spreads = []
        kalman_betas = []

        for t in range(n):
            y_t = y.iloc[t]
            x_t = x.iloc[t]

            if not (np.isfinite(y_t) and np.isfinite(x_t)):
                continue

            # Prediction step
            # State doesn't change (random walk)
            P = P + Q

            # Observation: y = beta*x + alpha + noise
            # H = [x, 1]
            H = np.array([x_t, 1.0])

            # Predicted observation
            y_pred = beta * x_t + alpha

            # Innovation
            innovation = y_t - y_pred

            # Innovation covariance
            S = H @ P @ H.T + R

            # Kalman gain
            K = P @ H.T / S

            # Update state
            state = np.array([beta, alpha])
            state = state + K * innovation
            beta, alpha = state[0], state[1]

            # Update covariance
            P = (np.eye(2) - np.outer(K, H)) @ P

            # Record spread and beta
            spread_t = y_t - beta * x_t - alpha
            kalman_spreads.append(spread_t)
            kalman_betas.append(beta)

        if len(kalman_spreads) < 50:
            return {
                'kalman_spread_adf_pvalue': 1.0,
                'beta_divergence': np.inf,
                'kalman_beta_final': np.nan,
                'kalman_spread_half_life': np.inf,
                'valid': False,
                'reason': 'Insufficient valid observations',
            }

        kalman_spread_series = pd.Series(kalman_spreads)
        final_kalman_beta = kalman_betas[-1]

        # ADF test on Kalman spread
        try:
            adf_result = adfuller(kalman_spread_series.dropna(), maxlag=5)
            adf_pvalue = float(adf_result[1])
        except Exception:
            adf_pvalue = 1.0

        # Beta divergence
        if abs(ols_beta) > 1e-6:
            beta_divergence = abs(final_kalman_beta - ols_beta) / abs(ols_beta)
        else:
            beta_divergence = abs(final_kalman_beta - ols_beta)

        # Half-life of Kalman spread
        half_life = self.calculate_half_life(kalman_spread_series)

        # Determine validity
        valid = (adf_pvalue < adf_threshold) and (beta_divergence < max_beta_divergence)
        reason = None

        if adf_pvalue >= adf_threshold:
            reason = f'Kalman spread not stationary (p={adf_pvalue:.3f})'
        elif beta_divergence >= max_beta_divergence:
            reason = f'Beta divergence too high ({beta_divergence:.2f})'

        return {
            'kalman_spread_adf_pvalue': adf_pvalue,
            'beta_divergence': beta_divergence,
            'kalman_beta_final': final_kalman_beta,
            'kalman_spread_half_life': half_life,
            'valid': valid,
            'reason': reason,
        }

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

    def compute_rolling_coint_stats(
        self,
        spread: pd.Series,
        window_bars: Optional[int] = None,
        sample_interval: Optional[int] = None,
    ) -> Dict:
        """
        Rolling ADF test to detect cointegration breakdowns.

        Problem #3 Fix: Cointegration in crypto is regime-dependent. This function
        tests stationarity over rolling windows to detect instability.

        Parameters
        ----------
        spread : pd.Series
            The spread series (Y - beta*X)
        window_bars : int, optional
            Rolling window size in bars. Default: 7 days worth of bars.
        sample_interval : int, optional
            How often to sample (in bars). Default: window/7 (daily for weekly window).

        Returns
        -------
        dict with:
            - failure_rate: fraction of windows that failed ADF (p > 0.05)
            - mean_pvalue: average p-value across windows
            - max_pvalue: maximum p-value (worst stationarity)
            - n_windows: number of windows tested
        """
        if len(spread) < 100:
            return {
                'failure_rate': 0.0,
                'mean_pvalue': 0.0,
                'max_pvalue': 0.0,
                'n_windows': 0,
            }

        # Default: 7-day window (assuming 1-min bars = 10080 bars)
        # For flexibility, use ~10% of data as window
        if window_bars is None:
            window_bars = max(500, len(spread) // 10)

        if sample_interval is None:
            sample_interval = max(1, window_bars // 7)

        adf_pvalues = []

        for i in range(window_bars, len(spread), sample_interval):
            chunk = spread.iloc[i - window_bars:i].dropna()

            if len(chunk) < 50:
                continue

            try:
                result = adfuller(chunk, maxlag=5, autolag=None)
                adf_pvalues.append(result[1])  # p-value
            except Exception:
                continue

        if not adf_pvalues:
            return {
                'failure_rate': 0.0,
                'mean_pvalue': 0.0,
                'max_pvalue': 0.0,
                'n_windows': 0,
            }

        failure_rate = sum(p > 0.05 for p in adf_pvalues) / len(adf_pvalues)

        return {
            'failure_rate': float(failure_rate),
            'mean_pvalue': float(np.mean(adf_pvalues)),
            'max_pvalue': float(np.max(adf_pvalues)),
            'n_windows': len(adf_pvalues),
        }

    def check_subwindow_stability(
        self,
        y: pd.Series,
        x: pd.Series,
        n_subwindows: int = 3,
        min_pass_rate: float = 1.0,
        max_half_life_cv: float = 0.5,
    ) -> Dict:
        """
        Split training data into n subwindows and verify cointegration holds in each.

        Issue #1 Fix: A pair might pass cointegration on the full training window
        but fail in individual subwindows. This indicates the relationship is not
        stable and likely regime-dependent.

        Parameters
        ----------
        y : pd.Series
            Dependent variable (log price of Y)
        x : pd.Series
            Independent variable (log price of X)
        n_subwindows : int
            Number of subwindows to test (default 3).
        min_pass_rate : float
            Minimum fraction of windows that must pass (default 1.0 = all must pass).
        max_half_life_cv : float
            Maximum coefficient of variation for half-life across windows.
            If half-life varies too much, the relationship is unstable.

        Returns
        -------
        dict with:
            - passed: bool - whether pair passes stability filter
            - pass_rate: fraction of subwindows that passed
            - subwindow_results: list of per-window results
            - half_life_cv: coefficient of variation of half-life
            - reason: rejection reason if failed
        """
        n = len(y)
        if n < n_subwindows * 50:  # Need at least 50 points per subwindow
            return {
                'passed': False,
                'pass_rate': 0.0,
                'subwindow_results': [],
                'half_life_cv': np.inf,
                'reason': 'Insufficient data for subwindow analysis',
            }

        window_size = n // n_subwindows
        results = []

        for i in range(n_subwindows):
            start = i * window_size
            end = start + window_size if i < n_subwindows - 1 else n

            y_sub = y.iloc[start:end]
            x_sub = x.iloc[start:end]

            # Drop NaN
            valid = y_sub.notna() & x_sub.notna()
            y_valid = y_sub[valid]
            x_valid = x_sub[valid]

            if len(y_valid) < 30:
                results.append({
                    'window': i,
                    'passed': False,
                    'p_value': 1.0,
                    'half_life': np.inf,
                    'beta': np.nan,
                    'reason': 'Insufficient data',
                })
                continue

            # Test cointegration in this subwindow
            try:
                _, p_val, _ = coint(y_valid, x_valid, maxlag=self.max_autolag)

                # Compute beta and half-life
                X = sm.add_constant(x_valid)
                model = sm.OLS(y_valid, X).fit()
                beta = float(model.params.iloc[1])
                spread = y_valid - beta * x_valid
                half_life = self.calculate_half_life(spread)

                # Check if this window passes
                passed = (
                    p_val < self.p_value_threshold and
                    self.min_half_life <= half_life <= self.max_half_life
                )

                results.append({
                    'window': i,
                    'passed': passed,
                    'p_value': float(p_val),
                    'half_life': float(half_life),
                    'beta': beta,
                    'reason': None if passed else f'p={p_val:.3f}, hl={half_life:.0f}',
                })
            except Exception as e:
                results.append({
                    'window': i,
                    'passed': False,
                    'p_value': 1.0,
                    'half_life': np.inf,
                    'beta': np.nan,
                    'reason': str(e),
                })

        # Compute pass rate
        n_passed = sum(1 for r in results if r['passed'])
        pass_rate = n_passed / n_subwindows

        # Compute half-life coefficient of variation
        half_lives = [r['half_life'] for r in results
                      if r['passed'] and np.isfinite(r['half_life'])]

        if len(half_lives) >= 2:
            hl_mean = np.mean(half_lives)
            hl_std = np.std(half_lives)
            half_life_cv = hl_std / hl_mean if hl_mean > 0 else np.inf
        else:
            half_life_cv = 0.0  # Not enough data to compute CV

        # Determine overall pass/fail
        passed = (pass_rate >= min_pass_rate) and (half_life_cv <= max_half_life_cv)
        reason = None

        if pass_rate < min_pass_rate:
            reason = f'Subwindow pass rate {pass_rate:.0%} < required {min_pass_rate:.0%}'
        elif half_life_cv > max_half_life_cv:
            reason = f'Half-life CV {half_life_cv:.2f} > max {max_half_life_cv:.2f}'

        return {
            'passed': passed,
            'pass_rate': pass_rate,
            'subwindow_results': results,
            'half_life_cv': half_life_cv,
            'reason': reason,
        }

    def compute_beta_stability(
        self,
        y: pd.Series,
        x: pd.Series,
        window_bars: Optional[int] = None,
    ) -> Dict:
        """
        Track how much beta drifts over time using rolling OLS.

        Problem #3 Fix: If beta drifts significantly, the "cointegration" is really
        just a moving linear fit that can break suddenly.

        Parameters
        ----------
        y : pd.Series
            Dependent variable (log price of Y)
        x : pd.Series
            Independent variable (log price of X)
        window_bars : int, optional
            Rolling window for beta estimation. Default: 10% of data.

        Returns
        -------
        dict with:
            - beta_std: standard deviation of rolling beta
            - beta_range: max - min of rolling beta
            - beta_drift: |beta_end - beta_start| / |beta_start|
            - beta_series: the rolling beta values (for plotting)
        """
        if len(y) < 100 or len(x) < 100:
            return {
                'beta_std': 0.0,
                'beta_range': 0.0,
                'beta_drift': 0.0,
                'beta_series': pd.Series(dtype=float),
            }

        if window_bars is None:
            window_bars = max(100, len(y) // 10)

        betas = []
        indices = []

        for i in range(window_bars, len(y), window_bars // 4):
            y_chunk = y.iloc[i - window_bars:i]
            x_chunk = x.iloc[i - window_bars:i]

            # Align and drop NaN
            valid = y_chunk.notna() & x_chunk.notna()
            y_valid = y_chunk[valid]
            x_valid = x_chunk[valid]

            if len(y_valid) < 20:
                continue

            try:
                X = sm.add_constant(x_valid)
                res = sm.OLS(y_valid, X).fit()
                beta = res.params.iloc[1]
                betas.append(beta)
                indices.append(y.index[i - 1])
            except Exception:
                continue

        if len(betas) < 2:
            return {
                'beta_std': 0.0,
                'beta_range': 0.0,
                'beta_drift': 0.0,
                'beta_series': pd.Series(dtype=float),
            }

        beta_series = pd.Series(betas, index=indices)
        beta_start = betas[0]
        beta_end = betas[-1]

        # Compute drift as relative change
        if abs(beta_start) > 1e-6:
            beta_drift = abs(beta_end - beta_start) / abs(beta_start)
        else:
            beta_drift = 0.0

        return {
            'beta_std': float(np.std(betas)),
            'beta_range': float(np.max(betas) - np.min(betas)),
            'beta_drift': float(beta_drift),
            'beta_series': beta_series,
        }

    def _test_single_direction(
        self,
        s1: str,
        s2: str,
        train_x: pd.Series,
        train_y: pd.Series,
        val_x: pd.Series,
        val_y: pd.Series,
        check_rolling_coint: bool = True,
        check_beta_stability: bool = True,
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
        if half_life < self.min_half_life or half_life > self.max_half_life:
            return None, f"Bad Half-Life ({half_life:.1f} bars, need {self.min_half_life}-{self.max_half_life})"

        # Rolling cointegration check (Problem #3 fix)
        rolling_stats = {}
        if check_rolling_coint:
            rolling_stats = self.compute_rolling_coint_stats(train_spread)
            if rolling_stats['n_windows'] > 0:
                if rolling_stats['failure_rate'] > self.DEFAULT_MAX_COINT_FAILURE_RATE:
                    return None, f"Rolling coint unstable (failure_rate={rolling_stats['failure_rate']:.2f})"

        # Beta stability check (Problem #3 fix)
        beta_stats = {}
        if check_beta_stability:
            beta_stats = self.compute_beta_stability(train_y, train_x)
            if beta_stats['beta_drift'] > self.DEFAULT_MAX_BETA_DRIFT:
                return None, f"Beta drift too high ({beta_stats['beta_drift']:.2f})"

        current_z = float(val_z_scores.iloc[-1])

        result = {
            "pair": f"{s1}-{s2}",
            "coin_y": s1,
            "coin_x": s2,
            "p_value": pvalue,
            "hedge_ratio": hedge_ratio,
            "half_life_bars": half_life,
            "spread_vol": vol,
            "current_z_score": current_z,
        }

        # Add diagnostics to result
        if rolling_stats:
            result["rolling_coint_failure_rate"] = rolling_stats.get('failure_rate', 0.0)
            result["rolling_coint_mean_pvalue"] = rolling_stats.get('mean_pvalue', 0.0)
        if beta_stats:
            result["beta_drift"] = beta_stats.get('beta_drift', 0.0)
            result["beta_std"] = beta_stats.get('beta_std', 0.0)

        return (result, None)

    def test_pair(
        self,
        s1: str,
        s2: str,
        train_x: pd.Series,
        train_y: pd.Series,
        val_x: pd.Series,
        val_y: pd.Series,
        check_rolling_coint: bool = True,
        check_beta_stability: bool = True,
    ):
        """
        Test cointegration in BOTH directions and return the better result
        (lower p-value if both pass).

        Parameters
        ----------
        check_rolling_coint : bool
            If True, run rolling ADF tests to detect cointegration instability.
        check_beta_stability : bool
            If True, check for excessive beta drift over time.
        """
        # Direction 1: s1 ~ s2
        result1, reason1 = self._test_single_direction(
            s1, s2, train_x, train_y, val_x, val_y,
            check_rolling_coint=check_rolling_coint,
            check_beta_stability=check_beta_stability,
        )

        # Direction 2: s2 ~ s1 (swap)
        result2, reason2 = self._test_single_direction(
            s2, s1, train_y, train_x, val_y, val_x,
            check_rolling_coint=check_rolling_coint,
            check_beta_stability=check_beta_stability,
        )

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

    def _test_pair_parallel(
        self,
        s1: str,
        s2: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        check_rolling_coint: bool,
        check_beta_stability: bool,
    ) -> Optional[dict]:
        """
        Helper method for parallel pair testing.
        Returns the result dict if pair is valid, None otherwise.
        """
        try:
            result = self.test_pair(
                s1,
                s2,
                train_x=train_data[s2],
                train_y=train_data[s1],
                val_x=val_data[s2],
                val_y=val_data[s1],
                check_rolling_coint=check_rolling_coint,
                check_beta_stability=check_beta_stability,
            )
            if isinstance(result, dict):
                return result
            return None
        except Exception as e:
            # Return error info for logging later
            return {'error': str(e), 'pair': f"{s1}-{s2}"}

    def _scan_with_log_prices(
        self,
        prices: pd.DataFrame,
        train_ratio: float,
        check_rolling_coint: bool = False,
        check_beta_stability: bool = False,
        volume_data: Optional[pd.DataFrame] = None,
        min_daily_volume_usd: float = 0.0,
    ) -> pd.DataFrame:
        """
        Shared scanning logic once we already have log prices.

        Parameters
        ----------
        prices : pd.DataFrame
            log(price_matrix) with columns as symbols
        train_ratio : float
            Fraction of data to use for training
        check_rolling_coint : bool
            If True, run rolling ADF tests to detect cointegration instability
        check_beta_stability : bool
            If True, check for excessive beta drift over time
        volume_data : pd.DataFrame, optional
            Dollar volume data for liquidity pre-filtering (same columns as prices)
        min_daily_volume_usd : float
            Minimum average daily volume in USD to include a coin (Phase 5A)
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

        # Phase 5A: Pre-compute liquid coins set for efficiency
        liquid_coins_set: Optional[set] = None
        if volume_data is not None and min_daily_volume_usd > 0:
            # Compute average daily volume per coin
            avg_volume = volume_data.mean()
            liquid_coins_set = set(avg_volume[avg_volume >= min_daily_volume_usd].index)
            logger.info(
                "Liquidity filter: %d/%d coins pass min volume $%.0f/day",
                len(liquid_coins_set), len(avg_volume), min_daily_volume_usd
            )

        valid_pairs = []
        total_pairs_checked = 0

        for cluster_id, coins in self.cluster_map.items():
            available_coins = [c for c in coins if c in prices.columns]

            # Phase 5A: Apply liquidity filter
            if liquid_coins_set is not None:
                available_coins = [c for c in available_coins if c in liquid_coins_set]

            n = len(available_coins)
            if n < 2:
                continue

            logger.info(f"Checking Cluster {cluster_id} ({n} coins)...")
            # Correlation prefilter: compute absolute correlation matrix for this cluster
            cluster_prices = prices[available_coins]
            corr = cluster_prices.corr().abs()

            # Build list of pair candidates after correlation filtering
            pair_candidates = []
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

                    pair_candidates.append((s1, s2))

            # Parallel test all candidates
            if pair_candidates:
                logger.info(f"  Testing {len(pair_candidates)} pairs (after correlation filter) using {-1} CPU cores...")
                results = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                    delayed(self._test_pair_parallel)(
                        s1, s2, train_data, val_data,
                        check_rolling_coint, check_beta_stability
                    )
                    for s1, s2 in pair_candidates
                )

                # Collect results and log matches
                for result in results:
                    if result is not None:
                        if 'error' in result:
                            logger.warning(f"Error testing {result['pair']}: {result['error']}")
                        else:
                            valid_pairs.append(result)
                            logger.info(
                                f"  MATCH: {result['pair']} "
                                f"(P={result['p_value']:.4f}, Z={result['current_z_score']:.2f}, "
                                f"HL={result['half_life_bars']:.1f} bars)"
                            )

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

    def find_pairs_from_matrix(
        self,
        price_matrix: pd.DataFrame,
        train_ratio: float = 0.90,
        check_rolling_coint: bool = False,  # Disabled by default - expensive and can be too strict
        check_beta_stability: bool = False,  # Disabled by default - expensive and can be too strict
        volume_data: Optional[pd.DataFrame] = None,  # Phase 5A: liquidity pre-filter
        min_daily_volume_usd: float = 0.0,  # Phase 5A: minimum daily volume
    ) -> pd.DataFrame:
        """
        Phase 5 backtest entrypoint:
        scans for cointegrated pairs using an already-prepared price matrix (train_df).

        price_matrix must be wide:
            index = timestamps
            columns = symbols
            values = prices (close)

        This prevents data source drift and ensures you scan ONLY on backtest train window.

        Parameters
        ----------
        check_rolling_coint : bool
            If True, run rolling ADF tests to detect cointegration instability.
        check_beta_stability : bool
            If True, check for excessive beta drift over time.
        volume_data : pd.DataFrame, optional
            Dollar volume data for liquidity pre-filtering (columns = symbols)
            Phase 5A improvement: filter illiquid coins before expensive coint tests.
        min_daily_volume_usd : float
            Minimum average daily volume in USD to include a coin.
            Default 0.0 means no filtering.
        """
        logger.info("SCANNING FOR COINTEGRATED PAIRS (matrix mode)...")
        log_prices = self._to_log_price_matrix(price_matrix)
        return self._scan_with_log_prices(
            log_prices,
            train_ratio=train_ratio,
            check_rolling_coint=check_rolling_coint,
            check_beta_stability=check_beta_stability,
            volume_data=volume_data,
            min_daily_volume_usd=min_daily_volume_usd,
        )

    def find_pairs_two_stage(
        self,
        price_matrix: pd.DataFrame,
        discovery_lookback_days: int = 120,
        validation_lookback_days: int = 30,
        discovery_p_value: float = 0.05,
        validation_p_value: float = 0.02,
        validation_min_half_life: int = 40,
        validation_max_half_life: int = 500,
        require_both: bool = True,
    ) -> pd.DataFrame:
        """
        Two-stage pair selection for crypto cointegration.

        Stage 1 (Discovery): Use longer lookback to find candidate pairs with
        looser thresholds. This captures pairs that have shown cointegration
        historically.

        Stage 2 (Validation): Require relationship to hold in RECENT data only
        with tighter thresholds. This filters out pairs where the relationship
        has broken down.

        This addresses regime-dependent cointegration in crypto markets where
        relationships can break down over time.

        Parameters
        ----------
        price_matrix : pd.DataFrame
            Full price matrix with timestamps as index, symbols as columns.
        discovery_lookback_days : int
            Lookback for Stage 1 (candidate discovery), typically 90-180d.
        validation_lookback_days : int
            Lookback for Stage 2 (recent validation), typically 30-45d.
        discovery_p_value : float
            P-value threshold for Stage 1 (can be looser).
        validation_p_value : float
            P-value threshold for Stage 2 (must be tighter).
        validation_min_half_life : int
            Min half-life required in validation window.
        validation_max_half_life : int
            Max half-life required in validation window.
        require_both : bool
            If True, pair must pass BOTH stages. If False, discovery-only pairs
            are included but flagged.

        Returns
        -------
        pd.DataFrame
            Valid pairs with columns including stage1_*, stage2_*, passed_both.
        """
        logger.info(
            "TWO-STAGE PAIR SELECTION: discovery=%dd (p<%.3f), validation=%dd (p<%.3f)",
            discovery_lookback_days,
            discovery_p_value,
            validation_lookback_days,
            validation_p_value,
        )

        # Convert to log prices
        log_prices = self._to_log_price_matrix(price_matrix)
        if log_prices.empty:
            logger.warning("Empty price matrix after log transformation.")
            return pd.DataFrame()

        # Determine date ranges
        end_date = log_prices.index.max()
        discovery_start = end_date - pd.Timedelta(days=discovery_lookback_days)
        validation_start = end_date - pd.Timedelta(days=validation_lookback_days)

        discovery_data = log_prices[log_prices.index >= discovery_start]
        validation_data = log_prices[log_prices.index >= validation_start]

        logger.info(
            "Data ranges: discovery=%d rows (%s to %s), validation=%d rows (%s to %s)",
            len(discovery_data),
            discovery_data.index.min(),
            discovery_data.index.max(),
            len(validation_data),
            validation_data.index.min(),
            validation_data.index.max(),
        )

        # --- STAGE 1: Discovery ---
        logger.info("Stage 1: Discovery scan with p_value < %.3f...", discovery_p_value)

        # Temporarily adjust thresholds for discovery
        original_p_value = self.p_value_threshold
        self.p_value_threshold = discovery_p_value

        # Use existing scan logic for Stage 1
        stage1_results = self._scan_with_log_prices(
            discovery_data,
            train_ratio=0.85,
            check_rolling_coint=False,
            check_beta_stability=False,
        )

        self.p_value_threshold = original_p_value

        if stage1_results.empty:
            logger.warning("Stage 1 found 0 candidates. Consider looser discovery thresholds.")
            return pd.DataFrame()

        logger.info("Stage 1: %d candidate pairs discovered.", len(stage1_results))

        # --- STAGE 2: Validation on Recent Data ---
        logger.info(
            "Stage 2: Validation on recent %d days with p_value < %.3f...",
            validation_lookback_days,
            validation_p_value,
        )

        validated_pairs = []

        for _, row in stage1_results.iterrows():
            pair = row["pair"]
            coin_y = row["coin_y"]
            coin_x = row["coin_x"]

            if coin_y not in validation_data.columns or coin_x not in validation_data.columns:
                logger.debug("Pair %s: missing columns in validation data.", pair)
                continue

            y_val = validation_data[coin_y]
            x_val = validation_data[coin_x]

            # Split validation window into mini train/test
            val_train_len = int(len(y_val) * 0.75)
            y_train, y_test = y_val.iloc[:val_train_len], y_val.iloc[val_train_len:]
            x_train, x_test = x_val.iloc[:val_train_len], x_val.iloc[val_train_len:]

            # Re-test cointegration on recent data
            result, reason = self._test_single_direction(
                coin_y,
                coin_x,
                train_x=x_train,
                train_y=y_train,
                val_x=x_test,
                val_y=y_test,
                check_rolling_coint=True,
                check_beta_stability=True,
            )

            stage2_passed = False
            stage2_info = {
                "stage2_p_value": np.nan,
                "stage2_half_life": np.nan,
                "stage2_hedge_ratio": np.nan,
            }

            if isinstance(result, dict):
                stage2_info = {
                    "stage2_p_value": result.get("p_value", np.nan),
                    "stage2_half_life": result.get("half_life_bars", np.nan),
                    "stage2_hedge_ratio": result.get("hedge_ratio", np.nan),
                }
                # Check tighter thresholds
                p_ok = result.get("p_value", 1.0) < validation_p_value
                hl_ok = validation_min_half_life <= result.get("half_life_bars", 0) <= validation_max_half_life
                if p_ok and hl_ok:
                    stage2_passed = True

            # Combine results
            combined = {
                "pair": pair,
                "coin_y": coin_y,
                "coin_x": coin_x,
                # Stage 1 results (from discovery)
                "stage1_p_value": row.get("p_value", np.nan),
                "stage1_half_life": row.get("half_life_bars", np.nan),
                "stage1_hedge_ratio": row.get("hedge_ratio", np.nan),
                # Stage 2 results
                "stage2_passed": stage2_passed,
                **stage2_info,
                # Combined flag
                "passed_both": stage2_passed,
                # Carry forward other useful columns from stage 1
                "current_z_score": row.get("current_z_score", np.nan),
            }

            if require_both and not stage2_passed:
                logger.debug("Pair %s: REJECTED (failed Stage 2 validation).", pair)
                continue

            validated_pairs.append(combined)

        df_validated = pd.DataFrame(validated_pairs)

        if not df_validated.empty:
            # Sort by stage2 p-value (best validation first)
            if "stage2_p_value" in df_validated.columns:
                df_validated = df_validated.sort_values("stage2_p_value").reset_index(drop=True)

            n_passed_both = df_validated["passed_both"].sum()
            logger.info(
                "Stage 2: %d pairs validated (%d passed both stages).",
                len(df_validated),
                n_passed_both,
            )
        else:
            logger.warning("Stage 2: 0 pairs passed validation.")

        return df_validated


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
