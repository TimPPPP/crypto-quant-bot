import numpy as np
import pandas as pd
import logging
from collections import deque
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.adaptive.market_regime import MarketRegime, VolatilityRegime, TrendRegime

logger = logging.getLogger("KalmanFilter")


# =============================================================================
# ADAPTIVE KALMAN PARAMETER FUNCTIONS
# =============================================================================

def compute_adaptive_kalman_params(
    regime: "MarketRegime",
    base_delta: float = 1e-6,
    base_R: float = 1e-2,
    delta_mult_min: float = 0.3,
    delta_mult_max: float = 3.0,
    r_mult_min: float = 0.5,
    r_mult_max: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute adaptive Kalman filter parameters based on market regime.

    Adaptation logic:
    - High volatility → Increase R (trust observations less, more measurement noise)
    - Trending market → Increase delta (allow beta to adapt faster to changing relationships)
    - Mean-reverting market → Decrease delta (trust historical beta, less process noise)

    Args:
        regime: MarketRegime from market_regime.py
        base_delta: Baseline process noise (Q scaling factor)
        base_R: Baseline measurement noise
        delta_mult_min: Minimum delta multiplier
        delta_mult_max: Maximum delta multiplier
        r_mult_min: Minimum R multiplier
        r_mult_max: Maximum R multiplier

    Returns:
        (adapted_delta, adapted_R)
    """
    # Import here to avoid circular imports
    from src.adaptive.market_regime import VolatilityRegime, TrendRegime

    # --- R Adaptation (Measurement Noise) ---
    # High vol: increase R to trust observations less (more noise in prices)
    # Low vol: decrease R to trust observations more
    if regime.volatility == VolatilityRegime.HIGH:
        # Scale up based on how extreme the percentile is
        r_mult = 2.0 + (regime.btc_vol_percentile - 0.75) * 4.0  # 2.0 to 3.0
    elif regime.volatility == VolatilityRegime.LOW:
        r_mult = 0.7  # Trust observations more in calm markets
    else:
        r_mult = 1.0  # Normal

    r_mult = np.clip(r_mult, r_mult_min, r_mult_max)
    adapted_R = base_R * r_mult

    # --- Delta Adaptation (Process Noise) ---
    # Trending: increase delta to let beta adapt faster to changing relationships
    # Mean-reverting: decrease delta to maintain stable beta estimates
    if regime.trend == TrendRegime.TRENDING:
        # Scale up based on autocorrelation strength
        delta_mult = 2.0 + abs(regime.market_autocorr) * 5.0  # 2.0 to ~2.75
    elif regime.trend == TrendRegime.MEAN_REVERTING:
        delta_mult = 0.5  # Very stable beta
    else:
        delta_mult = 1.0  # Normal

    delta_mult = np.clip(delta_mult, delta_mult_min, delta_mult_max)
    adapted_delta = base_delta * delta_mult

    logger.debug(
        f"Adaptive Kalman: R={adapted_R:.2e} (mult={r_mult:.2f}), "
        f"delta={adapted_delta:.2e} (mult={delta_mult:.2f})"
    )

    return adapted_delta, adapted_R


def get_adaptive_params_from_config(
    regime: "MarketRegime",
    config: Optional[Dict] = None,
) -> Tuple[float, float]:
    """
    Get adaptive Kalman parameters using config values.

    Args:
        regime: MarketRegime object
        config: Optional config dict (uses defaults if None)

    Returns:
        (adapted_delta, adapted_R)
    """
    if config is None:
        config = {}

    return compute_adaptive_kalman_params(
        regime=regime,
        base_delta=config.get("KALMAN_BASE_DELTA", 1e-6),
        base_R=config.get("KALMAN_BASE_R", 1e-2),
        delta_mult_min=config.get("KALMAN_DELTA_MULT_MIN", 0.3),
        delta_mult_max=config.get("KALMAN_DELTA_MULT_MAX", 3.0),
        r_mult_min=config.get("KALMAN_R_MULT_MIN", 0.5),
        r_mult_max=config.get("KALMAN_R_MULT_MAX", 3.0),
    )


class KalmanFilterRegime:
    """
    Adaptive Signal Engine for Pairs Trading.

    Features:
    1. Sticky Beta: Low Process Noise (Q) prevents overfitting
    2. Rolling Z-Score: Uses Realized Volatility for robust signals
    3. Burn-In: Ignores first N ticks to let filter stabilize
    4. Numerical Stability: Bounds on Kalman gain and covariance
    5. Half-Life Scaled Vol Window: Consistent z-score distribution across pairs
    """

    # Default parameters (can be overridden via constructor)
    DEFAULT_DELTA = 1e-6      # Process noise
    DEFAULT_R = 1e-2          # Measurement noise
    DEFAULT_WINDOW = 30       # Rolling window for volatility (legacy default)
    DEFAULT_ENTRY_Z = 2.0     # Z-score threshold for entry signal
    DEFAULT_MIN_SPREAD = 0.003  # Minimum spread error for signal (0.3%)

    # Vol window scaling defaults (aligned with config_backtest)
    DEFAULT_VOL_WINDOW_HL_MULT = 1.0  # vol_window = half_life * multiplier
    DEFAULT_MIN_VOL_WINDOW = 60       # 1 hour at 1-min bars
    DEFAULT_MAX_VOL_WINDOW = 1440     # 24 hours at 1-min bars
    DEFAULT_VOL_METHOD = "ewma"
    DEFAULT_VOL_EWMA_ALPHA = 0.2
    DEFAULT_VOL_MAD_SCALE = 1.4826

    def __init__(
        self,
        delta: float = None,
        R: float = None,
        rolling_window: int = None,
        entry_z_threshold: float = None,
        min_spread_pct: float = None,
        half_life_bars: float = None,
        vol_window_hl_mult: float = None,
        vol_method: str = None,
        vol_ewma_alpha: float = None,
        vol_mad_scale: float = None,
    ):
        """
        Initialize Kalman Filter for swing trading.

        Args:
            delta: Process noise (lower = stiffer filter, holds trades longer)
            R: Measurement noise (higher = more tolerant of price blips)
            rolling_window: Window size for realized volatility calculation.
                           If half_life_bars is provided and rolling_window is not,
                           the window will be scaled from half_life.
            entry_z_threshold: Z-score threshold for generating entry signals
            min_spread_pct: Minimum spread error percentage for signals
            half_life_bars: (NEW) Half-life in bars for this pair. Used to scale
                           the volatility window for consistent z-score behavior.
            vol_window_hl_mult: Multiplier for half_life -> vol_window scaling.
                               Default 1.0 means vol_window = half_life.
        """
        # Use defaults if not specified
        delta = delta if delta is not None else self.DEFAULT_DELTA
        R = R if R is not None else self.DEFAULT_R

        # Half-life scaled volatility window (Problem #2 fix)
        # If half_life_bars provided and rolling_window not explicitly set,
        # compute rolling_window from half_life for consistent z-score scaling
        if half_life_bars is not None and rolling_window is None:
            hl_mult = vol_window_hl_mult if vol_window_hl_mult is not None else self.DEFAULT_VOL_WINDOW_HL_MULT
            computed_window = int(half_life_bars * hl_mult)
            # Clamp to reasonable bounds
            rolling_window = max(self.DEFAULT_MIN_VOL_WINDOW,
                               min(computed_window, self.DEFAULT_MAX_VOL_WINDOW))
        else:
            rolling_window = rolling_window if rolling_window is not None else self.DEFAULT_WINDOW

        # Store half_life for reference
        self.half_life_bars = half_life_bars

        self.entry_z_threshold = entry_z_threshold if entry_z_threshold is not None else self.DEFAULT_ENTRY_Z
        self.min_spread_pct = min_spread_pct if min_spread_pct is not None else self.DEFAULT_MIN_SPREAD
        self.vol_method = (vol_method or self.DEFAULT_VOL_METHOD).lower()
        self.vol_ewma_alpha = vol_ewma_alpha if vol_ewma_alpha is not None else self.DEFAULT_VOL_EWMA_ALPHA
        self.vol_mad_scale = vol_mad_scale if vol_mad_scale is not None else self.DEFAULT_VOL_MAD_SCALE

        # State dimensions
        self.n_dim = 2

        # State vector [beta, alpha]
        self.x = np.zeros(self.n_dim)

        # Covariance matrix
        self.P = np.eye(self.n_dim)

        # Process Noise - Sticky beta (slower adaptation)
        self.delta = delta
        self.Q = np.eye(self.n_dim) * delta
        self.Q[0, 0] = delta / 100  # Beta changes 100x slower than alpha

        # Measurement Noise
        self.R = R

        # Rolling Statistics for realized volatility
        self.rolling_window = rolling_window
        self.error_history = deque(maxlen=rolling_window)
        self.burn_in = rolling_window

        # Track latest values for external access (used by main.py)
        self.latest_z = 0.0
        self.latest_error = 0.0
        self.latest_std = 0.0
        self.is_warmed_up = False

        # Numerical stability bounds
        self._max_kalman_gain = 1.0
        self._min_covariance = 1e-10
        self._max_covariance = 1e6

    def update(self, price_y: float, price_x: float) -> Dict:
        """
        Run one Kalman filter update step.

        Args:
            price_y: Log price of dependent asset (Y)
            price_x: Log price of independent asset (X)

        Returns:
            Dict with hedge_ratio, intercept, spread_error, spread_std, z_score, is_signal
        """
        # Validate inputs
        if not np.isfinite(price_y) or not np.isfinite(price_x):
            logger.warning("Invalid price input (NaN/Inf), skipping update")
            return self._get_current_state(is_signal=False)

        # Observation Matrix H = [x, 1]
        H = np.array([price_x, 1.0])

        # --- KALMAN PREDICT ---
        P_pred = self.P + self.Q

        # Ensure covariance stays bounded
        P_pred = np.clip(P_pred, self._min_covariance, self._max_covariance)

        # --- KALMAN UPDATE ---
        y_pred = np.dot(H, self.x)
        error = price_y - y_pred

        # Innovation covariance
        S = np.dot(H, np.dot(P_pred, H.T)) + self.R

        # Kalman gain with stability bound
        if S > 1e-10:
            K = np.dot(P_pred, H.T) / S
            K = np.clip(K, -self._max_kalman_gain, self._max_kalman_gain)
        else:
            K = np.zeros(self.n_dim)

        # Update state
        self.x = self.x + (K * error)

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n_dim) - np.outer(K, H)
        self.P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.R
        self.P = np.clip(self.P, self._min_covariance, self._max_covariance)

        # Store error for realized volatility
        self.error_history.append(error)

        # --- SIGNAL GENERATION ---
        beta = self.x[0]
        alpha = self.x[1]

        # Check warmup status
        self.is_warmed_up = len(self.error_history) >= self.burn_in

        # Calculate realized volatility (std dev of recent errors)
        if self.is_warmed_up:
            realized_std = self._compute_realized_std()
        else:
            # During warmup, use a conservative estimate
            realized_std = 0.01  # 1% default to avoid division issues

        # Robust Z-Score with safety check
        if realized_std > 1e-8:
            z_score = error / realized_std
        else:
            z_score = 0.0

        # Update latest values for external access
        self.latest_z = z_score
        self.latest_error = error
        self.latest_std = realized_std

        # --- SIGNAL GATE ---
        is_signal = False

        # Only generate signals after burn-in
        if self.is_warmed_up:
            # Configurable thresholds
            z_threshold_met = abs(z_score) > self.entry_z_threshold
            spread_threshold_met = abs(error) > self.min_spread_pct

            if z_threshold_met and spread_threshold_met:
                is_signal = True

        return {
            'hedge_ratio': beta,
            'intercept': alpha,
            'spread_error': error,
            'spread_std': realized_std,
            'z_score': z_score,
            'is_signal': is_signal,
            'is_warmed_up': self.is_warmed_up
        }

    def _get_current_state(self, is_signal: bool = False) -> Dict:
        """Return current state without updating."""
        return {
            'hedge_ratio': self.x[0],
            'intercept': self.x[1],
            'spread_error': self.latest_error,
            'spread_std': self.latest_std,
            'z_score': self.latest_z,
            'is_signal': is_signal,
            'is_warmed_up': self.is_warmed_up
        }

    def _compute_realized_std(self) -> float:
        arr = np.asarray(self.error_history, dtype="float64")
        if len(arr) == 0:
            return 0.0
        if self.vol_method == "ewma":
            alpha = float(self.vol_ewma_alpha)
            var = 0.0
            for v in arr:
                var = alpha * (v * v) + (1.0 - alpha) * var
            return float(np.sqrt(var))
        if self.vol_method == "mad":
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            return float(mad * self.vol_mad_scale)
        return float(np.std(arr))

    def get_state_dict(self) -> Dict:
        """Export filter state for persistence."""
        return {
            'x': self.x.tolist(),
            'P': self.P.tolist(),
            'R': self.R,
            'delta': self.delta,
            'error_history': list(self.error_history),
            'rolling_window': self.rolling_window,
            'entry_z_threshold': self.entry_z_threshold,
            'min_spread_pct': self.min_spread_pct,
            'half_life_bars': self.half_life_bars,
            'vol_method': self.vol_method,
            'vol_ewma_alpha': self.vol_ewma_alpha,
            'vol_mad_scale': self.vol_mad_scale,
        }

    def load_state_dict(self, state: Dict) -> bool:
        """
        Restore filter state from persistence.

        Args:
            state: Dictionary from get_state_dict()

        Returns:
            True if successful, False otherwise
        """
        try:
            self.x = np.array(state['x'])
            self.P = np.array(state['P'])
            self.R = state.get('R', self.DEFAULT_R)

            # Restore delta and rebuild Q
            self.delta = state.get('delta', self.DEFAULT_DELTA)
            self.Q = np.eye(self.n_dim) * self.delta
            self.Q[0, 0] = self.delta / 100

            # Restore rolling window
            self.rolling_window = state.get('rolling_window', self.DEFAULT_WINDOW)
            self.error_history = deque(maxlen=self.rolling_window)

            # Restore error history
            for err in state.get('error_history', []):
                self.error_history.append(err)

            # Restore thresholds
            self.entry_z_threshold = state.get('entry_z_threshold', self.DEFAULT_ENTRY_Z)
            self.min_spread_pct = state.get('min_spread_pct', self.DEFAULT_MIN_SPREAD)
            self.vol_method = state.get('vol_method', self.DEFAULT_VOL_METHOD)
            self.vol_ewma_alpha = state.get('vol_ewma_alpha', self.DEFAULT_VOL_EWMA_ALPHA)
            self.vol_mad_scale = state.get('vol_mad_scale', self.DEFAULT_VOL_MAD_SCALE)

            # Restore half-life (may be None for legacy states)
            self.half_life_bars = state.get('half_life_bars', None)

            # Update warmup status
            self.is_warmed_up = len(self.error_history) >= self.burn_in

            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

if __name__ == "__main__":
    print("Testing Kalman Filter...")

    # 1. Generate Synthetic Cointegrated Data
    np.random.seed(42)
    n_steps = 300
    X = np.linspace(10, 12, n_steps)
    noise = np.random.normal(0, 0.005, n_steps)

    # Add shock at step 250 to test signal generation
    noise[250:260] += 0.02

    Y = 1.2 * X + noise

    # 2. Run Filter
    kf = KalmanFilterRegime(delta=1e-5, R=1e-4, rolling_window=30)

    print(f"{'Step':<5} | {'True Beta':<10} | {'Est Beta':<10} | {'Error':<10} | {'Z-Score':<10} | {'Signal'}")
    print("-" * 75)

    signals_generated = 0
    for i, (x_val, y_val) in enumerate(zip(X, Y)):
        res = kf.update(y_val, x_val)

        if res['is_signal']:
            signals_generated += 1

        # Print warmup and shock windows
        if i < 5 or (245 < i < 265):
            signal_icon = "SIGNAL" if res['is_signal'] else "-"
            warmed = "Y" if res['is_warmed_up'] else "N"
            print(f"{i:<5} | {'1.200':<10} | {res['hedge_ratio']:<10.4f} | {res['spread_error']:<10.4f} | {res['z_score']:<10.2f} | {signal_icon} (warm:{warmed})")

    print("-" * 75)
    print(f"Total signals generated: {signals_generated}")
    print(f"Final beta estimate: {kf.x[0]:.4f} (true: 1.2)")
    print(f"Latest Z-score (via attribute): {kf.latest_z:.2f}")

    # 3. Test state persistence
    print("\nTesting state persistence...")
    state = kf.get_state_dict()
    kf2 = KalmanFilterRegime()
    kf2.load_state_dict(state)
    print(f"Restored beta: {kf2.x[0]:.4f}")
    print(f"Restored error history length: {len(kf2.error_history)}")
    print("State persistence test passed!" if kf2.x[0] == kf.x[0] else "State persistence FAILED!")
