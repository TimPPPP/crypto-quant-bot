"""
src/adaptive/market_regime.py

Market regime detection for adaptive strategy parameters.

Detects two orthogonal regime dimensions:
1. Volatility regime (LOW/NORMAL/HIGH) based on BTC realized volatility percentile
2. Trend regime (TRENDING/MEAN_REVERTING/NEUTRAL) based on market-wide autocorrelation

These regimes are used to:
- Adjust half-life search range for pair selection
- Adapt Kalman filter Q/R parameters
- Modify trend overlay sensitivity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("MarketRegime")


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TrendRegime(Enum):
    """Trend regime classification based on market autocorrelation."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"


@dataclass
class MarketRegime:
    """
    Complete market regime state.

    Attributes:
        volatility: Current volatility regime
        trend: Current trend regime
        btc_vol_percentile: BTC volatility percentile (0-1)
        btc_vol_annualized: BTC annualized volatility
        market_autocorr: Market-wide lag-1 autocorrelation
        recommended_min_half_life: Suggested minimum half-life for pair scanning
        recommended_max_half_life: Suggested maximum half-life for pair scanning
    """
    volatility: VolatilityRegime
    trend: TrendRegime
    btc_vol_percentile: float
    btc_vol_annualized: float
    market_autocorr: float
    recommended_min_half_life: int
    recommended_max_half_life: int

    def __str__(self) -> str:
        return (
            f"MarketRegime(vol={self.volatility.value}, trend={self.trend.value}, "
            f"btc_vol_pctl={self.btc_vol_percentile:.2f}, autocorr={self.market_autocorr:.3f}, "
            f"hl_range=({self.recommended_min_half_life}, {self.recommended_max_half_life}))"
        )


# Default configuration (can be overridden via config_backtest.py)
DEFAULT_REGIME_CONFIG = {
    # Lookback for regime detection
    "lookback_bars": 336,  # ~2 weeks at hourly, ~1 week at 15min

    # Volatility regime thresholds (percentiles of historical BTC vol)
    "vol_low_percentile": 0.25,
    "vol_high_percentile": 0.75,

    # Trend regime thresholds (lag-1 autocorrelation)
    "trend_trending_thresh": 0.15,       # Strong positive autocorr = trending
    "trend_mean_revert_thresh": -0.10,   # Negative autocorr = mean-reverting

    # Half-life ranges by regime
    # Format: (min_half_life, max_half_life) in bars
    "half_life_default": (60, 960),      # Normal: balanced range
    "half_life_high_vol": (120, 1200),   # High vol: slower reversion, larger HL
    "half_life_low_vol": (40, 720),      # Low vol: faster reversion possible
    "half_life_trending": (40, 480),     # Trending: shorter HL (faster adaptation)
    "half_life_mean_revert": (80, 1440), # Mean-reverting: can use longer HL
}


def compute_btc_volatility_percentile(
    btc_returns: np.ndarray,
    lookback: int,
    vol_window: int = 24,
) -> Tuple[float, float]:
    """
    Compute BTC realized volatility and its historical percentile.

    Args:
        btc_returns: Array of BTC returns (can be longer than lookback for percentile calc)
        lookback: Bars to use for percentile calculation
        vol_window: Rolling window for volatility calculation

    Returns:
        (percentile, annualized_vol)
    """
    if len(btc_returns) < vol_window + 1:
        return 0.5, 0.0

    # Compute rolling volatility
    returns = pd.Series(btc_returns)
    rolling_vol = returns.rolling(vol_window).std()

    # Get historical distribution (use available data up to lookback)
    use_len = min(len(rolling_vol), lookback)
    vol_history = rolling_vol.iloc[-use_len:].dropna()

    if len(vol_history) < 2:
        return 0.5, 0.0

    current_vol = vol_history.iloc[-1]
    percentile = (vol_history < current_vol).mean()

    # Annualize (assume 15-min bars by default: 35040 bars/year)
    # For hourly: 8760 bars/year; for 1-min: 525600 bars/year
    # Use a generic factor that gets adjusted by BAR_FREQ
    bars_per_year = 35040  # 15-min default
    annualized_vol = current_vol * np.sqrt(bars_per_year)

    return float(percentile), float(annualized_vol)


def compute_market_autocorrelation(
    returns_matrix: np.ndarray,
    lookback: int,
) -> float:
    """
    Compute market-wide lag-1 autocorrelation.

    Uses cross-sectional average of per-asset autocorrelations to get
    a robust measure of market-wide mean-reversion tendency.

    Args:
        returns_matrix: (T, N) array of returns for N assets
        lookback: Bars to use for autocorrelation calculation

    Returns:
        Average lag-1 autocorrelation across assets
    """
    if returns_matrix.shape[0] < lookback + 2:
        return 0.0

    # Use recent lookback period
    recent = returns_matrix[-lookback:, :]

    # Compute lag-1 autocorrelation for each asset
    autocorrs = []
    for j in range(recent.shape[1]):
        col = recent[:, j]
        # Skip assets with too many NaNs
        valid = ~np.isnan(col)
        if valid.sum() < 20:
            continue
        col_clean = col[valid]
        if len(col_clean) < 20:
            continue

        # Lag-1 autocorrelation
        lag0 = col_clean[:-1]
        lag1 = col_clean[1:]

        if np.std(lag0) < 1e-10 or np.std(lag1) < 1e-10:
            continue

        corr = np.corrcoef(lag0, lag1)[0, 1]
        if np.isfinite(corr):
            autocorrs.append(corr)

    if len(autocorrs) == 0:
        return 0.0

    return float(np.median(autocorrs))


def classify_volatility_regime(
    percentile: float,
    low_thresh: float = 0.25,
    high_thresh: float = 0.75,
) -> VolatilityRegime:
    """Classify volatility regime from percentile."""
    if percentile < low_thresh:
        return VolatilityRegime.LOW
    elif percentile > high_thresh:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.NORMAL


def classify_trend_regime(
    autocorr: float,
    trending_thresh: float = 0.15,
    mean_revert_thresh: float = -0.10,
) -> TrendRegime:
    """Classify trend regime from autocorrelation."""
    if autocorr > trending_thresh:
        return TrendRegime.TRENDING
    elif autocorr < mean_revert_thresh:
        return TrendRegime.MEAN_REVERTING
    else:
        return TrendRegime.NEUTRAL


def get_optimal_half_life_range(
    vol_regime: VolatilityRegime,
    trend_regime: TrendRegime,
    config: Optional[dict] = None,
) -> Tuple[int, int]:
    """
    Get recommended half-life range based on current regime.

    Priority:
    1. Trend regime takes precedence (it directly impacts mean-reversion)
    2. Volatility regime adjusts within trend-based range

    Args:
        vol_regime: Current volatility regime
        trend_regime: Current trend regime
        config: Optional config dict (uses DEFAULT_REGIME_CONFIG if None)

    Returns:
        (min_half_life, max_half_life) in bars
    """
    cfg = config or DEFAULT_REGIME_CONFIG

    # Start with trend-based range (primary factor)
    if trend_regime == TrendRegime.TRENDING:
        base_range = cfg.get("half_life_trending", (40, 480))
    elif trend_regime == TrendRegime.MEAN_REVERTING:
        base_range = cfg.get("half_life_mean_revert", (80, 1440))
    else:
        # Neutral trend: use volatility-based range
        if vol_regime == VolatilityRegime.HIGH:
            base_range = cfg.get("half_life_high_vol", (120, 1200))
        elif vol_regime == VolatilityRegime.LOW:
            base_range = cfg.get("half_life_low_vol", (40, 720))
        else:
            base_range = cfg.get("half_life_default", (60, 960))

    return base_range


def detect_market_regime(
    price_matrix: np.ndarray,
    btc_index: int = 0,
    lookback_bars: int = 336,
    config: Optional[dict] = None,
) -> MarketRegime:
    """
    Detect current market regime from price data.

    Args:
        price_matrix: (T, N) array of prices for N assets
        btc_index: Column index for BTC (used for volatility regime)
        lookback_bars: Bars to use for regime detection
        config: Optional config dict with regime thresholds

    Returns:
        MarketRegime with current state and recommendations
    """
    cfg = config or DEFAULT_REGIME_CONFIG

    # Default values if insufficient data
    if price_matrix.shape[0] < 50:
        logger.warning("Insufficient data for regime detection, using defaults")
        return MarketRegime(
            volatility=VolatilityRegime.NORMAL,
            trend=TrendRegime.NEUTRAL,
            btc_vol_percentile=0.5,
            btc_vol_annualized=0.0,
            market_autocorr=0.0,
            recommended_min_half_life=60,
            recommended_max_half_life=960,
        )

    # Compute returns
    returns_matrix = np.diff(np.log(price_matrix + 1e-10), axis=0)

    # 1. Volatility regime from BTC
    btc_idx = min(btc_index, price_matrix.shape[1] - 1)
    btc_returns = returns_matrix[:, btc_idx]
    vol_pctl, vol_ann = compute_btc_volatility_percentile(
        btc_returns,
        lookback=lookback_bars,
        vol_window=24,
    )

    vol_regime = classify_volatility_regime(
        vol_pctl,
        low_thresh=cfg.get("vol_low_percentile", 0.25),
        high_thresh=cfg.get("vol_high_percentile", 0.75),
    )

    # 2. Trend regime from market-wide autocorrelation
    market_autocorr = compute_market_autocorrelation(
        returns_matrix,
        lookback=lookback_bars,
    )

    trend_regime = classify_trend_regime(
        market_autocorr,
        trending_thresh=cfg.get("trend_trending_thresh", 0.15),
        mean_revert_thresh=cfg.get("trend_mean_revert_thresh", -0.10),
    )

    # 3. Get recommended half-life range
    min_hl, max_hl = get_optimal_half_life_range(vol_regime, trend_regime, cfg)

    regime = MarketRegime(
        volatility=vol_regime,
        trend=trend_regime,
        btc_vol_percentile=vol_pctl,
        btc_vol_annualized=vol_ann,
        market_autocorr=market_autocorr,
        recommended_min_half_life=min_hl,
        recommended_max_half_life=max_hl,
    )

    logger.info(f"Detected: {regime}")
    return regime


def detect_regime_from_dataframe(
    df: pd.DataFrame,
    btc_column: str = "BTC",
    lookback_bars: int = 336,
    config: Optional[dict] = None,
) -> MarketRegime:
    """
    Convenience function to detect regime from a DataFrame of prices.

    Args:
        df: DataFrame with columns as symbols and rows as timestamps
        btc_column: Column name for BTC
        lookback_bars: Bars to use for regime detection
        config: Optional config dict

    Returns:
        MarketRegime with current state
    """
    # Convert to numpy, get BTC index
    price_matrix = df.values

    if btc_column in df.columns:
        btc_index = list(df.columns).index(btc_column)
    else:
        # Try to find a column containing "BTC"
        btc_cols = [c for c in df.columns if "BTC" in str(c).upper()]
        btc_index = list(df.columns).index(btc_cols[0]) if btc_cols else 0

    return detect_market_regime(
        price_matrix,
        btc_index=btc_index,
        lookback_bars=lookback_bars,
        config=config,
    )


if __name__ == "__main__":
    # Test the regime detection
    np.random.seed(42)

    # Simulate price data: 500 bars, 10 assets
    n_bars, n_assets = 500, 10
    base_prices = np.linspace(100, 110, n_bars).reshape(-1, 1)
    noise = np.random.randn(n_bars, n_assets) * 0.5

    # BTC in column 0 with some trending behavior
    btc_trend = np.cumsum(np.random.randn(n_bars) * 0.3)
    prices = base_prices + noise
    prices[:, 0] = 100 + btc_trend

    print("Testing market regime detection...")
    regime = detect_market_regime(prices, btc_index=0, lookback_bars=200)
    print(f"Detected regime: {regime}")

    # Test with mean-reverting market
    print("\nTesting with mean-reverting synthetic data...")
    mean_revert_noise = np.zeros((n_bars, n_assets))
    for j in range(n_assets):
        for i in range(1, n_bars):
            mean_revert_noise[i, j] = -0.3 * mean_revert_noise[i-1, j] + np.random.randn() * 0.5
    prices_mr = base_prices + mean_revert_noise
    prices_mr[:, 0] = 100 + np.cumsum(mean_revert_noise[:, 0]) * 0.1

    regime_mr = detect_market_regime(prices_mr, btc_index=0, lookback_bars=200)
    print(f"Mean-reverting regime: {regime_mr}")
