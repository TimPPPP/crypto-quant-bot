"""
src/features/trend_overlay.py

Momentum/trend overlay for mean-reversion signal filtering.

Purpose:
- Suppress mean-reversion entries when spread is trending (prevents counter-trend losses)
- Provide trend score to reduce confidence in counter-trend trades

Key indicators:
1. MA slope (normalized): Direction and strength of spread trend
2. Price vs MA (z-score): Position relative to moving average
3. Trend score: 1.0 = no trend (safe), 0.0 = strong trend (risky)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("TrendOverlay")


@dataclass
class TrendState:
    """
    Trend state for a single spread at a specific time.

    Attributes:
        ma_slope: Normalized MA slope (positive = trending up)
        price_vs_ma: Z-score of price relative to MA
        trend_score: 1.0 = no trend (safe), 0.0 = strong trend (risky)
        suppress_long: True if should suppress long entries (spread trending up)
        suppress_short: True if should suppress short entries (spread trending down)
    """
    ma_slope: float
    price_vs_ma: float
    trend_score: float
    suppress_long: bool
    suppress_short: bool


# Default configuration
DEFAULT_TREND_CONFIG = {
    "ma_period": 20,              # Period for moving average
    "slope_strong_thresh": 0.002, # Slope threshold for "strong trend"
    "suppress_z_thresh": 1.0,     # Price vs MA z-score for suppression
    "score_penalty_weight": 0.3,  # How much trend affects score (0.3 = 30% max penalty)
}


def compute_ma_slope(
    spread: np.ndarray,
    ma_period: int = 20,
) -> np.ndarray:
    """
    Compute normalized moving average slope.

    The slope is normalized by the rolling std to make it comparable
    across different spread scales.

    Args:
        spread: 1D array of spread values
        ma_period: Period for moving average

    Returns:
        Array of normalized slopes (same length as input, with NaN prefix)
    """
    n = len(spread)
    if n < ma_period + 1:
        return np.full(n, np.nan)

    # Convert to pandas for easy rolling
    s = pd.Series(spread)

    # Moving average
    ma = s.rolling(ma_period).mean()

    # MA change (slope)
    ma_change = ma - ma.shift(1)

    # Normalize by rolling std of spread
    rolling_std = s.rolling(ma_period).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    normalized_slope = ma_change / rolling_std

    return normalized_slope.values


def compute_price_vs_ma(
    spread: np.ndarray,
    ma_period: int = 20,
) -> np.ndarray:
    """
    Compute z-score of spread relative to its moving average.

    Args:
        spread: 1D array of spread values
        ma_period: Period for moving average

    Returns:
        Array of z-scores (same length as input)
    """
    n = len(spread)
    if n < ma_period:
        return np.full(n, np.nan)

    s = pd.Series(spread)

    ma = s.rolling(ma_period).mean()
    rolling_std = s.rolling(ma_period).std().replace(0, np.nan)

    z_score = (s - ma) / rolling_std

    return z_score.values


def compute_trend_score(
    ma_slope: np.ndarray,
    slope_strong_thresh: float = 0.002,
) -> np.ndarray:
    """
    Compute trend score: 1.0 = no trend (safe), 0.0 = strong trend (risky).

    Args:
        ma_slope: Array of normalized MA slopes
        slope_strong_thresh: Slope magnitude at which trend_score = 0

    Returns:
        Array of trend scores [0, 1]
    """
    # Linear decay from 1.0 at slope=0 to 0.0 at slope=±slope_strong_thresh
    abs_slope = np.abs(ma_slope)
    trend_score = 1.0 - np.clip(abs_slope / slope_strong_thresh, 0, 1)

    return trend_score


def compute_suppression_masks(
    ma_slope: np.ndarray,
    price_vs_ma: np.ndarray,
    slope_strong_thresh: float = 0.002,
    z_thresh: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute suppression masks for long and short entries.

    Suppression logic:
    - Suppress LONG if spread trending UP (positive slope) AND price above MA
      (Why: Spread is in uptrend, going long expects it to fall = counter-trend)

    - Suppress SHORT if spread trending DOWN (negative slope) AND price below MA
      (Why: Spread is in downtrend, going short expects it to rise = counter-trend)

    Args:
        ma_slope: Normalized MA slopes
        price_vs_ma: Z-scores of price vs MA
        slope_strong_thresh: Slope threshold for suppression
        z_thresh: Price vs MA z-score threshold

    Returns:
        (suppress_long, suppress_short) boolean arrays
    """
    # Strong uptrend: positive slope above threshold AND price above MA
    suppress_long = (ma_slope > slope_strong_thresh) & (price_vs_ma > z_thresh)

    # Strong downtrend: negative slope below threshold AND price below MA
    suppress_short = (ma_slope < -slope_strong_thresh) & (price_vs_ma < -z_thresh)

    return suppress_long.astype(bool), suppress_short.astype(bool)


def compute_spread_trend_state(
    spread: np.ndarray,
    ma_period: int = 20,
    slope_strong_thresh: float = 0.002,
    z_thresh: float = 1.0,
) -> pd.DataFrame:
    """
    Compute complete trend state for a spread time series.

    Args:
        spread: 1D array of spread values (T,)
        ma_period: Moving average period
        slope_strong_thresh: Slope threshold for "strong trend"
        z_thresh: Z-score threshold for suppression

    Returns:
        DataFrame with columns:
        - ma_slope: Normalized MA slope
        - price_vs_ma: Z-score relative to MA
        - trend_score: 1.0 = no trend, 0.0 = strong trend
        - suppress_long: Boolean mask
        - suppress_short: Boolean mask
    """
    ma_slope = compute_ma_slope(spread, ma_period)
    price_vs_ma = compute_price_vs_ma(spread, ma_period)
    trend_score = compute_trend_score(ma_slope, slope_strong_thresh)
    suppress_long, suppress_short = compute_suppression_masks(
        ma_slope, price_vs_ma, slope_strong_thresh, z_thresh
    )

    return pd.DataFrame({
        "ma_slope": ma_slope,
        "price_vs_ma": price_vs_ma,
        "trend_score": trend_score,
        "suppress_long": suppress_long,
        "suppress_short": suppress_short,
    })


def compute_trend_overlay_matrix(
    spreads_matrix: np.ndarray,
    ma_period: int = 20,
    slope_strong_thresh: float = 0.002,
    z_thresh: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute trend overlay for multiple spreads (vectorized).

    Args:
        spreads_matrix: (T, P) array of spread values for P pairs
        ma_period: Moving average period
        slope_strong_thresh: Slope threshold for trend detection
        z_thresh: Z-score threshold for suppression

    Returns:
        Tuple of:
        - trend_scores: (T, P) array of trend scores [0, 1]
        - suppress_long: (T, P) boolean array
        - suppress_short: (T, P) boolean array
    """
    T, P = spreads_matrix.shape

    trend_scores = np.full((T, P), 1.0)      # Default: no trend
    suppress_long = np.zeros((T, P), dtype=bool)
    suppress_short = np.zeros((T, P), dtype=bool)

    for j in range(P):
        spread = spreads_matrix[:, j]

        # Skip if all NaN
        if np.all(np.isnan(spread)):
            continue

        ma_slope = compute_ma_slope(spread, ma_period)
        price_vs_ma = compute_price_vs_ma(spread, ma_period)
        ts = compute_trend_score(ma_slope, slope_strong_thresh)
        sl, ss = compute_suppression_masks(
            ma_slope, price_vs_ma, slope_strong_thresh, z_thresh
        )

        # Handle NaN in trend score
        ts = np.nan_to_num(ts, nan=1.0)

        trend_scores[:, j] = ts
        suppress_long[:, j] = sl
        suppress_short[:, j] = ss

    return trend_scores, suppress_long, suppress_short


def apply_trend_overlay_to_entries(
    entries_long: np.ndarray,
    entries_short: np.ndarray,
    suppress_long: np.ndarray,
    suppress_short: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply trend suppression to entry signals.

    Args:
        entries_long: (T, P) boolean array of long entry signals
        entries_short: (T, P) boolean array of short entry signals
        suppress_long: (T, P) boolean suppression mask for longs
        suppress_short: (T, P) boolean suppression mask for shorts

    Returns:
        (filtered_long, filtered_short) entry arrays with counter-trend signals removed
    """
    filtered_long = entries_long & ~suppress_long
    filtered_short = entries_short & ~suppress_short

    return filtered_long, filtered_short


def apply_trend_score_penalty(
    signal_scores: np.ndarray,
    trend_scores: np.ndarray,
    penalty_weight: float = 0.3,
) -> np.ndarray:
    """
    Apply trend-based penalty to signal scores.

    Formula: adjusted_score = score * (1 - penalty_weight + penalty_weight * trend_score)
    - trend_score = 1.0 (no trend): no penalty
    - trend_score = 0.0 (strong trend): penalty_weight reduction

    Args:
        signal_scores: (T, P) array of signal confidence scores [0, 1]
        trend_scores: (T, P) array of trend scores [0, 1]
        penalty_weight: Maximum penalty (0.3 = 30% max reduction)

    Returns:
        Adjusted signal scores
    """
    # Multiplier ranges from (1 - penalty_weight) to 1.0
    # When trend_score = 0: multiplier = 1 - penalty_weight
    # When trend_score = 1: multiplier = 1.0
    multiplier = (1.0 - penalty_weight) + penalty_weight * trend_scores

    return signal_scores * multiplier


def get_trend_state_at_bar(
    spread: np.ndarray,
    bar_idx: int,
    ma_period: int = 20,
    slope_strong_thresh: float = 0.002,
    z_thresh: float = 1.0,
) -> TrendState:
    """
    Get trend state for a single bar (useful for real-time).

    Args:
        spread: Full spread history up to current bar
        bar_idx: Current bar index
        ma_period: Moving average period
        slope_strong_thresh: Slope threshold
        z_thresh: Z-score threshold

    Returns:
        TrendState for the current bar
    """
    if bar_idx < ma_period:
        return TrendState(
            ma_slope=0.0,
            price_vs_ma=0.0,
            trend_score=1.0,
            suppress_long=False,
            suppress_short=False,
        )

    # Use data up to current bar
    spread_history = spread[:bar_idx + 1]

    ma_slope = compute_ma_slope(spread_history, ma_period)[-1]
    price_vs_ma = compute_price_vs_ma(spread_history, ma_period)[-1]

    if np.isnan(ma_slope):
        ma_slope = 0.0
    if np.isnan(price_vs_ma):
        price_vs_ma = 0.0

    trend_score = float(compute_trend_score(np.array([ma_slope]), slope_strong_thresh)[0])

    suppress_long = (ma_slope > slope_strong_thresh) and (price_vs_ma > z_thresh)
    suppress_short = (ma_slope < -slope_strong_thresh) and (price_vs_ma < -z_thresh)

    return TrendState(
        ma_slope=float(ma_slope),
        price_vs_ma=float(price_vs_ma),
        trend_score=trend_score,
        suppress_long=suppress_long,
        suppress_short=suppress_short,
    )


if __name__ == "__main__":
    # Test the trend overlay
    np.random.seed(42)

    # Generate synthetic spread with trend
    n = 200
    t = np.arange(n)

    # Trending spread (upward then downward)
    trend = np.concatenate([
        np.linspace(0, 0.1, 80),
        np.linspace(0.1, -0.05, 70),
        np.linspace(-0.05, 0.02, 50),
    ])
    noise = np.random.randn(n) * 0.01
    spread = trend + noise

    print("Testing trend overlay computation...")
    df = compute_spread_trend_state(spread)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nSample at bar 50 (uptrend):")
    print(df.iloc[50])

    print(f"\nSample at bar 120 (downtrend):")
    print(df.iloc[120])

    # Test vectorized version
    spreads_matrix = np.column_stack([spread, spread * -0.5, spread + 0.1])
    trend_scores, supp_long, supp_short = compute_trend_overlay_matrix(spreads_matrix)

    print(f"\nVectorized output shapes:")
    print(f"  trend_scores: {trend_scores.shape}")
    print(f"  suppress_long: {supp_long.shape}")

    # Test signal penalty
    signal_scores = np.full_like(trend_scores, 0.8)
    adjusted = apply_trend_score_penalty(signal_scores, trend_scores, penalty_weight=0.3)
    print(f"\nSignal score penalty range: {adjusted.min():.3f} to {adjusted.max():.3f}")
