# src/backtest/position_sizing.py
"""
Advanced Position Sizing for Pairs Trading

This module implements sophisticated position sizing that goes beyond simple
fixed-size or linear conviction scaling. It combines three key components:

1. NON-LINEAR CONVICTION SIZING
   - Converts signal scores to position sizes using non-linear functions
   - Options: power, sigmoid, kelly, or stepped sizing
   - Allows aggressive sizing for high-conviction signals

2. CORRELATION-BASED ADJUSTMENT
   - Reduces position size when correlated with existing positions
   - Prevents concentration risk from similar exposures
   - Uses rolling return correlations between pairs

3. VOLATILITY TARGETING
   - Scales positions to target a specific volatility level
   - Automatically reduces size in high-vol regimes
   - Maintains consistent risk across market conditions

Position Size Formula:
    final_size = base_size * conviction_mult * correlation_adj * vol_adj

Where:
    - base_size: Capital allocation per pair
    - conviction_mult: Non-linear function of signal score
    - correlation_adj: Penalty for correlated positions (0.5 to 1.0)
    - vol_adj: target_vol / realized_vol (capped)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("position_sizing")


class ConvictionMethod(Enum):
    """Methods for converting signal score to conviction multiplier."""
    LINEAR = "linear"           # size = min + (max - min) * score
    POWER = "power"             # size = min + (max - min) * score^power
    EXPONENTIAL = "exponential" # size using exponential curve (steeper for low scores)
    SIGMOID = "sigmoid"         # size = min + (max - min) * sigmoid(k * (score - mid))
    KELLY = "kelly"             # size based on Kelly criterion
    STEPPED = "stepped"         # discrete steps based on score thresholds


@dataclass
class PositionSizingConfig:
    """Configuration for advanced position sizing."""

    # Base sizing
    base_capital_per_pair: float = 1.0

    # === Non-linear Conviction Sizing ===
    conviction_method: str = "power"  # linear, power, sigmoid, kelly, stepped
    conviction_min: float = 0.3       # Minimum size multiplier
    conviction_max: float = 1.2       # Maximum size multiplier (can exceed 1.0 for high conviction)

    # Power method parameters
    conviction_power: float = 2.0     # >1 = conservative (penalize low scores), <1 = aggressive

    # Exponential method parameters (steeper than power for low scores)
    conviction_steepness: float = 3.0  # Controls steepness of exponential curve

    # Threshold-based penalty (applies extra reduction for very low scores)
    low_score_threshold: float = 0.5   # Below this, apply extra penalty
    low_score_harsh_penalty: bool = True  # If True, score < threshold gets harsh penalty

    # Sigmoid method parameters
    sigmoid_k: float = 10.0           # Steepness of sigmoid curve
    sigmoid_midpoint: float = 0.65    # Score at which size = (min + max) / 2

    # Kelly method parameters
    kelly_fraction: float = 0.25      # Fraction of full Kelly to use (0.25 = quarter Kelly)
    kelly_win_rate_floor: float = 0.45  # Minimum assumed win rate
    kelly_payoff_ratio: float = 1.5   # Average win / average loss

    # Stepped method parameters
    step_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.65, 0.8, 0.9])
    step_sizes: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.75, 1.0, 1.2])

    # === Correlation-Based Adjustment ===
    enable_correlation_adjustment: bool = True
    correlation_lookback_bars: int = 96   # Bars for rolling correlation (4 days at hourly)
    correlation_penalty_threshold: float = 0.5  # Start penalizing above this correlation
    correlation_max_penalty: float = 0.5  # Maximum reduction (1 - this = minimum multiplier)
    max_correlated_positions: int = 3     # Max positions with correlation > threshold

    # === Volatility Targeting ===
    enable_volatility_targeting: bool = True
    target_annual_volatility: float = 0.15  # 15% annual vol target
    volatility_lookback_bars: int = 48      # Bars for realized vol calculation
    vol_adjustment_min: float = 0.5         # Minimum vol adjustment multiplier
    vol_adjustment_max: float = 2.0         # Maximum vol adjustment multiplier
    vol_floor: float = 0.001                # Minimum volatility to avoid division issues

    # === Portfolio-Level Constraints ===
    max_total_exposure: float = 1.0         # Maximum sum of all position sizes
    max_single_position: float = 0.25       # Maximum single position as fraction of portfolio
    min_position_size: float = 0.1          # Minimum position size (floor)

    # === Regime-Aware Sizing ===
    enable_regime_sizing: bool = True
    regime_high_vol_size_mult: float = 0.7   # 30% reduction in high vol regimes
    regime_low_vol_size_mult: float = 1.2    # 20% increase in low vol regimes
    regime_trending_size_mult: float = 0.8   # 20% reduction when trending

    # === Performance-Based Sizing ===
    enable_performance_sizing: bool = False  # Start disabled
    performance_lookback_trades: int = 20
    performance_size_min: float = 0.5
    performance_size_max: float = 1.5

    # === Drawdown-Based Sizing ===
    enable_drawdown_sizing: bool = False     # Start disabled
    drawdown_threshold: float = 0.03         # Start reducing at 3% drawdown
    drawdown_max_reduction: float = 0.5      # Reduce to 50% at max drawdown
    drawdown_max_level: float = 0.10         # Consider 10% as "max" drawdown


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    final_size: float
    conviction_mult: float
    correlation_adj: float
    vol_adj: float
    diagnostics: Dict[str, float] = field(default_factory=dict)


class PositionSizer:
    """
    Advanced position sizing engine.

    Combines conviction scoring, correlation adjustment, and volatility targeting
    to produce optimal position sizes for each trade.
    """

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()

        # State tracking for correlation calculation
        self._return_history: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._realized_vol: Dict[str, float] = {}

    def compute_conviction_multiplier(
        self,
        signal_score: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """
        Convert signal score to conviction multiplier using configured method.

        Parameters
        ----------
        signal_score : float
            Signal quality score (0-1)
        win_rate : float, optional
            Historical win rate for Kelly calculation
        avg_win : float, optional
            Average winning trade return for Kelly
        avg_loss : float, optional
            Average losing trade return (positive value) for Kelly

        Returns
        -------
        float : Conviction multiplier (typically 0.3 to 1.2)
        """
        cfg = self.config
        method = cfg.conviction_method.lower()

        # Normalize score to [0, 1]
        score = max(0.0, min(1.0, signal_score))

        if method == "linear":
            mult = cfg.conviction_min + (cfg.conviction_max - cfg.conviction_min) * score

        elif method == "power":
            # Power function: score^power
            # power > 1: penalize low scores (conservative)
            # power < 1: boost low scores (aggressive)
            normalized = score ** cfg.conviction_power

            # Apply extra penalty for very low scores
            if cfg.low_score_harsh_penalty and score < cfg.low_score_threshold:
                # Linear reduction to near-zero at score=0
                harsh_mult = score / cfg.low_score_threshold
                normalized = normalized * harsh_mult

            mult = cfg.conviction_min + (cfg.conviction_max - cfg.conviction_min) * normalized

        elif method == "exponential":
            # Exponential: steeper penalty than power for low scores
            # Score 0.65 → ~0.55, Score 0.30 → ~0.11
            normalized = np.exp((score - 1.0) * cfg.conviction_steepness)
            mult = cfg.conviction_min + (cfg.conviction_max - cfg.conviction_min) * normalized

        elif method == "sigmoid":
            # Sigmoid: smooth S-curve transition
            # At midpoint, mult = (min + max) / 2
            x = cfg.sigmoid_k * (score - cfg.sigmoid_midpoint)
            sigmoid = 1.0 / (1.0 + np.exp(-x))
            mult = cfg.conviction_min + (cfg.conviction_max - cfg.conviction_min) * sigmoid

        elif method == "kelly":
            # Kelly criterion: f* = (p * b - q) / b
            # where p = win prob, q = 1-p, b = win/loss ratio
            if win_rate is not None and avg_win is not None and avg_loss is not None:
                p = max(cfg.kelly_win_rate_floor, win_rate)
                b = avg_win / max(0.001, avg_loss)
            else:
                # Use score as proxy for win probability
                p = max(cfg.kelly_win_rate_floor, 0.4 + 0.3 * score)  # Map [0,1] to [0.4, 0.7]
                b = cfg.kelly_payoff_ratio

            q = 1.0 - p
            kelly_fraction = (p * b - q) / b
            kelly_fraction = max(0.0, kelly_fraction)  # Can't be negative

            # Apply fractional Kelly and scale to our range
            adjusted_kelly = kelly_fraction * cfg.kelly_fraction
            mult = cfg.conviction_min + (cfg.conviction_max - cfg.conviction_min) * min(1.0, adjusted_kelly * 2)

        elif method == "stepped":
            # Discrete steps based on thresholds
            thresholds = cfg.step_thresholds
            sizes = cfg.step_sizes

            mult = sizes[0]  # Default to lowest
            for i, thresh in enumerate(thresholds):
                if score >= thresh:
                    mult = sizes[i + 1]
                else:
                    break

        else:
            logger.warning(f"Unknown conviction method '{method}', using linear")
            mult = cfg.conviction_min + (cfg.conviction_max - cfg.conviction_min) * score

        return float(mult)

    def compute_correlation_adjustment(
        self,
        pair: str,
        current_positions: List[str],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute position size adjustment based on correlation with existing positions.

        Reduces size when new position is highly correlated with existing ones
        to maintain portfolio diversification.

        Parameters
        ----------
        pair : str
            Pair being sized
        current_positions : list of str
            Currently open positions
        correlation_matrix : pd.DataFrame, optional
            Pre-computed correlation matrix. If None, uses internal state.

        Returns
        -------
        adjustment : float
            Multiplier (0.5 to 1.0)
        diagnostics : dict
            Breakdown of correlation impacts
        """
        cfg = self.config

        if not cfg.enable_correlation_adjustment:
            return 1.0, {}

        if not current_positions:
            return 1.0, {"n_positions": 0}

        corr_matrix = correlation_matrix if correlation_matrix is not None else self._correlation_matrix

        if corr_matrix is None or pair not in corr_matrix.columns:
            return 1.0, {"reason": "no_correlation_data"}

        # Find correlations with current positions
        correlations = {}
        high_corr_count = 0
        max_corr = 0.0

        for pos in current_positions:
            if pos in corr_matrix.columns and pos != pair:
                corr = abs(corr_matrix.loc[pair, pos])
                correlations[pos] = corr
                max_corr = max(max_corr, corr)
                if corr > cfg.correlation_penalty_threshold:
                    high_corr_count += 1

        # Compute adjustment
        # Guard against division by zero when threshold >= 1.0
        # Cleaner penalty logic - compute penalties separately, then combine
        base_penalty = 0.0
        count_penalty = 0.0

        # Method 1: Penalize based on maximum correlation
        if max_corr > cfg.correlation_penalty_threshold:
            # Guard against division by zero
            if cfg.correlation_penalty_threshold >= 0.99:
                # Near 1.0, apply maximum penalty
                base_penalty = cfg.correlation_max_penalty
            else:
                # Linear penalty above threshold
                excess = (max_corr - cfg.correlation_penalty_threshold) / (1.0 - cfg.correlation_penalty_threshold)
                base_penalty = excess * cfg.correlation_max_penalty

        # Method 2: Additional penalty if too many correlated positions
        if high_corr_count >= cfg.max_correlated_positions:
            count_penalty = 0.2  # Additional 20% reduction

        # Combine penalties (capped at max_penalty)
        total_penalty = min(base_penalty + count_penalty, cfg.correlation_max_penalty)
        adjustment = 1.0 - total_penalty

        diagnostics = {
            "max_correlation": max_corr,
            "high_corr_count": high_corr_count,
            "n_positions": len(current_positions),
        }

        return float(adjustment), diagnostics

    def compute_volatility_adjustment(
        self,
        pair: str,
        realized_vol: Optional[float] = None,
        spread_returns: Optional[pd.Series] = None,
        bars_per_year: float = 8760.0,  # Hourly bars
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute position size adjustment to target specific volatility level.

        Scales position inversely with realized volatility to maintain
        consistent risk across different market conditions.

        Parameters
        ----------
        pair : str
            Pair being sized
        realized_vol : float, optional
            Pre-computed realized volatility
        spread_returns : pd.Series, optional
            Historical spread returns for calculation
        bars_per_year : float
            Number of bars per year for annualization

        Returns
        -------
        adjustment : float
            Multiplier (typically 0.5 to 2.0)
        diagnostics : dict
            Volatility calculation details
        """
        cfg = self.config

        if not cfg.enable_volatility_targeting:
            return 1.0, {}

        # Get realized volatility
        if realized_vol is not None:
            vol = realized_vol
        elif spread_returns is not None and len(spread_returns) >= cfg.volatility_lookback_bars:
            # Compute from recent history
            recent = spread_returns.tail(cfg.volatility_lookback_bars)
            vol = recent.std() * np.sqrt(bars_per_year)
        elif pair in self._realized_vol:
            vol = self._realized_vol[pair]
        else:
            return 1.0, {"reason": "no_volatility_data"}

        # Explicit NaN check - max(0.001, NaN) = NaN
        if not np.isfinite(vol) or vol <= 0:
            vol = cfg.vol_floor
        else:
            vol = max(cfg.vol_floor, vol)

        # Compute adjustment: target_vol / realized_vol
        # If realized_vol > target, reduce size
        # If realized_vol < target, increase size (up to cap)
        adjustment = cfg.target_annual_volatility / vol

        # Apply bounds
        adjustment = max(cfg.vol_adjustment_min, min(cfg.vol_adjustment_max, adjustment))

        diagnostics = {
            "realized_vol_annual": vol,
            "target_vol": cfg.target_annual_volatility,
            "raw_adjustment": cfg.target_annual_volatility / vol,
        }

        return float(adjustment), diagnostics

    def compute_position_size(
        self,
        pair: str,
        signal_score: float,
        current_positions: Optional[List[str]] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        realized_vol: Optional[float] = None,
        spread_returns: Optional[pd.Series] = None,
        current_total_exposure: float = 0.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        bars_per_year: float = 8760.0,
    ) -> PositionSizeResult:
        """
        Compute final position size combining all adjustment factors.

        Parameters
        ----------
        pair : str
            Pair being sized
        signal_score : float
            Signal quality score (0-1)
        current_positions : list of str, optional
            Currently open positions for correlation check
        correlation_matrix : pd.DataFrame, optional
            Correlation matrix for adjustment
        realized_vol : float, optional
            Realized volatility for vol targeting
        spread_returns : pd.Series, optional
            Historical returns for calculation
        current_total_exposure : float
            Current sum of position sizes
        win_rate, avg_win, avg_loss : float, optional
            Historical trade stats for Kelly sizing
        bars_per_year : float
            Bars per year for annualization

        Returns
        -------
        PositionSizeResult with final size and component breakdowns
        """
        cfg = self.config

        # 1. Conviction multiplier
        conviction_mult = self.compute_conviction_multiplier(
            signal_score=signal_score,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        # 2. Correlation adjustment
        correlation_adj, corr_diag = self.compute_correlation_adjustment(
            pair=pair,
            current_positions=current_positions or [],
            correlation_matrix=correlation_matrix,
        )

        # 3. Volatility adjustment
        vol_adj, vol_diag = self.compute_volatility_adjustment(
            pair=pair,
            realized_vol=realized_vol,
            spread_returns=spread_returns,
            bars_per_year=bars_per_year,
        )

        # Combine all factors
        raw_size = cfg.base_capital_per_pair * conviction_mult * correlation_adj * vol_adj

        # Apply portfolio constraints
        # Constraint 1: Single position cap
        final_size = min(raw_size, cfg.max_single_position)

        # Constraint 2: Total exposure cap
        remaining_capacity = cfg.max_total_exposure - current_total_exposure
        if remaining_capacity < final_size:
            final_size = max(0.0, remaining_capacity)

        diagnostics = {
            "raw_size": raw_size,
            "signal_score": signal_score,
            **corr_diag,
            **vol_diag,
        }

        return PositionSizeResult(
            final_size=final_size,
            conviction_mult=conviction_mult,
            correlation_adj=correlation_adj,
            vol_adj=vol_adj,
            diagnostics=diagnostics,
        )

    def update_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        lookback_bars: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Update internal correlation matrix from return history.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Return time series (T × pairs)
        lookback_bars : int, optional
            Lookback window. Uses config default if not specified.

        Returns
        -------
        pd.DataFrame : Updated correlation matrix
        """
        lookback = lookback_bars or self.config.correlation_lookback_bars

        # Use recent returns
        if len(returns_df) > lookback:
            recent = returns_df.tail(lookback)
        else:
            recent = returns_df

        # Compute correlation matrix
        self._correlation_matrix = recent.corr()
        self._return_history = returns_df

        return self._correlation_matrix

    def update_realized_volatility(
        self,
        spread_returns: pd.DataFrame,
        lookback_bars: Optional[int] = None,
        bars_per_year: float = 8760.0,
    ) -> Dict[str, float]:
        """
        Update realized volatility estimates for all pairs.

        Parameters
        ----------
        spread_returns : pd.DataFrame
            Spread returns time series (T × pairs)
        lookback_bars : int, optional
            Lookback window
        bars_per_year : float
            Annualization factor

        Returns
        -------
        dict : Mapping pair -> annualized volatility
        """
        lookback = lookback_bars or self.config.volatility_lookback_bars

        for pair in spread_returns.columns:
            series = spread_returns[pair].dropna()
            if len(series) >= lookback:
                recent = series.tail(lookback)
                vol = recent.std() * np.sqrt(bars_per_year)
                self._realized_vol[pair] = vol

        return self._realized_vol.copy()


def compute_position_sizes_vectorized(
    signal_scores: pd.DataFrame,
    spread_returns: Optional[pd.DataFrame],
    config: PositionSizingConfig,
    correlation_matrix: Optional[pd.DataFrame] = None,
    bars_per_year: float = 8760.0,
) -> pd.DataFrame:
    """
    Compute position sizes for all pairs in a vectorized manner.

    This is the high-performance version for backtesting that processes
    all pairs at once without tracking open positions (simplified).

    Parameters
    ----------
    signal_scores : pd.DataFrame
        Signal scores (T × pairs)
    spread_returns : pd.DataFrame
        Spread returns (T × pairs)
    config : PositionSizingConfig
        Sizing configuration
    correlation_matrix : pd.DataFrame, optional
        Static correlation matrix for adjustment
    bars_per_year : float
        Annualization factor

    Returns
    -------
    pd.DataFrame : Position sizes (T × pairs)
    """
    # 1. Conviction multiplier (vectorized)
    scores = signal_scores.clip(0, 1)

    method = config.conviction_method.lower()

    if method == "linear":
        conv_mult = config.conviction_min + (config.conviction_max - config.conviction_min) * scores

    elif method == "power":
        normalized = scores ** config.conviction_power
        # Apply extra penalty for very low scores
        if config.low_score_harsh_penalty:
            harsh_mask = scores < config.low_score_threshold
            harsh_mult = scores / config.low_score_threshold
            # Apply only where score < threshold
            normalized = normalized.where(~harsh_mask, normalized * harsh_mult)
        conv_mult = config.conviction_min + (config.conviction_max - config.conviction_min) * normalized

    elif method == "exponential":
        # Aggressive exponential curve: rewards top signals heavily
        # Uses sigmoid midpoint (0.65) as the baseline score
        # Score at 0.65 → min size (0.2)
        # Score at 1.0 → max size (3.0)
        # Steep exponential growth between 0.65 and 1.0
        threshold = config.sigmoid_midpoint  # 0.65 - aligns with MIN_SIGNAL_SCORE
        k = config.conviction_steepness  # 5.0 for aggressive

        # Normalize score to [0, 1] from threshold to 1.0
        # Scores below threshold get clamped to 0 (minimum size)
        score_norm = ((scores - threshold) / (1.0 - threshold)).clip(0, 1)

        # Exponential curve: exp(k*x) - 1, normalized to [0, 1]
        # At score_norm=0: exp_factor = 0
        # At score_norm=1: exp_factor = 1
        exp_factor = (np.exp(k * score_norm) - 1) / (np.exp(k) - 1)

        # Map to size range [min, max]
        # Score 0.65 → 0.2, Score 1.0 → 3.0
        conv_mult = config.conviction_min + (config.conviction_max - config.conviction_min) * exp_factor

    elif method == "sigmoid":
        x = config.sigmoid_k * (scores - config.sigmoid_midpoint)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        conv_mult = config.conviction_min + (config.conviction_max - config.conviction_min) * sigmoid

    elif method == "stepped":
        conv_mult = pd.DataFrame(
            config.step_sizes[0],
            index=scores.index,
            columns=scores.columns
        )
        for i, thresh in enumerate(config.step_thresholds):
            conv_mult = conv_mult.where(scores < thresh, config.step_sizes[i + 1])

    else:  # Default to linear
        conv_mult = config.conviction_min + (config.conviction_max - config.conviction_min) * scores

    # 2. Volatility adjustment (vectorized)
    # Use min_periods to avoid NaN for early bars, fill remaining NaN
    if config.enable_volatility_targeting and spread_returns is not None:
        lookback = config.volatility_lookback_bars
        # Use expanding window for early bars (min 10 bars or 25% of lookback)
        min_periods = max(10, lookback // 4)
        rolling_vol = spread_returns.rolling(lookback, min_periods=min_periods).std() * np.sqrt(bars_per_year)

        # Fill any remaining NaN with median of non-NaN values per column
        median_vol = rolling_vol.median()
        # Use column-wise fillna
        for col in rolling_vol.columns:
            col_median = median_vol.get(col, config.vol_floor * 100)  # Reasonable default
            if not np.isfinite(col_median) or col_median <= 0:
                col_median = config.vol_floor * 100  # ~10% annual vol
            rolling_vol[col] = rolling_vol[col].fillna(col_median)

        # Final safety: floor to vol_floor
        rolling_vol = rolling_vol.clip(lower=config.vol_floor)

        vol_adj = config.target_annual_volatility / rolling_vol
        vol_adj = vol_adj.clip(config.vol_adjustment_min, config.vol_adjustment_max)
    else:
        vol_adj = 1.0

    # 3. Correlation adjustment (simplified - per-pair static adjustment)
    # Guard against division by zero and ensure correct DataFrame broadcasting
    if config.enable_correlation_adjustment and correlation_matrix is not None:
        # Compute average absolute correlation for each pair
        avg_corr = correlation_matrix.abs().mean()

        # Guard against division by zero when threshold >= 1.0
        if config.correlation_penalty_threshold >= 0.99:
            # Near 1.0, apply maximum penalty to high-corr pairs
            penalty = (avg_corr > 0.9).astype(float) * config.correlation_max_penalty
        else:
            # Penalize highly correlated pairs
            excess = (avg_corr - config.correlation_penalty_threshold).clip(lower=0)
            penalty = excess / (1.0 - config.correlation_penalty_threshold) * config.correlation_max_penalty

        corr_adj = (1.0 - penalty).clip(lower=1.0 - config.correlation_max_penalty)

        # Proper broadcasting - create array then tile to all rows
        corr_adj_values = np.array([corr_adj.get(pair, 1.0) for pair in scores.columns])
        corr_adj_df = pd.DataFrame(
            np.tile(corr_adj_values, (len(scores), 1)),
            index=scores.index,
            columns=scores.columns
        )
    else:
        corr_adj_df = 1.0

    # Combine all factors
    position_sizes = config.base_capital_per_pair * conv_mult * vol_adj * corr_adj_df

    # Apply both lower and upper bounds
    position_sizes = position_sizes.clip(
        lower=config.min_position_size,
        upper=config.max_single_position
    )

    return position_sizes


# =============================================================================
# NEW FEATURE: Regime-Aware Position Sizing
# =============================================================================

def compute_regime_adjustment(
    volatility_regime: str,
    trend_regime: str,
    config: PositionSizingConfig,
) -> float:
    """
    Compute position size adjustment based on market regime.

    Parameters
    ----------
    volatility_regime : str
        One of "high", "normal", "low"
    trend_regime : str
        One of "trending", "mean_reverting", "neutral"
    config : PositionSizingConfig
        Sizing configuration

    Returns
    -------
    float : Regime adjustment multiplier (typically 0.5 to 1.5)
    """
    if not config.enable_regime_sizing:
        return 1.0

    # Volatility regime adjustment
    vol_regime = volatility_regime.lower() if volatility_regime else "normal"
    if vol_regime == "high":
        vol_mult = config.regime_high_vol_size_mult  # Reduce in high vol
    elif vol_regime == "low":
        vol_mult = config.regime_low_vol_size_mult   # Increase in low vol
    else:
        vol_mult = 1.0

    # Trend regime adjustment (for mean-reversion strategies)
    trend = trend_regime.lower() if trend_regime else "neutral"
    if trend == "trending":
        trend_mult = config.regime_trending_size_mult  # Reduce when trending
    else:
        trend_mult = 1.0

    return vol_mult * trend_mult


def apply_regime_adjustment_vectorized(
    position_sizes: pd.DataFrame,
    volatility_regimes: pd.Series,
    trend_regimes: pd.Series,
    config: PositionSizingConfig,
) -> pd.DataFrame:
    """
    Apply regime-based adjustment to position sizes (vectorized).

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Base position sizes (T × pairs)
    volatility_regimes : pd.Series
        Volatility regime at each timestamp ("high", "normal", "low")
    trend_regimes : pd.Series
        Trend regime at each timestamp ("trending", "mean_reverting", "neutral")
    config : PositionSizingConfig
        Sizing configuration

    Returns
    -------
    pd.DataFrame : Adjusted position sizes
    """
    if not config.enable_regime_sizing:
        return position_sizes

    # Create adjustment series
    vol_adj = pd.Series(1.0, index=position_sizes.index)
    trend_adj = pd.Series(1.0, index=position_sizes.index)

    # Volatility adjustment
    if volatility_regimes is not None:
        vol_adj = vol_adj.where(
            volatility_regimes != "high",
            config.regime_high_vol_size_mult
        )
        vol_adj = vol_adj.where(
            volatility_regimes != "low",
            config.regime_low_vol_size_mult
        )

    # Trend adjustment
    if trend_regimes is not None:
        trend_adj = trend_adj.where(
            trend_regimes != "trending",
            config.regime_trending_size_mult
        )

    # Combine and apply
    regime_mult = vol_adj * trend_adj
    return position_sizes.multiply(regime_mult, axis=0)


# =============================================================================
# NEW FEATURE: Performance-Based Position Sizing
# =============================================================================

def compute_performance_adjustment(
    recent_returns: pd.Series,
    config: PositionSizingConfig,
) -> float:
    """
    Compute position size adjustment based on recent trading performance.

    Reduces size for pairs/strategies that have been losing recently,
    increases for consistent winners.

    Parameters
    ----------
    recent_returns : pd.Series
        Recent trade returns (per-trade PnL)
    config : PositionSizingConfig
        Sizing configuration

    Returns
    -------
    float : Performance adjustment multiplier
    """
    if not config.enable_performance_sizing:
        return 1.0

    lookback = config.performance_lookback_trades

    if recent_returns is None or len(recent_returns) < lookback:
        return 1.0

    recent = recent_returns.tail(lookback)

    # Sharpe-like metric
    std = recent.std()
    if std > 0 and np.isfinite(std):
        recent_sharpe = recent.mean() / std
    else:
        recent_sharpe = 0.0

    # Map to multiplier: Sharpe -2 → min, Sharpe +2 → max
    # Linear mapping centered at 0
    mult = 1.0 + recent_sharpe * 0.25

    return float(np.clip(mult, config.performance_size_min, config.performance_size_max))


def apply_performance_adjustment_vectorized(
    position_sizes: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    config: PositionSizingConfig,
) -> pd.DataFrame:
    """
    Apply performance-based adjustment to position sizes (vectorized).

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Base position sizes (T × pairs)
    cumulative_returns : pd.DataFrame
        Cumulative returns per pair (T × pairs)
    config : PositionSizingConfig
        Sizing configuration

    Returns
    -------
    pd.DataFrame : Adjusted position sizes
    """
    if not config.enable_performance_sizing:
        return position_sizes

    if cumulative_returns is None or cumulative_returns.empty:
        return position_sizes

    lookback = config.performance_lookback_trades
    min_mult = config.performance_size_min
    max_mult = config.performance_size_max

    # Compute period returns from cumulative
    period_returns = cumulative_returns.diff()

    # Rolling Sharpe-like metric
    rolling_mean = period_returns.rolling(lookback, min_periods=lookback // 2).mean()
    rolling_std = period_returns.rolling(lookback, min_periods=lookback // 2).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    rolling_sharpe = rolling_mean / rolling_std
    rolling_sharpe = rolling_sharpe.fillna(0)

    # Map to multiplier
    perf_mult = 1.0 + rolling_sharpe * 0.25
    perf_mult = perf_mult.clip(min_mult, max_mult)

    return position_sizes * perf_mult


# =============================================================================
# NEW FEATURE: Drawdown-Based Position Sizing
# =============================================================================

def compute_drawdown_adjustment(
    equity_curve: pd.Series,
    config: PositionSizingConfig,
) -> float:
    """
    Compute position size adjustment based on current drawdown.

    Reduces exposure during drawdowns to prevent further losses.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity curve
    config : PositionSizingConfig
        Sizing configuration

    Returns
    -------
    float : Drawdown adjustment multiplier (0.5 to 1.0)
    """
    if not config.enable_drawdown_sizing:
        return 1.0

    if equity_curve is None or len(equity_curve) < 2:
        return 1.0

    # Compute current drawdown
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    current_dd = drawdown.iloc[-1]

    if not np.isfinite(current_dd) or current_dd < config.drawdown_threshold:
        return 1.0

    # Linear reduction from 1.0 at threshold to (1-max_reduction) at max_dd
    dd_range = config.drawdown_max_level - config.drawdown_threshold
    if dd_range <= 0:
        return 1.0

    excess_dd = (current_dd - config.drawdown_threshold) / dd_range
    reduction = min(excess_dd * config.drawdown_max_reduction, config.drawdown_max_reduction)

    return 1.0 - reduction


def apply_drawdown_adjustment_vectorized(
    position_sizes: pd.DataFrame,
    equity_curve: pd.Series,
    config: PositionSizingConfig,
) -> pd.DataFrame:
    """
    Apply drawdown-based adjustment to position sizes (vectorized).

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Base position sizes (T × pairs)
    equity_curve : pd.Series
        Portfolio equity curve
    config : PositionSizingConfig
        Sizing configuration

    Returns
    -------
    pd.DataFrame : Adjusted position sizes
    """
    if not config.enable_drawdown_sizing:
        return position_sizes

    if equity_curve is None or len(equity_curve) < 2:
        return position_sizes

    # Compute drawdown series
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    drawdown = drawdown.fillna(0)

    # Compute adjustment at each timestamp
    dd_threshold = config.drawdown_threshold
    dd_max = config.drawdown_max_level
    max_reduction = config.drawdown_max_reduction

    dd_range = dd_max - dd_threshold
    if dd_range <= 0:
        return position_sizes

    # Vectorized adjustment calculation
    excess_dd = ((drawdown - dd_threshold) / dd_range).clip(lower=0)
    reduction = (excess_dd * max_reduction).clip(upper=max_reduction)
    dd_mult = 1.0 - reduction

    # Align index and apply
    dd_mult = dd_mult.reindex(position_sizes.index).ffill().fillna(1.0)
    return position_sizes.multiply(dd_mult, axis=0)


# =============================================================================
# Combined Position Sizing with All Features
# =============================================================================

def compute_position_sizes_with_features(
    signal_scores: pd.DataFrame,
    spread_returns: Optional[pd.DataFrame],
    config: PositionSizingConfig,
    correlation_matrix: Optional[pd.DataFrame] = None,
    volatility_regimes: Optional[pd.Series] = None,
    trend_regimes: Optional[pd.Series] = None,
    cumulative_returns: Optional[pd.DataFrame] = None,
    equity_curve: Optional[pd.Series] = None,
    spread_volatilities: Optional[pd.DataFrame] = None,
    enable_risk_parity: bool = False,
    target_vol_contribution: float = 0.125,
    final_max_single: float = 0.25,
    final_max_total: float = 1.0,
    bars_per_year: float = 8760.0,
) -> pd.DataFrame:
    """
    Compute position sizes with all features enabled.

    This is the main entry point that combines:
    1. Base conviction sizing
    2. Correlation adjustment
    3. Volatility targeting
    4. Regime-aware sizing
    5. Performance-based adjustment
    6. Drawdown-based sizing
    7. Risk parity adjustment
    8. FINAL hard clamp - runs LAST, no exceptions

    Parameters
    ----------
    signal_scores : pd.DataFrame
        Signal scores (T × pairs)
    spread_returns : pd.DataFrame
        Spread returns (T × pairs)
    config : PositionSizingConfig
        Sizing configuration
    correlation_matrix : pd.DataFrame, optional
        Static correlation matrix
    volatility_regimes : pd.Series, optional
        Volatility regime at each timestamp
    trend_regimes : pd.Series, optional
        Trend regime at each timestamp
    cumulative_returns : pd.DataFrame, optional
        Cumulative returns for performance adjustment
    equity_curve : pd.Series, optional
        Portfolio equity curve for drawdown adjustment
    spread_volatilities : pd.DataFrame, optional
        Spread volatilities for risk parity (T × pairs)
    enable_risk_parity : bool
        Whether to apply risk parity sizing (default False)
    target_vol_contribution : float
        Target volatility contribution per position for risk parity
    final_max_single : float
        FINAL hard clamp: max single position (default 0.25)
    final_max_total : float
        FINAL hard clamp: max total exposure (default 1.0)
    bars_per_year : float
        Annualization factor

    Returns
    -------
    pd.DataFrame : Final position sizes (T × pairs)
    """
    # 1. Base position sizes (with conviction, correlation, vol targeting)
    position_sizes = compute_position_sizes_vectorized(
        signal_scores=signal_scores,
        spread_returns=spread_returns,
        config=config,
        correlation_matrix=correlation_matrix,
        bars_per_year=bars_per_year,
    )

    # 2. Regime adjustment
    if config.enable_regime_sizing and (volatility_regimes is not None or trend_regimes is not None):
        position_sizes = apply_regime_adjustment_vectorized(
            position_sizes=position_sizes,
            volatility_regimes=volatility_regimes,
            trend_regimes=trend_regimes,
            config=config,
        )

    # 3. Performance adjustment
    if config.enable_performance_sizing and cumulative_returns is not None:
        position_sizes = apply_performance_adjustment_vectorized(
            position_sizes=position_sizes,
            cumulative_returns=cumulative_returns,
            config=config,
        )

    # 4. Drawdown adjustment
    if config.enable_drawdown_sizing and equity_curve is not None:
        position_sizes = apply_drawdown_adjustment_vectorized(
            position_sizes=position_sizes,
            equity_curve=equity_curve,
            config=config,
        )

    # 5. Risk parity adjustment
    # Rebalance positions so each contributes equal volatility to portfolio
    if enable_risk_parity and spread_volatilities is not None:
        position_sizes = apply_risk_parity_sizing(
            position_sizes=position_sizes,
            spread_volatilities=spread_volatilities,
            target_vol_contribution=target_vol_contribution,
            blend_factor=1.0,  # Full risk parity
        )
        logger.debug("Applied risk parity sizing")

    # 6. FINAL HARD CLAMP
    # This is the LAST step - no position ever exceeds these limits
    # regardless of what any earlier stage computed
    position_sizes = apply_final_hard_clamp(
        position_sizes=position_sizes,
        max_single=final_max_single,
        max_total=final_max_total,
        min_position=config.min_position_size,
    )

    return position_sizes


# =============================================================================
# ML-BASED RISK PREDICTION INTEGRATION
# =============================================================================

def apply_risk_prediction_adjustment(
    position_sizes: pd.DataFrame,
    risk_predictions: Dict[str, pd.DataFrame],
    historical_mae_median: float = 0.01,
    historical_vol_median: float = 0.005,
    size_reduction_max: float = 0.5,
    size_increase_max: float = 0.5,
    low_risk_threshold: float = 0.3,
    high_risk_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Asymmetric risk adjustment: reward low-risk AND punish high-risk.

    This is the key to achieving higher returns:
    - Low risk (score < low_threshold): INCREASE size up to (1 + size_increase_max)
    - Neutral (between thresholds): No adjustment (1.0x)
    - High risk (score > high_threshold): DECREASE size down to (1 - size_reduction_max)

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Base position sizes (T × P)
    risk_predictions : Dict[str, pd.DataFrame]
        Predictions from RiskPredictor.predict()
    historical_mae_median : float
        Historical median MAE for normalization
    historical_vol_median : float
        Historical median forward vol for normalization
    size_reduction_max : float
        Maximum size reduction for high risk (0.5 = down to 0.5x)
    size_increase_max : float
        Maximum size increase for low risk (0.5 = up to 1.5x)
    low_risk_threshold : float
        Risk score below this = low risk = increase size
    high_risk_threshold : float
        Risk score above this = high risk = decrease size
    confidence_threshold : float
        Below this confidence, use fallback heuristics

    Returns
    -------
    pd.DataFrame
        Risk-adjusted position sizes (T × P)
    """
    # Get predictions, filling missing with defaults
    predicted_mae = risk_predictions.get("predicted_mae", pd.DataFrame(0.0, index=position_sizes.index, columns=position_sizes.columns))
    predicted_vol = risk_predictions.get("predicted_vol", pd.DataFrame(0.0, index=position_sizes.index, columns=position_sizes.columns))
    stopout_prob = risk_predictions.get("stopout_prob", pd.DataFrame(0.5, index=position_sizes.index, columns=position_sizes.columns))
    confidence = risk_predictions.get("confidence", pd.DataFrame(0.0, index=position_sizes.index, columns=position_sizes.columns))

    # Normalize predictions to 0-1 scale
    mae_norm = (predicted_mae / (3 * historical_mae_median)).clip(0, 1)
    vol_norm = (predicted_vol / (3 * historical_vol_median)).clip(0, 1)

    # Handle NaN
    mae_norm = mae_norm.fillna(0.5)
    vol_norm = vol_norm.fillna(0.5)
    stopout_prob = stopout_prob.fillna(0.5)
    confidence = confidence.fillna(0.0)

    # Weighted risk score: higher = riskier (0 to 1 scale)
    risk_score = 0.4 * mae_norm + 0.3 * vol_norm + 0.3 * stopout_prob

    # ASYMMETRIC ADJUSTMENT: reward low-risk, punish high-risk
    # Low risk zone: increase size (risk_score < low_threshold)
    # Formula: mult = 1 + (low_threshold - risk_score) / low_threshold * size_increase_max
    # At risk_score=0: mult = 1 + size_increase_max (e.g., 1.5x)
    # At risk_score=low_threshold: mult = 1.0
    low_risk_mult = 1.0 + (low_risk_threshold - risk_score) / low_risk_threshold * size_increase_max

    # High risk zone: decrease size (risk_score > high_threshold)
    # Formula: mult = 1 - (risk_score - high_threshold) / (1 - high_threshold) * size_reduction_max
    # At risk_score=high_threshold: mult = 1.0
    # At risk_score=1.0: mult = 1 - size_reduction_max (e.g., 0.5x)
    high_risk_mult = 1.0 - (risk_score - high_risk_threshold) / (1.0 - high_risk_threshold) * size_reduction_max

    # Combine based on zone
    final_mult = np.where(
        risk_score < low_risk_threshold,
        np.clip(low_risk_mult, 1.0, 1.0 + size_increase_max),
        np.where(
            risk_score > high_risk_threshold,
            np.clip(high_risk_mult, 1.0 - size_reduction_max, 1.0),
            1.0  # Neutral zone
        )
    )

    # Fallback for low confidence: use simple vol-based heuristic (conservative)
    low_conf_mask = confidence < confidence_threshold
    # Fallback is less aggressive - only reduces, doesn't increase
    fallback_mult = np.where(
        vol_norm < 0.3,
        1.0 + (0.3 - vol_norm) * 0.5,  # Slight increase for low vol
        np.where(vol_norm > 0.5, 1.0 - (vol_norm - 0.5) * 0.6, 1.0)  # Reduce for high vol
    )
    fallback_mult = np.clip(fallback_mult, 0.7, 1.3)

    # Blend based on confidence
    blend_weight = (confidence / confidence_threshold).clip(0, 1)
    final_mult = np.where(
        low_conf_mask,
        fallback_mult * (1 - blend_weight) + final_mult * blend_weight,
        final_mult,
    )

    adjusted_sizes = position_sizes * final_mult

    # Compute stats for logging
    pct_increased = 100 * (final_mult > 1.0).mean() if hasattr(final_mult, 'mean') else 0.0
    pct_decreased = 100 * (final_mult < 1.0).mean() if hasattr(final_mult, 'mean') else 0.0

    logger.info(
        "Asymmetric risk adjustment: mean_mult=%.3f, min=%.3f, max=%.3f, pct_increased=%.1f%%, pct_decreased=%.1f%%",
        np.nanmean(final_mult),
        np.nanmin(final_mult),
        np.nanmax(final_mult),
        pct_increased,
        pct_decreased,
    )

    return adjusted_sizes


# =============================================================================
# POSITION SIZING HARDENING
# Add hard clamps after all multipliers, risk parity, and
# concentration metrics to prevent over-exposure.
# =============================================================================


@dataclass(frozen=True)
class ConcentrationMetrics:
    """Portfolio concentration metrics for monitoring."""

    hhi: float  # Herfindahl-Hirschman Index (sum of squared weights)
    effective_positions: float  # 1/HHI - effective number of equal positions
    max_single_weight: float  # Largest single position weight
    total_exposure: float  # Sum of all position sizes
    n_positions: int  # Count of non-zero positions
    is_concentrated: bool  # True if HHI > threshold


def apply_final_hard_clamp(
    position_sizes: pd.DataFrame,
    max_single: float = 0.25,
    max_total: float = 1.0,
    min_position: float = 0.0,
) -> pd.DataFrame:
    """
    Apply FINAL hard clamp on position sizes - this runs AFTER all other adjustments.

    This is the last safety gate before positions are used. No position can ever
    exceed max_single, and total exposure cannot exceed max_total, regardless
    of what earlier stages computed.

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Position sizes after all other adjustments (T × pairs)
    max_single : float
        Maximum single position as fraction of portfolio (default 0.25 = 25%)
    max_total : float
        Maximum sum of all position sizes (default 1.0 = 100%)
    min_position : float
        Minimum position size (positions below this are zeroed)

    Returns
    -------
    pd.DataFrame
        Clamped position sizes
    """
    # 1. Apply minimum threshold - zero out tiny positions
    if min_position > 0:
        position_sizes = position_sizes.where(position_sizes >= min_position, 0.0)

    # 2. Clamp each position to max_single
    position_sizes = position_sizes.clip(upper=max_single)

    # 3. Scale down if total exposure exceeds max_total (row-wise)
    row_totals = position_sizes.sum(axis=1)

    # Only scale rows that exceed max_total
    needs_scaling = row_totals > max_total
    if needs_scaling.any():
        # Compute scaling factor per row
        scale_factors = max_total / row_totals.where(needs_scaling, max_total)
        scale_factors = scale_factors.clip(upper=1.0)

        # Apply row-wise scaling
        position_sizes = position_sizes.multiply(scale_factors, axis=0)

    logger.debug(
        f"Final hard clamp applied: max_single={max_single:.2%}, "
        f"max_total={max_total:.2%}, rows_scaled={needs_scaling.sum()}"
    )

    return position_sizes


def compute_risk_parity_weights(
    spread_volatilities: pd.DataFrame,
    target_vol_contribution: Optional[float] = None,
    vol_floor: float = 0.001,
) -> pd.DataFrame:
    """
    Compute risk parity weights so each position contributes equal volatility.

    Risk parity sizing: position_weight ∝ 1/volatility

    This ensures that high-volatility spreads get smaller positions so they
    contribute the same risk as low-volatility spreads.

    Parameters
    ----------
    spread_volatilities : pd.DataFrame
        Spread volatilities (T × pairs) - annualized or same units
    target_vol_contribution : float, optional
        Target volatility contribution per position. If None, computed to
        achieve equal-risk across all non-zero positions.
    vol_floor : float
        Minimum volatility to avoid extreme sizing (default 0.1%)

    Returns
    -------
    pd.DataFrame
        Risk parity weights (T × pairs) that sum to ~1.0 per row
    """
    # Floor volatility to prevent extreme weights
    vol_floored = spread_volatilities.clip(lower=vol_floor)

    # Replace NaN with median (per row) to handle missing data
    row_median = vol_floored.median(axis=1)
    vol_filled = vol_floored.apply(lambda col: col.fillna(row_median), axis=0)

    # Inverse volatility weights (higher vol = lower weight)
    inv_vol = 1.0 / vol_filled

    # Normalize to sum to 1.0 per row
    row_sums = inv_vol.sum(axis=1)
    # Avoid division by zero
    row_sums = row_sums.replace(0, 1.0)

    risk_parity_weights = inv_vol.divide(row_sums, axis=0)

    # If target_vol_contribution specified, scale to that
    if target_vol_contribution is not None:
        n_positions = (risk_parity_weights > 0).sum(axis=1)
        total_target = target_vol_contribution * n_positions
        risk_parity_weights = risk_parity_weights.multiply(total_target, axis=0)

    return risk_parity_weights


def compute_concentration_metrics(
    position_sizes: pd.DataFrame,
    hhi_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Compute portfolio concentration metrics at each timestamp.

    Metrics computed:
    - HHI (Herfindahl-Hirschman Index): Sum of squared portfolio weights
      - HHI = 1.0: Single position (maximum concentration)
      - HHI = 1/N: Equal weights across N positions
    - Effective positions: 1/HHI (equivalent number of equal-sized positions)
    - Maximum single weight
    - Total exposure

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Position sizes (T × pairs)
    hhi_threshold : float
        Threshold above which portfolio is considered concentrated

    Returns
    -------
    pd.DataFrame
        Metrics per timestamp with columns:
        ['hhi', 'effective_positions', 'max_weight', 'total_exposure',
         'n_positions', 'is_concentrated']
    """
    # Normalize to weights (sum to 1)
    row_totals = position_sizes.sum(axis=1)
    row_totals = row_totals.replace(0, 1.0)  # Avoid div by zero

    weights = position_sizes.divide(row_totals, axis=0)

    # HHI = sum of squared weights
    hhi = (weights ** 2).sum(axis=1)

    # Effective positions = 1/HHI
    eff_pos = 1.0 / hhi.replace(0, 1.0)

    # Max single weight
    max_weight = weights.max(axis=1)

    # Total exposure (unnormalized)
    total_exposure = position_sizes.sum(axis=1)

    # Count non-zero positions
    n_positions = (position_sizes > 0).sum(axis=1)

    # Is concentrated flag
    is_concentrated = hhi > hhi_threshold

    metrics = pd.DataFrame({
        "hhi": hhi,
        "effective_positions": eff_pos,
        "max_weight": max_weight,
        "total_exposure": total_exposure,
        "n_positions": n_positions,
        "is_concentrated": is_concentrated,
    })

    return metrics


def apply_risk_parity_sizing(
    position_sizes: pd.DataFrame,
    spread_volatilities: pd.DataFrame,
    target_vol_contribution: float = 0.125,
    blend_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Apply risk parity adjustment to position sizes.

    Blends between original sizing and pure risk parity based on blend_factor.

    Parameters
    ----------
    position_sizes : pd.DataFrame
        Original position sizes (T × pairs)
    spread_volatilities : pd.DataFrame
        Spread volatilities (T × pairs)
    target_vol_contribution : float
        Target volatility contribution per position (default 12.5% for 8 positions)
    blend_factor : float
        0.0 = keep original, 1.0 = pure risk parity (default 1.0)

    Returns
    -------
    pd.DataFrame
        Risk-parity adjusted position sizes
    """
    if blend_factor <= 0.0 or spread_volatilities is None or spread_volatilities.empty:
        return position_sizes

    # Compute risk parity weights
    rp_weights = compute_risk_parity_weights(
        spread_volatilities=spread_volatilities,
        target_vol_contribution=target_vol_contribution,
    )

    # Scale risk parity weights to match original total exposure
    original_total = position_sizes.sum(axis=1)
    rp_total = rp_weights.sum(axis=1).replace(0, 1.0)
    rp_scaled = rp_weights.multiply(original_total / rp_total, axis=0)

    # Blend
    if blend_factor >= 1.0:
        return rp_scaled
    else:
        return position_sizes * (1 - blend_factor) + rp_scaled * blend_factor


def get_concentration_summary(
    metrics: pd.DataFrame,
) -> ConcentrationMetrics:
    """
    Get summary concentration metrics (average across time).

    Parameters
    ----------
    metrics : pd.DataFrame
        Output from compute_concentration_metrics()

    Returns
    -------
    ConcentrationMetrics
        Summary statistics
    """
    return ConcentrationMetrics(
        hhi=float(metrics["hhi"].mean()),
        effective_positions=float(metrics["effective_positions"].mean()),
        max_single_weight=float(metrics["max_weight"].mean()),
        total_exposure=float(metrics["total_exposure"].mean()),
        n_positions=int(metrics["n_positions"].median()),
        is_concentrated=bool(metrics["is_concentrated"].mean() > 0.5),
    )


def compute_dynamic_stops(
    predicted_vol: pd.DataFrame,
    base_stop_pct: float,
    base_time_stop_bars: int,
    vol_scale_min: float = 0.7,
    vol_scale_max: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute dynamic stop parameters based on predicted volatility.

    Low predicted vol → widen stops (more room to breathe)
    High predicted vol → tighten stops (cut losses faster)

    Parameters
    ----------
    predicted_vol : pd.DataFrame
        Forward volatility predictions (T × P)
    base_stop_pct : float
        Base stop loss percentage (e.g., 0.03 for 3%)
    base_time_stop_bars : int
        Base time stop in bars
    vol_scale_min : float
        Minimum scaling factor (for high vol)
    vol_scale_max : float
        Maximum scaling factor (for low vol)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (adjusted_stop_pct, adjusted_time_stop_bars) matrices
    """
    # Compute median vol for reference
    median_vol = predicted_vol.median().median()
    if not np.isfinite(median_vol) or median_vol <= 0:
        median_vol = 0.005  # Default

    # Vol ratio: current / median
    vol_ratio = predicted_vol / median_vol
    vol_ratio = vol_ratio.clip(0.5, 2.0)  # Reasonable bounds

    # Scale factor: inverse relationship (low vol → higher scale → wider stops)
    vol_scale = 1.0 / vol_ratio
    vol_scale = vol_scale.clip(vol_scale_min, vol_scale_max)

    # Apply to stops
    adjusted_stop_pct = base_stop_pct * vol_scale
    adjusted_time_stop = (base_time_stop_bars * vol_scale).astype(int)

    return adjusted_stop_pct, adjusted_time_stop
