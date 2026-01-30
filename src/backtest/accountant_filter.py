# src/backtest/accountant_filter.py
"""
Accountant Filter: Entry/Exit mask computation for pairs trading.

This module computes trading signals based on z-scores and expected profit.
It supports three scoring modes:

1. LEGACY BINARY FILTERS (USE_SIGNAL_SCORING=False):
   - Hard AND filters on each condition
   - Entry = (|z| > entry_z) AND (profit > hurdle) AND ...

2. RULE-BASED SCORING (USE_SIGNAL_SCORING=True, USE_ML_SIGNAL_SCORING=False):
   - Weighted combination of factors
   - Allows borderline signals if other factors are strong
   - Weights are manually tuned

3. ML-BASED SCORING (USE_ML_SIGNAL_SCORING=True):
   - LightGBM classifier trained on historical trades
   - Learns optimal feature weights from data
   - Walk-forward training to avoid lookahead bias
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.features.inflection_detector import compute_inflection_mask

from src.backtest import config_backtest as cfg

if TYPE_CHECKING:
    from src.models.ml_signal_scorer import MLSignalScorer

logger = logging.getLogger("backtest.accountant_filter")


@dataclass(frozen=True)
class TradeMasks:
    """
    Outputs are aligned as:
      index  -> time (t)
      columns -> pair_id (e.g., "ETH-BTC")
    """
    entries: pd.DataFrame          # bool
    exits: pd.DataFrame            # bool
    expected_profit: pd.DataFrame  # float
    expected_hold_bars: pd.DataFrame  # float, expected holding time in bars
    signal_score: Optional[pd.DataFrame] = None  # float 0-1, for conviction sizing
    scoring_method: str = "none"  # "rule_based", "ml_based", or "legacy"
    # Continuous exposure: position size [-1, 1] instead of binary entry
    continuous_size: Optional[pd.DataFrame] = None  # float, for gradual position building
    # Dynamic entry threshold per bar/pair
    dynamic_entry_z: Optional[pd.DataFrame] = None  # float, varies with spread vol


def _cfg_get(name: str, default):
    """Get config value with a safe fallback, raise clean error if missing and no default."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if default is not None:
        return default
    raise AttributeError(f"Missing required config: cfg.{name}")


def _rolling_lag1_autocorr(z_score: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling lag-1 autocorrelation per pair.
    """
    lag = z_score.shift(1)
    mean = z_score.rolling(window).mean()
    mean_lag = lag.rolling(window).mean()
    cov = (z_score * lag).rolling(window).mean() - (mean * mean_lag)
    std = z_score.rolling(window).std()
    std_lag = lag.rolling(window).std()
    denom = std * std_lag
    corr = cov / denom
    return corr


def _z_trend_confirm(z_score: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Require z-score slope to turn toward zero.
    """
    slope = z_score - z_score.shift(lookback)
    return ((z_score > 0) & (slope < 0)) | ((z_score < 0) & (slope > 0))


def _compute_spread_velocity(z_score: pd.DataFrame, lookback: int = 3) -> tuple:
    """
    Compute spread velocity (1st derivative) and entry quality indicators.

    For mean-reversion, the BEST entries are when:
    1. Z just hit an extreme (fresh signal) - not when already reverting
    2. Z is still expanding or just peaked (not late to the trade)

    Returns:
        velocity: Rate of change of z-score (negative = reverting for long z)
        acceleration: Rate of change of velocity
        is_fresh_extreme: Just crossed entry threshold (best entry)
        is_stale: Has been reverting for a while (avoid - late entry)
    """
    abs_z = z_score.abs()

    # Velocity: dz/dt (change per bar)
    velocity = (z_score - z_score.shift(lookback)) / lookback

    # Acceleration: d²z/dt²
    acceleration = (velocity - velocity.shift(lookback)) / lookback

    # Is reverting: velocity sign opposite to z-score sign
    is_reverting = ((z_score > 0) & (velocity < 0)) | ((z_score < 0) & (velocity > 0))

    # FRESH EXTREME: z is expanding or just peaked
    # Best entries are when abs(z) is still growing or just stopped
    # (not when it's been declining toward mean for several bars)
    abs_velocity = abs_z - abs_z.shift(lookback)
    is_fresh_extreme = (abs_velocity >= -0.1 * lookback)  # Allow small decline

    # STALE SIGNAL: has been reverting for multiple lookback periods
    # If z was even more extreme 2*lookback bars ago and has been declining
    past_abs_z = abs_z.shift(2 * lookback)
    is_stale = (past_abs_z > abs_z) & is_reverting

    return velocity, acceleration, is_fresh_extreme, is_stale


def compute_signal_score(
    z_score: pd.DataFrame,
    spread_volatility: pd.DataFrame,
    expected_profit: pd.DataFrame,
    expected_hold: pd.DataFrame,
    *,
    kalman_gain: Optional[pd.DataFrame] = None,
    beta_uncertainty: Optional[pd.DataFrame] = None,
    entry_z: float = 2.0,
    max_entry_z: float = 4.0,
    velocity_lookback: int = 3,
) -> pd.DataFrame:
    """
    Compute a confidence score (0-1) for each potential entry signal.

    Combines multiple factors into a single score, allowing borderline
    signals to pass if other factors are strong.

    Score components:
    1. Z-magnitude score: How far from mean (normalized to entry threshold)
    2. Velocity score: Is spread actively reverting?
    3. Profit potential score: Expected profit from OU model
    4. Stability score: Beta stability + Kalman gain quality

    Returns:
        signal_score: DataFrame of scores (0-1) for each bar/pair
    """
    # Get weights from config
    w_z = float(getattr(cfg, "SCORE_WEIGHT_Z_MAGNITUDE", 0.25))
    w_vel = float(getattr(cfg, "SCORE_WEIGHT_VELOCITY", 0.25))
    w_profit = float(getattr(cfg, "SCORE_WEIGHT_PROFIT_POTENTIAL", 0.25))
    w_stab = float(getattr(cfg, "SCORE_WEIGHT_STABILITY", 0.25))

    abs_z = z_score.abs()

    # 1. Z-magnitude score (0-1)
    # Score increases as |z| goes from entry_z to max_entry_z, then decreases
    # Peak score at ~(entry_z + max_entry_z) / 2
    z_range = max_entry_z - entry_z
    z_normalized = (abs_z - entry_z) / z_range  # 0 at entry_z, 1 at max_entry_z
    # Bell curve: peak at 0.5 (middle of range)
    z_score_component = 1.0 - 2.0 * (z_normalized - 0.5).abs()
    z_score_component = z_score_component.clip(0, 1)
    # Must be above entry threshold to get any score
    z_score_component = z_score_component.where(abs_z >= entry_z, 0.0)

    # 2. Velocity/Freshness score (0-1)
    # Best entries are FRESH extremes (z just hit threshold), not late entries
    velocity, acceleration, is_fresh_extreme, is_stale = _compute_spread_velocity(
        z_score, velocity_lookback
    )
    # Fresh extreme gets high score (best entry point)
    vel_score_component = is_fresh_extreme.astype(float) * 0.8
    # Stale signals (already reverting for a while) get penalty
    vel_score_component = vel_score_component - is_stale.astype(float) * 0.3
    vel_score_component = vel_score_component.clip(0, 1)

    # 3. Profit potential score (0-1)
    # Normalize expected profit to 0-1 range
    # Typical good profit: 0.5-2% per trade
    profit_normalized = (expected_profit / 0.02).clip(0, 1)  # 2% = perfect score
    profit_score_component = profit_normalized.fillna(0)

    # 4. Stability score (0-1)
    stab_score_component = pd.DataFrame(1.0, index=z_score.index, columns=z_score.columns)

    if kalman_gain is not None:
        # Lower Kalman gain = more stable beta estimate
        gain_threshold = float(getattr(cfg, "KALMAN_GAIN_THRESHOLD", 0.5))
        gain_penalty = (kalman_gain / gain_threshold).clip(0, 1)
        stab_score_component = stab_score_component * (1.0 - 0.5 * gain_penalty)

    if beta_uncertainty is not None:
        # Lower P[0,0] = more confident beta estimate
        max_uncertainty = float(getattr(cfg, "MAX_BETA_UNCERTAINTY", 1.0))
        uncertainty_penalty = (beta_uncertainty / max_uncertainty).clip(0, 1)
        stab_score_component = stab_score_component * (1.0 - 0.5 * uncertainty_penalty)

    # Combine weighted scores
    total_score = (
        w_z * z_score_component +
        w_vel * vel_score_component +
        w_profit * profit_score_component +
        w_stab * stab_score_component
    )

    # Final normalization to 0-1
    total_score = total_score.clip(0, 1)

    return total_score


def _freq_minutes(freq: str) -> float:
    f = str(freq).lower().strip()
    if f in ("1min", "1m", "min", "t"):
        return 1.0
    if f.endswith("h"):
        return float(f.replace("h", "")) * 60.0
    if f.endswith("hour"):
        return float(f.replace("hour", "")) * 60.0
    if f.endswith("hours"):
        return float(f.replace("hours", "")) * 60.0
    if f.endswith("min"):
        return float(f.replace("min", ""))
    if f.endswith("m"):
        return float(f.replace("m", ""))
    raise ValueError(f"Unsupported freq '{freq}'.")


def _bars_per_day(freq: str) -> float:
    mins = _freq_minutes(freq)
    return (24.0 * 60.0) / mins


def compute_expected_round_trip_cost(
    bars_per_day: float = 96.0,  # 15-min bars default
) -> float:
    """
    Compute expected round-trip trading cost using actual config values.

    This replaces the hardcoded 28 bps transaction cost with actual config-based costs.
    Ensures the OU model's expected net profit matches what will be realized in pnl_engine.

    Costs include:
    - Fees: 4 legs × effective_fee_rate (using FEE_MODEL)
    - Slippage: 4 legs × SLIPPAGE_RATE
    - Adverse selection buffer (optional market impact)

    Parameters
    ----------
    bars_per_day : float
        Number of bars per day (for potential future use with vol-adjusted slippage)

    Returns
    -------
    float
        Total expected cost as fraction (e.g., 0.002 = 20 bps)
    """
    # 1. Effective fee rate based on FEE_MODEL
    fee_model = getattr(cfg, "FEE_MODEL", "taker_only")
    if fee_model == "maker_taker_mix":
        maker_prob = getattr(cfg, "MAKER_FILL_PROBABILITY", 0.70)
        maker_rate = getattr(cfg, "MAKER_FEE_RATE", 0.0002)
        taker_rate = getattr(cfg, "TAKER_FEE_RATE", 0.0005)
        effective_fee = maker_prob * maker_rate + (1 - maker_prob) * taker_rate
    elif fee_model == "maker_only":
        effective_fee = getattr(cfg, "MAKER_FEE_RATE", 0.0002)
    else:  # taker_only
        effective_fee = getattr(cfg, "TAKER_FEE_RATE", 0.0005)

    # 2. Slippage rate per leg
    slippage_rate = getattr(cfg, "SLIPPAGE_RATE", 0.0002)  # 2 bps default

    # 3. Pair trade = 4 legs (2 entry + 2 exit, each leg is one coin)
    legs = 4
    total_fee_cost = legs * effective_fee
    total_slippage_cost = legs * slippage_rate

    # 4. Adverse selection buffer for market impact
    # Accounts for: partial fills, legging risk, unfavorable fills during volatile spikes
    adverse_selection_bps = getattr(cfg, "ADVERSE_SELECTION_BPS", 0.0)
    adverse_selection = adverse_selection_bps / 10_000.0

    total_cost = total_fee_cost + total_slippage_cost + adverse_selection

    logger.debug(
        "Expected round-trip cost: %.4f (%.1f bps) = fees %.1f bps + slippage %.1f bps + adverse %.1f bps",
        total_cost, total_cost * 10000,
        total_fee_cost * 10000, total_slippage_cost * 10000, adverse_selection * 10000
    )

    return total_cost


def compute_ou_expected_profit(
    z_score: pd.DataFrame,
    spread_volatility: pd.DataFrame,
    half_life_bars: Union[float, Dict[str, float], pd.Series],
    *,
    exit_z: float = 0.6,
    transaction_cost: Optional[float] = None,  # None = use config costs
    use_config_costs: bool = True,  # Use actual config fees/slippage
    funding_cost_per_bar: Optional[Union[float, pd.DataFrame]] = None,
    bars_per_day: float = 1440.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute OU-based expected profit that accounts for time-to-revert and costs.

    OU-based expected profit model: The simple expected_profit = vol * 0.75 doesn't account for:
    - How long it takes to mean-revert (slower = more funding cost)
    - The conditional expectation given current z-score
    - Transaction costs that eat into profit

    For an OU process with half-life H:
    - Mean-reversion rate λ = ln(2) / H
    - Expected time to hit exit_z from current z: first-passage time approximation
    - Expected reversion: z * σ * (1 - exp(-λ * τ)) where τ is expected hold time

    Cost-Aware Enhancement:
    - When use_config_costs=True (default), uses compute_expected_round_trip_cost()
      to compute actual expected costs from FEE_MODEL, SLIPPAGE_BPS, etc.
    - This ensures E[net] = E[gross] - (fees + slippage + funding) is realistic

    Parameters
    ----------
    z_score : pd.DataFrame
        Current z-scores (T × P)
    spread_volatility : pd.DataFrame
        Spread volatility (T × P)
    half_life_bars : float, dict, or Series
        Half-life in bars. Can be:
        - Single float (same for all pairs)
        - Dict mapping pair names to half-lives
        - Series with pair names as index
    exit_z : float
        Z-score threshold for exit (mean reversion target)
    transaction_cost : float, optional
        Round-trip transaction cost as fraction. If None:
        - use_config_costs=True: compute from FEE_MODEL/SLIPPAGE
        - use_config_costs=False: fallback to 28 bps
    use_config_costs : bool
        If True and transaction_cost is None, use compute_expected_round_trip_cost()
    funding_cost_per_bar : float or DataFrame, optional
        Funding cost per bar. If None, uses cfg defaults.
        Can be DataFrame for per-pair funding.
    bars_per_day : float
        Number of bars per day (1440 for 1-min bars)

    Returns
    -------
    expected_profit : pd.DataFrame
        Net expected profit after costs (T × P)
    expected_hold_bars : pd.DataFrame
        Expected holding time in bars (T × P)
    """
    T, P = z_score.shape
    pairs = z_score.columns

    # Compute transaction cost from config if not provided
    if transaction_cost is None:
        if use_config_costs:
            transaction_cost = compute_expected_round_trip_cost(bars_per_day=bars_per_day)
            logger.info(
                "Using config-based transaction cost: %.4f (%.1f bps)",
                transaction_cost, transaction_cost * 10000
            )
        else:
            transaction_cost = 0.0028  # Legacy fallback: 28 bps

    # Convert half_life to per-pair Series
    if isinstance(half_life_bars, (int, float)):
        hl_series = pd.Series(half_life_bars, index=pairs)
    elif isinstance(half_life_bars, dict):
        hl_series = pd.Series(half_life_bars).reindex(pairs)
        # Fill missing with default
        default_hl = _cfg_get("MIN_HALF_LIFE_BARS", 400) * 2  # Conservative default
        hl_series = hl_series.fillna(default_hl)
    else:
        hl_series = half_life_bars.reindex(pairs)

    # Mean-reversion rate: λ = ln(2) / half_life
    lambda_series = np.log(2) / hl_series

    # Funding cost per bar
    if funding_cost_per_bar is None:
        daily_funding = _cfg_get("FUNDING_DRAG_BASE_DAILY", 0.0001)
        funding_per_bar = daily_funding / bars_per_day
    elif isinstance(funding_cost_per_bar, pd.DataFrame):
        funding_per_bar = funding_cost_per_bar
    else:
        funding_per_bar = float(funding_cost_per_bar)

    # Compute expected profit for each cell
    abs_z = z_score.abs()

    # Expected holding time approximation:
    # For OU process, E[τ | z_0] ≈ half_life * ln(|z_0| / exit_z) for |z_0| > exit_z
    # This is a rough approximation of first-passage time
    # More conservative: use fraction of half-life scaled by z distance
    z_ratio = abs_z / exit_z
    z_ratio = z_ratio.clip(lower=1.01)  # Avoid log(1) = 0

    # Expected hold time: half_life * factor based on z distance
    # For z=2.3, exit_z=0.6: ratio=3.83, ln(3.83)=1.34 → ~1.34 half-lives
    expected_hold_bars = pd.DataFrame(index=z_score.index, columns=pairs, dtype=float)
    for pair in pairs:
        hl = hl_series[pair]
        hold_factor = np.log(z_ratio[pair])  # ln(|z|/exit_z)
        # Clamp to reasonable range: 0.5 to 3.0 half-lives
        hold_factor = hold_factor.clip(lower=0.5, upper=3.0)
        expected_hold_bars[pair] = hl * hold_factor

    # Expected reversion magnitude (OU conditional expectation)
    # E[reversion] = |z| * σ * (1 - exp(-λ * τ))
    # where τ is expected holding time
    expected_profit = pd.DataFrame(index=z_score.index, columns=pairs, dtype=float)

    for pair in pairs:
        lam = lambda_series[pair]
        tau = expected_hold_bars[pair]
        z_abs = abs_z[pair]
        vol = spread_volatility[pair]

        # Fraction of z expected to revert
        revert_frac = 1.0 - np.exp(-lam * tau)

        # Expected profit from reversion (in spread units)
        gross_profit = z_abs * vol * revert_frac

        # Subtract costs
        # 1. Transaction cost (fixed per trade)
        # 2. Funding cost (proportional to holding time)
        if isinstance(funding_per_bar, pd.DataFrame):
            funding_cost = funding_per_bar[pair] * tau
        else:
            funding_cost = funding_per_bar * tau

        net_profit = gross_profit - transaction_cost - funding_cost

        expected_profit[pair] = net_profit

    return expected_profit, expected_hold_bars


def compute_masks(
    z_score: pd.DataFrame,
    spread_volatility: pd.DataFrame,
    *,
    beta: Optional[pd.DataFrame] = None,
    entry_z: Optional[float] = None,
    exit_z: Optional[float] = None,
    stop_loss_z: Optional[float] = None,
    max_entry_z: Optional[float] = None,
    min_profit_hurdle: Optional[float] = None,
    expected_revert_mult: Optional[float] = None,
    use_ou_model: bool = False,
    half_life_bars: Optional[Union[float, Dict[str, float], pd.Series]] = None,
    transaction_cost: Optional[float] = None,
    funding_cost_per_bar: Optional[Union[float, pd.DataFrame]] = None,
    freq: Optional[str] = None,
    ml_scorer: Optional["MLSignalScorer"] = None,
    kalman_gain: Optional[pd.DataFrame] = None,
) -> TradeMasks:
    """
    Step F: Accountant Filter (vectorized masks).

    Two modes for expected profit calculation:

    1. Simple mode (use_ou_model=False, default):
       expected_profit = spread_volatility * EXPECTED_REVERT_MULT

    2. OU model mode (use_ou_model=True):
       Uses Ornstein-Uhlenbeck process to compute expected profit that accounts for:
       - Time-to-mean-revert based on half-life
       - Transaction costs
       - Funding costs during holding period
       Requires half_life_bars to be provided.

    entries = (abs(z) > ENTRY_Z) & (expected_profit > MIN_PROFIT_HURDLE)
    exits   = (abs(z) < EXIT_Z) | (abs(z) > STOP_LOSS_Z)

    Notes
    -----
    - This module does NOT enforce blocking. Blocking belongs to pnl_engine (Step G).
    - NaN policy:
        - entries NaN -> False (never enter on unknown signal)
        - exits NaN   -> False (exit logic is handled by pnl_engine's "end of data" rule)

    Parameters
    ----------
    use_ou_model : bool
        If True, use OU-based expected profit model .
        Requires half_life_bars to be provided.
    beta : pd.DataFrame, optional
        Per-pair beta used to align expected profit to return units.
    max_entry_z : float, optional
        Upper bound on entry z-score to avoid extreme mean-reversion bets.
    freq : str, optional
        Bar frequency (used to scale funding in OU model).
    half_life_bars : float, dict, or Series, optional
        Half-life in bars for OU model. Required if use_ou_model=True.
    transaction_cost : float, optional
        Round-trip transaction cost for OU model. Default 0.0028 (28 bps).
    funding_cost_per_bar : float or DataFrame, optional
        Per-bar funding cost for OU model.

    Returns
    -------
    TradeMasks with DataFrames aligned to z_score index/columns.
    """
    if z_score is None or spread_volatility is None:
        raise ValueError("z_score and spread_volatility must not be None.")
    if z_score.empty or spread_volatility.empty:
        raise ValueError("z_score and spread_volatility must be non-empty DataFrames.")

    # Alignment checks (critical for correctness)
    if not z_score.index.equals(spread_volatility.index):
        raise ValueError("Index mismatch: z_score and spread_volatility must share the same time index.")
    if not z_score.columns.equals(spread_volatility.columns):
        raise ValueError("Column mismatch: z_score and spread_volatility must share the same pair columns.")

    # Load knobs (with safe fallbacks where reasonable)
    entry_z = float(entry_z if entry_z is not None else _cfg_get("ENTRY_Z", None))
    exit_z = float(exit_z if exit_z is not None else _cfg_get("EXIT_Z", None))
    stop_loss_z = float(stop_loss_z if stop_loss_z is not None else _cfg_get("STOP_LOSS_Z", None))
    if max_entry_z is None:
        max_entry_z = getattr(cfg, "MAX_ENTRY_Z", None)
    max_entry_z = float(max_entry_z) if max_entry_z is not None else None
    min_profit_hurdle = float(
        min_profit_hurdle if min_profit_hurdle is not None else _cfg_get("MIN_PROFIT_HURDLE", 0.002)
    )

    # Compute expected profit
    if use_ou_model:
        if half_life_bars is None:
            raise ValueError("half_life_bars is required when use_ou_model=True")

        if freq is None:
            freq = getattr(cfg, "SIGNAL_TIMEFRAME", None) or getattr(cfg, "BAR_FREQ", "1min")
        bars_per_day = _bars_per_day(freq)
        # Let compute_ou_expected_profit use config-based costs if transaction_cost is None
        expected_profit, expected_hold = compute_ou_expected_profit(
            z_score=z_score,
            spread_volatility=spread_volatility,
            half_life_bars=half_life_bars,
            exit_z=exit_z,
            transaction_cost=transaction_cost,  # Pass through; None triggers config-based costs
            funding_cost_per_bar=funding_cost_per_bar,
            bars_per_day=bars_per_day,
        )
        logger.info("Using OU-based expected profit model ")
    else:
        # Simple model (legacy)
        expected_revert_mult = float(
            expected_revert_mult if expected_revert_mult is not None else _cfg_get("EXPECTED_REVERT_MULT", 0.75)
        )
        expected_profit = spread_volatility * expected_revert_mult
        # Placeholder for expected hold (not computed in simple mode)
        expected_hold = pd.DataFrame(
            np.nan, index=z_score.index, columns=z_score.columns
        )

    abs_z = z_score.abs()

    # Align expected profit to return units when beta is provided
    if beta is not None:
        if not beta.index.equals(z_score.index) or not beta.columns.equals(z_score.columns):
            raise ValueError("beta must align with z_score index/columns.")
        expected_profit = expected_profit / (1.0 + beta.abs())

    # Check scoring mode
    use_signal_scoring = bool(getattr(cfg, "USE_SIGNAL_SCORING", False))
    use_ml_scoring = bool(getattr(cfg, "USE_ML_SIGNAL_SCORING", False))
    ml_fallback = bool(getattr(cfg, "ML_FALLBACK_TO_RULE_SCORING", True))

    signal_score_df = None  # Will be populated if scoring is enabled
    scoring_method = "legacy"

    # Determine which scoring method to use
    if use_ml_scoring and ml_scorer is not None and ml_scorer.is_trained:
        # === ML-BASED SCORING ===
        scoring_method = "ml_based"
        # Use ML-specific threshold if set, otherwise fall back to general threshold
        ml_min_score = getattr(cfg, "ML_MIN_SIGNAL_SCORE", None)
        min_score = float(ml_min_score if ml_min_score is not None else getattr(cfg, "MIN_SIGNAL_SCORE", 0.55))

        logger.info("Using ML-based signal scoring")

        signal_score_df = ml_scorer.predict_scores(
            z_score=z_score,
            spread_volatility=spread_volatility,
            expected_profit=expected_profit,
            expected_hold_bars=expected_hold,
            half_life_bars=half_life_bars if half_life_bars is not None else 500,
            kalman_gain=kalman_gain,
            beta=beta,
            entry_z=entry_z,
            max_entry_z=max_entry_z if max_entry_z is not None else 4.0,
        )

        # Log ML score distribution (for debugging)
        at_entry_z = abs_z > entry_z
        scores_at_threshold = signal_score_df.where(at_entry_z)
        if scores_at_threshold.stack().notna().any():
            score_stats = scores_at_threshold.stack().dropna()
            logger.info(
                "ML score distribution at |z|>%.1f: min=%.3f, median=%.3f, max=%.3f, count=%d",
                entry_z, score_stats.min(), score_stats.median(), score_stats.max(), len(score_stats)
            )

        # Entry requires: above entry threshold AND ML score above minimum
        entries = (abs_z > entry_z) & (signal_score_df >= min_score)

        # Also require minimum profit (but lower threshold since ML handles quality)
        entries = entries & (expected_profit > min_profit_hurdle)

        # Upper bound on entry z-score
        if max_entry_z is not None:
            entries = entries & (abs_z < max_entry_z)

        # Log ML scoring stats
        entry_scores = signal_score_df.where(entries)
        if entries.any().any():
            avg_score = entry_scores.stack().mean()
            logger.info("ML scoring: avg_entry_score=%.3f, min_required=%.3f", avg_score, min_score)
        else:
            logger.warning("ML scoring produced 0 entries. Consider lowering MIN_SIGNAL_SCORE.")

    elif use_ml_scoring and ml_fallback:
        # ML enabled but scorer not ready, fall back to rule-based
        logger.info("ML scorer not trained, falling back to rule-based scoring")
        use_signal_scoring = True  # Force rule-based scoring

    if scoring_method == "legacy" and use_signal_scoring:
        # === RULE-BASED SIGNAL SCORING ===
        # Instead of binary AND filters, compute a confidence score
        scoring_method = "rule_based"
        min_score = float(getattr(cfg, "MIN_SIGNAL_SCORE", 0.55))
        velocity_lookback = int(getattr(cfg, "VELOCITY_LOOKBACK", 3))

        signal_score_df = compute_signal_score(
            z_score=z_score,
            spread_volatility=spread_volatility,
            expected_profit=expected_profit,
            expected_hold=expected_hold,
            kalman_gain=kalman_gain,
            beta_uncertainty=None,  # Could be passed from signal generation
            entry_z=entry_z,
            max_entry_z=max_entry_z if max_entry_z is not None else 4.0,
            velocity_lookback=velocity_lookback,
        )

        # Entry requires: above entry threshold AND score above minimum
        entries = (abs_z > entry_z) & (signal_score_df >= min_score)

        # Also require minimum profit (but lower threshold since scoring handles quality)
        entries = entries & (expected_profit > min_profit_hurdle)

        # Upper bound on entry z-score
        if max_entry_z is not None:
            entries = entries & (abs_z < max_entry_z)

        # Log scoring stats
        entry_scores = signal_score_df.where(entries)
        if entries.any().any():
            avg_score = entry_scores.stack().mean()
            logger.info("Rule-based scoring: avg_entry_score=%.3f, min_required=%.3f", avg_score, min_score)

    elif scoring_method == "legacy":
        # === LEGACY: Binary Filter System ===
        entries = (abs_z > entry_z) & (expected_profit > min_profit_hurdle)
        min_spread_vol = float(getattr(cfg, "MIN_SPREAD_VOL", 0.0))
        if min_spread_vol > 0.0:
            entries = entries & (spread_volatility > min_spread_vol)
        if getattr(cfg, "CROSS_REVERT_ENTRY", False):
            prev_abs_z = abs_z.shift(1)
            reverting = abs_z < prev_abs_z
            entries = entries & (prev_abs_z > entry_z) & reverting
        if getattr(cfg, "ENABLE_Z_TREND_CONFIRM", False):
            lookback = int(getattr(cfg, "Z_TREND_LOOKBACK", 3))
            if lookback > 0:
                trend_ok = _z_trend_confirm(z_score, lookback)
                entries = entries & trend_ok
        if getattr(cfg, "ENABLE_MEAN_REVERSION_REGIME", False):
            lookback = int(getattr(cfg, "REGIME_ROLLING_BARS", 96))
            max_autocorr = float(getattr(cfg, "REGIME_MAX_LAG1_AUTOCORR", 0.2))
            lag1_corr = _rolling_lag1_autocorr(z_score, lookback)
            regime_ok = (lag1_corr <= max_autocorr) | lag1_corr.isna()
            entries = entries & regime_ok
        if getattr(cfg, "ENABLE_EXPECTED_HOLD_FILTER", False) and expected_hold is not None:
            min_hold = int(getattr(cfg, "MIN_EXPECTED_HOLD_BARS", 0))
            max_hold = int(getattr(cfg, "MAX_EXPECTED_HOLD_BARS", 0))
            if min_hold > 0:
                entries = entries & (expected_hold >= min_hold)
            if max_hold > 0:
                entries = entries & (expected_hold <= max_hold)
        if max_entry_z is not None:
            entries = entries & (abs_z < max_entry_z)

        # === INFLECTION POINT DETECTION (Option C Redesign) ===
        # Apply inflection filter if enabled: wait for z-score peak before entry
        if getattr(cfg, "ENABLE_INFLECTION_FILTER", False):
            logger.info("Applying inflection point filter...")

            # Apply inflection filter to each pair independently
            inflection_entries = pd.DataFrame(False, index=z_score.index, columns=z_score.columns)

            for pair in z_score.columns:
                z_series = z_score[pair].values

                # Compute inflection mask for this pair
                inflection_mask = compute_inflection_mask(
                    z_score=z_series,
                    entry_threshold=entry_z,
                    min_confidence=getattr(cfg, "INFLECTION_MIN_CONFIDENCE", 0.5),
                    min_bars_since_extreme=getattr(cfg, "INFLECTION_MIN_BARS_SINCE_EXTREME", 2),
                    max_bars_since_extreme=getattr(cfg, "INFLECTION_MAX_BARS_SINCE_EXTREME", 10),
                    velocity_reversal_threshold=getattr(cfg, "INFLECTION_VELOCITY_THRESHOLD", -0.05),
                )

                inflection_entries[pair] = inflection_mask

            # Count how many signals passed the filter
            threshold_crosses = (abs_z > entry_z).sum().sum()
            inflection_passes = inflection_entries.sum().sum()

            logger.info(
                "Inflection filter: %d/%d signals kept (%.1f%%)",
                inflection_passes,
                threshold_crosses,
                100.0 * inflection_passes / max(1, threshold_crosses),
            )

            # Require BOTH threshold crossing AND inflection point
            entries = entries & inflection_entries

    # ==========================================================================
    # CONTINUOUS EXPOSURE & DYNAMIC ENTRY (ROI Optimization)
    # ==========================================================================
    continuous_size_df = None
    dynamic_entry_z_df = None

    # Compute dynamic entry threshold if enabled
    enable_dynamic_entry = bool(getattr(cfg, "ENABLE_DYNAMIC_ENTRY_Z", False))
    if enable_dynamic_entry:
        # Use config-based costs if transaction_cost not provided
        txn_cost = transaction_cost if transaction_cost is not None else compute_expected_round_trip_cost(bars_per_day=bars_per_day)
        dynamic_entry_z_df = compute_dynamic_entry_threshold(
            spread_volatility=spread_volatility,
            cost_per_trade=txn_cost,
        )
        # Re-compute entries with dynamic threshold
        # Entry when |z| > dynamic_threshold (instead of fixed entry_z)
        entries_dynamic = (abs_z > dynamic_entry_z_df) & (expected_profit > min_profit_hurdle)
        if max_entry_z is not None:
            entries_dynamic = entries_dynamic & (abs_z < max_entry_z)
        # Merge with existing entries (union: entry if either condition met)
        entries = entries | entries_dynamic
        logger.info(
            "Dynamic entry threshold: avg=%.2f, min=%.2f, max=%.2f",
            dynamic_entry_z_df.values.mean(),
            dynamic_entry_z_df.values.min(),
            dynamic_entry_z_df.values.max(),
        )

    # Compute continuous exposure sizing if enabled
    enable_continuous = bool(getattr(cfg, "ENABLE_CONTINUOUS_EXPOSURE", False))
    if enable_continuous:
        continuous_size_df = compute_continuous_exposure_size(
            z_score=z_score,
            spread_volatility=spread_volatility,
            expected_profit=expected_profit,
        )
        # For continuous mode, "entries" marks when to START or ADJUST position
        # Convert continuous size to entry signal (any non-zero size is an entry)
        continuous_entries = continuous_size_df.abs() > 0.01  # Small threshold
        # Merge with binary entries
        entries = entries | continuous_entries
        logger.info(
            "Continuous exposure: bars_with_position=%d (%.1f%%)",
            int(continuous_entries.any(axis=1).sum()),
            100.0 * continuous_entries.any(axis=1).mean(),
        )

    exits = (abs_z < exit_z) | (abs_z > stop_loss_z)

    # ==========================================================================
    # IMPROVEMENT #5: VOLATILITY FILTER
    # ==========================================================================
    # Only enter trades when spread volatility is within acceptable range
    enable_vol_filter = bool(getattr(cfg, "ENABLE_VOLATILITY_FILTER", False))
    if enable_vol_filter:
        min_vol_bps = float(getattr(cfg, "MIN_SPREAD_VOLATILITY_BPS", 15.0))
        max_vol_bps = float(getattr(cfg, "MAX_SPREAD_VOLATILITY_BPS", 500.0))
        # Convert volatility to bps (spread_volatility is typically in decimal form)
        vol_bps = spread_volatility * 10000
        vol_filter = (vol_bps >= min_vol_bps) & (vol_bps <= max_vol_bps)
        entries_before = int(entries.to_numpy().sum())
        entries = entries & vol_filter
        entries_after = int(entries.to_numpy().sum())
        n_filtered = entries_before - entries_after
        if n_filtered > 0:
            logger.info(
                "Volatility filter removed %d entries (min=%.1f bps, max=%.1f bps)",
                n_filtered, min_vol_bps, max_vol_bps
            )

    # Strict boolean masks; never enter/exit on NaNs
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    n_entries = int(entries.to_numpy().sum())
    n_exits = int(exits.to_numpy().sum())

    logger.info(
        "Accountant masks computed: entries_true=%d exits_true=%d (pairs=%d, rows=%d)",
        n_entries,
        n_exits,
        entries.shape[1],
        entries.shape[0],
    )

    if use_ou_model and n_entries > 0:
        # Log average expected profit for entries
        entry_profits = expected_profit.where(entries)
        avg_profit = entry_profits.stack().mean()
        avg_hold = expected_hold.where(entries).stack().mean()
        logger.info(
            "OU model stats: avg_expected_profit=%.4f, avg_expected_hold_bars=%.0f",
            avg_profit,
            avg_hold,
        )

    return TradeMasks(
        entries=entries,
        exits=exits,
        expected_profit=expected_profit,
        expected_hold_bars=expected_hold,
        signal_score=signal_score_df,
        scoring_method=scoring_method,
        continuous_size=continuous_size_df,
        dynamic_entry_z=dynamic_entry_z_df,
    )


# =============================================================================
# CONTINUOUS EXPOSURE & DYNAMIC ENTRY (ROI Optimization)
# =============================================================================

def compute_continuous_exposure_size(
    z_score: pd.DataFrame,
    spread_volatility: pd.DataFrame,
    expected_profit: pd.DataFrame,
    *,
    start_z: Optional[float] = None,
    full_z: Optional[float] = None,
    min_profit_hurdle: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute continuous position size based on z-score magnitude.

    Instead of binary entry (0 or 1), this returns a size in [0, 1]
    that scales with how extreme the z-score is.

    Formula: size = clip(alpha * (|z| - z_start), 0, 1) * sign(-z)
    where alpha = 1 / (z_full - z_start)

    Parameters
    ----------
    z_score : pd.DataFrame
        Z-scores for each pair
    spread_volatility : pd.DataFrame
        Spread volatility for each pair
    expected_profit : pd.DataFrame
        Expected profit for each entry
    start_z : float, optional
        Z-score at which to start building position. Default from config.
    full_z : float, optional
        Z-score at which to reach full position. Default from config.
    min_profit_hurdle : float, optional
        Minimum expected profit to allow entry. Default from config.

    Returns
    -------
    pd.DataFrame
        Position size in [-1, 1] where sign indicates direction
        (negative z -> long spread -> positive size)
    """
    # Load config values
    start_z = start_z if start_z is not None else float(getattr(cfg, "CONTINUOUS_ENTRY_START_Z", 1.5))
    full_z = full_z if full_z is not None else float(getattr(cfg, "CONTINUOUS_ENTRY_FULL_Z", 2.5))
    min_profit = min_profit_hurdle if min_profit_hurdle is not None else float(getattr(cfg, "MIN_PROFIT_HURDLE", 0.005))

    abs_z = z_score.abs()
    z_sign = np.sign(z_score)

    # Scaling factor
    alpha = 1.0 / (full_z - start_z) if full_z > start_z else 1.0

    # Position size magnitude: 0 at start_z, 1 at full_z
    size_magnitude = alpha * (abs_z - start_z)
    size_magnitude = size_magnitude.clip(lower=0.0, upper=1.0)

    # Direction: short spread when z > 0 (mean reversion), long when z < 0
    # size > 0 means long spread (long Y, short X)
    # size < 0 means short spread (short Y, long X)
    position_size = size_magnitude * (-z_sign)

    # Zero out positions where expected profit is below hurdle
    if expected_profit is not None:
        insufficient_profit = expected_profit < min_profit
        position_size = position_size.where(~insufficient_profit, 0.0)

    # Zero out positions below start threshold (safety)
    position_size = position_size.where(abs_z >= start_z, 0.0)

    return position_size


def compute_dynamic_entry_threshold(
    spread_volatility: pd.DataFrame,
    cost_per_trade: float = 0.0028,  # 28 bps default
    *,
    min_entry_z: Optional[float] = None,
    base_entry_z: Optional[float] = None,
    cost_multiplier: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute dynamic entry threshold based on spread volatility and costs.

    When spread volatility is high, we can afford to trade at lower z-scores
    because the edge (in dollar terms) is larger relative to fixed costs.

    Formula: entry_z = max(min_z, base_z + k * cost_bps / spread_vol_bps)

    Parameters
    ----------
    spread_volatility : pd.DataFrame
        Per-bar spread volatility
    cost_per_trade : float
        Round-trip transaction cost (as fraction)
    min_entry_z : float, optional
        Minimum entry z-score. Default from config.
    base_entry_z : float, optional
        Base entry z-score. Default from config.
    cost_multiplier : float, optional
        Scaling factor for cost adjustment. Default from config.

    Returns
    -------
    pd.DataFrame
        Dynamic entry threshold per bar/pair
    """
    min_z = min_entry_z if min_entry_z is not None else float(getattr(cfg, "DYNAMIC_ENTRY_Z_MIN", 1.3))
    base_z = base_entry_z if base_entry_z is not None else float(getattr(cfg, "DYNAMIC_ENTRY_Z_BASE", 1.5))
    k = cost_multiplier if cost_multiplier is not None else float(getattr(cfg, "DYNAMIC_ENTRY_Z_COST_MULT", 0.5))

    # Convert cost to bps
    cost_bps = cost_per_trade * 10000  # 28 bps

    # Convert spread vol to bps (per bar)
    # spread_vol is typically ~0.001 to 0.01 per bar for 15-min
    spread_vol_bps = spread_volatility * 10000

    # Avoid division by zero
    spread_vol_bps = spread_vol_bps.clip(lower=1.0)

    # Dynamic threshold: lower when vol is high, higher when vol is low
    dynamic_z = base_z + k * (cost_bps / spread_vol_bps)

    # Ensure minimum threshold
    dynamic_z = dynamic_z.clip(lower=min_z)

    return dynamic_z


def compute_pair_cluster(pairs: list) -> dict:
    """
    Group pairs by shared coins to identify correlation clusters.

    Pairs sharing a coin (e.g., ETH-BTC and ETH-SOL both have ETH)
    are likely correlated and should have position limits.

    Parameters
    ----------
    pairs : list
        List of pair names like "ETH-BTC"

    Returns
    -------
    dict
        Mapping of coin -> list of pairs containing that coin
    """
    clusters = {}
    for pair in pairs:
        parts = pair.split("-")
        if len(parts) == 2:
            coin1, coin2 = parts
            if coin1 not in clusters:
                clusters[coin1] = []
            if coin2 not in clusters:
                clusters[coin2] = []
            clusters[coin1].append(pair)
            clusters[coin2].append(pair)
    return clusters


def apply_cluster_cap(
    entries: pd.DataFrame,
    active_positions: pd.DataFrame,
    max_per_cluster: int = 3,
) -> pd.DataFrame:
    """
    Filter entries to respect cluster position limits.

    Prevents loading up on correlated pairs (e.g., all SOL pairs).

    Parameters
    ----------
    entries : pd.DataFrame
        Entry signals (bool)
    active_positions : pd.DataFrame
        Currently active positions (bool or int)
    max_per_cluster : int
        Max positions per cluster

    Returns
    -------
    pd.DataFrame
        Filtered entries
    """
    pairs = entries.columns.tolist()
    clusters = compute_pair_cluster(pairs)

    filtered_entries = entries.copy()

    for idx in entries.index:
        # Count active positions per cluster
        active = active_positions.loc[idx] if idx in active_positions.index else pd.Series(0, index=pairs)
        cluster_counts = {}

        for coin, coin_pairs in clusters.items():
            count = sum(1 for p in coin_pairs if active.get(p, 0) > 0)
            cluster_counts[coin] = count

        # Filter entries that would exceed cluster cap
        for pair in pairs:
            if not filtered_entries.loc[idx, pair]:
                continue

            parts = pair.split("-")
            if len(parts) != 2:
                continue

            coin1, coin2 = parts
            # Check if either coin's cluster is at limit
            if cluster_counts.get(coin1, 0) >= max_per_cluster or \
               cluster_counts.get(coin2, 0) >= max_per_cluster:
                filtered_entries.loc[idx, pair] = False

    return filtered_entries


# =============================================================================
# CARRY FILTER
# =============================================================================

def compute_expected_funding_cost(
    pair: str,
    direction: int,  # +1 long spread, -1 short spread
    expected_hold_bars: float,
    funding_rates: pd.DataFrame,
    current_idx: int,
    bars_per_funding_period: int = 480,  # 8 hours at 1-min bars
    lookback_periods: int = 21,  # 21 funding periods (7 days)
) -> float:
    """
    Estimate expected funding cost over holding period.

    Funding bias consideration: Pairs trading with perpetuals can have systematic funding
    differentials. This function estimates funding cost to filter bad entries.

    For LONG spread (long Y, short X):
    - Pay funding on Y short position
    - Receive funding on X long position (but we're shorting X!)
    - Actually: long Y (pay if funding positive), short X (receive if funding positive)
    - Net funding = funding_Y (if long) - funding_X (if short)
    - For long spread: we long Y, short X
    - If funding_Y > 0, we pay on long Y
    - If funding_X > 0, we receive on short X
    - Net = funding_Y - funding_X

    Parameters
    ----------
    pair : str
        Pair identifier (e.g., "ETH-BTC").
    direction : int
        +1 for long spread (long Y, short X), -1 for short spread.
    expected_hold_bars : float
        Expected holding time in bars.
    funding_rates : pd.DataFrame
        Funding rates per coin. Index = time, columns = coins.
        Rates should be in decimal form (e.g., 0.0001 = 0.01%).
    current_idx : int
        Current bar index (row position in DataFrame).
    bars_per_funding_period : int
        Bars per funding payment (8 hours = 480 for 1-min, 32 for 15-min).
    lookback_periods : int
        Number of past funding periods to average.

    Returns
    -------
    float
        Expected funding cost as fraction of notional.
        Positive = we pay, negative = we receive.
    """
    if funding_rates is None or funding_rates.empty:
        return 0.0

    # Parse pair
    sep = "-"
    if sep not in pair:
        return 0.0
    coin_y, coin_x = pair.split(sep, 1)

    # Check if both coins have funding data
    if coin_y not in funding_rates.columns or coin_x not in funding_rates.columns:
        return 0.0

    # Get historical funding rates
    lookback_bars = lookback_periods * bars_per_funding_period
    start_idx = max(0, current_idx - lookback_bars)

    if start_idx >= current_idx:
        return 0.0

    # Use iloc for positional indexing
    try:
        funding_y = funding_rates[coin_y].iloc[start_idx:current_idx].mean()
        funding_x = funding_rates[coin_x].iloc[start_idx:current_idx].mean()
    except Exception:
        return 0.0

    if not (np.isfinite(funding_y) and np.isfinite(funding_x)):
        return 0.0

    # Net funding per period
    # Long spread (long Y, short X): pay funding_Y, receive funding_X
    # Short spread (short Y, long X): receive funding_Y, pay funding_X
    if direction > 0:  # Long spread
        net_funding_per_period = funding_y - funding_x
    else:  # Short spread
        net_funding_per_period = funding_x - funding_y

    # Scale to expected holding period
    n_funding_periods = expected_hold_bars / bars_per_funding_period

    # Use 1.5x multiplier for conservative estimate (funding can spike)
    expected_funding_cost = 1.5 * net_funding_per_period * n_funding_periods

    return float(expected_funding_cost)


def apply_carry_filter(
    entries: pd.DataFrame,
    expected_profit: pd.DataFrame,
    z_score: pd.DataFrame,
    expected_hold_bars: pd.DataFrame,
    funding_rates: Optional[pd.DataFrame] = None,
    carry_filter_mult: float = 1.5,
    prefer_positive_carry: bool = True,
    bars_per_funding_period: int = 480,
) -> pd.DataFrame:
    """
    Filter entries where expected profit < expected funding cost.

    Funding bias consideration: Trades that "work" on price movement can still lose
    money due to systematic funding payments.

    Parameters
    ----------
    entries : pd.DataFrame
        Entry mask (T × P) bool.
    expected_profit : pd.DataFrame
        Expected profit per entry (T × P) float.
    z_score : pd.DataFrame
        Z-score matrix for direction determination.
    expected_hold_bars : pd.DataFrame
        Expected holding time per entry.
    funding_rates : pd.DataFrame, optional
        Funding rates per coin.
    carry_filter_mult : float
        Required edge multiplier. Entry allowed if:
        expected_profit > carry_filter_mult * expected_funding_cost
    prefer_positive_carry : bool
        If True, prefer the direction with lower funding cost when
        both directions are valid (not implemented in simple version).
    bars_per_funding_period : int
        Bars per funding payment period.

    Returns
    -------
    pd.DataFrame
        Filtered entry mask.
    """
    if funding_rates is None or funding_rates.empty:
        logger.debug("No funding rates provided, skipping carry filter")
        return entries

    filtered = entries.copy()
    pairs = entries.columns.tolist()
    n_filtered = 0

    for pair in pairs:
        if pair not in filtered.columns:
            continue

        # Get entry indices for this pair
        pair_entries = entries[pair]
        entry_indices = pair_entries[pair_entries].index

        for t in entry_indices:
            try:
                # Get position in DataFrame
                t_pos = entries.index.get_loc(t)

                # Direction from z-score
                z = z_score.loc[t, pair]
                direction = -1 if z > 0 else 1  # z > 0 means short spread

                # Expected profit
                exp_profit = expected_profit.loc[t, pair]

                # Expected hold
                exp_hold = expected_hold_bars.loc[t, pair]
                if not np.isfinite(exp_hold) or exp_hold <= 0:
                    exp_hold = 500  # Default

                # Compute expected funding cost
                funding_cost = compute_expected_funding_cost(
                    pair=pair,
                    direction=direction,
                    expected_hold_bars=exp_hold,
                    funding_rates=funding_rates,
                    current_idx=t_pos,
                    bars_per_funding_period=bars_per_funding_period,
                )

                # Filter: expected profit must exceed funding cost * multiplier
                if np.isfinite(exp_profit) and np.isfinite(funding_cost):
                    if funding_cost > 0 and exp_profit < carry_filter_mult * funding_cost:
                        filtered.loc[t, pair] = False
                        n_filtered += 1

            except Exception as e:
                logger.warning(f"Carry filter error for {pair} at {t}: {e}")
                continue

    if n_filtered > 0:
        logger.info(f"Carry filter removed {n_filtered} entries due to funding cost")

    return filtered


# =============================================================================
# OU MODEL V2: CALIBRATED EXPECTED PROFIT # =============================================================================

def compute_ou_expected_profit_v2(
    z_score: pd.DataFrame,
    spread_volatility: pd.DataFrame,
    half_life_bars: Union[float, Dict[str, float], pd.Series],
    *,
    exit_z: float = 0.6,
    transaction_cost: float = 0.0028,
    funding_cost_per_bar: Optional[Union[float, pd.DataFrame]] = None,
    bars_per_day: float = 96.0,
    # NEW: Rolling estimation
    use_rolling: bool = True,
    rolling_window_days: int = 14,
    # NEW: Sanity clamping
    max_expected_hold_bars: int = 200,
    min_expected_profit_pct: float = 0.001,
    max_expected_profit_pct: float = 0.05,
    # NEW: Calibration adjustment
    calibration_discount: float = 0.5,  # 50% haircut based on backtest results
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    V2 OU-based expected profit with calibration and sanity clamping.

    Improvements over V1:
    1. Rolling half-life estimation (recent behavior, not stale)
    2. Sanity clamping on expected hold time and profit
    3. Calibration discount based on historical over-optimism

    The original OU model predicted 2.75-3.15% profit per trade but reality
    was much worse. This version applies a 50% haircut to account for:
    - Model misspecification
    - Regime changes during trades
    - Adverse selection effects

    Parameters
    ----------
    calibration_discount : float
        Multiply expected profit by this factor. 0.5 means 50% of raw estimate.
        Based on backtest analysis showing OU predictions were 2x too optimistic.

    Returns
    -------
    expected_profit : pd.DataFrame
        Calibrated and clamped profit estimate
    expected_hold_bars : pd.DataFrame
        Clamped hold time estimate
    """
    T, P = z_score.shape
    pairs = z_score.columns

    # Convert half_life to per-pair Series
    if isinstance(half_life_bars, (int, float)):
        hl_series = pd.Series(half_life_bars, index=pairs)
    elif isinstance(half_life_bars, dict):
        hl_series = pd.Series(half_life_bars).reindex(pairs)
        default_hl = _cfg_get("MIN_HALF_LIFE_BARS", 200)
        hl_series = hl_series.fillna(default_hl)
    else:
        hl_series = half_life_bars.reindex(pairs)

    # Optionally use rolling half-life estimation
    if use_rolling:
        rolling_window = rolling_window_days * int(bars_per_day)
        # Compute rolling half-life from z-score autocorrelation
        # This captures recent mean-reversion behavior
        rolling_hl_df = pd.DataFrame(index=z_score.index, columns=pairs, dtype=float)

        for pair in pairs:
            z = z_score[pair].dropna()
            if len(z) < rolling_window:
                rolling_hl_df[pair] = hl_series[pair]
                continue

            # Rolling autocorrelation-based half-life
            z_lag = z.shift(1)
            rolling_cov = z.rolling(rolling_window).cov(z_lag)
            rolling_var = z.rolling(rolling_window).var()
            phi = rolling_cov / rolling_var.replace(0, np.nan)

            # half_life = -ln(2) / ln(phi) for phi in (0, 1)
            phi_clipped = phi.clip(lower=0.01, upper=0.99)
            rolling_hl = -np.log(2) / np.log(phi_clipped)

            # Clamp to reasonable range
            rolling_hl = rolling_hl.clip(lower=50, upper=500)
            rolling_hl = rolling_hl.fillna(hl_series[pair])
            rolling_hl_df[pair] = rolling_hl

        # Use rolling half-life for lambda calculation
        lambda_df = np.log(2) / rolling_hl_df
    else:
        lambda_series = np.log(2) / hl_series
        lambda_df = pd.DataFrame(
            {pair: lambda_series[pair] for pair in pairs},
            index=z_score.index
        )

    # Funding cost per bar
    if funding_cost_per_bar is None:
        daily_funding = _cfg_get("FUNDING_DRAG_BASE_DAILY", 0.0001)
        funding_per_bar = daily_funding / bars_per_day
    elif isinstance(funding_cost_per_bar, pd.DataFrame):
        funding_per_bar = funding_cost_per_bar
    else:
        funding_per_bar = float(funding_cost_per_bar)

    abs_z = z_score.abs()

    # Expected holding time
    z_ratio = abs_z / exit_z
    z_ratio = z_ratio.clip(lower=1.01)

    expected_hold_bars_df = pd.DataFrame(index=z_score.index, columns=pairs, dtype=float)
    expected_profit_df = pd.DataFrame(index=z_score.index, columns=pairs, dtype=float)

    for pair in pairs:
        if use_rolling:
            hl = rolling_hl_df[pair]
            lam = lambda_df[pair]
        else:
            hl = hl_series[pair]
            lam = lambda_df[pair] if isinstance(lambda_df, pd.DataFrame) else lambda_df

        hold_factor = np.log(z_ratio[pair])
        hold_factor = hold_factor.clip(lower=0.5, upper=3.0)

        if isinstance(hl, pd.Series):
            expected_hold = hl * hold_factor
        else:
            expected_hold = hl * hold_factor

        # SANITY CLAMP 1: Cap expected hold time
        expected_hold_clamped = expected_hold.clip(upper=max_expected_hold_bars)
        expected_hold_bars_df[pair] = expected_hold_clamped

        # Compute expected profit
        z_abs = abs_z[pair]
        vol = spread_volatility[pair]

        revert_frac = 1.0 - np.exp(-lam * expected_hold_clamped)
        gross_profit = z_abs * vol * revert_frac

        # Subtract costs
        if isinstance(funding_per_bar, pd.DataFrame):
            funding_cost = funding_per_bar[pair] * expected_hold_clamped
        else:
            funding_cost = funding_per_bar * expected_hold_clamped

        net_profit = gross_profit - transaction_cost - funding_cost

        # SANITY CLAMP 2: Cap expected profit
        net_profit_clamped = net_profit.clip(
            lower=min_expected_profit_pct,
            upper=max_expected_profit_pct
        )

        # CALIBRATION DISCOUNT: Apply haircut based on historical over-optimism
        net_profit_calibrated = net_profit_clamped * calibration_discount

        expected_profit_df[pair] = net_profit_calibrated

    logger.info(
        "OU V2 model: calibration_discount=%.1f%%, max_hold=%d bars, profit_range=[%.2f%%, %.2f%%]",
        calibration_discount * 100,
        max_expected_hold_bars,
        min_expected_profit_pct * 100,
        max_expected_profit_pct * 100,
    )

    return expected_profit_df, expected_hold_bars_df


# =============================================================================
# ENTRY COOLDOWN # =============================================================================

def apply_entry_cooldown(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    cooldown_bars: Optional[int] = None,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Prevent re-entry within cooldown period after exit.

    This reduces trading churn and gives trades time to develop
    rather than rapidly entering and exiting the same pair.

    Parameters
    ----------
    entries : pd.DataFrame
        Entry signals (bool, T × P)
    exits : pd.DataFrame
        Exit signals (bool, T × P)
    cooldown_bars : int, optional
        Number of bars to wait before re-entry. Default from config.
    freq : str
        Bar frequency for time delta calculation

    Returns
    -------
    pd.DataFrame
        Filtered entries with cooldown applied
    """
    if cooldown_bars is None:
        cooldown_bars = int(getattr(cfg, "ENTRY_COOLDOWN_BARS", 24))  # 6 hours at 15-min

    if cooldown_bars <= 0:
        return entries

    # Get time delta for cooldown period
    minutes_per_bar = _freq_minutes(freq)
    cooldown_minutes = cooldown_bars * minutes_per_bar

    cooled_entries = entries.copy()
    n_blocked = 0

    for pair in entries.columns:
        # Find exit timestamps for this pair
        exit_mask = exits[pair] if pair in exits.columns else pd.Series(False, index=exits.index)
        exit_times = exit_mask[exit_mask].index

        for exit_time in exit_times:
            cooldown_end = exit_time + pd.Timedelta(minutes=cooldown_minutes)

            # Block entries during cooldown window
            cooldown_mask = (cooled_entries.index > exit_time) & (cooled_entries.index <= cooldown_end)
            blocked = cooled_entries.loc[cooldown_mask, pair].sum()
            n_blocked += blocked
            cooled_entries.loc[cooldown_mask, pair] = False

    if n_blocked > 0:
        logger.info(
            "Entry cooldown blocked %d entries (cooldown=%d bars = %.1f hours)",
            n_blocked, cooldown_bars, cooldown_bars * minutes_per_bar / 60
        )

    return cooled_entries


def apply_smart_cooldown(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    exit_reasons: Optional[pd.DataFrame] = None,
    cooldown_signal_bars: Optional[int] = None,
    cooldown_stop_loss_bars: Optional[int] = None,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Apply different cooldowns based on exit type .

    Exit types have different meanings:
    - Signal exit: Relationship worked, shorter cooldown (3 hours)
    - Stop-loss exit: Relationship may be broken, longer cooldown (12 hours)

    Parameters
    ----------
    entries : pd.DataFrame
        Entry signals (bool, T × P)
    exits : pd.DataFrame
        Exit signals (bool, T × P)
    exit_reasons : pd.DataFrame, optional
        Exit reason codes per bar/pair:
        - 1 = signal exit (z reverted to target)
        - 2 = time stop
        - 3 = stop loss
        - 4 = forced/other
        If None, falls back to standard cooldown.
    cooldown_signal_bars : int, optional
        Cooldown bars after signal exit. Default 12 (3 hours at 15-min).
    cooldown_stop_loss_bars : int, optional
        Cooldown bars after stop-loss exit. Default 48 (12 hours at 15-min).
    freq : str
        Bar frequency for time delta calculation

    Returns
    -------
    pd.DataFrame
        Filtered entries with smart cooldown applied
    """
    if cooldown_signal_bars is None:
        cooldown_signal_bars = int(getattr(cfg, "COOLDOWN_AFTER_SIGNAL_BARS", 12))  # 3 hours
    if cooldown_stop_loss_bars is None:
        cooldown_stop_loss_bars = int(getattr(cfg, "COOLDOWN_AFTER_STOP_LOSS_BARS", 48))  # 12 hours

    # Fall back to standard cooldown if no exit reasons provided
    if exit_reasons is None:
        default_cooldown = int(getattr(cfg, "ENTRY_COOLDOWN_BARS", 24))
        return apply_entry_cooldown(entries, exits, default_cooldown, freq)

    minutes_per_bar = _freq_minutes(freq)
    cooldown_signal_minutes = cooldown_signal_bars * minutes_per_bar
    cooldown_stop_loss_minutes = cooldown_stop_loss_bars * minutes_per_bar

    cooled_entries = entries.copy()
    n_blocked_signal = 0
    n_blocked_stop_loss = 0
    n_blocked_other = 0

    for pair in entries.columns:
        # Find exit timestamps for this pair
        exit_mask = exits[pair] if pair in exits.columns else pd.Series(False, index=exits.index)
        exit_times = exit_mask[exit_mask].index

        # Get exit reasons for this pair
        reasons = exit_reasons[pair] if pair in exit_reasons.columns else pd.Series(0, index=exits.index)

        for exit_time in exit_times:
            # Get exit reason at this time
            exit_reason = reasons.loc[exit_time] if exit_time in reasons.index else 0

            # Choose cooldown based on exit reason
            if exit_reason == 3:  # Stop-loss exit
                cooldown_minutes = cooldown_stop_loss_minutes
            elif exit_reason == 1:  # Signal exit
                cooldown_minutes = cooldown_signal_minutes
            else:  # Time stop, forced, or unknown - use signal cooldown
                cooldown_minutes = cooldown_signal_minutes

            # Apply cooldown
            cooldown_end = exit_time + pd.Timedelta(minutes=cooldown_minutes)
            cooldown_mask = (cooled_entries.index > exit_time) & (cooled_entries.index <= cooldown_end)
            blocked = cooled_entries.loc[cooldown_mask, pair].sum()

            if blocked > 0:
                if exit_reason == 3:
                    n_blocked_stop_loss += blocked
                elif exit_reason == 1:
                    n_blocked_signal += blocked
                else:
                    n_blocked_other += blocked

            cooled_entries.loc[cooldown_mask, pair] = False

    total_blocked = n_blocked_signal + n_blocked_stop_loss + n_blocked_other
    if total_blocked > 0:
        logger.info(
            "Smart cooldown blocked %d entries: "
            "signal_exits=%d (cooldown=%d bars), stop_loss=%d (cooldown=%d bars), other=%d",
            total_blocked,
            n_blocked_signal, cooldown_signal_bars,
            n_blocked_stop_loss, cooldown_stop_loss_bars,
            n_blocked_other,
        )

    return cooled_entries


# =============================================================================
# ENTRY QUALITY FILTERS # =============================================================================

def apply_slope_filter(
    z_scores: pd.DataFrame,
    entries: pd.DataFrame,
    entry_z: Optional[float] = None,
) -> pd.DataFrame:
    """
    Slope/Turning-Point filter: Only enter if |z| is high AND no longer increasing.

    The BEST mean-reversion entries are when z-score has peaked and started reverting.
    This filters out entries where z is still expanding (early entry = higher stop-out risk).

    Logic: Enter when abs(z_t) >= entry_z AND abs(z_t) < abs(z_{t-1})

    Parameters
    ----------
    z_scores : pd.DataFrame
        Z-score series per pair
    entries : pd.DataFrame
        Entry signals (bool)
    entry_z : float, optional
        Entry threshold. Default from config.

    Returns
    -------
    pd.DataFrame
        Filtered entries (slope filter applied)
    """
    if entry_z is None:
        entry_z = float(getattr(cfg, "ENTRY_Z", 2.0))

    abs_z = z_scores.abs()
    abs_z_prev = abs_z.shift(1)

    # Z is turning (peaked and starting to revert)
    # abs(z_t) < abs(z_{t-1}) means |z| is decreasing (z moving toward zero)
    turning = abs_z < abs_z_prev

    # Fill NaN from shift with False (no filter at first bar)
    turning = turning.fillna(False)

    filtered_entries = entries & turning

    n_before = int(entries.sum().sum())
    n_after = int(filtered_entries.sum().sum())
    if n_before > n_after:
        logger.info(
            "Slope filter: blocked %d/%d entries (%.1f%%) - requiring z to be turning",
            n_before - n_after, n_before, 100 * (n_before - n_after) / max(n_before, 1)
        )

    return filtered_entries


def apply_confirmation_filter(
    z_scores: pd.DataFrame,
    entries: pd.DataFrame,
    entry_z: Optional[float] = None,
    confirmation_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Confirmation filter: Require z to stay beyond threshold for N bars.

    This avoids single-bar spikes that quickly reverse - ensures the signal is persistent.

    Parameters
    ----------
    z_scores : pd.DataFrame
        Z-score series per pair
    entries : pd.DataFrame
        Entry signals (bool)
    entry_z : float, optional
        Entry threshold. Default from config.
    confirmation_bars : int, optional
        Number of consecutive bars above threshold required. Default from config.

    Returns
    -------
    pd.DataFrame
        Filtered entries (confirmation filter applied)
    """
    if entry_z is None:
        entry_z = float(getattr(cfg, "ENTRY_Z", 2.0))
    if confirmation_bars is None:
        confirmation_bars = int(getattr(cfg, "CONFIRMATION_BARS", 2))

    if confirmation_bars <= 1:
        return entries  # No confirmation needed

    abs_z = z_scores.abs()

    # Z stayed above threshold for last N bars
    above_threshold = abs_z >= entry_z

    # Rolling minimum over confirmation window (1 if all True, 0 if any False)
    confirmed = above_threshold.rolling(confirmation_bars, min_periods=confirmation_bars).min() == 1

    # Fill NaN from rolling with False
    confirmed = confirmed.fillna(False)

    filtered_entries = entries & confirmed

    n_before = int(entries.sum().sum())
    n_after = int(filtered_entries.sum().sum())
    if n_before > n_after:
        logger.info(
            "Confirmation filter: blocked %d/%d entries (%.1f%%) - requiring %d bars above z=%.1f",
            n_before - n_after, n_before, 100 * (n_before - n_after) / max(n_before, 1),
            confirmation_bars, entry_z
        )

    return filtered_entries


def apply_inflection_filter(
    z_scores: pd.DataFrame,
    entries: pd.DataFrame,
    entry_z: Optional[float] = None,
    lookback: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inflection point filter: Only enter when z-score has peaked and is starting to revert.

    This is an enhanced version of slope_filter that requires the z-score to have
    reached a local maximum recently and started turning back toward zero.

    The BEST mean-reversion entries are when:
    1. |z| >= entry_z (signal strength)
    2. |z_t| < |z_{t-1}| (turning back toward mean)
    3. Recent lookback period had a local maximum (z was expanding, now contracting)

    Parameters
    ----------
    z_scores : pd.DataFrame
        Z-score series per pair
    entries : pd.DataFrame
        Entry signals (bool)
    entry_z : float, optional
        Entry threshold. Default from config.
    lookback : int, optional
        Lookback for local max detection. Default from config.

    Returns
    -------
    pd.DataFrame
        Filtered entries (inflection filter applied)
    """
    if entry_z is None:
        entry_z = float(getattr(cfg, "ENTRY_Z", 2.0))
    if lookback is None:
        lookback = int(getattr(cfg, "INFLECTION_LOOKBACK_BARS", 3))

    abs_z = z_scores.abs()

    # Condition 1: Above threshold (already in entries, but be explicit)
    above_threshold = abs_z >= entry_z

    # Condition 2: z is decreasing (turning back toward mean)
    # abs(z_t) < abs(z_{t-1}) means |z| is decreasing (z moving toward zero)
    turning = abs_z < abs_z.shift(1)

    # Condition 3: Recently peaked - current z is below the max of the last lookback bars
    # This ensures z recently hit a local maximum and is now coming down
    recent_max = abs_z.rolling(lookback + 1, min_periods=2).max()
    recently_peaked = abs_z < recent_max

    # Combine: must be above threshold, turning, and recently peaked
    # Note: turning & recently_peaked together ensure z is at an inflection point
    inflection_mask = above_threshold & turning & recently_peaked

    # Fill NaN from shift/rolling with False
    inflection_mask = inflection_mask.fillna(False)

    filtered_entries = entries & inflection_mask

    n_before = int(entries.sum().sum())
    n_after = int(filtered_entries.sum().sum())
    if n_before > n_after:
        logger.info(
            "Inflection filter: blocked %d/%d entries (%.1f%%) - requiring z at turning point",
            n_before - n_after, n_before, 100 * (n_before - n_after) / max(n_before, 1)
        )

    return filtered_entries


def apply_stale_signal_filter(
    z_scores: pd.DataFrame,
    entries: pd.DataFrame,
    entry_z: Optional[float] = None,
    max_stale_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stale signal filter: Reject entries if z has been above threshold too long.

    Z-score can be high for many bars without reverting. Late entries have lower
    win rate because the easy reversion has already happened.

    Parameters
    ----------
    z_scores : pd.DataFrame
        Z-score series per pair
    entries : pd.DataFrame
        Entry signals (bool)
    entry_z : float, optional
        Entry threshold. Default from config.
    max_stale_bars : int, optional
        Maximum bars z can be above threshold. Default from config.

    Returns
    -------
    pd.DataFrame
        Filtered entries (stale signals removed)
    """
    if entry_z is None:
        entry_z = float(getattr(cfg, "ENTRY_Z", 2.0))
    if max_stale_bars is None:
        max_stale_bars = int(getattr(cfg, "MAX_STALE_SIGNAL_BARS", 8))

    if max_stale_bars <= 0:
        return entries  # No filter if max_stale_bars is 0 or negative

    abs_z = z_scores.abs()
    above_threshold = abs_z >= entry_z

    # Count consecutive bars above threshold for each column separately
    # Using a simple approach: count how many of the last max_stale_bars were above threshold
    # A signal is fresh if it WASN'T above threshold for all of the prior max_stale_bars

    # Check if we just crossed the threshold (wasn't above in previous bar)
    just_crossed = above_threshold & ~above_threshold.shift(1).fillna(False)

    # Or: signal is fresh if it's been above for <= max_stale_bars
    # Count consecutive True values using rolling sum approach
    # A bar is "stale" if all of the previous max_stale_bars were above threshold
    consecutive_above = above_threshold.rolling(max_stale_bars + 1, min_periods=1).sum()

    # Fresh = currently above AND not stale (haven't been above for full max_stale_bars)
    # If consecutive_above == max_stale_bars + 1, then we've been above for too long
    is_fresh = above_threshold & (consecutive_above <= max_stale_bars)

    filtered_entries = entries & is_fresh

    n_before = int(entries.sum().sum())
    n_after = int(filtered_entries.sum().sum())
    if n_before > n_after:
        logger.info(
            "Stale signal filter: blocked %d/%d entries (%.1f%%) - z above threshold > %d bars",
            n_before - n_after, n_before, 100 * (n_before - n_after) / max(n_before, 1),
            max_stale_bars
        )

    return filtered_entries


def apply_expected_profit_filter(
    entries: pd.DataFrame,
    expected_profit: pd.DataFrame,
    min_expected_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Expected profit filter: Only enter if OU model predicts meaningful profit.

    Parameters
    ----------
    entries : pd.DataFrame
        Entry signals (bool)
    expected_profit : pd.DataFrame
        Expected profit per entry
    min_expected_pct : float, optional
        Minimum expected profit percentage. Default from config.

    Returns
    -------
    pd.DataFrame
        Filtered entries
    """
    if min_expected_pct is None:
        min_expected_pct = float(getattr(cfg, "MIN_OU_EXPECTED_PROFIT_PCT", 0.003))

    if min_expected_pct <= 0:
        return entries

    profit_ok = expected_profit >= min_expected_pct
    filtered_entries = entries & profit_ok

    n_before = int(entries.sum().sum())
    n_after = int(filtered_entries.sum().sum())
    if n_before > n_after:
        logger.info(
            "Expected profit filter: blocked %d/%d entries (%.1f%%) - expected profit < %.2f%%",
            n_before - n_after, n_before, 100 * (n_before - n_after) / max(n_before, 1),
            min_expected_pct * 100
        )

    return filtered_entries


def apply_entry_quality_filters(
    z_scores: pd.DataFrame,
    entries: pd.DataFrame,
    entry_z: Optional[float] = None,
    enable_slope: Optional[bool] = None,
    enable_confirmation: Optional[bool] = None,
    confirmation_bars: Optional[int] = None,
    enable_inflection: Optional[bool] = None,
    inflection_lookback: Optional[int] = None,
    enable_stale_filter: Optional[bool] = None,
    max_stale_bars: Optional[int] = None,
    expected_profit: Optional[pd.DataFrame] = None,
    min_expected_profit_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply all entry quality filters .

    Combines multiple filters to reduce stop-loss hits and improve signal quality:
    - Slope filter: z must be turning (not still expanding)
    - Confirmation filter: z must stay above threshold for N bars
    - Inflection filter: z must be at local max turning point
    - Stale signal filter: z must not have been extreme too long
    - Expected profit filter: OU model must predict meaningful profit

    Parameters
    ----------
    z_scores : pd.DataFrame
        Z-score series per pair
    entries : pd.DataFrame
        Entry signals (bool)
    entry_z : float, optional
        Entry threshold
    enable_slope : bool, optional
        Enable slope filter. Default from config.
    enable_confirmation : bool, optional
        Enable confirmation filter. Default from config.
    confirmation_bars : int, optional
        Bars for confirmation. Default from config.
    enable_inflection : bool, optional
        Enable inflection filter. Default from config.
    inflection_lookback : int, optional
        Lookback for inflection filter. Default from config.
    enable_stale_filter : bool, optional
        Enable stale signal filter. Default from config.
    max_stale_bars : int, optional
        Max stale bars. Default from config.
    expected_profit : pd.DataFrame, optional
        Expected profit for expected profit filter.
    min_expected_profit_pct : float, optional
        Min expected profit. Default from config.

    Returns
    -------
    pd.DataFrame
        Filtered entries
    """
    if enable_slope is None:
        enable_slope = bool(getattr(cfg, "ENABLE_SLOPE_FILTER", False))
    if enable_confirmation is None:
        enable_confirmation = bool(getattr(cfg, "ENABLE_CONFIRMATION_FILTER", False))
    if enable_inflection is None:
        enable_inflection = bool(getattr(cfg, "ENABLE_INFLECTION_FILTER", False))
    if enable_stale_filter is None:
        enable_stale_filter = bool(getattr(cfg, "ENABLE_STALE_SIGNAL_FILTER", False))

    filtered = entries.copy()

    # Apply inflection filter (strongest, supersedes slope filter)
    if enable_inflection:
        filtered = apply_inflection_filter(z_scores, filtered, entry_z, inflection_lookback)
    elif enable_slope:
        # Only apply slope if inflection is not enabled (inflection is stronger)
        filtered = apply_slope_filter(z_scores, filtered, entry_z)

    if enable_confirmation:
        filtered = apply_confirmation_filter(z_scores, filtered, entry_z, confirmation_bars)

    if enable_stale_filter:
        filtered = apply_stale_signal_filter(z_scores, filtered, entry_z, max_stale_bars)

    # Apply expected profit filter if data is provided
    if expected_profit is not None:
        min_pct = min_expected_profit_pct or float(getattr(cfg, "MIN_OU_EXPECTED_PROFIT_PCT", 0.0))
        if min_pct > 0:
            filtered = apply_expected_profit_filter(filtered, expected_profit, min_pct)

    return filtered


# =============================================================================
# EXPECTED PROFIT RANKING # =============================================================================

def rank_expected_profits(
    expected_profit: pd.DataFrame,
    min_rank_percentile: float = 0.5,
) -> pd.DataFrame:
    """
    Rank expected profits across pairs at each timestamp.

    Instead of using expected profit as a binary gate, rank it and only
    allow entries in the top percentile of opportunities.

    Parameters
    ----------
    expected_profit : pd.DataFrame
        Expected profit per pair per timestamp
    min_rank_percentile : float
        Minimum percentile rank to allow entry (0.5 = top 50%)

    Returns
    -------
    pd.DataFrame
        Boolean mask where True = expected profit in top percentile
    """
    # Rank across pairs at each timestamp (higher profit = higher rank)
    profit_rank = expected_profit.rank(axis=1, pct=True)

    # Only allow top N percentile
    profit_gate = profit_rank >= min_rank_percentile

    logger.info(
        "Expected profit ranking: min_pct=%.1f%%, allowed_fraction=%.1f%%",
        min_rank_percentile * 100,
        profit_gate.mean().mean() * 100,
    )

    return profit_gate


# Backward-compatible wrapper for older tests/scripts
def compute_accountant_masks(*args, **kwargs):
    masks = compute_masks(*args, **kwargs)
    return masks.entries, masks.exits, masks.expected_profit
