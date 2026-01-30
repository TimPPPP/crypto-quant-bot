# src/backtest/signal_generation.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg
from src.models.kalman import KalmanFilterRegime

logger = logging.getLogger("backtest.signal_generation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


@dataclass(frozen=True)
class SignalFrames:
    z_score: pd.DataFrame
    spread_volatility: pd.DataFrame
    beta: pd.DataFrame
    # Trend overlay fields (optional - only populated if enabled)
    trend_score: Optional[pd.DataFrame] = None
    suppress_long: Optional[pd.DataFrame] = None
    suppress_short: Optional[pd.DataFrame] = None


def _parse_pair(pair: str) -> Tuple[str, str]:
    sep = getattr(cfg, "PAIR_ID_SEPARATOR", "-")
    if sep not in pair:
        raise ValueError(f"Invalid pair '{pair}'. Expected format like 'ETH-BTC'.")
    y, x = pair.split(sep, 1)
    y, x = y.strip(), x.strip()
    if not y or not x:
        raise ValueError(f"Invalid pair '{pair}'.")
    return y, x


def _safe_log(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float64", copy=False)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0] = np.nan
    return np.log(arr)


def _robust_vol(err_hist) -> float:
    # err_hist is a deque inside KalmanFilterRegime
    if err_hist is None or len(err_hist) == 0:
        return np.nan
    arr = np.asarray(err_hist, dtype="float64")
    method = getattr(cfg, "VOL_METHOD", "std").lower()
    if method == "ewma":
        alpha = float(getattr(cfg, "VOL_EWMA_ALPHA", 0.2))
        var = 0.0
        for v in arr:
            var = alpha * (v * v) + (1.0 - alpha) * var
        return float(np.sqrt(var))
    if method == "mad":
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = float(getattr(cfg, "VOL_MAD_SCALE", 1.4826))
        return mad * scale
    return float(np.std(arr))


def _compute_kalman_gain(kf, x_t: float) -> float:
    """
    Compute the Kalman gain K[0] for beta (first state variable).

    K = P * H' / (H * P * H' + R)

    High K indicates the filter is rapidly adjusting beta (potential regime change).
    """
    H = np.array([x_t, 1.0], dtype="float64")
    P = kf.P
    R = kf.R

    # Innovation variance: S = H * P * H' + R
    S = float(np.dot(H, np.dot(P, H.T))) + R
    if S <= 1e-10:
        return 0.0

    # Kalman gain for beta (first element)
    K_beta = float(np.dot(P, H.T)[0]) / S
    return abs(K_beta)


def _apply_signal_quality_filters(
    z_raw: float,
    kalman_gain: float,
    beta_uncertainty: float,
) -> float:
    """
    Apply signal quality filters (Priority 4).

    1. Regime filter: suppress signals when Kalman gain is high (beta unstable)
    2. Beta uncertainty scaling: reduce z-score confidence when P is high
    """
    # Check regime filter
    enable_regime = getattr(cfg, "ENABLE_REGIME_FILTER", True)
    gain_threshold = getattr(cfg, "KALMAN_GAIN_THRESHOLD", 0.3)

    if enable_regime and kalman_gain > gain_threshold:
        # Suppress signal during regime changes
        return np.nan

    # Check beta uncertainty scaling
    enable_uncertainty = getattr(cfg, "ENABLE_BETA_UNCERTAINTY_SCALING", True)
    max_uncertainty = getattr(cfg, "MAX_BETA_UNCERTAINTY", 0.5)

    if enable_uncertainty and beta_uncertainty > max_uncertainty:
        # Scale down z-score proportionally to uncertainty
        # Higher uncertainty -> smaller effective z-score
        scale = max_uncertainty / beta_uncertainty
        return z_raw * scale

    return z_raw


def _init_kalman_from_state(state_dict: Dict) -> KalmanFilterRegime:
    """
    Restore a KalmanFilterRegime from a persisted state dict.
    Uses model as the source of truth.
    """
    kf = KalmanFilterRegime(
        delta=state_dict.get("delta", cfg.KALMAN_DELTA),
        R=state_dict.get("R", cfg.KALMAN_R),
        rolling_window=state_dict.get("rolling_window", cfg.LOOKBACK_WINDOW),
        entry_z_threshold=state_dict.get("entry_z_threshold", cfg.ENTRY_Z),
        min_spread_pct=state_dict.get("min_spread_pct", 0.0),
    )

    ok = kf.load_state_dict(state_dict)
    if not ok:
        raise ValueError("Failed to load Kalman state dict.")
    return kf


def generate_signals_for_pair(
    test_df: pd.DataFrame,
    pair: str,
    warm_state: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate causal (t-1) z-scores, spread volatility, and beta for one pair.

    Causality rules enforced:
    - Use beta/alpha from state at time t-1 to predict y_t
    - Compute spread_volatility[t] from error_history up to t-1
    - Then call kf.update(y_t, x_t) to advance to time t
    """
    coin_y, coin_x = _parse_pair(pair)

    if coin_y not in test_df.columns or coin_x not in test_df.columns:
        raise KeyError(f"Missing columns for pair {pair}: need {coin_y} and {coin_x}")

    # Pull prices
    y_px = pd.to_numeric(test_df[coin_y], errors="coerce").to_numpy(dtype="float64")
    x_px = pd.to_numeric(test_df[coin_x], errors="coerce").to_numpy(dtype="float64")

    y = _safe_log(y_px)
    x = _safe_log(x_px)

    kf = _init_kalman_from_state(warm_state)

    n = len(test_df)
    z = np.full(n, np.nan, dtype="float64")
    vol = np.full(n, np.nan, dtype="float64")
    beta = np.full(n, np.nan, dtype="float64")

    # Causal loop
    for t in range(n):
        if not np.isfinite(y[t]) or not np.isfinite(x[t]):
            # Skip update entirely to prevent NaN contamination of error_history
            continue

        # ---- Predict using state at t-1 ----
        # Observation matrix H = [price_x, 1]
        H = np.array([x[t], 1.0], dtype="float64")
        y_pred = float(np.dot(H, kf.x))
        err = float(y[t] - y_pred)

        # ---- Volatility from history up to t-1 (strictly causal) ----
        std_prev = _robust_vol(kf.error_history)
        if not np.isfinite(std_prev) or std_prev <= 1e-8:
            # Conservative fallback to avoid division explosions
            std_prev = 0.01

        z_raw = err / std_prev

        # ---- Signal quality filters (Priority 4) ----
        # Get Kalman gain (before update) to detect regime changes
        kalman_gain = _compute_kalman_gain(kf, x[t])

        # Get beta uncertainty from P matrix
        beta_uncertainty = float(kf.P[0, 0]) if hasattr(kf, 'P') else 0.0

        # Apply quality filters (regime detection + uncertainty scaling)
        z[t] = _apply_signal_quality_filters(z_raw, kalman_gain, beta_uncertainty)

        vol[t] = std_prev
        beta[t] = float(kf.x[0])

        # ---- Now advance filter to incorporate time t ----
        # (kf.update will append err and update x/P)
        kf.update(y[t], x[t])

    return z, vol, beta


def generate_signals(
    test_df: pd.DataFrame,
    valid_pairs: Iterable[str],
    warm_states: Dict[str, Dict],
) -> SignalFrames:
    """
    Generate signals for all pairs on the test set.

    Returns three aligned DataFrames:
    - z_score[t, pair]
    - spread_volatility[t, pair]
    - beta[t, pair]
    """
    if test_df.empty:
        raise ValueError("test_df is empty.")

    pairs = list(valid_pairs)
    if not pairs:
        raise ValueError("valid_pairs is empty.")

    z_cols = {}
    vol_cols = {}
    beta_cols = {}

    logger.info("Generating signals on test set for %d pairs...", len(pairs))

    for pair in pairs:
        if pair not in warm_states:
            logger.warning("Skipping %s: no warm state found.", pair)
            continue

        try:
            z, vol, b = generate_signals_for_pair(test_df, pair, warm_states[pair])
            z_cols[pair] = z
            vol_cols[pair] = vol
            beta_cols[pair] = b
        except Exception as e:
            logger.warning("Failed signal gen for %s: %s", pair, e)

    if not z_cols:
        raise ValueError("No signals produced (all pairs skipped/failed).")

    z_df = pd.DataFrame(z_cols, index=test_df.index)
    vol_df = pd.DataFrame(vol_cols, index=test_df.index)
    beta_df = pd.DataFrame(beta_cols, index=test_df.index)

    logger.info("✅ Signal generation complete. Shape: %s (time x pairs)", z_df.shape)
    return SignalFrames(z_score=z_df, spread_volatility=vol_df, beta=beta_df)


def compute_trend_overlay_for_signals(
    test_df: pd.DataFrame,
    valid_pairs: List[str],
    warm_states: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute trend overlay for all pairs based on their spreads.

    This function computes spreads using warm-state betas and applies
    trend overlay analysis to detect trending spreads.

    Args:
        test_df: Price data with columns for each coin
        valid_pairs: List of pair IDs
        warm_states: Dictionary of warm Kalman states per pair

    Returns:
        (trend_score_df, suppress_long_df, suppress_short_df)
    """
    from src.features.trend_overlay import compute_spread_trend_state

    # Get trend overlay config
    ma_period = getattr(cfg, "TREND_MA_PERIOD", 20)
    slope_thresh = getattr(cfg, "TREND_MA_SLOPE_STRONG", 0.002)
    z_thresh = getattr(cfg, "TREND_Z_THRESH", 1.0)

    trend_scores = {}
    suppress_long = {}
    suppress_short = {}

    for pair in valid_pairs:
        if pair not in warm_states:
            continue

        try:
            # Parse pair
            coin_y, coin_x = _parse_pair(pair)

            if coin_y not in test_df.columns or coin_x not in test_df.columns:
                continue

            # Get prices
            y_px = pd.to_numeric(test_df[coin_y], errors="coerce").to_numpy(dtype="float64")
            x_px = pd.to_numeric(test_df[coin_x], errors="coerce").to_numpy(dtype="float64")

            # Log prices
            y = _safe_log(y_px)
            x = _safe_log(x_px)

            # Get beta from warm state
            state = warm_states[pair]
            beta = state.get("x", [1.0, 0.0])[0]  # x[0] is hedge ratio

            # Compute spread: Y - beta * X
            spread = y - beta * x

            # Compute trend overlay
            trend_state = compute_spread_trend_state(
                spread,
                ma_period=ma_period,
                slope_strong_thresh=slope_thresh,
                z_thresh=z_thresh,
            )

            trend_scores[pair] = trend_state["trend_score"].values
            suppress_long[pair] = trend_state["suppress_long"].values
            suppress_short[pair] = trend_state["suppress_short"].values

        except Exception as e:
            logger.warning(f"Failed to compute trend overlay for {pair}: {e}")
            continue

    if not trend_scores:
        # Return empty DataFrames if no pairs processed
        empty = pd.DataFrame(index=test_df.index)
        return empty, empty, empty

    trend_score_df = pd.DataFrame(trend_scores, index=test_df.index)
    suppress_long_df = pd.DataFrame(suppress_long, index=test_df.index)
    suppress_short_df = pd.DataFrame(suppress_short, index=test_df.index)

    # Fill NaN trend scores with 1.0 (no penalty)
    trend_score_df = trend_score_df.fillna(1.0)

    return trend_score_df, suppress_long_df, suppress_short_df


def generate_signals_with_trend_overlay(
    test_df: pd.DataFrame,
    valid_pairs: Iterable[str],
    warm_states: Dict[str, Dict],
    enable_trend_overlay: bool = True,
) -> SignalFrames:
    """
    Generate signals for all pairs including trend overlay.

    This is an enhanced version of generate_signals() that also computes
    trend overlay indicators for each pair.

    Args:
        test_df: Price data
        valid_pairs: List of pair IDs
        warm_states: Warm Kalman states
        enable_trend_overlay: Whether to compute trend overlay

    Returns:
        SignalFrames with z_score, spread_volatility, beta, and trend overlay
    """
    # Generate base signals
    pairs = list(valid_pairs)
    base_signals = generate_signals(test_df, pairs, warm_states)

    # Check if trend overlay is enabled
    if not enable_trend_overlay:
        return base_signals

    enable_cfg = getattr(cfg, "ENABLE_TREND_OVERLAY", True)
    if not enable_cfg:
        return base_signals

    # Compute trend overlay
    logger.info("Computing trend overlay for %d pairs...", len(pairs))
    trend_score_df, suppress_long_df, suppress_short_df = compute_trend_overlay_for_signals(
        test_df, pairs, warm_states
    )

    # Return enhanced SignalFrames
    return SignalFrames(
        z_score=base_signals.z_score,
        spread_volatility=base_signals.spread_volatility,
        beta=base_signals.beta,
        trend_score=trend_score_df,
        suppress_long=suppress_long_df,
        suppress_short=suppress_short_df,
    )


def save_signals_parquet(run_dir: Path, signals: SignalFrames) -> Path:
    """
    Optional: save signals to results/run_id/signals.parquet
    """
    run_dir = Path(run_dir)

    # Expect cfg.get_run_paths(run_dir) from your config plan
    if not hasattr(cfg, "get_run_paths"):
        raise AttributeError("config_backtest.py missing get_run_paths(run_dir).")

    paths = cfg.get_run_paths(run_dir)
    out_path = paths.get("signals", run_dir / "signals.parquet")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Store as a single Parquet with column groups by prefix
    frames = {
        "z_score": signals.z_score,
        "spread_volatility": signals.spread_volatility,
        "beta": signals.beta,
    }

    # Add trend overlay if present
    if signals.trend_score is not None and not signals.trend_score.empty:
        frames["trend_score"] = signals.trend_score
    if signals.suppress_long is not None and not signals.suppress_long.empty:
        frames["suppress_long"] = signals.suppress_long
    if signals.suppress_short is not None and not signals.suppress_short.empty:
        frames["suppress_short"] = signals.suppress_short

    df = pd.concat(frames, axis=1)
    df.to_parquet(out_path)
    logger.info("💾 Saved signals: %s", out_path)
    return out_path
