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


def _std_history(err_hist) -> float:
    # err_hist is a deque inside KalmanFilterRegime
    if err_hist is None or len(err_hist) == 0:
        return np.nan
    return float(np.std(np.asarray(err_hist, dtype="float64")))


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
            # Still advance the filter? Your model skips invalid inputs anyway.
            kf.update(y[t], x[t])
            continue

        # ---- Predict using state at t-1 ----
        # Observation matrix H = [price_x, 1]
        H = np.array([x[t], 1.0], dtype="float64")
        y_pred = float(np.dot(H, kf.x))
        err = float(y[t] - y_pred)

        # ---- Volatility from history up to t-1 (strictly causal) ----
        std_prev = _std_history(kf.error_history)
        if not np.isfinite(std_prev) or std_prev <= 1e-8:
            # Conservative fallback to avoid division explosions
            std_prev = 0.01

        z[t] = err / std_prev
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
    df = pd.concat(
        {
            "z_score": signals.z_score,
            "spread_volatility": signals.spread_volatility,
            "beta": signals.beta,
        },
        axis=1,
    )
    df.to_parquet(out_path)
    logger.info("💾 Saved signals: %s", out_path)
    return out_path
