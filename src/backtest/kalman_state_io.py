# src/backtest/kalman_state_io.py

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg
from src.models.kalman import KalmanFilterRegime

logger = logging.getLogger("backtest.kalman_state_io")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Warm-start validation thresholds
VALID_BETA_MIN = 0.01  # Reject if |beta| < 0.01 (essentially zero hedge ratio)
VALID_BETA_MAX = 50.0  # Reject if |beta| > 50.0 (extreme leverage)
MAX_P_DIAGONAL = 5.0   # Reject if P[0,0] > 5.0 (beta highly uncertain)


def _parse_pair(pair: str) -> Tuple[str, str]:
    """
    Parse "Y-X" format into (coin_y, coin_x).

    Keeps your existing naming scheme consistent (ETH-BTC).
    """
    if cfg.PAIR_ID_SEPARATOR not in pair:
        raise ValueError(f"Invalid pair format '{pair}'. Expected like 'ETH-BTC'.")
    coin_y, coin_x = pair.split(cfg.PAIR_ID_SEPARATOR, 1)
    coin_y, coin_x = coin_y.strip(), coin_x.strip()
    if not coin_y or not coin_x:
        raise ValueError(f"Invalid pair format '{pair}'.")
    return coin_y, coin_x


def _log_prices(series: pd.Series) -> np.ndarray:
    """
    Convert price series to log prices safely.

    - Coerces to float
    - Non-positive -> NaN (log invalid)
    - Returns numpy float64 array
    """
    arr = pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()
    arr[arr <= 0] = np.nan
    return np.log(arr)


def _validate_warm_state(state: Dict, pair: str) -> Tuple[bool, str]:
    """
    Validate that a warm-start state is usable for trading.

    Checks:
    1. Beta is within reasonable range [VALID_BETA_MIN, VALID_BETA_MAX]
    2. Kalman filter has converged (P matrix diagonal is small)

    Returns
    -------
    (is_valid, reason) : Tuple[bool, str]
        is_valid: True if state passes all checks
        reason: Empty string if valid, or rejection reason if invalid
    """
    # Check beta exists and is valid
    x = state.get("x")
    if x is None or len(x) == 0:
        return False, "No beta estimate"

    beta = float(x[0])

    if not np.isfinite(beta):
        return False, f"Beta is not finite: {beta}"

    if abs(beta) < VALID_BETA_MIN:
        return False, f"Beta too small: |{beta:.3f}| < {VALID_BETA_MIN}"

    if abs(beta) > VALID_BETA_MAX:
        return False, f"Beta too large: |{beta:.3f}| > {VALID_BETA_MAX}"

    # Check P matrix convergence
    P = state.get("P")
    if P is not None:
        try:
            P_arr = np.asarray(P)
            if P_arr.size > 0:
                p_diag = float(P_arr[0, 0])
                if not np.isfinite(p_diag):
                    return False, f"P matrix not finite"
                if p_diag > MAX_P_DIAGONAL:
                    return False, f"Beta not converged: P[0,0]={p_diag:.3f} > {MAX_P_DIAGONAL}"
        except (IndexError, TypeError, ValueError):
            # If P matrix can't be read properly, skip this check
            pass

    return True, ""


def _get_bars_per_day() -> int:
    """Get number of bars per day based on configured timeframe."""
    freq = getattr(cfg, "SIGNAL_TIMEFRAME", "15min")
    freq_to_bars = {
        "1min": 1440,
        "5min": 288,
        "15min": 96,
        "30min": 48,
        "1h": 24,
        "4h": 6,
    }
    return freq_to_bars.get(freq, 96)  # Default to 15min


def compute_warm_states(
    train_df: pd.DataFrame,
    valid_pairs: Iterable[str],
    *,
    pair_half_lives: Optional[Dict[str, float]] = None,
    delta: Optional[float] = None,
    R: Optional[float] = None,
    rolling_window: Optional[int] = None,
    entry_z_threshold: Optional[float] = None,
    min_spread_pct: Optional[float] = None,
    vol_window_hl_mult: Optional[float] = None,
    warmup_days: Optional[int] = None,
    warmup_half_life_mult: Optional[float] = None,
    warmup_min_days: int = 7,
    warmup_max_days: int = 45,
) -> Dict[str, Dict]:
    """
    Run Kalman through train_df for each pair and return persisted states.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training price data with coins as columns
    valid_pairs : Iterable[str]
        List of pair identifiers like "ETH-BTC"
    pair_half_lives : dict, optional
        Per-pair half-life in bars, e.g., {"ETH-BTC": 500.0}.
        If provided, the volatility window will be scaled from half-life
        for more consistent z-score behavior.
    delta : float, optional
        Kalman process noise (default from config)
    R : float, optional
        Kalman measurement noise (default from config)
    rolling_window : int, optional
        Explicit rolling window override. If None and pair_half_lives provided,
        window is computed from half_life * vol_window_hl_mult.
    entry_z_threshold : float, optional
        Z-score entry threshold
    min_spread_pct : float, optional
        Minimum spread percentage for signals
    vol_window_hl_mult : float, optional
        Multiplier for half_life -> vol_window scaling (default from config)
    warmup_days : int, optional
        If provided, only use the last `warmup_days` of train_df for warming.
        This prevents stale data from polluting the beta estimate.
    warmup_half_life_mult : float, optional
        If provided (and pair_half_lives available), compute warmup window as:
        warmup = half_life * warmup_half_life_mult.
        Takes precedence over warmup_days for pairs with known half-life.
    warmup_min_days : int
        Floor for half-life-based warmup calculation.
    warmup_max_days : int
        Ceiling for half-life-based warmup calculation.

    Returns
    -------
    states : dict
        { "ETH-BTC": state_dict, ... }
        where state_dict contains x, P, error_history, rolling_window, half_life_bars, etc.
        (via KalmanFilterRegime.get_state_dict()).
    """
    if train_df.empty:
        raise ValueError("train_df is empty; cannot warm-start Kalman.")

    states: Dict[str, Dict] = {}
    pairs = list(valid_pairs)
    pair_half_lives = pair_half_lives or {}

    # Get vol window scaling multiplier from config if not specified
    hl_mult = vol_window_hl_mult if vol_window_hl_mult is not None else getattr(
        cfg, 'VOL_WINDOW_HALF_LIFE_MULT', 1.0
    )

    logger.info("Warming Kalman on train set for %d pairs...", len(pairs))
    if pair_half_lives:
        logger.info("Using half-life scaled vol windows (mult=%.2f)", hl_mult)
    if warmup_days is not None or warmup_half_life_mult is not None:
        logger.info(
            "Using warmup window: days=%s, half_life_mult=%s, min=%d, max=%d",
            warmup_days,
            warmup_half_life_mult,
            warmup_min_days,
            warmup_max_days,
        )

    bars_per_day = _get_bars_per_day()

    for pair in pairs:
        coin_y, coin_x = _parse_pair(pair)

        if coin_y not in train_df.columns or coin_x not in train_df.columns:
            logger.warning("Skipping %s: missing %s or %s in train_df columns.", pair, coin_y, coin_x)
            continue

        # Determine warmup window for this pair
        warmup_df = train_df  # Default: use full training data

        if warmup_days is not None or warmup_half_life_mult is not None:
            pair_warmup_days = warmup_days  # Default to explicit warmup_days

            # If half-life-based warmup is enabled and we have half-life for this pair
            if warmup_half_life_mult is not None and pair in pair_half_lives:
                half_life_bars = pair_half_lives[pair]
                hl_days = half_life_bars / bars_per_day
                pair_warmup_days = int(hl_days * warmup_half_life_mult)
                pair_warmup_days = max(warmup_min_days, min(warmup_max_days, pair_warmup_days))
                logger.debug(
                    "Pair %s: half_life=%.1f bars (%.1f days), warmup=%d days",
                    pair,
                    half_life_bars,
                    hl_days,
                    pair_warmup_days,
                )

            if pair_warmup_days is not None:
                warmup_start = train_df.index.max() - pd.Timedelta(days=pair_warmup_days)
                warmup_df = train_df[train_df.index >= warmup_start]

                min_warmup_rows = 100  # Need at least 100 bars for stable Kalman warmup
                if len(warmup_df) < min_warmup_rows:
                    logger.warning(
                        "Pair %s: warmup window too small (%d rows), using full train data.",
                        pair,
                        len(warmup_df),
                    )
                    warmup_df = train_df

        y = _log_prices(warmup_df[coin_y])
        x = _log_prices(warmup_df[coin_x])

        # Get per-pair half-life if available
        half_life = pair_half_lives.get(pair, None)

        # Determine rolling window:
        # 1. If explicit rolling_window given, use it
        # 2. If half_life available, compute from half_life
        # 3. Otherwise use config default
        if rolling_window is not None:
            rw = rolling_window
        elif half_life is not None:
            # Scale window from half-life
            rw = None  # Let KalmanFilterRegime compute it
        else:
            rw = cfg.LOOKBACK_WINDOW

        # Initialize filter using config defaults unless overridden
        kf = KalmanFilterRegime(
            delta=delta if delta is not None else cfg.KALMAN_DELTA,
            R=R if R is not None else cfg.KALMAN_R,
            rolling_window=rw,
            entry_z_threshold=entry_z_threshold if entry_z_threshold is not None else cfg.ENTRY_Z,
            min_spread_pct=min_spread_pct if min_spread_pct is not None else 0.0,
            half_life_bars=half_life,
            vol_window_hl_mult=hl_mult,
        )

        # Run through train window
        n = len(y)
        for i in range(n):
            # update() already skips NaN/Inf safely
            kf.update(price_y=y[i], price_x=x[i])

        # Persist full state (includes x, P, error_history, half_life_bars)
        st = kf.get_state_dict()

        # Quick sanity metadata (optional but useful)
        st["pair"] = pair
        st["coin_y"] = coin_y
        st["coin_x"] = coin_x
        st["latest_beta"] = float(st["x"][0]) if "x" in st and len(st["x"]) > 0 else None

        # Validate warm-start quality (Priority 3: reject unconverged filters)
        is_valid, reject_reason = _validate_warm_state(st, pair)
        if not is_valid:
            logger.warning("Rejecting %s: %s", pair, reject_reason)
            continue

        states[pair] = st

    if not states:
        raise ValueError("No warm states produced. Check valid_pairs and train_df columns.")

    logger.info("✅ Warm-start states computed for %d pairs.", len(states))
    return states


def save_warm_states(run_dir: Path, states: Dict[str, Dict]) -> Path:
    """
    Save warm-start states to results/run_id/warm_states.pkl.
    """
    run_dir = Path(run_dir)
    paths = cfg.get_run_paths(run_dir)
    out_path = paths["warm_states"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("💾 Saved warm states: %s", out_path)
    return out_path


def load_warm_states(path: Path) -> Dict[str, Dict]:
    """
    Load previously saved warm-start states.
    """
    path = Path(path)
    with path.open("rb") as f:
        states = pickle.load(f)
    if not isinstance(states, dict):
        raise TypeError("warm_states.pkl did not contain a dict.")
    return states
