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


def compute_warm_states(
    train_df: pd.DataFrame,
    valid_pairs: Iterable[str],
    *,
    delta: Optional[float] = None,
    R: Optional[float] = None,
    rolling_window: Optional[int] = None,
    entry_z_threshold: Optional[float] = None,
    min_spread_pct: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Run Kalman through train_df for each pair and return persisted states.

    Returns
    -------
    states : dict
        { "ETH-BTC": state_dict, ... }
        where state_dict contains x, P, error_history, rolling_window, etc.
        (via KalmanFilterRegime.get_state_dict()).
    """
    if train_df.empty:
        raise ValueError("train_df is empty; cannot warm-start Kalman.")

    states: Dict[str, Dict] = {}
    pairs = list(valid_pairs)

    logger.info("Warming Kalman on train set for %d pairs...", len(pairs))

    for pair in pairs:
        coin_y, coin_x = _parse_pair(pair)

        if coin_y not in train_df.columns or coin_x not in train_df.columns:
            logger.warning("Skipping %s: missing %s or %s in train_df columns.", pair, coin_y, coin_x)
            continue

        y = _log_prices(train_df[coin_y])
        x = _log_prices(train_df[coin_x])

        # Initialize filter using config defaults unless overridden
        kf = KalmanFilterRegime(
            delta=delta if delta is not None else cfg.KALMAN_DELTA,
            R=R if R is not None else cfg.KALMAN_R,
            rolling_window=rolling_window if rolling_window is not None else cfg.LOOKBACK_WINDOW,
            entry_z_threshold=entry_z_threshold if entry_z_threshold is not None else cfg.ENTRY_Z,
            min_spread_pct=min_spread_pct if min_spread_pct is not None else 0.0,  # backtest doesn't need live signal gate
        )

        # Run through train window
        n = len(y)
        for i in range(n):
            # update() already skips NaN/Inf safely
            kf.update(price_y=y[i], price_x=x[i])

        # Persist full state (includes x, P, error_history)
        st = kf.get_state_dict()

        # Quick sanity metadata (optional but useful)
        st["pair"] = pair
        st["coin_y"] = coin_y
        st["coin_x"] = coin_x
        st["latest_beta"] = float(st["x"][0]) if "x" in st and len(st["x"]) > 0 else None

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
