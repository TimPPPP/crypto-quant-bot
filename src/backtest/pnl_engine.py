# src/backtest/pnl_engine.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.pnl_engine")


SlippageModel = Literal["fixed", "vol_adjusted"]
PnlMode = Literal["price", "log"]


@dataclass(frozen=True)
class PnlResult:
    """
    returns_matrix:
      index   -> time (t)
      columns -> pair_id (e.g., "ETH-BTC")
      values  -> realized return at exit timestamps (0 otherwise)
    """
    returns_matrix: pd.DataFrame
    trades_count: pd.Series  # number of completed trades per pair


def _parse_pair(pair: str, sep: str = "-") -> Tuple[str, str]:
    if sep not in pair:
        raise ValueError(f"Invalid pair '{pair}'. Expected like 'ETH-BTC'.")
    y, x = pair.split(sep, 1)
    y, x = y.strip(), x.strip()
    if not y or not x:
        raise ValueError(f"Invalid pair '{pair}'.")
    return y, x


def _safe_log_prices(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float64", copy=False)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0] = np.nan
    return np.log(arr)


def build_pair_matrices(
    test_df: pd.DataFrame,
    pairs: Iterable[str],
    *,
    pair_sep: str = "-",
    pnl_mode: PnlMode = "price",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build aligned Y and X matrices of shape (T, P) for numba.

    pnl_mode:
      - "price": use raw prices
      - "log":   use log prices (more consistent if your beta is learned on logs)
    """
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("pairs is empty")

    T = len(test_df)
    P = len(pair_list)

    y_mat = np.full((T, P), np.nan, dtype="float64")
    x_mat = np.full((T, P), np.nan, dtype="float64")

    for j, pair in enumerate(pair_list):
        coin_y, coin_x = _parse_pair(pair, sep=pair_sep)
        if coin_y not in test_df.columns or coin_x not in test_df.columns:
            raise KeyError(f"Missing columns for {pair}: need {coin_y}, {coin_x}")

        y = pd.to_numeric(test_df[coin_y], errors="coerce").to_numpy(dtype="float64")
        x = pd.to_numeric(test_df[coin_x], errors="coerce").to_numpy(dtype="float64")

        if pnl_mode == "log":
            y = _safe_log_prices(y)
            x = _safe_log_prices(x)

        y_mat[:, j] = y
        x_mat[:, j] = x

    return y_mat, x_mat, pair_list


def _require_aligned(df: pd.DataFrame, ref_index: pd.Index, ref_cols: pd.Index, name: str) -> pd.DataFrame:
    if df is None:
        raise ValueError(f"{name} is required.")
    if not df.index.equals(ref_index):
        raise ValueError(f"{name} index must match test_df index.")
    if not df.columns.equals(ref_cols):
        raise ValueError(f"{name} columns must match pairs list exactly (same order).")
    return df


# ----------------------------- NUMBA CORE ---------------------------------- #

@njit
def _pnl_state_machine_numba(
    y_mat: np.ndarray,              # (T, P)
    x_mat: np.ndarray,              # (T, P)
    beta_mat: np.ndarray,           # (T, P)
    z_mat: np.ndarray,              # (T, P)
    entry_mask: np.ndarray,         # (T, P) bool
    exit_mask: np.ndarray,          # (T, P) bool
    spread_vol_mat: np.ndarray,     # (T, P) float, can be NaN
    fee_rate: float,
    slippage_rate: float,
    slippage_model_id: int,         # 0=fixed, 1=vol_adjusted
    slippage_vol_mult: float,       # used only when vol_adjusted
    capital_per_pair: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truth engine: per-pair blocking state machine.

    Returns
    -------
    returns_mat : (T, P) float
        Non-zero only at exit timestamps (net return per trade).
    trades_count : (P,) int
        Completed trades per pair.
    """
    T, P = y_mat.shape
    returns_mat = np.zeros((T, P), dtype=np.float64)
    trades_count = np.zeros(P, dtype=np.int64)

    LOOKING = 0
    IN_TRADE = 1

    state = np.zeros(P, dtype=np.int64)  # start LOOKING
    entry_y = np.zeros(P, dtype=np.float64)
    entry_x = np.zeros(P, dtype=np.float64)
    beta_entry = np.zeros(P, dtype=np.float64)
    dir_entry = np.ones(P, dtype=np.float64)  # +1 long spread, -1 short spread

    # Track last finite prices per pair (for safe forced close at end)
    last_y = np.full(P, np.nan, dtype=np.float64)
    last_x = np.full(P, np.nan, dtype=np.float64)

    for t in range(T):
        for j in range(P):
            y = y_mat[t, j]
            x = x_mat[t, j]

            if np.isfinite(y) and np.isfinite(x):
                last_y[j] = y
                last_x[j] = x

            if state[j] == LOOKING:
                if entry_mask[t, j]:
                    # Require finite inputs at entry
                    b = beta_mat[t, j]
                    z = z_mat[t, j]
                    if not (np.isfinite(y) and np.isfinite(x) and np.isfinite(b) and np.isfinite(z)):
                        continue

                    entry_y[j] = y
                    entry_x[j] = x
                    beta_entry[j] = b

                    # Direction: z>0 => short spread => dir = -1
                    #            z<0 => long spread  => dir = +1
                    dir_entry[j] = -1.0 if z > 0.0 else 1.0

                    state[j] = IN_TRADE

            else:  # IN_TRADE
                # Exit if mask says so OR if last bar (forced close)
                do_exit = exit_mask[t, j] or (t == T - 1)

                if do_exit:
                    # Use current prices if finite; otherwise last finite
                    y_exit = y if np.isfinite(y) else last_y[j]
                    x_exit = x if np.isfinite(x) else last_x[j]

                    if not (np.isfinite(y_exit) and np.isfinite(x_exit)):
                        # cannot value exit; drop trade (conservative)
                        state[j] = LOOKING
                        continue

                    b = beta_entry[j]

                    # Base spread PnL (static beta locked at entry)
                    # Note: this matches your master plan formula:
                    #   pnl_spread = (Y_t - Y_entry) - beta*(X_t - X_entry)
                    pnl_spread = (y_exit - entry_y[j]) - b * (x_exit - entry_x[j])

                    # Apply direction:
                    #   if z_entry > 0 (short spread), profit = -pnl_spread
                    pnl = dir_entry[j] * pnl_spread

                    # Notional approximation (1 unit Y, beta units X)
                    notional = abs(entry_y[j]) + abs(b) * abs(entry_x[j])
                    if not np.isfinite(notional) or notional <= 0.0:
                        notional = 0.0

                    # Fees: 4 legs total
                    fees = notional * 4.0 * fee_rate

                    # Slippage: same 4-leg notion, optionally volatility-scaled
                    slip = slippage_rate
                    if slippage_model_id == 1:
                        v = spread_vol_mat[t, j]
                        if np.isfinite(v) and v > 0.0:
                            slip = slippage_rate * (1.0 + slippage_vol_mult * v)
                    slippage_cost = notional * 4.0 * slip

                    net_pnl = pnl - fees - slippage_cost

                    # Convert to return
                    if capital_per_pair > 0.0:
                        returns_mat[t, j] = net_pnl / capital_per_pair

                    trades_count[j] += 1
                    state[j] = LOOKING

    return returns_mat, trades_count


# --------------------------- PUBLIC WRAPPER -------------------------------- #

def run_pnl_engine(
    test_df: pd.DataFrame,
    pairs: List[str],
    *,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    beta: pd.DataFrame,
    z_score: pd.DataFrame,
    spread_volatility: Optional[pd.DataFrame] = None,
    pnl_mode: PnlMode = "price",
    pair_sep: str = "-",
    fee_rate: Optional[float] = None,
    slippage_rate: float = 0.0,
    slippage_model: SlippageModel = "fixed",
    slippage_vol_mult: float = 0.0,
    capital_per_pair: float = 1.0,
) -> PnlResult:
    """
    Step G: Truth engine PnL simulator.

    Inputs are all aligned on:
      - time index == test_df.index
      - columns == pairs (same order)

    Returns a returns_matrix with non-zero values only at exit timestamps.
    """
    if test_df.empty:
        raise ValueError("test_df is empty.")
    if not pairs:
        raise ValueError("pairs is empty.")

    # Build matrices for Y/X per pair
    y_mat, x_mat, pair_list = build_pair_matrices(test_df, pairs, pair_sep=pair_sep, pnl_mode=pnl_mode)
    pair_index = pd.Index(pair_list)

    # Enforce alignment of all signal frames
    entries = _require_aligned(entries, test_df.index, pair_index, "entries")
    exits = _require_aligned(exits, test_df.index, pair_index, "exits")
    beta = _require_aligned(beta, test_df.index, pair_index, "beta")
    z_score = _require_aligned(z_score, test_df.index, pair_index, "z_score")

    if spread_volatility is None:
        spread_volatility = pd.DataFrame(np.nan, index=test_df.index, columns=pair_index)
    else:
        spread_volatility = _require_aligned(spread_volatility, test_df.index, pair_index, "spread_volatility")

    fee_rate = float(fee_rate if fee_rate is not None else cfg.FEE_RATE)

    slippage_model_id = 0 if slippage_model == "fixed" else 1

    # Convert to numpy for numba
    entry_mask = entries.to_numpy(dtype=np.bool_)
    exit_mask = exits.to_numpy(dtype=np.bool_)
    beta_mat = beta.to_numpy(dtype=np.float64)
    z_mat = z_score.to_numpy(dtype=np.float64)
    vol_mat = spread_volatility.to_numpy(dtype=np.float64)

    returns_mat, trades_count = _pnl_state_machine_numba(
        y_mat=y_mat,
        x_mat=x_mat,
        beta_mat=beta_mat,
        z_mat=z_mat,
        entry_mask=entry_mask,
        exit_mask=exit_mask,
        spread_vol_mat=vol_mat,
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        slippage_model_id=int(slippage_model_id),
        slippage_vol_mult=float(slippage_vol_mult),
        capital_per_pair=float(capital_per_pair),
    )

    returns_df = pd.DataFrame(returns_mat, index=test_df.index, columns=pair_index)
    trades_series = pd.Series(trades_count, index=pair_index, name="trades_count")

    logger.info(
        "PnL engine complete: total_trades=%d, total_nonzero_returns=%d",
        int(trades_series.sum()),
        int(np.count_nonzero(returns_mat)),
    )

    return PnlResult(returns_matrix=returns_df, trades_count=trades_series)
