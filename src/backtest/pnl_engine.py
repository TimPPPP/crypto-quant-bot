# src/backtest/pnl_engine.py
"""
PnL Engine for Pairs Trading Backtest

Numba-accelerated state machine that simulates trade execution,
tracking entries, exits, fees, slippage, and position sizing.

Supports:
- Normalized notional sizing (Problem #5 fix)
- Time-based stops (Problem #7 fix)
- Conviction-weighted position sizing
- Advanced position sizing (non-linear, correlation, volatility targeting)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.pnl_engine")


# =============================================================================
# FEE MODEL HELPERS
# =============================================================================

def compute_effective_fee_rate(
    fee_model: str = "taker_only",
    taker_rate: Optional[float] = None,
    maker_rate: Optional[float] = None,
    maker_fill_prob: Optional[float] = None,
) -> float:
    """
    Compute blended fee rate based on fee model.

    Parameters
    ----------
    fee_model : str
        One of "taker_only", "maker_only", "maker_taker_mix"
    taker_rate : float
        Taker fee rate (default from config)
    maker_rate : float
        Maker fee rate (default from config)
    maker_fill_prob : float
        Probability of filling at maker rate (default from config)

    Returns
    -------
    float
        Effective fee rate to use in PnL calculations
    """
    if taker_rate is None:
        taker_rate = getattr(cfg, "TAKER_FEE_RATE", 0.0005)
    if maker_rate is None:
        maker_rate = getattr(cfg, "MAKER_FEE_RATE", 0.0002)
    if maker_fill_prob is None:
        maker_fill_prob = getattr(cfg, "MAKER_FILL_PROBABILITY", 0.70)

    if fee_model == "taker_only":
        return taker_rate
    elif fee_model == "maker_only":
        return maker_rate
    elif fee_model == "maker_taker_mix":
        return maker_fill_prob * maker_rate + (1 - maker_fill_prob) * taker_rate
    else:
        logger.warning("Unknown fee model '%s', defaulting to taker_only", fee_model)
        return taker_rate


# =============================================================================
# PAIR QUALITY KILL SWITCH
# =============================================================================

@dataclass
class PairQualityTracker:
    """
    Tracks per-pair quality metrics for mid-window retirement (kill switch).

    This class evaluates pair performance after trades complete and can
    recommend retirement of poorly performing pairs.

    Example: MERL-SOL with 8/15 stop losses (-115% contribution) should
    be killed after ~6 trades showing >50% stop rate.
    """
    trade_count: Dict[str, int]
    win_count: Dict[str, int]
    stop_loss_count: Dict[str, int]
    cumulative_pnl: Dict[str, float]
    retired_pairs: set  # Set[str]

    @classmethod
    def create(cls, pairs: List[str]) -> "PairQualityTracker":
        """Create a new tracker for the given pairs."""
        return cls(
            trade_count={p: 0 for p in pairs},
            win_count={p: 0 for p in pairs},
            stop_loss_count={p: 0 for p in pairs},
            cumulative_pnl={p: 0.0 for p in pairs},
            retired_pairs=set(),
        )

    def record_trade(self, pair: str, pnl: float, exit_reason: int) -> None:
        """
        Record a completed trade for quality tracking.

        Parameters
        ----------
        pair : str
            Pair identifier (e.g., "ETH-BTC")
        pnl : float
            Trade P&L (net return)
        exit_reason : int
            Exit reason code (1=signal, 2=time_stop, 3=stop_loss, 4=forced)
        """
        if pair in self.retired_pairs:
            return
        if pair not in self.trade_count:
            return

        self.trade_count[pair] += 1
        self.cumulative_pnl[pair] += pnl
        if pnl > 0:
            self.win_count[pair] += 1
        if exit_reason == EXIT_REASON_STOP_LOSS:
            self.stop_loss_count[pair] += 1

    def evaluate_pair(
        self,
        pair: str,
        min_trades: int = 6,
        max_stop_rate: float = 0.50,
        min_expectancy: float = 0.0,
        min_win_rate: float = 0.25,
    ) -> bool:
        """
        Evaluate if a pair should be retired (killed).

        Returns True if the pair should be retired.

        Parameters
        ----------
        pair : str
            Pair to evaluate
        min_trades : int
            Minimum trades before evaluation (avoid early decisions)
        max_stop_rate : float
            Kill if stop_loss_rate > this threshold
        min_expectancy : float
            Kill if expectancy (avg pnl per trade) < this
        min_win_rate : float
            Kill if win_rate < this threshold

        Returns
        -------
        bool
            True if pair should be retired
        """
        if pair in self.retired_pairs:
            return False  # Already retired

        n = self.trade_count.get(pair, 0)
        if n < min_trades:
            return False  # Not enough data to decide

        stop_rate = self.stop_loss_count.get(pair, 0) / n
        win_rate = self.win_count.get(pair, 0) / n
        expectancy = self.cumulative_pnl.get(pair, 0.0) / n

        # Any of these conditions triggers retirement
        if stop_rate > max_stop_rate:
            logger.info(
                "Kill switch: %s retired due to high stop rate %.1f%% > %.1f%%",
                pair, stop_rate * 100, max_stop_rate * 100
            )
            return True
        if expectancy < min_expectancy:
            logger.info(
                "Kill switch: %s retired due to negative expectancy %.4f < %.4f",
                pair, expectancy, min_expectancy
            )
            return True
        if win_rate < min_win_rate:
            logger.info(
                "Kill switch: %s retired due to low win rate %.1f%% < %.1f%%",
                pair, win_rate * 100, min_win_rate * 100
            )
            return True

        return False

    def retire_pair(self, pair: str) -> None:
        """Mark a pair as retired (no more trades allowed)."""
        self.retired_pairs.add(pair)

    def get_stats(self, pair: str) -> Dict[str, float]:
        """Get quality statistics for a pair."""
        n = self.trade_count.get(pair, 0)
        if n == 0:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "stop_rate": 0.0,
                "expectancy": 0.0,
            }
        return {
            "trade_count": n,
            "win_rate": self.win_count.get(pair, 0) / n,
            "stop_rate": self.stop_loss_count.get(pair, 0) / n,
            "expectancy": self.cumulative_pnl.get(pair, 0.0) / n,
        }


def build_quality_tracker_from_pnl_result(
    pnl_result: "PnlResult",
) -> PairQualityTracker:
    """
    Build a PairQualityTracker from completed PnL result.

    This allows evaluating pair quality after a backtest window
    to inform pair selection for subsequent windows.
    """
    pairs = list(pnl_result.returns_matrix.columns)
    tracker = PairQualityTracker.create(pairs)

    returns_mat = pnl_result.returns_matrix.values
    exit_reason_mat = pnl_result.exit_reason_matrix.values

    for j, pair in enumerate(pairs):
        # Find all exit timestamps (non-zero exit reasons)
        exit_mask = exit_reason_mat[:, j] > 0
        exit_indices = np.where(exit_mask)[0]

        for t in exit_indices:
            pnl = returns_mat[t, j]
            exit_reason = int(exit_reason_mat[t, j])
            tracker.record_trade(pair, pnl, exit_reason)

    return tracker


SlippageModel = Literal["fixed", "vol_adjusted"]
PnlMode = Literal["price", "log"]


# Exit reason codes for Numba compatibility
EXIT_REASON_NONE = 0
EXIT_REASON_SIGNAL = 1
EXIT_REASON_TIME_STOP = 2
EXIT_REASON_STOP_LOSS = 3
EXIT_REASON_FORCED = 4  # End of backtest


@dataclass(frozen=True)
class PnlResult:
    """
    returns_matrix:
      index   -> time (t)
      columns -> pair_id (e.g., "ETH-BTC")
      values  -> realized return at exit timestamps (0 otherwise)

    mae_matrix:
      Max Adverse Excursion per trade (worst unrealized loss before exit).
      Non-zero only at exit timestamps. Positive value = how bad it got.

    mfe_matrix:
      Max Favorable Excursion per trade (best unrealized gain before exit).
      Non-zero only at exit timestamps. Positive value = peak profit.

    exit_reason_matrix:
      Exit reason code at exit timestamps:
        0 = no exit, 1 = signal, 2 = time_stop, 3 = stop_loss, 4 = forced

    P&L Attribution (Phase 0):
      gross_pnl_matrix: Gross spread P&L (price movement only, before costs)
      fees_matrix: Total fees paid (4 legs × fee_rate × position_weight)
      slippage_matrix: Total slippage cost (4 legs × slip × position_weight)
      hold_bars_matrix: Number of bars held for each trade

    Funding Cost Attribution:
      funding_costs_matrix: Funding costs per bar while in position (per pair)
      total_funding: Sum of all funding costs across all pairs and trades
    """
    returns_matrix: pd.DataFrame
    trades_count: pd.Series      # number of completed trades per pair
    time_stop_count: pd.Series   # number of trades closed due to time stop per pair
    stop_loss_count: pd.Series   # number of trades closed due to PnL stop per pair
    mae_matrix: pd.DataFrame     # max adverse excursion per trade
    mfe_matrix: pd.DataFrame     # max favorable excursion per trade
    exit_reason_matrix: pd.DataFrame  # exit reason codes per trade
    # P&L Attribution components (Phase 0)
    gross_pnl_matrix: pd.DataFrame   # gross spread P&L before costs
    fees_matrix: pd.DataFrame        # total fees paid
    slippage_matrix: pd.DataFrame    # total slippage cost
    hold_bars_matrix: pd.DataFrame   # bars held per trade
    # Funding Cost Attribution
    funding_costs_matrix: Optional[pd.DataFrame] = None  # funding costs per bar while in position
    total_funding: float = 0.0  # sum of all funding costs


def compute_time_stops_from_half_life(
    pairs: List[str],
    pair_half_lives: Dict[str, float],
    time_stop_mult: Optional[float] = None,
    default_time_stop_bars: Optional[int] = None,
) -> Dict[str, int]:
    """
    Compute per-pair time stop (max hold bars) from half-lives.

    This connects Problem #7 (time stops) with half-life data from pair selection.

    Parameters
    ----------
    pairs : list of str
        List of pair identifiers
    pair_half_lives : dict
        Per-pair half-life in bars, e.g., {"ETH-BTC": 500.0}
    time_stop_mult : float, optional
        Multiplier for half_life -> time_stop. Default from config.
        E.g., 3.0 means exit after 3× half-life without reversion.
    default_time_stop_bars : int, optional
        Default time stop for pairs without half-life data.

    Returns
    -------
    dict
        Per-pair time stop in bars, e.g., {"ETH-BTC": 1500}
    """
    if time_stop_mult is None:
        time_stop_mult = getattr(cfg, 'TIME_STOP_HALF_LIFE_MULT', 3.0)
    if default_time_stop_bars is None:
        default_time_stop_bars = getattr(cfg, 'DEFAULT_TIME_STOP_BARS', 1440)

    result = {}
    for pair in pairs:
        if pair in pair_half_lives:
            hl = pair_half_lives[pair]
            result[pair] = int(hl * time_stop_mult)
        else:
            result[pair] = default_time_stop_bars

    return result


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
    score_mat: np.ndarray,          # (T, P) float, signal score for conviction sizing
    size_mult_mat: np.ndarray,      # (T, P) float, pre-computed position size multipliers
    fee_rate: float,
    slippage_rate: float,
    slippage_model_id: int,         # 0=fixed, 1=vol_adjusted
    slippage_vol_mult: float,       # used only when vol_adjusted
    capital_per_pair: float,
    max_trades_per_pair: int,       # 0 = unlimited
    normalize_notional: bool,       # if True, use beta-normalized weights
    use_log_prices: bool,           # if True, y/x are log-prices
    pair_coin1_idx: np.ndarray,     # (P,) coin index for leg Y
    pair_coin2_idx: np.ndarray,     # (P,) coin index for leg X
    n_coins: int,
    max_positions_total: int,       # 0 = unlimited
    max_positions_per_coin: int,    # 0 = unlimited
    stop_loss_pct: float,           # 0 = disabled
    max_hold_bars_arr: np.ndarray,  # (P,) per-pair time stop in bars, 0 = disabled
    conviction_min: float,          # min conviction size (e.g., 0.5)
    conviction_max: float,          # max conviction size (e.g., 1.0)
    min_signal_score: float,        # threshold for normalization (e.g., 0.65)
    use_advanced_sizing: bool,      # if True, use size_mult_mat directly
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Truth engine: per-pair blocking state machine.

    Problem #5 Fix: When normalize_notional=True, positions are sized so that
    total gross exposure = 1.0 regardless of beta:
        w_Y = 1 / (1 + |beta|)
        w_X = |beta| / (1 + |beta|)

    Problem #7 Fix: Time-based stop via max_hold_bars_arr. If a trade is held
    for longer than max_hold_bars without reverting, it's force-closed.
    This prevents holding "dead" trades through regime changes.

    Returns
    -------
    returns_mat : (T, P) float
        Non-zero only at exit timestamps (net return per trade).
    trades_count : (P,) int
        Completed trades per pair.
    time_stop_count : (P,) int
        Number of trades closed due to time stop per pair.
    stop_loss_count : (P,) int
        Number of trades closed due to PnL stop per pair.
    mae_mat : (T, P) float
        Max adverse excursion at exit timestamps (positive = worst loss).
    mfe_mat : (T, P) float
        Max favorable excursion at exit timestamps (positive = best gain).
    exit_reason_mat : (T, P) int
        Exit reason codes: 0=none, 1=signal, 2=time_stop, 3=stop_loss, 4=forced.
    gross_pnl_mat : (T, P) float
        Gross spread P&L before costs (for attribution).
    fees_mat : (T, P) float
        Total fees paid per trade (for attribution).
    slippage_mat : (T, P) float
        Total slippage cost per trade (for attribution).
    hold_bars_mat : (T, P) int
        Number of bars held per trade (for attribution).
    """
    T, P = y_mat.shape
    returns_mat = np.zeros((T, P), dtype=np.float64)
    trades_count = np.zeros(P, dtype=np.int64)
    time_stop_count = np.zeros(P, dtype=np.int64)  # Track time stop exits
    stop_loss_count = np.zeros(P, dtype=np.int64)  # Track PnL stop exits

    # MAE/MFE tracking matrices (for ML risk prediction)
    mae_mat = np.zeros((T, P), dtype=np.float64)  # Max adverse excursion at exit
    mfe_mat = np.zeros((T, P), dtype=np.float64)  # Max favorable excursion at exit
    exit_reason_mat = np.zeros((T, P), dtype=np.int64)  # Exit reason codes

    # P&L Attribution matrices (Phase 0)
    gross_pnl_mat = np.zeros((T, P), dtype=np.float64)  # Gross P&L before costs
    fees_mat = np.zeros((T, P), dtype=np.float64)       # Total fees paid
    slippage_mat = np.zeros((T, P), dtype=np.float64)   # Total slippage cost
    hold_bars_mat = np.zeros((T, P), dtype=np.int64)    # Bars held per trade

    LOOKING = 0
    IN_TRADE = 1

    # Exit reason codes (must match constants defined above)
    EXIT_SIGNAL = 1
    EXIT_TIME = 2
    EXIT_STOP = 3
    EXIT_FORCED = 4

    state = np.zeros(P, dtype=np.int64)  # start LOOKING
    entry_y = np.zeros(P, dtype=np.float64)
    entry_x = np.zeros(P, dtype=np.float64)
    beta_entry = np.zeros(P, dtype=np.float64)
    dir_entry = np.ones(P, dtype=np.float64)  # +1 long spread, -1 short spread
    entry_bar = np.zeros(P, dtype=np.int64)  # Track entry time for time stop
    entry_score = np.zeros(P, dtype=np.float64)  # Track signal score at entry for conviction sizing
    entry_size_mult = np.ones(P, dtype=np.float64)  # Track advanced position size at entry

    # Running MAE/MFE tracking per pair (reset at each entry)
    running_mae = np.zeros(P, dtype=np.float64)  # Worst unrealized PnL (most negative)
    running_mfe = np.zeros(P, dtype=np.float64)  # Best unrealized PnL (most positive)
    entry_cost_paid = np.zeros(P, dtype=np.float64)  # Entry transaction cost for MAE calc

    # Portfolio-level exposure tracking
    active_positions = 0
    coin_counts = np.zeros(n_coins, dtype=np.int64)

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
                    # Skip if max trades per pair limit reached
                    if max_trades_per_pair > 0 and trades_count[j] >= max_trades_per_pair:
                        continue

                    # Portfolio constraints
                    if max_positions_total > 0 and active_positions >= max_positions_total:
                        continue
                    c1 = pair_coin1_idx[j]
                    c2 = pair_coin2_idx[j]
                    if max_positions_per_coin > 0:
                        if coin_counts[c1] >= max_positions_per_coin or coin_counts[c2] >= max_positions_per_coin:
                            continue

                    # Require finite inputs at entry
                    b = beta_mat[t, j]
                    z = z_mat[t, j]
                    if not (np.isfinite(y) and np.isfinite(x) and np.isfinite(b) and np.isfinite(z)):
                        continue

                    entry_y[j] = y
                    entry_x[j] = x
                    beta_entry[j] = b
                    entry_bar[j] = t  # Record entry time for time stop

                    # Store signal score at entry for conviction sizing
                    score = score_mat[t, j]
                    entry_score[j] = score if np.isfinite(score) else min_signal_score

                    # Store advanced position size multiplier at entry
                    size_mult = size_mult_mat[t, j]
                    entry_size_mult[j] = size_mult if np.isfinite(size_mult) else 1.0

                    # Direction: z>0 => short spread => dir = -1
                    #            z<0 => long spread  => dir = +1
                    dir_entry[j] = -1.0 if z > 0.0 else 1.0

                    # Reset MAE/MFE tracking for this trade
                    # Entry cost = 2 legs × (fee + slippage)
                    entry_slip = slippage_rate
                    if slippage_model_id == 1:
                        v = spread_vol_mat[t, j]
                        if np.isfinite(v) and v > 0.0:
                            entry_slip = slippage_rate * (1.0 + slippage_vol_mult * v)
                    entry_cost_paid[j] = 2.0 * (fee_rate + entry_slip)
                    running_mae[j] = -entry_cost_paid[j]  # Start at entry cost (we're already down)
                    running_mfe[j] = -entry_cost_paid[j]  # Start at entry cost

                    state[j] = IN_TRADE
                    active_positions += 1
                    coin_counts[c1] += 1
                    coin_counts[c2] += 1

            else:  # IN_TRADE
                # Update running MAE/MFE with current unrealized PnL
                y_cur = y if np.isfinite(y) else last_y[j]
                x_cur = x if np.isfinite(x) else last_x[j]
                if np.isfinite(y_cur) and np.isfinite(x_cur):
                    b_entry = beta_entry[j]
                    abs_b = abs(b_entry)
                    if normalize_notional:
                        denom = 1.0 + abs_b
                        w_y = 1.0 / denom
                        w_x = abs_b / denom
                        if use_log_prices:
                            ret_y = y_cur - entry_y[j]
                            ret_x = x_cur - entry_x[j]
                        else:
                            ret_y = (y_cur - entry_y[j]) / abs(entry_y[j]) if abs(entry_y[j]) > 1e-10 else 0.0
                            ret_x = (x_cur - entry_x[j]) / abs(entry_x[j]) if abs(entry_x[j]) > 1e-10 else 0.0
                        spread_ret = w_y * ret_y - w_x * ret_x
                        unrealized_pnl = dir_entry[j] * spread_ret - entry_cost_paid[j]
                    else:
                        pnl_spread = (y_cur - entry_y[j]) - b_entry * (x_cur - entry_x[j])
                        unrealized_pnl = dir_entry[j] * pnl_spread
                        if capital_per_pair > 0.0:
                            unrealized_pnl = unrealized_pnl / capital_per_pair
                        unrealized_pnl = unrealized_pnl - entry_cost_paid[j]

                    # Update running MAE (worst point) and MFE (best point)
                    if unrealized_pnl < running_mae[j]:
                        running_mae[j] = unrealized_pnl
                    if unrealized_pnl > running_mfe[j]:
                        running_mfe[j] = unrealized_pnl

                # Check time stop (Problem #7 fix)
                bars_held = t - entry_bar[j]
                max_hold = max_hold_bars_arr[j]
                time_stop_hit = (max_hold > 0) and (bars_held >= max_hold)

                # Mark-to-market stop loss (institutional risk control)
                stop_loss_hit = False
                if stop_loss_pct > 0.0:
                    y_cur = y if np.isfinite(y) else last_y[j]
                    x_cur = x if np.isfinite(x) else last_x[j]
                    if np.isfinite(y_cur) and np.isfinite(x_cur):
                        b_entry = beta_entry[j]
                        abs_b_entry = abs(b_entry)
                        if normalize_notional:
                            denom = 1.0 + abs_b_entry
                            w_y = 1.0 / denom
                            w_x = abs_b_entry / denom
                            if use_log_prices:
                                ret_y = y_cur - entry_y[j]
                                ret_x = x_cur - entry_x[j]
                            else:
                                ret_y = (y_cur - entry_y[j]) / abs(entry_y[j]) if abs(entry_y[j]) > 1e-10 else 0.0
                                ret_x = (x_cur - entry_x[j]) / abs(entry_x[j]) if abs(entry_x[j]) > 1e-10 else 0.0
                            spread_ret = w_y * ret_y - w_x * ret_x
                            pnl_mtm = dir_entry[j] * spread_ret
                            # MTM costs: 2× round-trip (entry + exit) on normalized notional
                            fees_mtm = 2.0 * fee_rate
                            slip_mtm = slippage_rate
                            if slippage_model_id == 1:
                                v = spread_vol_mat[t, j]
                                if np.isfinite(v) and v > 0.0:
                                    slip_mtm = slippage_rate * (1.0 + slippage_vol_mult * v)
                            slippage_cost = 2.0 * slip_mtm
                            net_mtm = pnl_mtm - fees_mtm - slippage_cost
                        else:
                            pnl_spread = (y_cur - entry_y[j]) - b_entry * (x_cur - entry_x[j])
                            pnl_mtm = dir_entry[j] * pnl_spread
                            # Legacy notional: sum of both legs
                            notional = abs(entry_y[j]) + abs_b_entry * abs(entry_x[j])
                            # Round-trip: entry + exit = 2 × notional
                            fees_mtm = notional * 2.0 * fee_rate
                            slip_mtm = slippage_rate
                            if slippage_model_id == 1:
                                v = spread_vol_mat[t, j]
                                if np.isfinite(v) and v > 0.0:
                                    slip_mtm = slippage_rate * (1.0 + slippage_vol_mult * v)
                            slippage_cost = notional * 2.0 * slip_mtm
                            net_mtm = pnl_mtm - fees_mtm - slippage_cost
                            if capital_per_pair > 0.0:
                                net_mtm = net_mtm / capital_per_pair
                            else:
                                net_mtm = 0.0
                        if net_mtm <= -stop_loss_pct:
                            stop_loss_hit = True

                # Exit if: signal says so OR time stop OR stop loss OR last bar (forced close)
                do_exit = exit_mask[t, j] or time_stop_hit or stop_loss_hit or (t == T - 1)

                if do_exit:
                    # Use current prices if finite; otherwise last finite
                    y_exit = y if np.isfinite(y) else last_y[j]
                    x_exit = x if np.isfinite(x) else last_x[j]

                    if not (np.isfinite(y_exit) and np.isfinite(x_exit)):
                        # cannot value exit; drop trade (conservative)
                        state[j] = LOOKING
                        continue

                    b = beta_entry[j]
                    abs_b = abs(b)

                    if normalize_notional:
                        # Problem #5 Fix: Normalized position sizing
                        # Weights sum to 1.0 for fixed gross exposure
                        denom = 1.0 + abs_b
                        w_y = 1.0 / denom
                        w_x = abs_b / denom

                        # Per-leg returns
                        # For log prices: log(exit) - log(entry)
                        # For raw prices: (exit - entry) / entry
                        if use_log_prices:
                            ret_y = y_exit - entry_y[j]
                            ret_x = x_exit - entry_x[j]
                        else:
                            ret_y = (y_exit - entry_y[j]) / abs(entry_y[j]) if abs(entry_y[j]) > 1e-10 else 0.0
                            ret_x = (x_exit - entry_x[j]) / abs(entry_x[j]) if abs(entry_x[j]) > 1e-10 else 0.0

                        # Portfolio return: long Y (weight w_y), short X (weight w_x)
                        # For long spread (dir=+1): profit when Y↑ and X↓
                        #   portfolio_ret = w_y * ret_y - w_x * ret_x
                        # For short spread (dir=-1): profit when Y↓ and X↑
                        #   portfolio_ret = -w_y * ret_y + w_x * ret_x = -(w_y * ret_y - w_x * ret_x)
                        spread_ret = w_y * ret_y - w_x * ret_x
                        pnl = dir_entry[j] * spread_ret

                        # Position sizing: either use pre-computed advanced sizes or simple conviction scaling
                        if use_advanced_sizing:
                            # Use pre-computed position size multiplier (from position_sizing module)
                            # This includes non-linear conviction, correlation adjustment, and vol targeting
                            size_mult = entry_size_mult[j]
                        else:
                            # Simple conviction-weighted sizing: scale position by signal score
                            # conviction = min + (max - min) * normalized_score
                            # where normalized_score = (score - min_threshold) / (1 - min_threshold)
                            score = entry_score[j]
                            if conviction_max > conviction_min and min_signal_score < 1.0:
                                norm_score = (score - min_signal_score) / (1.0 - min_signal_score)
                                norm_score = max(0.0, min(1.0, norm_score))  # clip to [0, 1]
                                size_mult = conviction_min + (conviction_max - conviction_min) * norm_score
                            else:
                                size_mult = 1.0

                        # Compute effective position weight (fraction of portfolio allocated to pair)
                        # This is the TOTAL pair allocation, not per-leg
                        position_weight = capital_per_pair * size_mult

                        # Fees: round-trip cost = entry_notional + exit_notional
                        # In normalized model: entry_notional = position_weight (split w_y, w_x)
                        #                      exit_notional = position_weight
                        # Total notional traded = 2 × position_weight
                        # Fee = total_notional × fee_rate = 2 × position_weight × fee_rate
                        fees = 2.0 * fee_rate * position_weight

                        # Slippage (same logic: 2× round-trip, scaled by position weight)
                        slip = slippage_rate
                        if slippage_model_id == 1:
                            v = spread_vol_mat[t, j]
                            if np.isfinite(v) and v > 0.0:
                                slip = slippage_rate * (1.0 + slippage_vol_mult * v)
                        slippage_cost = 2.0 * slip * position_weight

                        # Gross return scaled by position, then subtract proportional costs
                        gross_return = pnl * position_weight
                        net_return = gross_return - fees - slippage_cost

                        returns_mat[t, j] = net_return

                        # Record P&L attribution components
                        gross_pnl_mat[t, j] = gross_return
                        fees_mat[t, j] = fees
                        slippage_mat[t, j] = slippage_cost
                        hold_bars_mat[t, j] = t - entry_bar[j]

                    else:
                        # Legacy behavior (unnormalized)
                        # Base spread PnL (static beta locked at entry)
                        pnl_spread = (y_exit - entry_y[j]) - b * (x_exit - entry_x[j])

                        # Apply direction
                        pnl = dir_entry[j] * pnl_spread

                        # Notional approximation (1 unit Y, beta units X) - NOT normalized
                        notional = abs(entry_y[j]) + abs_b * abs(entry_x[j])
                        if not np.isfinite(notional) or notional <= 0.0:
                            notional = 0.0

                        # Round-trip costs: entry + exit = 2 × notional
                        fees = notional * 2.0 * fee_rate

                        # Slippage (same 2× round-trip logic)
                        slip = slippage_rate
                        if slippage_model_id == 1:
                            v = spread_vol_mat[t, j]
                            if np.isfinite(v) and v > 0.0:
                                slip = slippage_rate * (1.0 + slippage_vol_mult * v)
                        slippage_cost = notional * 2.0 * slip

                        net_pnl = pnl - fees - slippage_cost

                        # Position sizing (legacy path)
                        if use_advanced_sizing:
                            # Use pre-computed position size multiplier
                            size_mult = entry_size_mult[j]
                        else:
                            # Simple conviction-weighted sizing
                            score = entry_score[j]
                            if conviction_max > conviction_min and min_signal_score < 1.0:
                                norm_score = (score - min_signal_score) / (1.0 - min_signal_score)
                                norm_score = max(0.0, min(1.0, norm_score))
                                size_mult = conviction_min + (conviction_max - conviction_min) * norm_score
                            else:
                                size_mult = 1.0

                        # Convert to return
                        if capital_per_pair > 0.0:
                            returns_mat[t, j] = (net_pnl / capital_per_pair) * size_mult

                        # Record P&L attribution components (legacy path)
                        # Scale to be consistent with returns_mat
                        if capital_per_pair > 0.0:
                            gross_pnl_mat[t, j] = (pnl / capital_per_pair) * size_mult
                            fees_mat[t, j] = (fees / capital_per_pair) * size_mult
                            slippage_mat[t, j] = (slippage_cost / capital_per_pair) * size_mult
                        hold_bars_mat[t, j] = t - entry_bar[j]

                    # Track exit reason and record MAE/MFE
                    exit_reason = EXIT_SIGNAL  # Default to signal exit
                    if stop_loss_hit:
                        stop_loss_count[j] += 1
                        exit_reason = EXIT_STOP
                    elif time_stop_hit:
                        time_stop_count[j] += 1
                        exit_reason = EXIT_TIME
                    elif t == T - 1:
                        exit_reason = EXIT_FORCED

                    # Record MAE/MFE at exit timestamp
                    # MAE is stored as positive (how bad it got)
                    # MFE is stored as positive (how good it got)
                    mae_mat[t, j] = -running_mae[j] if running_mae[j] < 0 else 0.0
                    mfe_mat[t, j] = running_mfe[j] if running_mfe[j] > 0 else 0.0
                    exit_reason_mat[t, j] = exit_reason

                    trades_count[j] += 1
                    state[j] = LOOKING
                    if active_positions > 0:
                        active_positions -= 1
                    c1 = pair_coin1_idx[j]
                    c2 = pair_coin2_idx[j]
                    if coin_counts[c1] > 0:
                        coin_counts[c1] -= 1
                    if coin_counts[c2] > 0:
                        coin_counts[c2] -= 1

    return (returns_mat, trades_count, time_stop_count, stop_loss_count, mae_mat, mfe_mat, exit_reason_mat,
            gross_pnl_mat, fees_mat, slippage_mat, hold_bars_mat)


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
    signal_score: Optional[pd.DataFrame] = None,
    position_size_multiplier: Optional[pd.DataFrame] = None,
    funding_rates: Optional[pd.DataFrame] = None,  # NEW: funding rates per pair per bar
    pnl_mode: PnlMode = "price",
    pair_sep: str = "-",
    fee_rate: Optional[float] = None,
    slippage_rate: float = 0.0,
    slippage_model: SlippageModel = "fixed",
    slippage_vol_mult: float = 0.0,
    capital_per_pair: float = 1.0,
    max_trades_per_pair: Optional[int] = None,
    normalize_notional: Optional[bool] = None,
    max_positions_total: Optional[int] = None,
    max_positions_per_coin: Optional[int] = None,
    stop_loss_pct: Optional[float] = None,
    max_hold_bars: Optional[Dict[str, int]] = None,
    default_max_hold_bars: Optional[int] = None,
    enable_conviction_sizing: Optional[bool] = None,
    conviction_min: Optional[float] = None,
    conviction_max: Optional[float] = None,
    enable_advanced_sizing: Optional[bool] = None,
) -> PnlResult:
    """
    Step G: Truth engine PnL simulator.

    Inputs are all aligned on:
      - time index == test_df.index
      - columns == pairs (same order)

    Parameters
    ----------
    normalize_notional : bool, optional
        If True (default), normalize position weights so gross exposure = 1.0:
            w_Y = 1 / (1 + |beta|)
            w_X = |beta| / (1 + |beta|)
        This fixes Problem #5: position sizing is truly market-neutral and
        returns are scale-invariant.
        If False, uses legacy behavior where gross = 1 + |beta|.

    max_hold_bars : dict, optional
        Per-pair maximum holding time in bars before time stop triggers.
        E.g., {"ETH-BTC": 1500, "SOL-ETH": 1200}.
        If a pair is not in the dict, uses default_max_hold_bars.
        Set to 0 to disable time stop for that pair.
        This fixes Problem #7: prevents holding "dead" trades through regime changes.

    default_max_hold_bars : int, optional
        Default time stop for pairs not in max_hold_bars dict.
        If None, uses cfg.DEFAULT_TIME_STOP_BARS.
        Set to 0 to disable time stops by default.

    max_positions_total : int, optional
        Max concurrent positions across the portfolio (0 = unlimited).
    max_positions_per_coin : int, optional
        Max concurrent positions per coin (0 = unlimited).
    stop_loss_pct : float, optional
        Hard PnL stop per trade (fraction of allocated capital).
    position_size_multiplier : pd.DataFrame, optional
        Pre-computed position size multipliers (T × pairs). If provided along
        with enable_advanced_sizing=True, these values are used directly instead
        of computing conviction from signal_score. This allows using sophisticated
        sizing from the position_sizing module (non-linear conviction, correlation
        adjustment, volatility targeting).
    enable_advanced_sizing : bool, optional
        If True and position_size_multiplier is provided, use pre-computed sizes.
        If False or position_size_multiplier is None, fall back to simple conviction sizing.

    funding_rates : pd.DataFrame, optional
        Funding rates per pair per bar. Should be aligned with test_df index and
        have same columns as pairs. Funding is charged while a position is open.
        Positive funding = cost to holder (long pays short in typical perp markets).

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

    # Use effective fee rate from fee model if not explicitly provided
    if fee_rate is None:
        fee_rate = compute_effective_fee_rate(
            fee_model=getattr(cfg, "FEE_MODEL", "taker_only"),
        )
    fee_rate = float(fee_rate)
    max_trades = int(max_trades_per_pair if max_trades_per_pair is not None else cfg.MAX_TRADES_PER_PAIR)

    # Default to normalized notional (Problem #5 fix)
    if normalize_notional is None:
        normalize_notional = getattr(cfg, 'NORMALIZE_NOTIONAL', True)

    if max_positions_total is None:
        max_positions_total = getattr(cfg, "MAX_PORTFOLIO_POSITIONS", 0)
    if max_positions_per_coin is None:
        max_positions_per_coin = getattr(cfg, "MAX_POSITIONS_PER_COIN", 0)
    max_positions_total = int(max_positions_total)
    max_positions_per_coin = int(max_positions_per_coin)

    if stop_loss_pct is None:
        stop_loss_pct = getattr(cfg, "STOP_LOSS_PCT", 0.0)
    stop_loss_pct = float(stop_loss_pct)

    # Build per-pair max hold bars array (Problem #7 fix)
    if default_max_hold_bars is None:
        default_max_hold_bars = getattr(cfg, 'DEFAULT_TIME_STOP_BARS', 1440)
    max_hold_bars = max_hold_bars or {}

    max_hold_bars_arr = np.zeros(len(pair_list), dtype=np.int64)
    for j, pair in enumerate(pair_list):
        max_hold_bars_arr[j] = max_hold_bars.get(pair, default_max_hold_bars)

    slippage_model_id = 0 if slippage_model == "fixed" else 1

    # Conviction-weighted position sizing
    if enable_conviction_sizing is None:
        enable_conviction_sizing = getattr(cfg, "ENABLE_CONVICTION_SIZING", False)
    if conviction_min is None:
        conviction_min = getattr(cfg, "MIN_CONVICTION_SIZE", 0.5)
    if conviction_max is None:
        conviction_max = getattr(cfg, "MAX_CONVICTION_SIZE", 1.0)
    min_signal_score = getattr(cfg, "MIN_SIGNAL_SCORE", 0.65)

    # If conviction sizing disabled, set min=max=1.0 to have no effect
    if not enable_conviction_sizing:
        conviction_min = 1.0
        conviction_max = 1.0

    # Prepare signal score matrix
    if signal_score is None:
        # Default: all scores = min_signal_score (no scaling)
        score_mat = np.full((len(test_df), len(pair_list)), min_signal_score, dtype=np.float64)
    else:
        signal_score = _require_aligned(signal_score, test_df.index, pair_index, "signal_score")
        score_mat = signal_score.to_numpy(dtype=np.float64)

    # Advanced position sizing
    if enable_advanced_sizing is None:
        enable_advanced_sizing = getattr(cfg, "ENABLE_ADVANCED_SIZING", False)

    use_advanced_sizing = bool(enable_advanced_sizing) and (position_size_multiplier is not None)

    if use_advanced_sizing:
        position_size_multiplier = _require_aligned(
            position_size_multiplier, test_df.index, pair_index, "position_size_multiplier"
        )
        size_mult_mat = position_size_multiplier.to_numpy(dtype=np.float64)
        # Fill NaN with 1.0 (no adjustment)
        size_mult_mat = np.where(np.isfinite(size_mult_mat), size_mult_mat, 1.0)
        logger.info(
            "Advanced sizing enabled: size_mult range=[%.3f, %.3f], mean=%.3f",
            np.nanmin(size_mult_mat), np.nanmax(size_mult_mat), np.nanmean(size_mult_mat)
        )
    else:
        # Default: all multipliers = 1.0 (no external adjustment)
        size_mult_mat = np.ones((len(test_df), len(pair_list)), dtype=np.float64)

    # Build coin index mapping for portfolio constraints
    coin_set = set()
    for pair in pair_list:
        cy, cx = _parse_pair(pair, sep=pair_sep)
        coin_set.add(cy)
        coin_set.add(cx)
    coins = sorted(coin_set)
    coin_idx = {c: i for i, c in enumerate(coins)}
    pair_coin1_idx = np.zeros(len(pair_list), dtype=np.int64)
    pair_coin2_idx = np.zeros(len(pair_list), dtype=np.int64)
    for j, pair in enumerate(pair_list):
        cy, cx = _parse_pair(pair, sep=pair_sep)
        pair_coin1_idx[j] = coin_idx[cy]
        pair_coin2_idx[j] = coin_idx[cx]

    # Convert to numpy for numba
    entry_mask = entries.to_numpy(dtype=np.bool_)
    exit_mask = exits.to_numpy(dtype=np.bool_)
    beta_mat = beta.to_numpy(dtype=np.float64)
    z_mat = z_score.to_numpy(dtype=np.float64)
    vol_mat = spread_volatility.to_numpy(dtype=np.float64)

    (returns_mat, trades_count, time_stop_count, stop_loss_count,
     mae_mat, mfe_mat, exit_reason_mat,
     gross_pnl_mat, fees_mat, slippage_mat, hold_bars_mat) = _pnl_state_machine_numba(
        y_mat=y_mat,
        x_mat=x_mat,
        beta_mat=beta_mat,
        z_mat=z_mat,
        entry_mask=entry_mask,
        exit_mask=exit_mask,
        spread_vol_mat=vol_mat,
        score_mat=score_mat,
        size_mult_mat=size_mult_mat,
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        slippage_model_id=int(slippage_model_id),
        slippage_vol_mult=float(slippage_vol_mult),
        capital_per_pair=float(capital_per_pair),
        max_trades_per_pair=int(max_trades),
        normalize_notional=bool(normalize_notional),
        use_log_prices=bool(pnl_mode == "log"),
        pair_coin1_idx=pair_coin1_idx,
        pair_coin2_idx=pair_coin2_idx,
        n_coins=len(coins),
        max_positions_total=int(max_positions_total),
        max_positions_per_coin=int(max_positions_per_coin),
        stop_loss_pct=float(stop_loss_pct),
        max_hold_bars_arr=max_hold_bars_arr,
        conviction_min=float(conviction_min),
        conviction_max=float(conviction_max),
        min_signal_score=float(min_signal_score),
        use_advanced_sizing=bool(use_advanced_sizing),
    )

    returns_df = pd.DataFrame(returns_mat, index=test_df.index, columns=pair_index)
    trades_series = pd.Series(trades_count, index=pair_index, name="trades_count")
    time_stop_series = pd.Series(time_stop_count, index=pair_index, name="time_stop_count")

    # Create MAE/MFE/exit_reason DataFrames
    mae_df = pd.DataFrame(mae_mat, index=test_df.index, columns=pair_index)
    mfe_df = pd.DataFrame(mfe_mat, index=test_df.index, columns=pair_index)
    exit_reason_df = pd.DataFrame(exit_reason_mat, index=test_df.index, columns=pair_index)

    # Create P&L Attribution DataFrames
    gross_pnl_df = pd.DataFrame(gross_pnl_mat, index=test_df.index, columns=pair_index)
    fees_df = pd.DataFrame(fees_mat, index=test_df.index, columns=pair_index)
    slippage_df = pd.DataFrame(slippage_mat, index=test_df.index, columns=pair_index)
    hold_bars_df = pd.DataFrame(hold_bars_mat, index=test_df.index, columns=pair_index)

    total_time_stops = int(time_stop_series.sum())
    total_stop_losses = int(stop_loss_count.sum())

    # Log MAE/MFE statistics for trades
    nonzero_mae = mae_mat[mae_mat > 0]
    if len(nonzero_mae) > 0:
        logger.info(
            "MAE stats: mean=%.4f, median=%.4f, max=%.4f, trades_with_mae=%d",
            np.mean(nonzero_mae), np.median(nonzero_mae), np.max(nonzero_mae), len(nonzero_mae)
        )

    logger.info(
        "PnL engine complete: total_trades=%d, time_stops=%d, stop_losses=%d, total_nonzero_returns=%d",
        int(trades_series.sum()),
        total_time_stops,
        total_stop_losses,
        int(np.count_nonzero(returns_mat)),
    )

    # -------------------------------------------------------------------------
    # Funding Cost Computation (post-hoc, based on position state)
    # -------------------------------------------------------------------------
    # Compute funding costs based on when positions were held.
    # We reconstruct in_trade mask from entry/exit reason to know when funding applies.
    funding_costs_df = None
    total_funding = 0.0

    if funding_rates is not None:
        try:
            # Align funding rates to pair index
            funding_rates_aligned = funding_rates.reindex(columns=pair_index, fill_value=0.0)
            funding_rates_aligned = funding_rates_aligned.reindex(index=test_df.index, fill_value=0.0)
            funding_mat = funding_rates_aligned.to_numpy(dtype=np.float64)

            # Reconstruct in-position mask from exit_reason_matrix
            # Position is open from entry until exit (inclusive of entry bar, exclusive of exit bar)
            # We'll track state per pair across time
            in_position = np.zeros((len(test_df), len(pair_list)), dtype=np.bool_)
            state = np.zeros(len(pair_list), dtype=np.int64)  # 0 = looking, 1 = in trade

            for t in range(len(test_df)):
                for j in range(len(pair_list)):
                    # Check if we entered this bar
                    if state[j] == 0 and entry_mask[t, j]:
                        state[j] = 1
                    # If in trade, mark it and check for exit
                    if state[j] == 1:
                        in_position[t, j] = True
                        # Check if we exited this bar
                        if exit_reason_mat[t, j] > 0:
                            state[j] = 0

            # Compute funding costs: funding_rate * position_weight while in position
            # For simplicity, assume position_weight = capital_per_pair (could enhance later)
            funding_costs_mat = np.where(in_position, funding_mat * capital_per_pair, 0.0)

            # Create DataFrame
            funding_costs_df = pd.DataFrame(
                funding_costs_mat, index=test_df.index, columns=pair_index
            )
            total_funding = float(np.sum(funding_costs_mat))

            if total_funding != 0.0:
                logger.info(
                    "Funding costs computed: total=%.6f, bars_with_position=%d",
                    total_funding, int(np.sum(in_position))
                )
        except Exception as e:
            logger.warning("Failed to compute funding costs: %s", e)
            funding_costs_df = None
            total_funding = 0.0

    return PnlResult(
        returns_matrix=returns_df,
        trades_count=trades_series,
        time_stop_count=time_stop_series,
        stop_loss_count=pd.Series(stop_loss_count, index=pair_index, name="stop_loss_count"),
        mae_matrix=mae_df,
        mfe_matrix=mfe_df,
        exit_reason_matrix=exit_reason_df,
        gross_pnl_matrix=gross_pnl_df,
        fees_matrix=fees_df,
        slippage_matrix=slippage_df,
        hold_bars_matrix=hold_bars_df,
        funding_costs_matrix=funding_costs_df,
        total_funding=total_funding,
    )
