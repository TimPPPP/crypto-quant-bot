from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import vectorbt as vbt

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.performance_report")


# ----------------------------- Data structures ----------------------------- #

@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    funding_drag_daily: float  # e.g., 0.0001 for 0.01% daily
    extra_slippage_per_exit: float  # applied only where portfolio_return != 0
    notes: str


@dataclass(frozen=True)
class ScenarioMetrics:
    name: str
    total_return: float
    sharpe: float              # Primary: trade-level Sharpe
    sharpe_bar_level: float    # Secondary: bar-level Sharpe (for comparison)
    calmar: float
    max_drawdown: float
    max_dd_duration_minutes: float
    corr_to_btc_returns: float
    corr_to_btc_equity: float
    n_trades: int              # Number of completed trades
    # Realistic simulation fields (optional)
    initial_capital: Optional[float] = None
    final_capital: Optional[float] = None
    total_pnl: Optional[float] = None
    max_drawdown_dollars: Optional[float] = None


# ----------------------------- Helpers ------------------------------------ #

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _freq_minutes(freq: str) -> float:
    """
    Convert a freq string like '1min', '1m', or '1h' to minutes.
    """
    f = freq.lower().strip()
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
    raise ValueError(f"Unsupported freq '{freq}'. Use '1min'/'1m' or '<N>min'.")


def _per_bar_funding_drag(daily_drag: float, freq: str) -> float:
    """
    Convert daily funding drag to per-bar return drag for given bar frequency.
    For 1-minute bars: daily_drag / (24*60).
    """
    mins = _freq_minutes(freq)
    bars_per_day = (24.0 * 60.0) / mins
    return float(daily_drag) / bars_per_day


def _equity_from_returns(returns: pd.Series, init_cash: float = 1.0) -> pd.Series:
    """
    Equity curve from simple returns.
    """
    r = returns.fillna(0.0).astype(float)
    return init_cash * (1.0 + r).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown as a positive fraction (e.g., 0.25 means -25%).
    """
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float((-dd.min()) if len(dd) else 0.0)


def _max_dd_duration_minutes(equity: pd.Series, freq: str) -> float:
    """
    Longest time spent under prior peak (peak-to-recovery duration).
    Returns duration in minutes.
    """
    if equity.empty:
        return 0.0

    eq = equity.astype(float)
    peak = eq.cummax()
    under = eq < peak

    # Identify contiguous "underwater" segments and measure their lengths
    max_len = 0
    cur_len = 0
    for u in under.to_numpy(dtype=bool):
        if u:
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
        else:
            cur_len = 0

    return float(max_len * _freq_minutes(freq))


def _annualization_factor(freq: str) -> float:
    """
    Annualization factor for Sharpe etc.
    For 1-minute: sqrt(365*24*60)
    """
    mins = _freq_minutes(freq)
    bars_per_year = (365.0 * 24.0 * 60.0) / mins
    return float(np.sqrt(bars_per_year))


def _calmar_ratio(equity: pd.Series, freq: str) -> float:
    """
    Calmar = annualized return / max drawdown.
    Uses equity curve and bar frequency.
    """
    if equity.empty:
        return 0.0

    md = _max_drawdown(equity)
    if md <= 1e-12:
        return float("inf")

    # Annualized return (CAGR-like) from start to end
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or not np.isfinite(start) or not np.isfinite(end):
        return 0.0

    mins = _freq_minutes(freq)
    years = (len(equity) * mins) / (365.0 * 24.0 * 60.0)
    if years <= 1e-6:  # Very short period, avoid overflow
        return 0.0

    try:
        ratio = end / start
        if ratio <= 0 or not np.isfinite(ratio):
            return 0.0
        ann_return = ratio ** (1.0 / years) - 1.0
        if not np.isfinite(ann_return):
            return 0.0
        return float(ann_return / md)
    except (OverflowError, ValueError):
        return 0.0


def _corr_safe(a: pd.Series, b: pd.Series) -> float:
    """
    Compute correlation on aligned, finite values. Returns 0.0 if undefined.
    """
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 3:
        return 0.0
    c = df.iloc[:, 0].corr(df.iloc[:, 1])
    return float(c) if np.isfinite(c) else 0.0


def _compute_in_trade_mask(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    max_trades_per_pair: Optional[int] = None,
    max_hold_bars: Optional[Dict[str, int]] = None,
    default_max_hold_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reconstruct an in-trade mask using the same blocking logic as the PnL engine.
    """
    if not entries.index.equals(exits.index):
        raise ValueError("entries/exits index mismatch for in-trade reconstruction.")
    if not entries.columns.equals(exits.columns):
        raise ValueError("entries/exits columns mismatch for in-trade reconstruction.")

    max_trades = max_trades_per_pair
    if max_trades is None:
        max_trades = int(getattr(cfg, "MAX_TRADES_PER_PAIR", 0))
    max_trades = int(max_trades)

    if default_max_hold_bars is None:
        default_max_hold_bars = int(getattr(cfg, "DEFAULT_TIME_STOP_BARS", 0))

    max_hold_bars = max_hold_bars or {}

    pairs = entries.columns
    T, P = entries.shape

    in_trade = np.zeros((T, P), dtype=bool)
    state = np.zeros(P, dtype=np.int8)
    entry_bar = np.zeros(P, dtype=np.int64)
    trades = np.zeros(P, dtype=np.int64)
    max_hold_arr = np.zeros(P, dtype=np.int64)

    for j, pair in enumerate(pairs):
        max_hold_arr[j] = int(max_hold_bars.get(pair, default_max_hold_bars))

    entry_mask = entries.to_numpy(dtype=np.bool_)
    exit_mask = exits.to_numpy(dtype=np.bool_)

    LOOKING = 0
    IN_TRADE = 1

    for t in range(T):
        for j in range(P):
            if state[j] == LOOKING:
                if entry_mask[t, j]:
                    if max_trades > 0 and trades[j] >= max_trades:
                        continue
                    state[j] = IN_TRADE
                    entry_bar[j] = t
                    in_trade[t, j] = True
            else:
                in_trade[t, j] = True
                bars_held = t - entry_bar[j]
                max_hold = max_hold_arr[j]
                time_stop_hit = (max_hold > 0) and (bars_held >= max_hold)
                if exit_mask[t, j] or time_stop_hit:
                    trades[j] += 1
                    state[j] = LOOKING

    return pd.DataFrame(in_trade, index=entries.index, columns=entries.columns)


def compute_sharpe_robust(
    returns: pd.Series,
    freq: str,
    method: str = "trade_level",
) -> float:
    """
    Compute Sharpe ratio with proper handling of sparse returns.

    For pairs trading where returns are only non-zero at trade exits:
    - bar_level: standard Sharpe on full time series (will be low due to many zeros)
    - trade_level: Sharpe on non-zero returns only, annualized by trade frequency

    Parameters
    ----------
    returns : pd.Series
        Return series (can be sparse with many zeros)
    freq : str
        Bar frequency (e.g., "1min")
    method : str
        "bar_level" for standard Sharpe, "trade_level" for trade-based Sharpe

    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    r = returns.fillna(0.0).astype(float)

    if len(r) < 2:
        return 0.0

    if method == "trade_level":
        # Only look at actual trades (non-zero returns)
        trade_returns = r[r != 0]
        n_trades = len(trade_returns)

        if n_trades < 2:
            return 0.0

        mean_ret = trade_returns.mean()
        std_ret = trade_returns.std()

        if std_ret < 1e-10:
            return 0.0

        # Annualize based on trade frequency
        # trades_per_year = n_trades / (total_bars / bars_per_year)
        total_bars = len(r)
        mins_per_bar = _freq_minutes(freq)
        bars_per_year = (365.0 * 24.0 * 60.0) / mins_per_bar

        # Avoid division by zero for very short periods
        if total_bars < 2:
            return 0.0

        trades_per_year = n_trades * (bars_per_year / total_bars)
        trade_ann_factor = np.sqrt(max(1.0, trades_per_year))

        sharpe = (mean_ret / std_ret) * trade_ann_factor
        return float(sharpe) if np.isfinite(sharpe) else 0.0

    else:  # bar_level - standard approach
        mean_ret = r.mean()
        std_ret = r.std()

        if std_ret < 1e-10:
            return 0.0

        ann_factor = _annualization_factor(freq)
        sharpe = (mean_ret / std_ret) * ann_factor
        return float(sharpe) if np.isfinite(sharpe) else 0.0


class _ReturnsWrapper:
    """
    Wrapper to provide Portfolio-like interface using vectorbt returns accessor.
    vectorbt 0.28+ removed Portfolio.from_returns(), so we use the returns accessor.
    """
    def __init__(self, returns: pd.Series, freq: str):
        self._returns = returns.fillna(0.0).astype(float)
        self._freq = freq
        self._acc = self._returns.vbt.returns
        # Pre-compute equity curve: cumulative product of (1 + r)
        self._equity = (1 + self._returns).cumprod()

    def value(self) -> pd.Series:
        return self._equity

    def sharpe_ratio(self) -> float:
        return float(self._acc.sharpe_ratio())


def _portfolio_from_returns(returns: pd.Series, freq: str) -> _ReturnsWrapper:
    """
    Returns a Portfolio-like wrapper using vectorbt returns accessor.
    """
    return _ReturnsWrapper(returns, freq=freq)


# ----------------------------- Funding rate loading ------------------------ #

def load_funding_rates() -> Optional[pd.DataFrame]:
    """
    Load funding rate matrix from parquet if available.

    Returns:
        DataFrame with timestamp index and coin symbols as columns, or None if not available.
    """
    funding_path = cfg.PATH_FUNDING_PARQUET
    if not funding_path.exists():
        logger.warning("Funding rates file not found: %s", funding_path)
        return None

    try:
        funding_df = pd.read_parquet(funding_path)
        logger.info("Loaded funding rates: %s timestamps x %s coins",
                   f"{len(funding_df):,}", f"{len(funding_df.columns):,}")
        return funding_df
    except Exception as exc:
        logger.error("Failed to load funding rates: %s", exc)
        return None


def _apply_real_funding(
    returns: pd.Series,
    pair_name: str,
    funding_rates: pd.DataFrame,
    freq: str,
    in_trade: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Apply real funding costs to returns based on actual funding rate data.

    For pairs trading (long coin1, short coin2):
    - Net funding cost = funding_rate_coin1 - funding_rate_coin2
    - Funding happens every 8 hours, so we distribute the cost across those bars

    Parameters:
    -----------
    returns : pd.Series
        Return series for this pair
    pair_name : str
        Pair identifier like "ETH-BTC"
    funding_rates : pd.DataFrame
        Funding rate matrix (timestamp x coins)
    freq : str
        Bar frequency (e.g., "1min")
    in_trade : Optional[pd.Series]
        Boolean mask of when a position is open (apply funding only then)

    Returns:
    --------
    pd.Series
        Returns with funding costs applied
    """
    r = returns.copy().fillna(0.0).astype(float)

    # Parse pair name to get individual coins
    separator = cfg.PAIR_ID_SEPARATOR  # e.g., "-"
    parts = pair_name.split(separator)
    if len(parts) != 2:
        logger.warning("Cannot parse pair name '%s', skipping real funding", pair_name)
        return r

    coin1, coin2 = parts

    # Check if both coins have funding data
    if coin1 not in funding_rates.columns or coin2 not in funding_rates.columns:
        logger.debug("Missing funding data for pair %s (coins: %s, %s)", pair_name, coin1, coin2)
        return r

    # Align funding rates to returns timeline
    funding_aligned = funding_rates[[coin1, coin2]].reindex(r.index).ffill()

    # Net funding cost: long coin1 - short coin2
    # Positive = we pay, Negative = we receive
    net_funding_8h = funding_aligned[coin1] - funding_aligned[coin2]

    # Convert 8-hour funding rate to per-bar cost
    # Funding happens every 8 hours = 480 minutes
    # For 1-minute bars: divide by 480
    mins_per_bar = _freq_minutes(freq)
    funding_interval_mins = 8.0 * 60.0  # 8 hours
    bars_per_funding_period = funding_interval_mins / mins_per_bar

    funding_per_bar = net_funding_8h / bars_per_funding_period
    if in_trade is not None:
        in_trade = in_trade.reindex(r.index).fillna(False)
        # DEBUG: Check how many bars we're in trade
        n_in_trade = in_trade.sum()
        funding_per_bar = funding_per_bar.where(in_trade, 0.0)
        logger.info(f"[DEBUG] {pair_name}: in_trade bars={n_in_trade}/{len(in_trade)}, funding_per_bar mean={funding_per_bar[in_trade].mean():.9f}")

    # DEBUG: Log funding impact
    funding_total = funding_per_bar.fillna(0.0).sum()
    r_input = r.sum()

    # Subtract funding cost from returns (cost is positive, so we subtract)
    r = r - funding_per_bar.fillna(0.0)

    # DEBUG: Check if funding was applied correctly
    r_output = r.sum()
    if abs(r_output - r_input) > 1e-9:
        logger.info(f"[DEBUG] _apply_real_funding {pair_name}: input={r_input:.6f}, funding={funding_total:.6f}, output={r_output:.6f}")

    return r


# ----------------------------- Scenario transforms ------------------------- #

def apply_scenario(
    base_returns: pd.Series,
    *,
    funding_drag_daily: float,
    extra_slippage_per_exit: float,
    freq: str,
    funding_rates: Optional[pd.DataFrame] = None,
    pair_name: Optional[str] = None,
    in_trade: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Apply scenario-level penalties.

    - funding drag: either real funding rates (if available) or constant drag
    - slippage stress: subtract an extra cost ONLY when a trade closes (nonzero returns)
      because your returns_matrix has spikes at exits.

    Parameters:
    -----------
    base_returns : pd.Series
        Base return series
    funding_drag_daily : float
        Daily funding drag (used if real funding unavailable)
    extra_slippage_per_exit : float
        Extra slippage applied on exit events
    freq : str
        Bar frequency
    funding_rates : Optional[pd.DataFrame]
        Real funding rate matrix (if USE_REAL_FUNDING is True)
    pair_name : Optional[str]
        Pair identifier (needed for real funding)
    in_trade : Optional[pd.Series]
        Boolean mask indicating when a position is open (apply funding only then)
    """
    r = base_returns.copy().fillna(0.0).astype(float)
    in_trade_mask = None
    if in_trade is not None:
        in_trade_mask = in_trade.reindex(r.index).fillna(False).astype(bool)

    # DEBUG: Log input and output
    input_sum = base_returns.sum()

    # IMPORTANT: Identify actual trade exits BEFORE applying funding
    # (funding adjustments make many bars non-zero, but we only want exit events for slippage)
    exit_events = base_returns.fillna(0.0) != 0.0

    # Funding: use real rates if available and enabled
    # IMPORTANT: Real funding is NOT applied in scenarios! Scenarios use hypothetical funding_drag_daily
    # for stress testing. Real funding should be tracked separately in PnL attribution.
    use_real = False  # Disabled: returns_matrix already reflects actual trading P&L
    if use_real and funding_rates is not None and pair_name is not None:
        logger.debug("Applying real funding for pair: %s", pair_name)
        r = _apply_real_funding(r, pair_name, funding_rates, freq, in_trade=in_trade_mask)
    else:
        # Apply constant funding drag for scenario stress testing
        # NOTE: This is ADDITIVE stress on top of any funding already in base returns.
        # For scenarios to differ, we MUST apply this drag even without perfect position tracking.
        drag_per_bar = _per_bar_funding_drag(funding_drag_daily, freq)
        if drag_per_bar != 0.0:
            if in_trade_mask is not None:
                # Ideal: apply only when in position
                r.loc[in_trade_mask] = r.loc[in_trade_mask] - drag_per_bar
            else:
                # Fallback: estimate in-trade periods from non-zero base returns
                # Trades have non-zero returns at exit; estimate ~avg_hold_bars of funding per trade
                # Apply funding as a fraction of time spent trading (conservative estimate)
                exit_mask = base_returns.fillna(0.0) != 0.0
                n_exits = exit_mask.sum()
                if n_exits > 0:
                    # Estimate: apply funding_drag to periods proportional to trades
                    # Use exit bars as proxy (each exit = one trade's funding exposure)
                    # This is approximate but ensures scenarios diverge
                    avg_hold_estimate = 10  # Assume ~10 bars avg hold time per trade
                    total_funding_drag = drag_per_bar * n_exits * avg_hold_estimate
                    # Distribute this drag across exit bars
                    r.loc[exit_mask] = r.loc[exit_mask] - (total_funding_drag / n_exits)
                    logger.debug(
                        "Applied estimated funding drag for %s: %.6f over %d exits",
                        pair_name, total_funding_drag, n_exits
                    )

    # Extra slippage only on actual trade exit events (identified BEFORE funding adjustments)
    if extra_slippage_per_exit != 0.0:
        r.loc[exit_events] = r.loc[exit_events] - float(extra_slippage_per_exit)

    # DEBUG: Log if sum changed
    output_sum = r.sum()
    if abs(output_sum - input_sum) > 1e-9:
        logger.info(f"[DEBUG] apply_scenario for {pair_name}: input={input_sum:.6f}, output={output_sum:.6f}, diff={output_sum-input_sum:.6f}")

    return r


def default_scenarios() -> Tuple[ScenarioSpec, ScenarioSpec, ScenarioSpec]:
    """
    Load scenario parameters from config. These define different cost assumptions
    for optimistic, base case, and stress testing.
    """
    # Read from config to allow tuning without code changes
    base_drag = getattr(cfg, "FUNDING_DRAG_BASE_DAILY", 0.0001)     # 0.01% daily
    stress_drag = getattr(cfg, "FUNDING_DRAG_STRESS_DAILY", 0.0002)  # 0.02% daily
    stress_extra_slip = getattr(cfg, "STRESS_EXTRA_SLIPPAGE_PER_EXIT", 0.0003)  # 3 bps per exit

    return (
        ScenarioSpec("optimistic", funding_drag_daily=0.0,     extra_slippage_per_exit=0.0,              notes="No funding drag"),
        ScenarioSpec("base_case",  funding_drag_daily=base_drag, extra_slippage_per_exit=0.0,            notes=f"{base_drag*100:.2f}% daily funding drag"),
        ScenarioSpec("stress",     funding_drag_daily=stress_drag, extra_slippage_per_exit=stress_extra_slip,
                    notes=f"{stress_drag*100:.2f}% daily funding drag + extra slippage per exit"),
    )


# ----------------------------- Metrics + plots ----------------------------- #

def compute_metrics(
    scenario_name: str,
    scenario_returns: pd.Series,
    btc_prices: Optional[pd.Series],
    *,
    freq: str,
    n_trades_override: Optional[int] = None,
) -> ScenarioMetrics:
    """
    Compute the required metrics for one scenario.

    Uses robust Sharpe calculation that handles sparse returns properly.
    Primary Sharpe is trade-level (more representative for pairs trading).
    """
    # Portfolio (vectorbt) for equity curve
    pf = _portfolio_from_returns(scenario_returns, freq=freq)

    equity = pf.value()
    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if len(equity) else 0.0

    # Robust Sharpe calculations
    # Trade-level: only considers non-zero returns (actual trades)
    # Bar-level: standard calculation on full series
    sharpe_trade = compute_sharpe_robust(scenario_returns, freq, method="trade_level")
    sharpe_bar = compute_sharpe_robust(scenario_returns, freq, method="bar_level")

    # Count actual trades (non-zero returns) unless overridden
    n_trades = int((scenario_returns != 0).sum())
    if n_trades_override is not None:
        n_trades = int(n_trades_override)

    max_dd = _max_drawdown(equity)
    calmar = _calmar_ratio(equity, freq=freq)
    dd_dur_min = _max_dd_duration_minutes(equity, freq=freq)

    corr_btc_ret = 0.0
    corr_btc_eq = 0.0
    if btc_prices is not None and not btc_prices.empty:
        btc_prices = btc_prices.reindex(equity.index).ffill()
        btc_ret = btc_prices.pct_change().fillna(0.0)
        btc_eq = _equity_from_returns(btc_ret, init_cash=1.0)

        strat_ret = equity.pct_change().fillna(0.0)
        corr_btc_ret = _corr_safe(strat_ret, btc_ret)
        corr_btc_eq = _corr_safe((equity / equity.iloc[0]), (btc_eq / btc_eq.iloc[0]))

    return ScenarioMetrics(
        name=scenario_name,
        total_return=total_return,
        sharpe=sharpe_trade,           # Primary: trade-level
        sharpe_bar_level=sharpe_bar,   # Secondary: bar-level for comparison
        calmar=float(calmar if np.isfinite(calmar) else 0.0),
        max_drawdown=float(max_dd),
        max_dd_duration_minutes=float(dd_dur_min),
        corr_to_btc_returns=float(corr_btc_ret),
        corr_to_btc_equity=float(corr_btc_eq),
        n_trades=n_trades,
    )


def _plot_equity_curves(
    out_path: Path,
    curves: Dict[str, pd.Series],
    btc_equity: Optional[pd.Series] = None,
) -> None:
    plt.figure()
    for name, s in curves.items():
        plt.plot(s.index, s.values, label=name)
    if btc_equity is not None:
        plt.plot(btc_equity.index, btc_equity.values, label="BTC_buy_hold")
    plt.title("Equity Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_drawdowns(out_path: Path, curves: Dict[str, pd.Series]) -> None:
    plt.figure()
    for name, eq in curves.items():
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        plt.plot(dd.index, dd.values, label=name)
    plt.title("Drawdowns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_returns_hist(out_path: Path, scenario_returns: Dict[str, pd.Series]) -> None:
    plt.figure()
    for name, r in scenario_returns.items():
        plt.hist(r.dropna().values, bins=80, alpha=0.5, label=name)
    plt.title("Return Distribution (Per Bar)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------- Public API ---------------------------------- #

def generate_performance_report(
    *,
    run_dir: Path,
    returns_matrix: pd.DataFrame,
    test_prices: Optional[pd.DataFrame] = None,
    btc_symbol: str = "BTC",
    freq: str = "1min",
    scenarios: Optional[Tuple[ScenarioSpec, ScenarioSpec, ScenarioSpec]] = None,
    entries: Optional[pd.DataFrame] = None,
    exits: Optional[pd.DataFrame] = None,
    max_hold_bars: Optional[Dict[str, int]] = None,
    position_size_multiplier: Optional[pd.DataFrame] = None,
    initial_capital: Optional[float] = None,
) -> Dict:
    """
    Step H: Performance report + stress scenarios.

    Inputs
    ------
    run_dir:
      results/run_id directory.
    returns_matrix:
      DataFrame indexed by time, columns are pairs, values are realized returns at exits (0 otherwise).
    test_prices (optional):
      If provided, used to compute BTC buy/hold equity curve for correlation checks.
      Expected wide matrix with columns containing btc_symbol (e.g., 'BTC').
    entries/exits (optional):
      If provided, funding drag is applied only while a position is open.
    position_size_multiplier (optional):
      DataFrame of position size multipliers per pair per bar (from risk prediction).
      If provided, returns are weighted by position sizes.
    initial_capital (optional):
      Starting capital in USD. If provided, enables realistic simulation mode
      with actual dollar values and compounding.

    Outputs
    -------
    Writes:
      - run_dir/metrics.json
      - run_dir/plots/*.png

    Returns:
      dict metrics payload.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    _ensure_dir(run_dir)
    _ensure_dir(plots_dir)

    if returns_matrix.empty:
        raise ValueError("returns_matrix is empty.")

    # Load funding rates if USE_REAL_FUNDING is enabled
    funding_rates = None
    use_real_funding = getattr(cfg, "USE_REAL_FUNDING", False)
    if use_real_funding:
        logger.info("Loading real funding rates...")
        funding_rates = load_funding_rates()
        if funding_rates is None:
            logger.warning("Real funding enabled but data unavailable, falling back to constant drag")

    # Portfolio return series = sum across pairs (your architecture assumes per-pair capital)
    # BUT: we need to apply funding per-pair BEFORE summing, since each pair has different funding
    # So we'll process each pair separately then sum

    # BTC series for correlation
    btc_prices = None
    if test_prices is not None and not test_prices.empty:
        if btc_symbol in test_prices.columns:
            btc_raw = pd.to_numeric(test_prices[btc_symbol], errors="coerce")
            # Drop duplicates to allow reindex, then ffill
            btc_deduped = btc_raw[~btc_raw.index.duplicated(keep="first")]
            # Create a mapping dict for looking up BTC prices
            btc_deduped = btc_deduped.sort_index().ffill()
            btc_map = btc_deduped.to_dict()
            # Map returns_matrix index to BTC prices (handles duplicates)
            btc_prices = pd.Series(
                [btc_map.get(ts, None) for ts in returns_matrix.index],
                index=returns_matrix.index
            ).ffill()
        else:
            logger.warning("BTC symbol '%s' not found in test_prices columns. Skipping BTC correlation.", btc_symbol)

    # Scenario set
    scenarios = scenarios or default_scenarios()

    scenario_equity: Dict[str, pd.Series] = {}
    scenario_returns: Dict[str, pd.Series] = {}
    metrics: Dict[str, Dict] = {}

    # Realistic simulation mode
    enable_realistic = bool(getattr(cfg, "ENABLE_REALISTIC_SIMULATION", False))
    if initial_capital is None:
        initial_capital = float(getattr(cfg, "INITIAL_CAPITAL", 100_000.0))
    max_position_pct = float(getattr(cfg, "MAX_POSITION_PCT", 0.15))
    max_positions = int(getattr(cfg, "MAX_PORTFOLIO_POSITIONS", 8))

    # Prepare position size multiplier (align with returns_matrix)
    if position_size_multiplier is not None:
        # Align columns and index
        common_cols = list(set(position_size_multiplier.columns) & set(returns_matrix.columns))
        pos_mult = position_size_multiplier[common_cols].reindex(returns_matrix.index).fillna(1.0)
    else:
        pos_mult = pd.DataFrame(1.0, index=returns_matrix.index, columns=returns_matrix.columns)

    # Optional: reconstruct in-trade masks for funding drag
    in_trade_masks: Optional[pd.DataFrame] = None
    if entries is not None and exits is not None:
        try:
            # Align entries/exits to returns_matrix index and columns
            common_cols = list(set(entries.columns) & set(exits.columns) & set(returns_matrix.columns))
            if not common_cols:
                logger.warning("No common columns between entries/exits/returns_matrix; skipping in-trade mask computation.")
            else:
                # Reindex to common structure
                aligned_entries = entries[common_cols].reindex(returns_matrix.index).fillna(False)
                aligned_exits = exits[common_cols].reindex(returns_matrix.index).fillna(False)

                in_trade_masks = _compute_in_trade_mask(
                    entries=aligned_entries,
                    exits=aligned_exits,
                    max_trades_per_pair=getattr(cfg, "MAX_TRADES_PER_PAIR", 0),
                    max_hold_bars=max_hold_bars,
                )
                logger.info("Computed in-trade masks for %d pairs", len(common_cols))
        except Exception as exc:
            logger.warning("Failed to reconstruct in-trade masks: %s", exc)

    # Precompute BTC equity if available
    btc_equity = None
    if btc_prices is not None and not btc_prices.empty:
        btc_ret = btc_prices.pct_change().fillna(0.0)
        btc_equity = _equity_from_returns(btc_ret, init_cash=1.0)

    # DEBUG: Log returns_matrix before processing
    logger.info(f"[DEBUG] returns_matrix shape: {returns_matrix.shape}, sum: {returns_matrix.values.sum():.6f}")

    # Process each scenario
    for spec in scenarios:
        # Apply scenario transforms per-pair, then sum
        pair_returns_with_costs = {}

        for pair_name in returns_matrix.columns:
            pair_base_returns = returns_matrix[pair_name].fillna(0.0)

            # Apply funding + slippage for this pair
            in_trade = None
            if in_trade_masks is not None and pair_name in in_trade_masks.columns:
                in_trade = in_trade_masks[pair_name]

            pair_returns_with_costs[pair_name] = apply_scenario(
                pair_base_returns,
                funding_drag_daily=spec.funding_drag_daily,
                extra_slippage_per_exit=spec.extra_slippage_per_exit,
                freq=freq,
                funding_rates=funding_rates,
                pair_name=pair_name,
                in_trade=in_trade,
            )

        # Aggregate portfolio returns (sum across pairs)
        portfolio_returns_df = pd.DataFrame(pair_returns_with_costs)

        # DEBUG: Calculate total funding impact
        total_funding_impact = portfolio_returns_df.values.sum() - returns_matrix.values.sum()
        logger.info(f"[DEBUG] Scenario '{spec.name}': Total funding impact = {total_funding_impact:.6f}")

        # NOTE: Do NOT multiply by position_size_multiplier here!
        # The returns_matrix already includes position sizing from the PnL engine
        # (see pnl_engine.py line 820: returns_mat[t, j] = (net_pnl / capital_per_pair) * size_mult)
        # Multiplying again would cause double-counting of position sizes.

        # In realistic simulation mode, scale returns by position allocation
        # Each position gets (1 / max_positions) of capital, then scaled by risk multiplier
        if enable_realistic:
            # Base allocation per position (equal weight among max positions)
            base_alloc = 1.0 / max_positions
            # Further cap by max_position_pct
            alloc_per_pos = min(base_alloc, max_position_pct)
            # Scale all pair returns by allocation
            portfolio_returns_df = portfolio_returns_df * alloc_per_pos

        r_s = portfolio_returns_df.sum(axis=1).astype(float).fillna(0.0)

        # DEBUG: Log what we're computing
        logger.info(f"[DEBUG] Scenario '{spec.name}': r_s sum = {r_s.sum():.6f}, len = {len(r_s)}")
        logger.info(f"[DEBUG] portfolio_returns_df shape: {portfolio_returns_df.shape}, sum: {portfolio_returns_df.values.sum():.6f}")

        scenario_returns[spec.name] = r_s

        # Compute equity curve with actual capital if in realistic mode
        init_cash_for_eq = initial_capital if enable_realistic else 1.0
        eq_s = _equity_from_returns(r_s, init_cash=init_cash_for_eq)
        scenario_equity[spec.name] = eq_s

        # Compute metrics
        m = compute_metrics(
            spec.name,
            r_s,
            btc_prices,
            freq=freq,
            n_trades_override=int((returns_matrix != 0).sum().sum()),
        )

        # Add realistic simulation metrics
        if enable_realistic:
            final_cap = float(eq_s.iloc[-1]) if len(eq_s) else initial_capital
            total_pnl = final_cap - initial_capital
            peak_eq = eq_s.cummax()
            max_dd_dollars = float((peak_eq - eq_s).max()) if len(eq_s) else 0.0

            m = ScenarioMetrics(
                name=m.name,
                total_return=m.total_return,
                sharpe=m.sharpe,
                sharpe_bar_level=m.sharpe_bar_level,
                calmar=m.calmar,
                max_drawdown=m.max_drawdown,
                max_dd_duration_minutes=m.max_dd_duration_minutes,
                corr_to_btc_returns=m.corr_to_btc_returns,
                corr_to_btc_equity=m.corr_to_btc_equity,
                n_trades=m.n_trades,
                initial_capital=initial_capital,
                final_capital=final_cap,
                total_pnl=total_pnl,
                max_drawdown_dollars=max_dd_dollars,
            )

        metrics[spec.name] = asdict(m)

    # Save metrics.json
    out_metrics = {
        "freq": freq,
        "btc_symbol": btc_symbol,
        "scenario_specs": [asdict(s) for s in scenarios],
        "metrics": metrics,
        "portfolio_return_definition": "sum(returns_matrix[pairs]) per bar; nonzero returns occur at trade exits",
    }

    # Add realistic simulation summary
    if enable_realistic:
        base_case_metrics = metrics.get("base_case", metrics.get("optimistic", {}))
        out_metrics["realistic_simulation"] = {
            "enabled": True,
            "initial_capital": initial_capital,
            "final_capital": base_case_metrics.get("final_capital"),
            "total_pnl": base_case_metrics.get("total_pnl"),
            "total_return_pct": base_case_metrics.get("total_return", 0) * 100,
            "max_drawdown_dollars": base_case_metrics.get("max_drawdown_dollars"),
            "max_drawdown_pct": base_case_metrics.get("max_drawdown", 0) * 100,
            "position_sizing_applied": position_size_multiplier is not None,
            "max_positions": max_positions,
            "max_position_pct": max_position_pct * 100,
        }

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    # Plots
    _plot_equity_curves(plots_dir / "equity_curves.png", scenario_equity, btc_equity=btc_equity)
    _plot_drawdowns(plots_dir / "drawdowns.png", scenario_equity)
    _plot_returns_hist(plots_dir / "returns_hist.png", scenario_returns)

    logger.info("✅ Report written: %s", metrics_path)
    logger.info("✅ Plots written under: %s", plots_dir)

    return out_metrics
