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
    sharpe: float
    calmar: float
    max_drawdown: float
    max_dd_duration_minutes: float
    corr_to_btc_returns: float
    corr_to_btc_equity: float


# ----------------------------- Helpers ------------------------------------ #

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _freq_minutes(freq: str) -> float:
    """
    Convert a freq string like '1min' or '1m' to minutes.
    """
    f = freq.lower().strip()
    if f in ("1min", "1m", "min", "t"):
        return 1.0
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
    if start <= 0:
        return 0.0

    mins = _freq_minutes(freq)
    years = (len(equity) * mins) / (365.0 * 24.0 * 60.0)
    if years <= 0:
        return 0.0

    ann_return = (end / start) ** (1.0 / years) - 1.0
    return float(ann_return / md)


def _corr_safe(a: pd.Series, b: pd.Series) -> float:
    """
    Compute correlation on aligned, finite values. Returns 0.0 if undefined.
    """
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 3:
        return 0.0
    c = df.iloc[:, 0].corr(df.iloc[:, 1])
    return float(c) if np.isfinite(c) else 0.0


def _portfolio_from_returns(returns: pd.Series, freq: str) -> vbt.Portfolio:
    """
    vectorbt portfolio using precomputed returns (no re-trading).
    """
    return vbt.Portfolio.from_returns(returns, freq=freq)


# ----------------------------- Scenario transforms ------------------------- #

def apply_scenario(
    base_returns: pd.Series,
    *,
    funding_drag_daily: float,
    extra_slippage_per_exit: float,
    freq: str,
) -> pd.Series:
    """
    Apply scenario-level penalties.

    - funding drag: subtract per-bar constant drag
    - slippage stress: subtract an extra cost ONLY when a trade closes (nonzero returns)
      because your returns_matrix has spikes at exits.
    """
    r = base_returns.copy().fillna(0.0).astype(float)

    # Funding drag per bar
    drag_per_bar = _per_bar_funding_drag(funding_drag_daily, freq)
    if drag_per_bar != 0.0:
        r = r - drag_per_bar

    # Extra slippage only on trade exit events
    if extra_slippage_per_exit != 0.0:
        exit_events = r != 0.0
        r.loc[exit_events] = r.loc[exit_events] - float(extra_slippage_per_exit)

    return r


def default_scenarios() -> Tuple[ScenarioSpec, ScenarioSpec, ScenarioSpec]:
    """
    You can override these via config if you want, but defaults match your plan.
    """
    base_drag = 0.0001   # 0.01% daily
    stress_drag = 0.0003 # 0.03% daily

    # Stress slippage: extra return hit per trade exit (heuristic).
    # If your returns are in "return per pair capital", this is directly comparable.
    stress_extra_slip = getattr(cfg, "STRESS_EXTRA_SLIPPAGE_PER_EXIT", 0.0005)  # 5 bps per exit event

    return (
        ScenarioSpec("optimistic", funding_drag_daily=0.0,     extra_slippage_per_exit=0.0,              notes="No funding drag"),
        ScenarioSpec("base_case",  funding_drag_daily=base_drag, extra_slippage_per_exit=0.0,            notes="0.01% daily funding drag"),
        ScenarioSpec("stress",     funding_drag_daily=stress_drag, extra_slippage_per_exit=stress_extra_slip,
                    notes="0.03% daily funding drag + extra slippage per exit"),
    )


# ----------------------------- Metrics + plots ----------------------------- #

def compute_metrics(
    scenario_name: str,
    scenario_returns: pd.Series,
    btc_prices: Optional[pd.Series],
    *,
    freq: str,
) -> ScenarioMetrics:
    """
    Compute the required metrics for one scenario.
    """
    # Portfolio (vectorbt) for Sharpe-like metrics
    pf = _portfolio_from_returns(scenario_returns, freq=freq)

    equity = pf.value()
    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if len(equity) else 0.0

    # Sharpe (vectorbt uses annualization based on freq internally; we still sanity-check)
    try:
        sharpe = float(pf.sharpe_ratio())
        if not np.isfinite(sharpe):
            sharpe = 0.0
    except Exception:
        sharpe = 0.0

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
        sharpe=sharpe,
        calmar=float(calmar if np.isfinite(calmar) else 0.0),
        max_drawdown=float(max_dd),
        max_dd_duration_minutes=float(dd_dur_min),
        corr_to_btc_returns=float(corr_btc_ret),
        corr_to_btc_equity=float(corr_btc_eq),
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

    # Portfolio return series = sum across pairs (your architecture assumes per-pair capital)
    portfolio_returns = returns_matrix.sum(axis=1).astype(float).fillna(0.0)

    # BTC series for correlation
    btc_prices = None
    if test_prices is not None and not test_prices.empty:
        if btc_symbol in test_prices.columns:
            btc_prices = pd.to_numeric(test_prices[btc_symbol], errors="coerce").reindex(portfolio_returns.index).ffill()
        else:
            logger.warning("BTC symbol '%s' not found in test_prices columns. Skipping BTC correlation.", btc_symbol)

    # Scenario set
    scenarios = scenarios or default_scenarios()

    scenario_equity: Dict[str, pd.Series] = {}
    scenario_returns: Dict[str, pd.Series] = {}
    metrics: Dict[str, Dict] = {}

    # Precompute BTC equity if available
    btc_equity = None
    if btc_prices is not None and not btc_prices.empty:
        btc_ret = btc_prices.pct_change().fillna(0.0)
        btc_equity = _equity_from_returns(btc_ret, init_cash=1.0)

    for spec in scenarios:
        r_s = apply_scenario(
            portfolio_returns,
            funding_drag_daily=spec.funding_drag_daily,
            extra_slippage_per_exit=spec.extra_slippage_per_exit,
            freq=freq,
        )
        scenario_returns[spec.name] = r_s
        eq_s = _equity_from_returns(r_s, init_cash=1.0)
        scenario_equity[spec.name] = eq_s

        m = compute_metrics(spec.name, r_s, btc_prices, freq=freq)
        metrics[spec.name] = asdict(m)

    # Save metrics.json
    out_metrics = {
        "freq": freq,
        "btc_symbol": btc_symbol,
        "scenario_specs": [asdict(s) for s in scenarios],
        "metrics": metrics,
        "portfolio_return_definition": "sum(returns_matrix[pairs]) per bar; nonzero returns occur at trade exits",
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
