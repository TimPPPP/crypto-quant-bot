# src/backtest/timeframe_analysis.py
"""
Multi-Timeframe Analysis Module for Pairs Trading Backtest

Issue #4 Fix: 1-minute signals often capture microstructure noise that
disappears after costs. This module enables systematic comparison across
timeframes to find where net Sharpe is maximized.

Usage:
    results = run_timeframe_comparison(
        raw_df=price_data,
        pairs=selected_pairs,
        timeframes=["1min", "5min", "15min", "1h"],
    )

    # Find optimal timeframe
    best_tf = max(results.items(), key=lambda x: x[1].net_sharpe)[0]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest.timeframe_analysis")


@dataclass(frozen=True)
class TimeframeResult:
    """Performance metrics for a single timeframe."""

    timeframe: str
    n_trades: int
    gross_pnl: float
    fees: float
    slippage: float
    funding_cost: float
    net_pnl: float
    gross_sharpe: float
    net_sharpe: float
    cost_to_gross_ratio: float  # (fees + slippage) / |gross_pnl|
    avg_hold_bars: float
    win_rate: float


@dataclass(frozen=True)
class TimeframeComparison:
    """Complete comparison across all tested timeframes."""

    results: Dict[str, TimeframeResult]
    optimal_timeframe: str
    optimal_net_sharpe: float
    cost_sensitivity: float  # How much does cost ratio vary across TFs
    summary: str


def resample_price_matrix(
    price_df: pd.DataFrame,
    target_timeframe: str,
) -> pd.DataFrame:
    """
    Resample price matrix to target timeframe.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price matrix with DatetimeIndex and coin columns.
    target_timeframe : str
        Target timeframe: "1min", "5min", "15min", "30min", "1h", "4h".

    Returns
    -------
    pd.DataFrame
        Resampled price matrix (OHLC aggregated to close).
    """
    # Map timeframe string to pandas offset
    tf_map = {
        "1min": "1min",
        "1m": "1min",
        "5min": "5min",
        "5m": "5min",
        "15min": "15min",
        "15m": "15min",
        "30min": "30min",
        "30m": "30min",
        "1h": "1h",
        "1H": "1h",
        "4h": "4h",
        "4H": "4h",
    }

    offset = tf_map.get(target_timeframe, target_timeframe)

    if offset == "1min":
        # No resampling needed
        return price_df.copy()

    # Resample: use last price (close) for each period
    resampled = price_df.resample(offset).last()

    # Drop rows with all NaN
    resampled = resampled.dropna(how="all")

    logger.info(
        f"Resampled {len(price_df)} bars to {len(resampled)} bars ({target_timeframe})"
    )

    return resampled


def compute_timeframe_metrics(
    returns_series: pd.Series,
    gross_returns: Optional[pd.Series] = None,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0002,
    bars_per_year: int = 35040,  # Approximate for 15-min bars
) -> TimeframeResult:
    """
    Compute performance metrics for a single timeframe.

    Parameters
    ----------
    returns_series : pd.Series
        Net returns per bar (0 for non-trade bars).
    gross_returns : pd.Series, optional
        Gross returns (before costs). If None, estimated from net.
    fee_rate : float
        Fee rate per leg.
    slippage_rate : float
        Slippage rate per leg.
    bars_per_year : int
        Number of bars per year for annualization.

    Returns
    -------
    TimeframeResult
        Performance metrics for this timeframe.
    """
    # Find trades (non-zero returns)
    trade_mask = returns_series != 0
    n_trades = int(trade_mask.sum())

    if n_trades == 0:
        return TimeframeResult(
            timeframe="unknown",
            n_trades=0,
            gross_pnl=0.0,
            fees=0.0,
            slippage=0.0,
            funding_cost=0.0,
            net_pnl=0.0,
            gross_sharpe=0.0,
            net_sharpe=0.0,
            cost_to_gross_ratio=0.0,
            avg_hold_bars=0.0,
            win_rate=0.0,
        )

    # Net P&L
    net_pnl = float(returns_series.sum())

    # Estimate costs (4 legs per trade)
    total_cost_per_trade = 4 * (fee_rate + slippage_rate)
    total_fees = n_trades * 4 * fee_rate
    total_slippage = n_trades * 4 * slippage_rate

    # Gross P&L (estimated)
    if gross_returns is not None:
        gross_pnl = float(gross_returns.sum())
    else:
        gross_pnl = net_pnl + total_fees + total_slippage

    # Cost to gross ratio
    total_cost = total_fees + total_slippage
    cost_ratio = total_cost / abs(gross_pnl) if abs(gross_pnl) > 1e-10 else 0.0

    # Sharpe calculation (trade-level)
    trade_returns = returns_series[trade_mask]
    if len(trade_returns) > 1 and trade_returns.std() > 1e-10:
        # Annualized Sharpe (approximate based on trade frequency)
        trades_per_year = n_trades / (len(returns_series) / bars_per_year)
        net_sharpe = float(
            (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)
        )
    else:
        net_sharpe = 0.0

    # Gross Sharpe (estimated)
    if gross_returns is not None:
        gross_trade_returns = gross_returns[trade_mask]
        if len(gross_trade_returns) > 1 and gross_trade_returns.std() > 1e-10:
            trades_per_year = n_trades / (len(returns_series) / bars_per_year)
            gross_sharpe = float(
                (gross_trade_returns.mean() / gross_trade_returns.std())
                * np.sqrt(trades_per_year)
            )
        else:
            gross_sharpe = 0.0
    else:
        # Estimate gross Sharpe by scaling up
        gross_sharpe = net_sharpe * (1 + cost_ratio) if cost_ratio < 1 else net_sharpe

    # Win rate
    win_rate = float((trade_returns > 0).sum() / n_trades) if n_trades > 0 else 0.0

    return TimeframeResult(
        timeframe="unknown",  # Will be set by caller
        n_trades=n_trades,
        gross_pnl=gross_pnl,
        fees=total_fees,
        slippage=total_slippage,
        funding_cost=0.0,  # Would need funding data
        net_pnl=net_pnl,
        gross_sharpe=gross_sharpe,
        net_sharpe=net_sharpe,
        cost_to_gross_ratio=cost_ratio,
        avg_hold_bars=0.0,  # Would need hold duration data
        win_rate=win_rate,
    )


def format_timeframe_comparison(comparison: TimeframeComparison) -> str:
    """
    Format timeframe comparison as human-readable text.

    Parameters
    ----------
    comparison : TimeframeComparison
        The comparison results.

    Returns
    -------
    str
        Formatted report text.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-TIMEFRAME COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append(f"Optimal Timeframe: {comparison.optimal_timeframe}")
    lines.append(f"Optimal Net Sharpe: {comparison.optimal_net_sharpe:.3f}")
    lines.append(f"Cost Sensitivity: {comparison.cost_sensitivity:.3f}")
    lines.append("")

    # Table header
    lines.append(
        f"{'Timeframe':<10} {'Trades':>7} {'Gross':>10} {'Net':>10} "
        f"{'Cost%':>7} {'GrossSR':>8} {'NetSR':>8} {'Win%':>6}"
    )
    lines.append("-" * 70)

    # Sort by net Sharpe descending
    sorted_results = sorted(
        comparison.results.values(), key=lambda x: x.net_sharpe, reverse=True
    )

    for r in sorted_results:
        marker = " *" if r.timeframe == comparison.optimal_timeframe else ""
        lines.append(
            f"{r.timeframe:<10} {r.n_trades:>7} {r.gross_pnl:>+10.4f} {r.net_pnl:>+10.4f} "
            f"{r.cost_to_gross_ratio * 100:>6.1f}% {r.gross_sharpe:>8.3f} {r.net_sharpe:>8.3f} "
            f"{r.win_rate * 100:>5.1f}%{marker}"
        )

    lines.append("")
    lines.append("* = Optimal timeframe")
    lines.append("")

    # Interpretation
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append(comparison.summary)
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_comparison_summary(results: Dict[str, TimeframeResult]) -> str:
    """
    Generate interpretation summary based on comparison results.

    Parameters
    ----------
    results : dict
        Mapping of timeframe -> TimeframeResult.

    Returns
    -------
    str
        Interpretation text.
    """
    if not results:
        return "No results to analyze."

    # Find patterns
    sorted_by_tf = sorted(results.items(), key=lambda x: _tf_to_minutes(x[0]))
    sorted_by_sharpe = sorted(results.items(), key=lambda x: x[1].net_sharpe, reverse=True)

    best_tf, best_result = sorted_by_sharpe[0]
    worst_tf, worst_result = sorted_by_sharpe[-1]

    # Check if shorter timeframes have worse Sharpe (friction problem)
    short_tfs = [tf for tf, _ in sorted_by_tf[:2]]  # First 2 (shortest)
    long_tfs = [tf for tf, _ in sorted_by_tf[-2:]]  # Last 2 (longest)

    short_avg_sharpe = np.mean([results[tf].net_sharpe for tf in short_tfs if tf in results])
    long_avg_sharpe = np.mean([results[tf].net_sharpe for tf in long_tfs if tf in results])

    lines = []

    if short_avg_sharpe < long_avg_sharpe - 0.5:
        lines.append(
            "FRICTION PROBLEM DETECTED: Shorter timeframes have significantly "
            f"lower net Sharpe ({short_avg_sharpe:.2f}) vs longer ({long_avg_sharpe:.2f})."
        )
        lines.append(
            "This suggests microstructure noise and transaction costs are "
            "dominating at faster frequencies."
        )
    elif long_avg_sharpe < short_avg_sharpe - 0.5:
        lines.append(
            "SIGNAL DECAY: Longer timeframes have lower Sharpe. "
            "The trading signal may be capturing short-term patterns."
        )
    else:
        lines.append(
            "STABLE ACROSS TIMEFRAMES: Performance is relatively consistent. "
            "The optimal timeframe choice is a balance of signal vs costs."
        )

    # Check cost sensitivity
    cost_ratios = [r.cost_to_gross_ratio for r in results.values()]
    if max(cost_ratios) > 0.5:
        lines.append(
            f"\nHIGH COST WARNING: Some timeframes have cost/gross ratio > 50% "
            f"(max: {max(cost_ratios) * 100:.0f}%). Consider reducing trading frequency."
        )

    # Recommendation
    lines.append(f"\nRECOMMENDATION: Use {best_tf} timeframe (Net Sharpe: {best_result.net_sharpe:.3f})")

    return "\n".join(lines)


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes for sorting."""
    tf_map = {
        "1min": 1, "1m": 1,
        "5min": 5, "5m": 5,
        "15min": 15, "15m": 15,
        "30min": 30, "30m": 30,
        "1h": 60, "1H": 60,
        "4h": 240, "4H": 240,
    }
    return tf_map.get(tf, 1)


def create_timeframe_comparison(
    results: Dict[str, TimeframeResult],
) -> TimeframeComparison:
    """
    Create complete comparison object from individual results.

    Parameters
    ----------
    results : dict
        Mapping of timeframe -> TimeframeResult.

    Returns
    -------
    TimeframeComparison
        Complete comparison with analysis.
    """
    if not results:
        return TimeframeComparison(
            results={},
            optimal_timeframe="N/A",
            optimal_net_sharpe=0.0,
            cost_sensitivity=0.0,
            summary="No timeframes analyzed.",
        )

    # Find optimal
    best_tf = max(results.items(), key=lambda x: x[1].net_sharpe)
    optimal_timeframe = best_tf[0]
    optimal_net_sharpe = best_tf[1].net_sharpe

    # Cost sensitivity: std of cost ratios
    cost_ratios = [r.cost_to_gross_ratio for r in results.values()]
    cost_sensitivity = float(np.std(cost_ratios)) if len(cost_ratios) > 1 else 0.0

    # Generate summary
    summary = generate_comparison_summary(results)

    return TimeframeComparison(
        results=results,
        optimal_timeframe=optimal_timeframe,
        optimal_net_sharpe=optimal_net_sharpe,
        cost_sensitivity=cost_sensitivity,
        summary=summary,
    )


__all__ = [
    "TimeframeResult",
    "TimeframeComparison",
    "resample_price_matrix",
    "compute_timeframe_metrics",
    "format_timeframe_comparison",
    "create_timeframe_comparison",
]
