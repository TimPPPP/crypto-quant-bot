"""
src/backtest/window_analysis.py

Phase 5B: Walk-Forward Window Analysis

Analyzes market conditions during walk-forward windows to understand what
differentiates winning vs losing windows. This helps identify regime patterns
and improve the regime filter.

Key insight from backtest: Windows 5 & 11 contributed -4.64% loss (62% of total).
Understanding what made these windows fail helps build better regime detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest.window_analysis")


@dataclass
class WindowMetrics:
    """Metrics for a single walk-forward window."""
    window_idx: int
    pnl: float
    trade_count: int
    stop_loss_count: int
    win_rate: float

    # Market conditions
    btc_return: float
    btc_volatility: float
    cross_dispersion: float
    avg_correlation: float

    # Derived
    stop_loss_rate: float = field(init=False)
    is_losing: bool = field(init=False)

    def __post_init__(self):
        self.stop_loss_rate = self.stop_loss_count / self.trade_count if self.trade_count > 0 else 0
        self.is_losing = self.pnl < -0.005  # > 0.5% loss


@dataclass
class WindowAnalysis:
    """
    Analyze market conditions during walk-forward windows.

    Compares winning vs losing windows to identify patterns that can
    improve regime detection and filtering.
    """

    # Threshold for "losing" window
    loss_threshold: float = -0.005  # -0.5%

    # Collected window data
    window_metrics: List[WindowMetrics] = field(default_factory=list)

    def record_window(
        self,
        window_idx: int,
        pnl: float,
        trade_count: int,
        stop_loss_count: int,
        win_rate: float,
        btc_prices: pd.Series,
        all_returns: pd.DataFrame,
    ) -> WindowMetrics:
        """
        Record metrics for a completed window.

        Parameters
        ----------
        window_idx : int
            Window index
        pnl : float
            Total P&L for the window
        trade_count : int
            Number of trades in the window
        stop_loss_count : int
            Number of stop-loss exits
        win_rate : float
            Fraction of winning trades
        btc_prices : pd.Series
            BTC prices during the test period
        all_returns : pd.DataFrame
            Returns for all coins during the test period

        Returns
        -------
        WindowMetrics
            Computed metrics for the window
        """
        # Compute BTC metrics
        btc_return = (btc_prices.iloc[-1] / btc_prices.iloc[0] - 1) if len(btc_prices) > 1 else 0.0
        btc_returns = btc_prices.pct_change().dropna()
        btc_volatility = btc_returns.std() * np.sqrt(96 * 365) if len(btc_returns) > 0 else 0.0  # Annualized

        # Compute cross-sectional dispersion
        cross_dispersion = all_returns.std(axis=1).mean() if not all_returns.empty else 0.0

        # Compute average pairwise correlation
        if not all_returns.empty and all_returns.shape[1] > 1:
            corr_matrix = all_returns.corr()
            # Get upper triangle values (excluding diagonal)
            upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            avg_correlation = upper_triangle.mean() if len(upper_triangle) > 0 else 0.0
        else:
            avg_correlation = 0.0

        metrics = WindowMetrics(
            window_idx=window_idx,
            pnl=pnl,
            trade_count=trade_count,
            stop_loss_count=stop_loss_count,
            win_rate=win_rate,
            btc_return=btc_return,
            btc_volatility=btc_volatility,
            cross_dispersion=cross_dispersion,
            avg_correlation=avg_correlation,
        )

        self.window_metrics.append(metrics)
        return metrics

    def get_comparison_report(self) -> pd.DataFrame:
        """
        Compare market conditions in losing vs winning windows.

        Returns
        -------
        pd.DataFrame
            Comparison of metrics between losing and winning windows
        """
        if not self.window_metrics:
            return pd.DataFrame()

        # Convert to DataFrame
        records = [
            {
                "window_idx": m.window_idx,
                "pnl": m.pnl,
                "trade_count": m.trade_count,
                "stop_loss_rate": m.stop_loss_rate,
                "win_rate": m.win_rate,
                "btc_return": m.btc_return,
                "btc_volatility": m.btc_volatility,
                "cross_dispersion": m.cross_dispersion,
                "avg_correlation": m.avg_correlation,
                "is_losing": m.is_losing,
            }
            for m in self.window_metrics
        ]
        df = pd.DataFrame(records)

        # Split into losing and winning
        losers = df[df["is_losing"]]
        winners = df[~df["is_losing"]]

        # Compute comparison
        metric_cols = [
            "pnl", "trade_count", "stop_loss_rate", "win_rate",
            "btc_return", "btc_volatility", "cross_dispersion", "avg_correlation"
        ]

        comparison_data = {}
        for col in metric_cols:
            comparison_data[col] = {
                "losing_mean": losers[col].mean() if len(losers) > 0 else np.nan,
                "winning_mean": winners[col].mean() if len(winners) > 0 else np.nan,
                "delta": (losers[col].mean() - winners[col].mean()) if len(losers) > 0 and len(winners) > 0 else np.nan,
                "losing_count": len(losers),
                "winning_count": len(winners),
            }

        comparison = pd.DataFrame(comparison_data).T
        return comparison

    def get_detailed_report(self) -> pd.DataFrame:
        """
        Get detailed per-window metrics.

        Returns
        -------
        pd.DataFrame
            Per-window metrics sorted by P&L
        """
        if not self.window_metrics:
            return pd.DataFrame()

        records = [
            {
                "window_idx": m.window_idx,
                "pnl": m.pnl,
                "pnl_pct": m.pnl * 100,
                "trade_count": m.trade_count,
                "stop_loss_rate": m.stop_loss_rate * 100,
                "win_rate": m.win_rate * 100,
                "btc_return": m.btc_return * 100,
                "btc_volatility": m.btc_volatility * 100,
                "cross_dispersion": m.cross_dispersion * 10000,  # bps
                "avg_correlation": m.avg_correlation,
                "is_losing": m.is_losing,
            }
            for m in self.window_metrics
        ]
        df = pd.DataFrame(records)
        return df.sort_values("pnl")

    def log_summary(self) -> None:
        """Log a summary of window analysis."""
        if not self.window_metrics:
            logger.info("Window analysis: No windows recorded")
            return

        n_losing = sum(1 for m in self.window_metrics if m.is_losing)
        n_winning = len(self.window_metrics) - n_losing

        losing_pnl = sum(m.pnl for m in self.window_metrics if m.is_losing)
        winning_pnl = sum(m.pnl for m in self.window_metrics if not m.is_losing)

        # Compute average conditions for losing windows
        losing_metrics = [m for m in self.window_metrics if m.is_losing]
        if losing_metrics:
            avg_btc_vol_losing = np.mean([m.btc_volatility for m in losing_metrics])
            avg_dispersion_losing = np.mean([m.cross_dispersion for m in losing_metrics])
            avg_stop_rate_losing = np.mean([m.stop_loss_rate for m in losing_metrics])
        else:
            avg_btc_vol_losing = avg_dispersion_losing = avg_stop_rate_losing = 0

        logger.info(
            "Window analysis: %d losing (%.2f%% pnl), %d winning (%.2f%% pnl)",
            n_losing, losing_pnl * 100, n_winning, winning_pnl * 100
        )

        if n_losing > 0:
            logger.info(
                "Losing window characteristics: btc_vol=%.1f%%, dispersion=%.1f bps, stop_rate=%.1f%%",
                avg_btc_vol_losing * 100, avg_dispersion_losing * 10000, avg_stop_rate_losing * 100
            )

        # Identify worst windows
        worst = sorted(self.window_metrics, key=lambda m: m.pnl)[:3]
        if worst:
            worst_info = ", ".join([f"W{m.window_idx}({m.pnl*100:.2f}%)" for m in worst])
            logger.info("Worst windows: %s", worst_info)

    def save_to_csv(self, output_path: str) -> None:
        """Save window analysis to CSV."""
        detailed = self.get_detailed_report()
        if not detailed.empty:
            detailed.to_csv(output_path, index=False)
            logger.info("Window analysis saved to %s", output_path)

    def get_regime_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations for regime filter parameters based on analysis.

        Returns
        -------
        Dict[str, Any]
            Recommended parameter adjustments
        """
        comparison = self.get_comparison_report()
        if comparison.empty:
            return {}

        recommendations = {}

        # Check BTC volatility difference
        btc_vol_delta = comparison.loc["btc_volatility", "delta"]
        if not np.isnan(btc_vol_delta) and btc_vol_delta > 0.1:
            recommendations["btc_volatility"] = {
                "observation": f"Losing windows have {btc_vol_delta*100:.1f}% higher BTC volatility",
                "suggestion": "Consider tightening REGIME_BTC_VOL_MAX_PERCENTILE",
            }

        # Check dispersion difference
        disp_delta = comparison.loc["cross_dispersion", "delta"]
        if not np.isnan(disp_delta) and disp_delta > 0.0001:
            recommendations["dispersion"] = {
                "observation": f"Losing windows have {disp_delta*10000:.1f} bps higher dispersion",
                "suggestion": "Consider tightening REGIME_DISPERSION_MAX_PERCENTILE",
            }

        # Check stop loss rate
        stop_delta = comparison.loc["stop_loss_rate", "delta"]
        if not np.isnan(stop_delta) and stop_delta > 0.1:
            recommendations["stop_loss"] = {
                "observation": f"Losing windows have {stop_delta*100:.1f}% higher stop-loss rate",
                "suggestion": "Consider stricter entry criteria or wider stops",
            }

        # Check correlation
        corr_delta = comparison.loc["avg_correlation", "delta"]
        if not np.isnan(corr_delta) and corr_delta < -0.1:
            recommendations["correlation"] = {
                "observation": f"Losing windows have {abs(corr_delta):.2f} lower average correlation",
                "suggestion": "Pairs break down when correlations drop; consider correlation-based regime filter",
            }

        return recommendations


def create_window_analysis() -> WindowAnalysis:
    """Factory function to create a WindowAnalysis instance."""
    return WindowAnalysis()
