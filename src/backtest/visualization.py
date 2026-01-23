# src/backtest/visualization.py
"""
Advanced visualization module for backtest results.

Provides comprehensive diagnostic plots beyond basic equity curves:
- Per-pair performance heatmaps
- Trade duration distributions
- Exit reason breakdowns
- Expected vs realized P&L analysis
- Correlation heatmaps
- Walk-forward progression charts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

logger = logging.getLogger("backtest.visualization")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_pair_performance_heatmap(
    pair_summaries: List[Dict],
    output_path: Path,
    metric: str = "total_net_pnl",
    title: str = "Pair Performance Heatmap",
) -> Path:
    """
    Create a heatmap showing per-pair performance metrics.

    Parameters
    ----------
    pair_summaries : List[Dict]
        List of pair summary dictionaries (from pnl_attribution)
    output_path : Path
        Where to save the plot
    metric : str
        Which metric to display ("total_net_pnl", "win_rate", "expectancy", etc.)
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    if not pair_summaries:
        logger.warning("No pair summaries provided for heatmap")
        return output_path

    # Extract unique coins and build matrix
    coins_y = set()
    coins_x = set()

    for s in pair_summaries:
        pair = s.get("pair", "")
        parts = pair.split("-")
        if len(parts) == 2:
            coins_y.add(parts[0])
            coins_x.add(parts[1])

    coins_y = sorted(coins_y)
    coins_x = sorted(coins_x)

    if not coins_y or not coins_x:
        logger.warning("Could not extract coins from pair summaries")
        return output_path

    # Build matrix
    matrix = pd.DataFrame(np.nan, index=coins_y, columns=coins_x)

    for s in pair_summaries:
        pair = s.get("pair", "")
        parts = pair.split("-")
        if len(parts) == 2:
            y, x = parts
            if y in matrix.index and x in matrix.columns:
                matrix.loc[y, x] = s.get(metric, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use diverging colormap centered at 0 for P&L metrics
    if "pnl" in metric.lower() or "expectancy" in metric.lower():
        vmax = max(abs(matrix.min().min()), abs(matrix.max().max()))
        if np.isnan(vmax) or vmax == 0:
            vmax = 1
        cmap = "RdYlGn"
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        cmap = "YlOrRd"
        norm = None

    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(coins_x)))
    ax.set_yticks(np.arange(len(coins_y)))
    ax.set_xticklabels(coins_x, rotation=45, ha="right")
    ax.set_yticklabels(coins_y)

    # Add value annotations
    for i in range(len(coins_y)):
        for j in range(len(coins_x)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                       color=text_color, fontsize=8)

    ax.set_xlabel("Short Leg (X)")
    ax.set_ylabel("Long Leg (Y)")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved pair performance heatmap: %s", output_path)
    return output_path


def plot_trade_duration_histogram(
    pair_summaries: List[Dict],
    output_path: Path,
    freq: str = "15min",
    title: str = "Trade Duration Distribution",
) -> Path:
    """
    Create histogram of trade durations across all pairs.

    Parameters
    ----------
    pair_summaries : List[Dict]
        List of pair summary dictionaries
    output_path : Path
        Where to save the plot
    freq : str
        Bar frequency for time conversion
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    # Convert frequency to hours
    freq_map = {
        "1min": 1/60, "5min": 5/60, "15min": 15/60, "30min": 30/60,
        "1h": 1, "1H": 1, "4h": 4, "1d": 24
    }
    hours_per_bar = freq_map.get(freq, 0.25)

    # Collect hold durations
    hold_hours = []
    for s in pair_summaries:
        avg_hold_bars = s.get("avg_hold_bars", 0)
        trade_count = s.get("trade_count", 0)
        if avg_hold_bars > 0 and trade_count > 0:
            # Weight by trade count
            for _ in range(trade_count):
                hold_hours.append(avg_hold_bars * hours_per_bar)

    if not hold_hours:
        logger.warning("No hold duration data available")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No trade duration data", ha="center", va="center")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins, patches = ax.hist(hold_hours, bins=30, edgecolor="black", alpha=0.7)

    # Color by duration bucket
    colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    # Add statistics
    median_hours = np.median(hold_hours)
    mean_hours = np.mean(hold_hours)

    ax.axvline(median_hours, color="red", linestyle="--", linewidth=2,
              label=f"Median: {median_hours:.1f}h")
    ax.axvline(mean_hours, color="orange", linestyle="-.", linewidth=2,
              label=f"Mean: {mean_hours:.1f}h")

    ax.set_xlabel("Trade Duration (hours)")
    ax.set_ylabel("Number of Trades")
    ax.set_title(title)
    ax.legend()

    # Add summary stats box
    stats_text = f"Total Trades: {len(hold_hours)}\n"
    stats_text += f"Min: {min(hold_hours):.1f}h\n"
    stats_text += f"Max: {max(hold_hours):.1f}h\n"
    stats_text += f"Std: {np.std(hold_hours):.1f}h"

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment="top", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved trade duration histogram: %s", output_path)
    return output_path


def plot_exit_reasons_pie(
    pair_summaries: List[Dict],
    output_path: Path,
    title: str = "Exit Reason Breakdown",
) -> Path:
    """
    Create pie chart showing exit reason distribution.

    Parameters
    ----------
    pair_summaries : List[Dict]
        List of pair summary dictionaries
    output_path : Path
        Where to save the plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    # Aggregate exit reasons
    signal_exits = sum(s.get("signal_exits", 0) for s in pair_summaries)
    time_stop_exits = sum(s.get("time_stop_exits", 0) for s in pair_summaries)
    stop_loss_exits = sum(s.get("stop_loss_exits", 0) for s in pair_summaries)
    forced_exits = sum(s.get("forced_exits", 0) for s in pair_summaries)

    # Filter out zeros
    labels = []
    sizes = []
    colors = []
    color_map = {
        "Signal (Mean Reversion)": "#2ecc71",  # Green
        "Time Stop": "#f39c12",  # Orange
        "Stop Loss": "#e74c3c",  # Red
        "Forced Exit": "#9b59b6",  # Purple
    }

    if signal_exits > 0:
        labels.append("Signal (Mean Reversion)")
        sizes.append(signal_exits)
        colors.append(color_map["Signal (Mean Reversion)"])
    if time_stop_exits > 0:
        labels.append("Time Stop")
        sizes.append(time_stop_exits)
        colors.append(color_map["Time Stop"])
    if stop_loss_exits > 0:
        labels.append("Stop Loss")
        sizes.append(stop_loss_exits)
        colors.append(color_map["Stop Loss"])
    if forced_exits > 0:
        labels.append("Forced Exit")
        sizes.append(forced_exits)
        colors.append(color_map["Forced Exit"])

    if not sizes:
        logger.warning("No exit reason data available")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No exit reason data", ha="center", va="center")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, explode=[0.02] * len(sizes)
    )
    ax1.set_title(title)

    # Bar chart with P&L breakdown by exit reason
    # Calculate P&L by exit type
    exit_pnl = {
        "Signal": 0.0,
        "Time Stop": 0.0,
        "Stop Loss": 0.0,
        "Forced": 0.0,
    }
    exit_counts = {
        "Signal": signal_exits,
        "Time Stop": time_stop_exits,
        "Stop Loss": stop_loss_exits,
        "Forced": forced_exits,
    }

    for s in pair_summaries:
        # Estimate P&L contribution by exit type (rough approximation)
        total_pnl = s.get("total_net_pnl", 0)
        total_exits = (s.get("signal_exits", 0) + s.get("time_stop_exits", 0) +
                      s.get("stop_loss_exits", 0) + s.get("forced_exits", 0))
        if total_exits > 0:
            pnl_per_exit = total_pnl / total_exits
            # Weight by exit type (simplified - assumes uniform P&L per exit)
            # In reality, signal exits are usually profitable, stop losses not
            exit_pnl["Signal"] += pnl_per_exit * s.get("signal_exits", 0)
            exit_pnl["Time Stop"] += pnl_per_exit * s.get("time_stop_exits", 0)
            exit_pnl["Stop Loss"] += pnl_per_exit * s.get("stop_loss_exits", 0)
            exit_pnl["Forced"] += pnl_per_exit * s.get("forced_exits", 0)

    # Bar chart
    bar_labels = [k for k, v in exit_counts.items() if v > 0]
    bar_values = [exit_counts[k] for k in bar_labels]
    bar_colors = [
        "#2ecc71" if "Signal" in l else
        "#f39c12" if "Time" in l else
        "#e74c3c" if "Stop" in l else "#9b59b6"
        for l in bar_labels
    ]

    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, edgecolor="black")

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ax2.set_xlabel("Exit Reason")
    ax2.set_ylabel("Number of Trades")
    ax2.set_title("Exit Counts by Type")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved exit reasons pie chart: %s", output_path)
    return output_path


def plot_expected_vs_realized_scatter(
    pair_summaries: List[Dict],
    output_path: Path,
    title: str = "Expected Profit vs Realized P&L",
) -> Path:
    """
    Scatter plot comparing expected profit per trade to realized P&L.

    Parameters
    ----------
    pair_summaries : List[Dict]
        List of pair summary dictionaries
    output_path : Path
        Where to save the plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    # Extract expectancy vs realized
    expectancies = []
    realized_pnl = []
    pair_names = []
    trade_counts = []

    for s in pair_summaries:
        if s.get("trade_count", 0) > 0:
            expectancies.append(s.get("expectancy", 0))
            realized_pnl.append(s.get("total_net_pnl", 0))
            pair_names.append(s.get("pair", ""))
            trade_counts.append(s.get("trade_count", 1))

    if not expectancies:
        logger.warning("No expectancy data available")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No expectancy data", ha="center", va="center")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scale dot size by trade count
    sizes = [max(30, min(500, tc * 20)) for tc in trade_counts]

    # Color by whether realized matched expectation
    colors = []
    for exp, real in zip(expectancies, realized_pnl):
        if exp > 0 and real > 0:
            colors.append("#2ecc71")  # Green: both positive
        elif exp < 0 and real < 0:
            colors.append("#f39c12")  # Orange: both negative (expected)
        elif exp > 0 and real < 0:
            colors.append("#e74c3c")  # Red: expected positive, got negative
        else:
            colors.append("#3498db")  # Blue: expected negative, got positive

    scatter = ax.scatter(expectancies, realized_pnl, s=sizes, c=colors, alpha=0.7, edgecolors="black")

    # Add diagonal line (perfect prediction)
    min_val = min(min(expectancies), min(realized_pnl))
    max_val = max(max(expectancies), max(realized_pnl))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect Prediction")

    # Add zero lines
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)

    # Label outliers
    for i, (exp, real, name) in enumerate(zip(expectancies, realized_pnl, pair_names)):
        if abs(real) > abs(max(realized_pnl)) * 0.5 or abs(exp) > abs(max(expectancies)) * 0.5:
            ax.annotate(name, (exp, real), fontsize=8, alpha=0.8)

    ax.set_xlabel("Expected Profit per Trade (Expectancy)")
    ax.set_ylabel("Realized Net P&L")
    ax.set_title(title)

    # Legend
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Both Positive"),
        Patch(facecolor="#f39c12", label="Both Negative"),
        Patch(facecolor="#e74c3c", label="Expected +, Got -"),
        Patch(facecolor="#3498db", label="Expected -, Got +"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Add correlation
    corr = np.corrcoef(expectancies, realized_pnl)[0, 1]
    ax.text(0.95, 0.05, f"Correlation: {corr:.3f}", transform=ax.transAxes,
           fontsize=10, verticalalignment="bottom", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved expected vs realized scatter: %s", output_path)
    return output_path


def plot_pair_correlation_heatmap(
    returns_matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Pair Returns Correlation",
) -> Path:
    """
    Create correlation heatmap for pair returns.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Returns matrix (T x pairs)
    output_path : Path
        Where to save the plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    # Compute correlation matrix on non-zero returns
    # Replace zeros with NaN for correlation (we only care about actual trades)
    returns_nonzero = returns_matrix.replace(0, np.nan)
    corr_matrix = returns_nonzero.corr()

    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        logger.warning("Not enough pairs for correlation heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Not enough pairs", ha="center", va="center")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_matrix.index, fontsize=8)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Correlation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved correlation heatmap: %s", output_path)
    return output_path


def plot_walk_forward_progression(
    window_metrics: List[Dict],
    output_path: Path,
    title: str = "Walk-Forward Performance Progression",
) -> Path:
    """
    Plot performance metrics across walk-forward windows.

    Parameters
    ----------
    window_metrics : List[Dict]
        List of per-window metric dictionaries
    output_path : Path
        Where to save the plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    if not window_metrics:
        logger.warning("No window metrics available")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "No window metrics", ha="center", va="center")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    # Extract metrics
    windows = list(range(1, len(window_metrics) + 1))
    returns = [w.get("total_return", 0) for w in window_metrics]
    trade_counts = [w.get("trade_count", 0) for w in window_metrics]
    win_rates = [w.get("win_rate", 0) for w in window_metrics]
    sharpes = [w.get("sharpe", 0) for w in window_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total return per window
    ax1 = axes[0, 0]
    colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in returns]
    ax1.bar(windows, returns, color=colors, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Return")
    ax1.set_title("Return by Window")

    # Cumulative return
    ax2 = axes[0, 1]
    cumulative = np.cumprod([1 + r for r in returns])
    ax2.plot(windows, cumulative, marker="o", linewidth=2, color="#3498db")
    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(windows, 1, cumulative, alpha=0.3,
                    where=[c >= 1 for c in cumulative], color="#2ecc71")
    ax2.fill_between(windows, 1, cumulative, alpha=0.3,
                    where=[c < 1 for c in cumulative], color="#e74c3c")
    ax2.set_xlabel("Window")
    ax2.set_ylabel("Cumulative Growth")
    ax2.set_title("Cumulative Performance")

    # Trade count per window
    ax3 = axes[1, 0]
    ax3.bar(windows, trade_counts, color="#9b59b6", edgecolor="black")
    ax3.set_xlabel("Window")
    ax3.set_ylabel("Trade Count")
    ax3.set_title("Trades per Window")

    # Win rate per window
    ax4 = axes[1, 1]
    ax4.plot(windows, [wr * 100 for wr in win_rates], marker="s", linewidth=2, color="#f39c12")
    ax4.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% (breakeven)")
    ax4.set_xlabel("Window")
    ax4.set_ylabel("Win Rate (%)")
    ax4.set_title("Win Rate by Window")
    ax4.legend()
    ax4.set_ylim(0, 100)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved walk-forward progression: %s", output_path)
    return output_path


def plot_pnl_waterfall(
    gross_pnl: float,
    fees: float,
    slippage: float,
    funding: float,
    net_pnl: float,
    output_path: Path,
    title: str = "P&L Attribution Waterfall",
) -> Path:
    """
    Create waterfall chart showing P&L breakdown.

    Parameters
    ----------
    gross_pnl : float
        Gross P&L (before costs)
    fees : float
        Total fees
    slippage : float
        Total slippage
    funding : float
        Funding P&L (can be positive or negative)
    net_pnl : float
        Net P&L
    output_path : Path
        Where to save the plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Components
    categories = ["Gross P&L", "Fees", "Slippage", "Funding", "Net P&L"]
    values = [gross_pnl, -fees, -slippage, funding, 0]  # Net will be calculated

    # Calculate cumulative for waterfall
    cumulative = [gross_pnl]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(net_pnl)

    # Bar positions
    x = np.arange(len(categories))

    # Colors
    colors = []
    for i, v in enumerate(values):
        if i == 0:  # Gross
            colors.append("#3498db" if v >= 0 else "#e74c3c")
        elif i == len(values) - 1:  # Net
            colors.append("#2ecc71" if net_pnl >= 0 else "#e74c3c")
        elif v < 0:  # Costs
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    # Create waterfall bars
    for i in range(len(categories)):
        if i == 0:
            bottom = 0
            height = cumulative[i]
        elif i == len(categories) - 1:
            bottom = 0
            height = net_pnl
        else:
            if values[i] >= 0:
                bottom = cumulative[i-1]
                height = values[i]
            else:
                bottom = cumulative[i]
                height = -values[i]

        ax.bar(x[i], height, bottom=bottom, color=colors[i], edgecolor="black", width=0.6)

        # Add value labels
        label_y = bottom + height / 2
        ax.text(x[i], label_y, f"{values[i] if i < len(values)-1 else net_pnl:+.4f}",
               ha="center", va="center", fontsize=9, fontweight="bold")

    # Connect bars with lines
    for i in range(len(categories) - 2):
        ax.plot([x[i] + 0.3, x[i+1] - 0.3], [cumulative[i], cumulative[i]], "k--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("P&L")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved P&L waterfall: %s", output_path)
    return output_path


def plot_win_loss_distribution(
    pair_summaries: List[Dict],
    output_path: Path,
    title: str = "Win/Loss Distribution by Pair",
) -> Path:
    """
    Create stacked bar chart showing wins vs losses per pair.

    Parameters
    ----------
    pair_summaries : List[Dict]
        List of pair summary dictionaries
    output_path : Path
        Where to save the plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    # Sort by trade count
    summaries = sorted(
        [s for s in pair_summaries if s.get("trade_count", 0) > 0],
        key=lambda x: x.get("trade_count", 0),
        reverse=True
    )[:20]  # Top 20 pairs

    if not summaries:
        logger.warning("No pair data for win/loss distribution")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, "No trade data", ha="center", va="center")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    pairs = [s.get("pair", "") for s in summaries]
    trade_counts = [s.get("trade_count", 0) for s in summaries]
    win_rates = [s.get("win_rate", 0) for s in summaries]

    wins = [int(tc * wr) for tc, wr in zip(trade_counts, win_rates)]
    losses = [tc - w for tc, w in zip(trade_counts, wins)]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(pairs))
    width = 0.7

    ax.bar(x, wins, width, label="Wins", color="#2ecc71", edgecolor="black")
    ax.bar(x, losses, width, bottom=wins, label="Losses", color="#e74c3c", edgecolor="black")

    # Add win rate labels
    for i, (w, l, wr) in enumerate(zip(wins, losses, win_rates)):
        ax.text(i, w + l + 0.5, f"{wr*100:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Pair")
    ax.set_ylabel("Number of Trades")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved win/loss distribution: %s", output_path)
    return output_path


def generate_all_visualizations(
    run_dir: Path,
    returns_matrix: pd.DataFrame,
    pair_summaries: List[Dict],
    window_metrics: Optional[List[Dict]] = None,
    attribution_summary: Optional[Dict] = None,
    freq: str = "15min",
) -> Dict[str, Path]:
    """
    Generate all diagnostic visualizations for a backtest run.

    Parameters
    ----------
    run_dir : Path
        Results directory
    returns_matrix : pd.DataFrame
        Returns matrix (T x pairs)
    pair_summaries : List[Dict]
        Per-pair summary data
    window_metrics : List[Dict], optional
        Per-window metrics for walk-forward
    attribution_summary : Dict, optional
        P&L attribution summary
    freq : str
        Bar frequency

    Returns
    -------
    Dict[str, Path]
        Mapping of plot names to file paths
    """
    plots_dir = run_dir / "plots"
    _ensure_dir(plots_dir)

    paths = {}

    # 1. Pair performance heatmap
    try:
        paths["pair_heatmap"] = plot_pair_performance_heatmap(
            pair_summaries=pair_summaries,
            output_path=plots_dir / "pair_performance_heatmap.png",
            metric="total_net_pnl",
            title="Pair Net P&L Heatmap (Long Y - Short X)",
        )
    except Exception as e:
        logger.warning("Failed to generate pair heatmap: %s", e)

    # 2. Trade duration histogram
    try:
        paths["duration_histogram"] = plot_trade_duration_histogram(
            pair_summaries=pair_summaries,
            output_path=plots_dir / "trade_duration_histogram.png",
            freq=freq,
        )
    except Exception as e:
        logger.warning("Failed to generate duration histogram: %s", e)

    # 3. Exit reasons pie chart
    try:
        paths["exit_reasons"] = plot_exit_reasons_pie(
            pair_summaries=pair_summaries,
            output_path=plots_dir / "exit_reasons_breakdown.png",
        )
    except Exception as e:
        logger.warning("Failed to generate exit reasons chart: %s", e)

    # 4. Expected vs realized scatter
    try:
        paths["expected_vs_realized"] = plot_expected_vs_realized_scatter(
            pair_summaries=pair_summaries,
            output_path=plots_dir / "expected_vs_realized.png",
        )
    except Exception as e:
        logger.warning("Failed to generate expected vs realized: %s", e)

    # 5. Correlation heatmap
    try:
        paths["correlation_heatmap"] = plot_pair_correlation_heatmap(
            returns_matrix=returns_matrix,
            output_path=plots_dir / "pair_correlation_heatmap.png",
        )
    except Exception as e:
        logger.warning("Failed to generate correlation heatmap: %s", e)

    # 6. Walk-forward progression (if available)
    if window_metrics:
        try:
            paths["walk_forward"] = plot_walk_forward_progression(
                window_metrics=window_metrics,
                output_path=plots_dir / "walk_forward_progression.png",
            )
        except Exception as e:
            logger.warning("Failed to generate walk-forward chart: %s", e)

    # 7. P&L waterfall (if attribution available)
    if attribution_summary:
        try:
            paths["pnl_waterfall"] = plot_pnl_waterfall(
                gross_pnl=attribution_summary.get("total_gross_pnl", 0),
                fees=attribution_summary.get("total_fees", 0),
                slippage=attribution_summary.get("total_slippage", 0),
                funding=attribution_summary.get("total_funding_pnl", 0),
                net_pnl=attribution_summary.get("total_net_pnl", 0),
                output_path=plots_dir / "pnl_waterfall.png",
            )
        except Exception as e:
            logger.warning("Failed to generate P&L waterfall: %s", e)

    # 8. Win/loss distribution
    try:
        paths["win_loss_distribution"] = plot_win_loss_distribution(
            pair_summaries=pair_summaries,
            output_path=plots_dir / "win_loss_distribution.png",
        )
    except Exception as e:
        logger.warning("Failed to generate win/loss distribution: %s", e)

    logger.info("Generated %d visualization plots in %s", len(paths), plots_dir)
    return paths


__all__ = [
    "plot_pair_performance_heatmap",
    "plot_trade_duration_histogram",
    "plot_exit_reasons_pie",
    "plot_expected_vs_realized_scatter",
    "plot_pair_correlation_heatmap",
    "plot_walk_forward_progression",
    "plot_pnl_waterfall",
    "plot_win_loss_distribution",
    "generate_all_visualizations",
]
