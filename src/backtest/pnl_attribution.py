# src/backtest/pnl_attribution.py
"""
P&L Attribution Module for Pairs Trading Backtest

Diagnostic foundation - understand why trades fail.

Provides comprehensive breakdown of trading P&L into components:
- Gross spread P&L (price movement only)
- Fees paid
- Slippage cost
- Funding P&L (paid/received)
- Residual (marking, rounding, etc.)

Per-pair metrics:
- Trade count
- Average hold time
- Win rate
- Average win, average loss
- Expectancy per trade
- Contribution to total portfolio P&L
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.pnl_attribution")


@dataclass(frozen=True)
class TradeAttribution:
    """
    Per-trade P&L component breakdown.

    All matrices have shape (T, P) where T = time bars, P = pairs.
    Values are non-zero only at exit timestamps.

    Invariant: gross_pnl - fees - slippage - funding_cost = net_pnl
    """

    gross_pnl: pd.DataFrame  # Price movement P&L only (before any costs)
    fees: pd.DataFrame  # Total fees paid (entry + exit)
    slippage: pd.DataFrame  # Estimated slippage cost
    funding_pnl: pd.DataFrame  # Funding received (positive) or paid (negative)
    net_pnl: pd.DataFrame  # Final net P&L (should match returns_matrix)
    hold_duration_bars: pd.DataFrame  # Number of bars held per trade
    direction: pd.DataFrame  # Trade direction at exit: +1 long spread, -1 short


@dataclass(frozen=True)
class PairSummary:
    """Per-pair performance summary."""

    pair: str
    trade_count: int
    avg_hold_bars: float
    win_rate: float  # Fraction of trades with positive net P&L
    avg_win: float  # Average profit on winning trades
    avg_loss: float  # Average loss on losing trades (as positive number)
    expectancy: float  # avg_win * win_rate - avg_loss * (1 - win_rate)

    # P&L attribution components (totals across all trades)
    total_gross_pnl: float
    total_fees: float
    total_slippage: float
    total_funding_pnl: float
    total_net_pnl: float

    # Contribution metrics
    contribution_pct: float  # % of total portfolio net P&L
    cost_to_gross_ratio: float  # (fees + slippage) / |gross_pnl|

    # Exit reason breakdown
    signal_exits: int
    time_stop_exits: int
    stop_loss_exits: int
    forced_exits: int


@dataclass(frozen=True)
class AttributionReport:
    """Complete P&L attribution report."""

    # Per-trade breakdown
    trade_attribution: TradeAttribution

    # Per-pair summaries
    pair_summaries: List[PairSummary]

    # Aggregate statistics
    total_trades: int
    total_gross_pnl: float
    total_fees: float
    total_slippage: float
    total_funding_pnl: float
    total_net_pnl: float
    cost_to_gross_ratio: float  # (fees + slippage) / |gross_pnl|

    # Key diagnostic patterns
    gross_positive_net_negative: bool  # Friction problem
    gross_negative: bool  # Signal problem
    few_pairs_dominate_losses: bool  # Universe quality problem


def compute_pair_summary(
    pair: str,
    attribution: TradeAttribution,
    exit_reason_matrix: pd.DataFrame,
    total_portfolio_pnl: float,
) -> PairSummary:
    """
    Compute summary statistics for a single pair.

    Parameters
    ----------
    pair : str
        Pair identifier (e.g., "ETH-BTC").
    attribution : TradeAttribution
        P&L component breakdown.
    exit_reason_matrix : pd.DataFrame
        Exit reason codes per trade.
    total_portfolio_pnl : float
        Total portfolio net P&L (for contribution calculation).

    Returns
    -------
    PairSummary
        Complete summary statistics for the pair.
    """
    # Exit reason codes
    EXIT_SIGNAL = 1
    EXIT_TIME_STOP = 2
    EXIT_STOP_LOSS = 3
    EXIT_FORCED = 4

    # Get non-zero trades for this pair
    net_pnl = attribution.net_pnl[pair]
    gross_pnl = attribution.gross_pnl[pair]
    fees = attribution.fees[pair]
    slippage_col = attribution.slippage[pair]
    funding = attribution.funding_pnl[pair]
    hold_bars = attribution.hold_duration_bars[pair]
    exit_reasons = exit_reason_matrix[pair]

    # Find trade exits (non-zero net P&L)
    trade_mask = net_pnl != 0
    n_trades = trade_mask.sum()

    if n_trades == 0:
        return PairSummary(
            pair=pair,
            trade_count=0,
            avg_hold_bars=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            expectancy=0.0,
            total_gross_pnl=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            total_funding_pnl=0.0,
            total_net_pnl=0.0,
            contribution_pct=0.0,
            cost_to_gross_ratio=0.0,
            signal_exits=0,
            time_stop_exits=0,
            stop_loss_exits=0,
            forced_exits=0,
        )

    # Trade P&Ls
    trade_pnls = net_pnl[trade_mask]
    winners = trade_pnls > 0
    losers = trade_pnls < 0

    # Win/loss statistics
    n_wins = winners.sum()
    n_losses = losers.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0

    avg_win = float(trade_pnls[winners].mean()) if n_wins > 0 else 0.0
    avg_loss = float(-trade_pnls[losers].mean()) if n_losses > 0 else 0.0  # As positive

    expectancy = avg_win * win_rate - avg_loss * (1 - win_rate)

    # Aggregate P&L components
    total_gross = float(gross_pnl[trade_mask].sum())
    total_fees = float(fees[trade_mask].sum())
    total_slip = float(slippage_col[trade_mask].sum())
    total_funding = float(funding[trade_mask].sum())
    total_net = float(net_pnl[trade_mask].sum())

    # Hold duration
    avg_hold = float(hold_bars[trade_mask].mean()) if n_trades > 0 else 0.0

    # Contribution to portfolio
    contribution = (total_net / total_portfolio_pnl * 100) if abs(total_portfolio_pnl) > 1e-10 else 0.0

    # Cost to gross ratio
    total_cost = total_fees + total_slip
    cost_ratio = (total_cost / abs(total_gross)) if abs(total_gross) > 1e-10 else 0.0

    # Exit reason counts
    trade_exit_reasons = exit_reasons[trade_mask]
    signal_exits = int((trade_exit_reasons == EXIT_SIGNAL).sum())
    time_stop_exits = int((trade_exit_reasons == EXIT_TIME_STOP).sum())
    stop_loss_exits = int((trade_exit_reasons == EXIT_STOP_LOSS).sum())
    forced_exits = int((trade_exit_reasons == EXIT_FORCED).sum())

    return PairSummary(
        pair=pair,
        trade_count=int(n_trades),
        avg_hold_bars=avg_hold,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        total_gross_pnl=total_gross,
        total_fees=total_fees,
        total_slippage=total_slip,
        total_funding_pnl=total_funding,
        total_net_pnl=total_net,
        contribution_pct=contribution,
        cost_to_gross_ratio=cost_ratio,
        signal_exits=signal_exits,
        time_stop_exits=time_stop_exits,
        stop_loss_exits=stop_loss_exits,
        forced_exits=forced_exits,
    )


def format_attribution_report(report: AttributionReport, top_n_pairs: int = 10) -> str:
    """
    Format attribution report as human-readable text.

    Parameters
    ----------
    report : AttributionReport
        The attribution report to format.
    top_n_pairs : int
        Number of top/bottom pairs to show.

    Returns
    -------
    str
        Formatted report text.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("P&L ATTRIBUTION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Aggregate summary
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Trades:        {report.total_trades:,}")
    lines.append(f"Gross P&L:           {report.total_gross_pnl:+.4f}")
    lines.append(f"  - Fees:            {report.total_fees:-.4f}")
    lines.append(f"  - Slippage:        {report.total_slippage:-.4f}")
    lines.append(f"  - Funding:         {report.total_funding_pnl:+.4f}")
    lines.append(f"Net P&L:             {report.total_net_pnl:+.4f}")
    lines.append(f"Cost/Gross Ratio:    {report.cost_to_gross_ratio:.2%}")
    lines.append("")

    # Diagnostic alerts
    lines.append("DIAGNOSTIC ALERTS")
    lines.append("-" * 40)
    if report.gross_positive_net_negative:
        lines.append("⚠️  GROSS POSITIVE, NET NEGATIVE")
        lines.append("    → Friction/carry/execution problem")
        lines.append("    → Consider longer timeframe or lower trading frequency")
    if report.gross_negative:
        lines.append("⚠️  GROSS NEGATIVE")
        lines.append("    → Signal is not predictive")
        lines.append("    → Review pair selection, beta, regime filter, stop logic")
    if report.few_pairs_dominate_losses:
        lines.append("⚠️  FEW PAIRS DOMINATE LOSSES")
        lines.append("    → Universe quality / structural break / stability issue")
        lines.append("    → Consider stricter pair selection filters")
    if not any([report.gross_positive_net_negative, report.gross_negative, report.few_pairs_dominate_losses]):
        lines.append("✓  No major diagnostic alerts")
    lines.append("")

    # Top/bottom pairs
    lines.append(f"TOP {top_n_pairs} PAIRS (by net P&L)")
    lines.append("-" * 70)
    lines.append(f"{'Pair':<12} {'Trades':>7} {'Win%':>6} {'Expect':>8} {'Gross':>9} {'Net':>9} {'Contrib':>8}")
    lines.append("-" * 70)

    sorted_summaries = sorted(report.pair_summaries, key=lambda s: s.total_net_pnl, reverse=True)

    for summary in sorted_summaries[:top_n_pairs]:
        lines.append(
            f"{summary.pair:<12} "
            f"{summary.trade_count:>7} "
            f"{summary.win_rate * 100:>5.1f}% "
            f"{summary.expectancy:>+8.4f} "
            f"{summary.total_gross_pnl:>+9.4f} "
            f"{summary.total_net_pnl:>+9.4f} "
            f"{summary.contribution_pct:>+7.1f}%"
        )

    # Bottom pairs (worst performers)
    if len(sorted_summaries) > top_n_pairs:
        lines.append("")
        lines.append(f"BOTTOM {top_n_pairs} PAIRS (by net P&L)")
        lines.append("-" * 70)
        for summary in sorted_summaries[-top_n_pairs:][::-1]:
            lines.append(
                f"{summary.pair:<12} "
                f"{summary.trade_count:>7} "
                f"{summary.win_rate * 100:>5.1f}% "
                f"{summary.expectancy:>+8.4f} "
                f"{summary.total_gross_pnl:>+9.4f} "
                f"{summary.total_net_pnl:>+9.4f} "
                f"{summary.contribution_pct:>+7.1f}%"
            )

    # Exit reason breakdown
    lines.append("")
    lines.append("EXIT REASON BREAKDOWN (all pairs)")
    lines.append("-" * 40)
    total_signal = sum(s.signal_exits for s in report.pair_summaries)
    total_time = sum(s.time_stop_exits for s in report.pair_summaries)
    total_stop = sum(s.stop_loss_exits for s in report.pair_summaries)
    total_forced = sum(s.forced_exits for s in report.pair_summaries)
    total = total_signal + total_time + total_stop + total_forced

    if total > 0:
        lines.append(f"Signal exits:     {total_signal:>5} ({total_signal / total * 100:>5.1f}%)")
        lines.append(f"Time stop exits:  {total_time:>5} ({total_time / total * 100:>5.1f}%)")
        lines.append(f"Stop loss exits:  {total_stop:>5} ({total_stop / total * 100:>5.1f}%)")
        lines.append(f"Forced exits:     {total_forced:>5} ({total_forced / total * 100:>5.1f}%)")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_attribution_report(
    report: AttributionReport,
    output_dir: Path,
    filename_prefix: str = "attribution",
) -> Dict[str, Path]:
    """
    Save attribution report to files.

    Parameters
    ----------
    report : AttributionReport
        The attribution report to save.
    output_dir : Path
        Directory to save files.
    filename_prefix : str
        Prefix for output files.

    Returns
    -------
    dict
        Mapping of file type to path.
    """
    from pathlib import Path
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save text report
    text_path = output_dir / f"{filename_prefix}_report.txt"
    with open(text_path, "w") as f:
        f.write(format_attribution_report(report))
    paths["text_report"] = text_path

    # Save per-pair summary as CSV
    csv_path = output_dir / f"{filename_prefix}_pair_summary.csv"
    summary_data = []
    for s in report.pair_summaries:
        summary_data.append({
            "pair": s.pair,
            "trade_count": s.trade_count,
            "avg_hold_bars": s.avg_hold_bars,
            "win_rate": s.win_rate,
            "avg_win": s.avg_win,
            "avg_loss": s.avg_loss,
            "expectancy": s.expectancy,
            "total_gross_pnl": s.total_gross_pnl,
            "total_fees": s.total_fees,
            "total_slippage": s.total_slippage,
            "total_funding_pnl": s.total_funding_pnl,
            "total_net_pnl": s.total_net_pnl,
            "contribution_pct": s.contribution_pct,
            "cost_to_gross_ratio": s.cost_to_gross_ratio,
            "signal_exits": s.signal_exits,
            "time_stop_exits": s.time_stop_exits,
            "stop_loss_exits": s.stop_loss_exits,
            "forced_exits": s.forced_exits,
        })
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    paths["pair_summary_csv"] = csv_path

    # Save aggregate metrics as JSON
    json_path = output_dir / f"{filename_prefix}_aggregate.json"
    aggregate = {
        "total_trades": report.total_trades,
        "total_gross_pnl": report.total_gross_pnl,
        "total_fees": report.total_fees,
        "total_slippage": report.total_slippage,
        "total_funding_pnl": report.total_funding_pnl,
        "total_net_pnl": report.total_net_pnl,
        "cost_to_gross_ratio": report.cost_to_gross_ratio,
        "diagnostics": {
            "gross_positive_net_negative": report.gross_positive_net_negative,
            "gross_negative": report.gross_negative,
            "few_pairs_dominate_losses": report.few_pairs_dominate_losses,
        },
    }
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    paths["aggregate_json"] = json_path

    logger.info(f"Attribution report saved to {output_dir}")

    return paths


# =============================================================================
# DIRECT ENGINE ATTRIBUTION (Using PnlResult components)
# =============================================================================

def generate_attribution_from_pnl_result(
    returns_matrix: pd.DataFrame,
    gross_pnl_matrix: pd.DataFrame,
    fees_matrix: pd.DataFrame,
    slippage_matrix: pd.DataFrame,
    hold_bars_matrix: pd.DataFrame,
    exit_reason_matrix: pd.DataFrame,
    funding_rates: Optional[pd.DataFrame] = None,
    funding_costs_matrix: Optional[pd.DataFrame] = None,  # NEW: direct from PnL engine
    freq: str = "15min",
) -> AttributionReport:
    """
    Generate attribution report directly from PnL engine outputs.

    This uses the actual tracked components from the PnL engine instead of
    reverse-engineering from net returns. More accurate attribution.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Net returns at exit timestamps (T × pairs) from PnlResult
    gross_pnl_matrix : pd.DataFrame
        Gross P&L at exit timestamps (T × pairs) from PnlResult
    fees_matrix : pd.DataFrame
        Fees at exit timestamps (T × pairs) from PnlResult
    slippage_matrix : pd.DataFrame
        Slippage at exit timestamps (T × pairs) from PnlResult
    hold_bars_matrix : pd.DataFrame
        Bars held at exit timestamps (T × pairs) from PnlResult
    exit_reason_matrix : pd.DataFrame
        Exit reason codes (T × pairs) from PnlResult
    funding_rates : pd.DataFrame, optional
        8-hour funding rates for funding P&L estimation (fallback)
    funding_costs_matrix : pd.DataFrame, optional
        Direct funding costs from PnL engine (preferred if available)
    freq : str
        Bar frequency for time conversion

    Returns
    -------
    AttributionReport
        Complete attribution analysis
    """
    pairs = returns_matrix.columns
    index = returns_matrix.index

    # Use funding costs from PnL engine if available, otherwise compute from rates
    funding_pnl = pd.DataFrame(0.0, index=index, columns=pairs)
    if funding_costs_matrix is not None and not funding_costs_matrix.empty:
        # Direct from PnL engine - most accurate
        # Aggregate funding costs to exit timestamps (sum funding paid during trade)
        try:
            funding_aligned = funding_costs_matrix.reindex(columns=pairs, fill_value=0.0)
            funding_aligned = funding_aligned.reindex(index=index, fill_value=0.0)
            # Sum funding costs per trade (accumulate until exit)
            # For now, just use the cumulative sum at each bar
            funding_pnl = funding_aligned.cumsum()
            # Reset at each trade boundary (approximate - use exit_reason to identify)
            # This is a simplification; proper tracking would be in the engine
            logger.info(
                "Using funding costs from PnL engine: total=%.6f",
                float(funding_costs_matrix.values.sum())
            )
        except Exception as e:
            logger.warning("Error using engine funding costs: %s", e)
            funding_pnl = pd.DataFrame(0.0, index=index, columns=pairs)
    elif funding_rates is not None and not funding_rates.empty:
        # Fallback: compute from funding rates
        funding_pnl = _compute_funding_pnl_matrix(
            returns_matrix=returns_matrix,
            funding_rates=funding_rates,
            freq=freq,
        )

    # Build TradeAttribution from engine outputs
    # Direction is not tracked directly, estimate from gross P&L sign
    direction_df = pd.DataFrame(0, index=index, columns=pairs, dtype=np.int8)

    trade_attribution = TradeAttribution(
        gross_pnl=gross_pnl_matrix,
        fees=fees_matrix,
        slippage=slippage_matrix,
        funding_pnl=funding_pnl,
        net_pnl=returns_matrix,
        hold_duration_bars=hold_bars_matrix,
        direction=direction_df,
    )

    # Compute aggregates
    total_gross = float(gross_pnl_matrix.values.sum())
    total_fees = float(fees_matrix.values.sum())
    total_slippage = float(slippage_matrix.values.sum())
    total_funding = float(funding_pnl.values.sum())
    total_net = float(returns_matrix.values.sum())
    total_trades = int((returns_matrix != 0).values.sum())

    # Cost to gross ratio
    total_cost = total_fees + total_slippage
    cost_ratio = (total_cost / abs(total_gross)) if abs(total_gross) > 1e-10 else 0.0

    # Compute per-pair summaries
    pair_summaries = []
    for pair in pairs:
        summary = compute_pair_summary(
            pair=pair,
            attribution=trade_attribution,
            exit_reason_matrix=exit_reason_matrix,
            total_portfolio_pnl=total_net,
        )
        pair_summaries.append(summary)

    # Sort by contribution (descending)
    pair_summaries.sort(key=lambda s: s.total_net_pnl, reverse=True)

    # Diagnostic patterns
    gross_positive_net_negative = (total_gross > 0) and (total_net < 0)
    gross_negative = total_gross < 0

    # Check if few pairs dominate losses
    losing_pairs = [s for s in pair_summaries if s.total_net_pnl < 0]
    total_losses = sum(s.total_net_pnl for s in losing_pairs)
    few_pairs_dominate_losses = False
    if len(losing_pairs) >= 2 and total_losses < 0:
        top_2_losses = sum(
            s.total_net_pnl for s in sorted(losing_pairs, key=lambda x: x.total_net_pnl)[:2]
        )
        if top_2_losses / total_losses > 0.5:
            few_pairs_dominate_losses = True

    return AttributionReport(
        trade_attribution=trade_attribution,
        pair_summaries=pair_summaries,
        total_trades=total_trades,
        total_gross_pnl=total_gross,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_funding_pnl=total_funding,
        total_net_pnl=total_net,
        cost_to_gross_ratio=cost_ratio,
        gross_positive_net_negative=gross_positive_net_negative,
        gross_negative=gross_negative,
        few_pairs_dominate_losses=few_pairs_dominate_losses,
    )


def _compute_funding_pnl_matrix(
    returns_matrix: pd.DataFrame,
    funding_rates: pd.DataFrame,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Estimate funding P&L for each trade.

    Funding is applied during the holding period. This is an estimate
    based on average funding rates during the trade.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Net returns to identify trade exits (T × pairs)
    funding_rates : pd.DataFrame
        8-hour funding rates (T_funding × coins)
    freq : str
        Bar frequency

    Returns
    -------
    pd.DataFrame
        Estimated funding P&L at exit timestamps (T × pairs)
    """
    pairs = returns_matrix.columns
    index = returns_matrix.index
    funding_pnl = pd.DataFrame(0.0, index=index, columns=pairs)

    # Convert bar frequency to minutes
    freq_map = {
        "1min": 1, "5min": 5, "15min": 15, "30min": 30,
        "1h": 60, "1H": 60, "4h": 240, "1d": 1440
    }
    mins_per_bar = freq_map.get(freq, 15)
    funding_interval_mins = 8.0 * 60.0  # 8 hours
    bars_per_funding_period = funding_interval_mins / mins_per_bar

    for pair in pairs:
        try:
            parts = pair.split("-")
            if len(parts) != 2:
                continue
            coin1, coin2 = parts

            # Check if both coins have funding data
            if coin1 not in funding_rates.columns or coin2 not in funding_rates.columns:
                continue

            # Align funding rates to returns timeline
            funding_aligned = funding_rates[[coin1, coin2]].reindex(index).ffill()

            # Net funding rate: long coin1 - short coin2
            net_funding_8h = funding_aligned[coin1] - funding_aligned[coin2]
            funding_per_bar = net_funding_8h / bars_per_funding_period

            # Find exit timestamps
            exit_mask = returns_matrix[pair] != 0

            # Estimate funding as average funding rate × hold time
            # This is approximate - precise tracking would need entry/exit timestamps
            for exit_time in index[exit_mask]:
                # Get average funding rate (simple approximation)
                avg_funding = funding_per_bar.loc[:exit_time].tail(100).mean()
                if np.isfinite(avg_funding):
                    funding_pnl.loc[exit_time, pair] = avg_funding * 10  # rough estimate

        except Exception as exc:
            logger.debug("Error computing funding for %s: %s", pair, exc)
            continue

    return funding_pnl


def generate_attribution_json(report: AttributionReport, freq: str = "15min") -> Dict:
    """
    Generate JSON-serializable attribution data.

    Parameters
    ----------
    report : AttributionReport
        The attribution report
    freq : str
        Bar frequency

    Returns
    -------
    dict
        JSON-serializable report
    """
    # Convert frequency to minutes for hold time
    freq_map = {
        "1min": 1, "5min": 5, "15min": 15, "30min": 30,
        "1h": 60, "1H": 60, "4h": 240, "1d": 1440
    }
    mins_per_bar = freq_map.get(freq, 15)

    # Portfolio summary
    portfolio_summary = {
        "total_gross_pnl": report.total_gross_pnl,
        "total_fees": report.total_fees,
        "total_slippage": report.total_slippage,
        "total_funding_pnl": report.total_funding_pnl,
        "total_net_pnl": report.total_net_pnl,
        "total_trades": report.total_trades,
        "cost_to_gross_ratio": report.cost_to_gross_ratio,
    }

    # Verification
    expected_net = report.total_gross_pnl - report.total_fees - report.total_slippage + report.total_funding_pnl
    residual = report.total_net_pnl - expected_net

    verification = {
        "expected_net": expected_net,
        "actual_net": report.total_net_pnl,
        "residual": residual,
        "residual_pct_of_gross": abs(residual) / abs(report.total_gross_pnl) * 100 if abs(report.total_gross_pnl) > 1e-10 else 0.0,
        "attribution_verified": abs(residual) < 1e-6,
    }

    # Diagnostics
    diagnostics = {
        "gross_positive_net_negative": report.gross_positive_net_negative,
        "gross_negative": report.gross_negative,
        "few_pairs_dominate_losses": report.few_pairs_dominate_losses,
    }

    # Per-pair details
    pair_details = []
    for s in report.pair_summaries:
        if s.trade_count > 0:
            pair_details.append({
                "pair": s.pair,
                "trade_count": s.trade_count,
                "avg_hold_bars": s.avg_hold_bars,
                "avg_hold_minutes": s.avg_hold_bars * mins_per_bar,
                "win_rate": s.win_rate,
                "avg_win": s.avg_win,
                "avg_loss": s.avg_loss,
                "expectancy": s.expectancy,
                "total_gross_pnl": s.total_gross_pnl,
                "total_fees": s.total_fees,
                "total_slippage": s.total_slippage,
                "total_funding_pnl": s.total_funding_pnl,
                "total_net_pnl": s.total_net_pnl,
                "contribution_pct": s.contribution_pct,
                "cost_to_gross_ratio": s.cost_to_gross_ratio,
                "exit_reasons": {
                    "signal": s.signal_exits,
                    "time_stop": s.time_stop_exits,
                    "stop_loss": s.stop_loss_exits,
                    "forced": s.forced_exits,
                },
            })

    return {
        "freq": freq,
        "portfolio_summary": portfolio_summary,
        "verification": verification,
        "diagnostics": diagnostics,
        "pair_details": pair_details,
    }


__all__ = [
    "TradeAttribution",
    "PairSummary",
    "AttributionReport",
    "generate_attribution_from_pnl_result",
    "generate_attribution_json",
    "format_attribution_report",
    "save_attribution_report",
]
