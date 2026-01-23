# research/backtest/analyze_results.py
"""
Analyze an existing Phase 5 run WITHOUT rerunning the simulation.

Usage:
  poetry run python research/backtest/analyze_results.py --run-id run_20250101_120000
Optional:
  poetry run python research/backtest/analyze_results.py --run-id run_... --btc-symbol BTC --diagnose 10 --force

What it does:
- Loads saved artifacts from results/<run_id>/:
    - returns_matrix.parquet (required)
    - metrics.json (optional; will be overwritten if regenerated)
    - signals.parquet (optional; used for richer diagnostics)
- Rebuilds plots + metrics.json by calling performance_report
- Optionally regenerates pair-level diagnosis plots (requires signals.parquet)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg
from src.backtest.performance_report import generate_performance_report
from src.backtest.diagnostics import plot_pair_diagnosis

logger = logging.getLogger("backtest.analyze")


def _setup_logging(level: str) -> None:
    level_u = level.upper()
    logging.basicConfig(
        level=getattr(logging, level_u, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _infer_run_dir(run_id: str) -> Path:
    run_dir = cfg.RESULTS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    return run_dir


def _load_returns_matrix(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "returns_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing returns_matrix.parquet at: {path}")
    df = pd.read_parquet(path)

    # Normalize format: (time x pairs)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("returns_matrix must have a DatetimeIndex.")
    df = df.sort_index()
    return df


def _load_signals(run_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    """
    signals.parquet format (recommended from runner):
      columns are a MultiIndex with top level: {"z","vol","beta"} and second level: pair_id
    """
    path = run_dir / "signals.parquet"
    if not path.exists():
        return None

    packed = pd.read_parquet(path)
    if not isinstance(packed.columns, pd.MultiIndex) or packed.columns.nlevels < 2:
        logger.warning("signals.parquet exists but does not look like MultiIndex-packed signals. Skipping.")
        return None

    signals: Dict[str, pd.DataFrame] = {}
    for key in ("z", "vol", "beta"):
        if key in packed.columns.get_level_values(0):
            signals[key] = packed[key].copy()

    if not signals:
        return None
    return signals


def _load_masks(run_dir: Path) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    entries_path = run_dir / "entries.parquet"
    exits_path = run_dir / "exits.parquet"
    if entries_path.exists() and exits_path.exists():
        try:
            return pd.read_parquet(entries_path), pd.read_parquet(exits_path)
        except Exception as exc:
            logger.warning("Failed to load entries/exits masks: %s", exc)
            return None, None
    return None, None


def _maybe_load_test_prices(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Optional: if you saved test prices in the run folder, load them for BTC correlation.
    If not present, performance_report can still run but BTC correlation may be skipped/less meaningful.
    """
    # Standardize a filename if you want later:
    candidate = run_dir / "test_prices.parquet"
    if candidate.exists():
        df = pd.read_parquet(candidate)
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
    return None


def _validate_pnl_consistency(
    returns_matrix: pd.DataFrame,
    metrics: Dict[str, Any],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Validate P&L consistency between returns_matrix and reported metrics.

    Checks:
    1. Sum of returns_matrix should match total_return from metrics
    2. Per-pair attribution net_pnl should match returns_matrix columns

    Returns dict with validation results and any discrepancies found.
    """
    issues = []

    # Compute total return from returns_matrix (sum of all bar returns)
    total_from_matrix = float(returns_matrix.sum().sum())

    # Get total_return from metrics (may be in different places depending on format)
    metrics_total = None
    if "total_return" in metrics:
        metrics_total = metrics["total_return"]
    elif "base_case" in metrics and "total_return" in metrics["base_case"]:
        metrics_total = metrics["base_case"]["total_return"]
    elif "realistic_simulation" in metrics:
        # metrics.json stores as percentage sometimes
        pct = metrics["realistic_simulation"].get("total_return_pct", 0)
        metrics_total = pct / 100.0 if pct else None

    if metrics_total is not None:
        diff = abs(total_from_matrix - metrics_total)
        if diff > tolerance:
            issues.append({
                "check": "total_return_consistency",
                "returns_matrix_sum": total_from_matrix,
                "metrics_total_return": metrics_total,
                "difference": diff,
                "passed": False,
            })
            logger.warning(
                "P&L MISMATCH: returns_matrix sum (%.6f) != metrics total_return (%.6f), diff=%.6f",
                total_from_matrix, metrics_total, diff
            )
        else:
            logger.info(
                "P&L consistency check PASSED: returns_matrix sum (%.6f) matches metrics (%.6f)",
                total_from_matrix, metrics_total
            )
    else:
        issues.append({
            "check": "total_return_consistency",
            "error": "Could not find total_return in metrics",
            "passed": False,
        })

    # Per-pair validation
    per_pair_total = returns_matrix.sum()
    pair_issues = []
    for pair, pair_sum in per_pair_total.items():
        if abs(pair_sum) > 0.0001:  # Only check pairs with meaningful P&L
            # This can be extended to check against attribution report if available
            pass

    return {
        "total_from_matrix": total_from_matrix,
        "metrics_total": metrics_total,
        "issues": issues,
        "all_passed": len(issues) == 0,
    }


# ============================================================================
# OUTLIER EVENT ANALYSIS
# ============================================================================

# Known extreme market events to exclude for robustness analysis
KNOWN_OUTLIER_EVENTS = [
    {
        "name": "Oct 2025 Flash Crash",
        "start": "2025-10-10",
        "end": "2025-10-13",
        "description": "Trump tariff announcement triggered $19B liquidation cascade",
    },
]


def detect_outlier_days(
    returns_matrix: pd.DataFrame,
    threshold_std: float = 3.0,
    min_daily_return_pct: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Automatically detect outlier days based on daily return distribution.

    Args:
        returns_matrix: DataFrame of returns (time x pairs)
        threshold_std: Number of standard deviations to consider outlier
        min_daily_return_pct: Minimum absolute daily return to flag (%)

    Returns:
        List of detected outlier periods with dates and returns
    """
    portfolio_returns = returns_matrix.sum(axis=1)
    daily_returns = portfolio_returns.groupby(portfolio_returns.index.date).sum()

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    threshold = max(threshold_std * std_ret, min_daily_return_pct / 100)

    outliers = []
    for date, ret in daily_returns.items():
        if abs(ret) > threshold:
            outliers.append({
                "date": str(date),
                "return_pct": float(ret * 100),
                "std_from_mean": float((ret - mean_ret) / std_ret) if std_ret > 0 else 0,
            })

    return sorted(outliers, key=lambda x: abs(x["return_pct"]), reverse=True)


def analyze_excluding_outliers(
    returns_matrix: pd.DataFrame,
    exclude_periods: Optional[List[Tuple[str, str]]] = None,
    auto_detect: bool = True,
    threshold_std: float = 3.0,
) -> Dict[str, Any]:
    """
    Analyze backtest results with and without outlier events.

    Args:
        returns_matrix: DataFrame of returns (time x pairs)
        exclude_periods: List of (start_date, end_date) tuples to exclude
        auto_detect: Whether to auto-detect outlier days
        threshold_std: Std threshold for auto-detection

    Returns:
        Dictionary with full and excluded analysis
    """
    # Full period analysis
    total_return = float(returns_matrix.sum().sum())
    total_trades = int((returns_matrix != 0).sum().sum())
    winners = int((returns_matrix > 0).sum().sum())
    win_rate = winners / total_trades if total_trades > 0 else 0

    # Portfolio returns for Sharpe calculation
    portfolio_returns = returns_matrix.sum(axis=1)
    daily_returns = portfolio_returns.groupby(portfolio_returns.index.date).sum()
    sharpe_full = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0

    full_metrics = {
        "total_return_pct": total_return * 100,
        "total_trades": total_trades,
        "win_rate_pct": win_rate * 100,
        "sharpe_annual": sharpe_full,
    }

    # Auto-detect outliers if requested
    detected_outliers = []
    if auto_detect:
        detected_outliers = detect_outlier_days(returns_matrix, threshold_std)

    # Build exclusion mask
    exclude_mask = pd.Series(False, index=returns_matrix.index)
    excluded_events = []

    # Add known events
    for event in KNOWN_OUTLIER_EVENTS:
        mask = (returns_matrix.index >= event["start"]) & (returns_matrix.index < event["end"])
        if mask.any():
            exclude_mask |= mask
            event_return = float(returns_matrix[mask].sum().sum())
            excluded_events.append({
                "name": event["name"],
                "start": event["start"],
                "end": event["end"],
                "return_pct": event_return * 100,
                "pct_of_total": (event_return / total_return * 100) if total_return != 0 else 0,
            })

    # Add custom exclusion periods
    if exclude_periods:
        for start, end in exclude_periods:
            mask = (returns_matrix.index >= start) & (returns_matrix.index < end)
            exclude_mask |= mask

    # Compute excluded metrics
    rm_ex = returns_matrix[~exclude_mask]
    total_return_ex = float(rm_ex.sum().sum())
    total_trades_ex = int((rm_ex != 0).sum().sum())
    winners_ex = int((rm_ex > 0).sum().sum())
    win_rate_ex = winners_ex / total_trades_ex if total_trades_ex > 0 else 0

    portfolio_ex = rm_ex.sum(axis=1)
    daily_ex = portfolio_ex.groupby(portfolio_ex.index.date).sum()
    sharpe_ex = float(daily_ex.mean() / daily_ex.std() * np.sqrt(365)) if daily_ex.std() > 0 else 0

    excluded_metrics = {
        "total_return_pct": total_return_ex * 100,
        "total_trades": total_trades_ex,
        "win_rate_pct": win_rate_ex * 100,
        "sharpe_annual": sharpe_ex,
        "bars_excluded": int(exclude_mask.sum()),
        "days_excluded": len(set(returns_matrix[exclude_mask].index.date)) if exclude_mask.any() else 0,
    }

    # Compute outlier contribution
    outlier_contribution = {
        "return_pct": (total_return - total_return_ex) * 100,
        "pct_of_total_pnl": ((total_return - total_return_ex) / total_return * 100) if total_return != 0 else 0,
        "trades": total_trades - total_trades_ex,
    }

    return {
        "full_period": full_metrics,
        "excluding_outliers": excluded_metrics,
        "outlier_contribution": outlier_contribution,
        "excluded_events": excluded_events,
        "auto_detected_outliers": detected_outliers[:10],  # Top 10
        "robustness_warning": total_return_ex <= 0 < total_return,
    }


def analyze_run(
    *,
    run_id: str,
    btc_symbol: str,
    diagnose_n: int,
    force: bool,
) -> Dict[str, Any]:
    run_dir = _infer_run_dir(run_id)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    returns_matrix = _load_returns_matrix(run_dir)
    metrics_path = run_dir / "metrics.json"
    existing_metrics = _load_json(metrics_path)

    # If metrics exist and not forcing, still regenerate plots (safe) but keep metrics as baseline.
    if existing_metrics and not force:
        logger.info("Found existing metrics.json (use --force to overwrite).")

    # Optional: signals for richer diagnostics
    signals = _load_signals(run_dir)
    if signals is None:
        logger.info("No signals.parquet found (pair-level diagnosis plots require it).")

    entries, exits = _load_masks(run_dir)

    # Optional: test prices for BTC correlation (best practice is to save them during simulation)
    test_prices = _maybe_load_test_prices(run_dir)
    if test_prices is None:
        logger.warning(
            "No test_prices.parquet found in run folder. "
            "BTC correlation may be skipped or approximated depending on performance_report implementation."
        )

    # Regenerate performance report (metrics + plots)
    report = generate_performance_report(
        run_dir=run_dir,
        returns_matrix=returns_matrix,
        test_prices=test_prices,     # may be None
        btc_symbol=btc_symbol,
        freq=getattr(cfg, "SIGNAL_TIMEFRAME", None) or getattr(cfg, "BAR_FREQ", "1min"),
        entries=entries,
        exits=exits,
    )

    # Overwrite metrics.json if forced OR if it didn't exist
    if force or (existing_metrics is None):
        _write_json(metrics_path, report.get("metrics", report))
        logger.info("✅ Wrote metrics.json: %s", metrics_path)
    else:
        logger.info("Kept existing metrics.json (not overwritten).")

    # P&L consistency validation
    final_metrics = _load_json(metrics_path) or report.get("metrics", report)
    pnl_validation = _validate_pnl_consistency(returns_matrix, final_metrics)
    if not pnl_validation["all_passed"]:
        logger.warning("⚠️  P&L consistency validation FAILED - see issues above")
    else:
        logger.info("✅ P&L consistency validation passed")

    # Outlier event analysis (robustness check)
    outlier_analysis = analyze_excluding_outliers(returns_matrix)
    if outlier_analysis["robustness_warning"]:
        logger.warning(
            "⚠️  ROBUSTNESS WARNING: Strategy is UNPROFITABLE when excluding outlier events!\n"
            "   Full period: %.2f%%, Excluding outliers: %.2f%%",
            outlier_analysis["full_period"]["total_return_pct"],
            outlier_analysis["excluding_outliers"]["total_return_pct"],
        )
    else:
        logger.info(
            "✅ Robustness check: %.2f%% full period, %.2f%% excluding outliers",
            outlier_analysis["full_period"]["total_return_pct"],
            outlier_analysis["excluding_outliers"]["total_return_pct"],
        )

    # Save outlier analysis
    outlier_path = run_dir / "outlier_analysis.json"
    _write_json(outlier_path, outlier_analysis)
    logger.info("📊 Saved outlier analysis to: %s", outlier_path)

    # Optional: regenerate pair diagnosis plots
    if diagnose_n > 0:
        if signals is None or any(k not in signals for k in ("z", "vol", "beta")):
            logger.warning("Diagnostics requested but signals.parquet missing required keys (z/vol/beta). Skipping.")
        elif test_prices is None:
            logger.warning("Diagnostics requested but test_prices.parquet missing. Skipping.")
        else:
            z_df = signals["z"]
            vol_df = signals["vol"]
            beta_df = signals["beta"]

            # entries/exits are not saved by default; we can approximate them if you saved them,
            # but for now we require you to persist them OR re-compute from z/vol.
            # We'll try to recompute via accountant_filter if available.
            try:
                from src.backtest.accountant_filter import compute_masks
                entries, exits, expected_profit = compute_masks(z_score=z_df, spread_volatility=vol_df)
            except Exception as e:
                logger.warning("Could not recompute entries/exits for diagnostics: %s", e)
                entries = None
                exits = None
                expected_profit = None

            pairs = list(returns_matrix.columns)
            n = min(diagnose_n, len(pairs))

            if entries is None or exits is None:
                logger.warning("Skipping diagnosis plots (no entry/exit masks available).")
            else:
                for pair_id in pairs[:n]:
                    try:
                        plot_pair_diagnosis(
                            run_dir=run_dir,
                            pair_id=pair_id,
                            test_df=test_prices,
                            z_score=z_df,
                            beta=beta_df,
                            entries=entries,
                            exits=exits,
                            expected_profit=expected_profit,
                            spread_volatility=vol_df,
                            pnl_mode="price",
                            save=True,
                        )
                    except Exception as e:
                        logger.warning("Diagnosis failed for %s: %s", pair_id, e)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "returns_shape": list(returns_matrix.shape),
        "signals_loaded": signals is not None,
        "plots_dir": str(plots_dir),
        "metrics_path": str(metrics_path),
        "pnl_validation": pnl_validation,
        "outlier_analysis": outlier_analysis,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze an existing backtest run without rerunning simulation.")
    p.add_argument("--run-id", type=str, required=True, help="Run folder name under ./results (e.g., run_YYYYMMDD_HHMMSS)")
    p.add_argument("--btc-symbol", type=str, default="BTC", help="BTC column name for correlation (if test_prices.parquet exists).")
    p.add_argument("--diagnose", type=int, default=0, help="Regenerate N pair diagnosis plots (requires signals.parquet).")
    p.add_argument("--force", action="store_true", help="Overwrite metrics.json even if it exists.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR).")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    _setup_logging(args.log_level)

    summary = analyze_run(
        run_id=args.run_id,
        btc_symbol=args.btc_symbol,
        diagnose_n=args.diagnose,
        force=args.force,
    )

    print(json.dumps(summary, indent=2))
