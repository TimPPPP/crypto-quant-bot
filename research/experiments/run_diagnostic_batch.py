#!/usr/bin/env python3
"""
Batch Experiment Runner (Phase 6) - Controlled A/B Testing.

This script runs multiple experiment configurations to isolate what works.
Each experiment modifies specific config parameters while keeping others constant.

Run:
    python research/experiments/run_diagnostic_batch.py
    # or with Docker:
    docker exec crypto_worker python research/experiments/run_diagnostic_batch.py

Experiments:
    A: Diagnostic - No regime filter, no cooldown (does strategy have edge?)
    B: Relaxed regime - Softer hard regime thresholds + smart cooldown
    C: Soft regime - 3-state soft regime + entry quality filters
    D: Sensitivity - Same as C but +0.05 dispersion threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("experiments.batch_runner")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Experiment configurations
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "run_A_diagnostic": {
        # Baseline: Does strategy have edge when allowed to trade?
        # Disable all restrictive filters to see raw signal quality
        "description": "Diagnostic: No regime filter, no cooldown - test raw signal edge",
        "config_overrides": {
            "ENABLE_REGIME_FILTER": False,
            "ENABLE_ENTRY_COOLDOWN": False,
            "ENABLE_SMART_COOLDOWN": False,
            "ENABLE_SOFT_REGIME": False,
            "ENABLE_TOPN_PAIR_SELECTION": True,
            "PAIR_SELECTION_TOP_N": 30,
            "PAIR_SELECTION_MIN_FLOOR": 0.25,
        },
    },
    "run_B_relaxed_regime": {
        # Relaxed hard regime with smart cooldown
        "description": "Relaxed hard regime + smart cooldown",
        "config_overrides": {
            "ENABLE_REGIME_FILTER": True,
            "ENABLE_SOFT_REGIME": False,
            "REGIME_BTC_VOL_MAX_PERCENTILE": 0.85,
            "REGIME_DISPERSION_MAX_PERCENTILE": 0.90,
            "ENABLE_SMART_COOLDOWN": True,
            "COOLDOWN_AFTER_SIGNAL_BARS": 12,
            "COOLDOWN_AFTER_STOP_LOSS_BARS": 48,
        },
    },
    "run_C_soft_regime": {
        # 3-state soft regime (recommended)
        "description": "Soft 3-state regime + entry quality filters",
        "config_overrides": {
            "ENABLE_REGIME_FILTER": True,
            "ENABLE_SOFT_REGIME": True,
            "REGIME_GREEN_BTC_VOL_MAX": 0.85,
            "REGIME_GREEN_DISPERSION_MAX": 0.90,
            "REGIME_YELLOW_BTC_VOL_MAX": 0.92,
            "REGIME_YELLOW_DISPERSION_MAX": 0.95,
            "ENABLE_SLOPE_FILTER": True,
            "ENABLE_CONFIRMATION_FILTER": True,
            "CONFIRMATION_BARS": 2,
        },
    },
    "run_D_sensitivity": {
        # Same as C but +0.05 dispersion threshold
        "description": "Sensitivity test: +0.05 dispersion thresholds",
        "config_overrides": {
            "ENABLE_REGIME_FILTER": True,
            "ENABLE_SOFT_REGIME": True,
            "REGIME_GREEN_BTC_VOL_MAX": 0.85,
            "REGIME_GREEN_DISPERSION_MAX": 0.95,  # +0.05
            "REGIME_YELLOW_BTC_VOL_MAX": 0.92,
            "REGIME_YELLOW_DISPERSION_MAX": 1.00,  # +0.05
            "ENABLE_SLOPE_FILTER": True,
            "ENABLE_CONFIRMATION_FILTER": True,
        },
    },
    "run_E_high_selectivity": {
        # Higher entry bar, wider regime, less cooldown
        # Goal: Fewer but better trades with higher gross P&L per trade
        "description": "High selectivity: stricter entry, wider regime, shorter cooldown",
        "config_overrides": {
            # Wider regime thresholds - let more trades through
            "ENABLE_REGIME_FILTER": True,
            "ENABLE_SOFT_REGIME": True,
            "REGIME_GREEN_BTC_VOL_MAX": 0.92,     # Very wide
            "REGIME_GREEN_DISPERSION_MAX": 0.95,  # Very wide
            "REGIME_YELLOW_BTC_VOL_MAX": 0.97,
            "REGIME_YELLOW_DISPERSION_MAX": 0.99,
            # Stricter entry to improve win rate
            "ENTRY_Z": 3.2,                        # Up from 2.8
            "EXIT_Z": 0.3,                         # Lower exit (let winners run)
            # Shorter cooldown
            "ENABLE_SMART_COOLDOWN": True,
            "COOLDOWN_AFTER_SIGNAL_BARS": 6,       # 1.5 hours (down from 3h)
            "COOLDOWN_AFTER_STOP_LOSS_BARS": 24,   # 6 hours (down from 12h)
            # More pairs
            "PAIR_SELECTION_TOP_N": 35,
            "PAIR_SELECTION_MIN_FLOOR": 0.20,
        },
    },
    "run_F_max_exposure": {
        # Maximum capital deployment - no regime filter, minimal cooldown
        # Goal: Test if more capital deployed = more absolute return
        "description": "Max exposure: no regime, minimal cooldown, many pairs",
        "config_overrides": {
            "ENABLE_REGIME_FILTER": False,
            "ENABLE_SOFT_REGIME": False,
            "ENABLE_ENTRY_COOLDOWN": True,
            "ENABLE_SMART_COOLDOWN": False,
            "ENTRY_COOLDOWN_BARS": 4,              # Only 1 hour cooldown
            # Stricter entry to compensate for no regime filter
            "ENTRY_Z": 3.0,
            "EXIT_Z": 0.4,
            # More pairs
            "PAIR_SELECTION_TOP_N": 40,
            "PAIR_SELECTION_MIN_FLOOR": 0.20,
        },
    },
}


def write_config_overrides(overrides: Dict[str, Any], output_path: Path) -> None:
    """Write config overrides to a JSON file for use by run_simulation.py."""
    with open(output_path, "w") as f:
        json.dump(overrides, f, indent=2)
    logger.info("Wrote config overrides to %s", output_path)


def run_experiment(
    experiment_name: str,
    experiment_config: Dict[str, Any],
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (used as run name)
    experiment_config : Dict[str, Any]
        Experiment configuration including config_overrides
    results_dir : Path
        Directory to store results

    Returns
    -------
    Dict[str, Any]
        Experiment results summary
    """
    logger.info("=" * 60)
    logger.info("Running experiment: %s", experiment_name)
    logger.info("Description: %s", experiment_config.get("description", "N/A"))
    logger.info("=" * 60)

    # Write config overrides to temp file
    overrides_path = results_dir / f"{experiment_name}_overrides.json"
    write_config_overrides(experiment_config.get("config_overrides", {}), overrides_path)

    # Build command
    run_simulation_path = PROJECT_ROOT / "research" / "backtest" / "run_simulation.py"
    cmd = [
        sys.executable,
        str(run_simulation_path),
        "--run-name", experiment_name,
        "--config-overrides", str(overrides_path),
    ]

    # Run the simulation
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode != 0:
            logger.error("Experiment %s failed with return code %d", experiment_name, result.returncode)
            logger.error("STDERR: %s", result.stderr[-2000:] if result.stderr else "N/A")
            return {
                "experiment": experiment_name,
                "status": "failed",
                "return_code": result.returncode,
                "elapsed_seconds": elapsed,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
            }

        logger.info("Experiment %s completed in %.1f seconds", experiment_name, elapsed)

        # Try to load metrics from results
        metrics_path = PROJECT_ROOT / "results" / experiment_name / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        return {
            "experiment": experiment_name,
            "status": "success",
            "elapsed_seconds": elapsed,
            "metrics": metrics,
        }

    except subprocess.TimeoutExpired:
        logger.error("Experiment %s timed out after 1 hour", experiment_name)
        return {
            "experiment": experiment_name,
            "status": "timeout",
            "elapsed_seconds": 3600,
        }
    except Exception as e:
        logger.error("Experiment %s failed with exception: %s", experiment_name, e)
        return {
            "experiment": experiment_name,
            "status": "error",
            "error": str(e),
        }


def compare_results(results: List[Dict[str, Any]]) -> None:
    """Print a comparison table of experiment results."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)

    # Header
    headers = ["Experiment", "Status", "Trades", "Gross P&L", "Net P&L", "Sharpe", "Stop %"]
    row_format = "{:<25} {:<10} {:>8} {:>12} {:>12} {:>8} {:>8}"
    logger.info(row_format.format(*headers))
    logger.info("-" * 80)

    for result in results:
        name = result.get("experiment", "N/A")
        status = result.get("status", "N/A")
        metrics = result.get("metrics", {})

        trades = metrics.get("total_trades", "N/A")
        gross_pnl = metrics.get("gross_pnl", "N/A")
        net_pnl = metrics.get("total_return", "N/A")
        sharpe = metrics.get("sharpe_ratio", "N/A")
        stop_rate = metrics.get("stop_loss_rate", "N/A")

        # Format numbers
        if isinstance(gross_pnl, (int, float)):
            gross_pnl = f"{gross_pnl:+.4f}"
        if isinstance(net_pnl, (int, float)):
            net_pnl = f"{net_pnl:+.4f}"
        if isinstance(sharpe, (int, float)):
            sharpe = f"{sharpe:.2f}"
        if isinstance(stop_rate, (int, float)):
            stop_rate = f"{stop_rate:.1%}"

        logger.info(row_format.format(
            name[:25], status[:10], str(trades), str(gross_pnl),
            str(net_pnl), str(sharpe), str(stop_rate)
        ))

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic batch experiments")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        help="Experiments to run (default: all)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "batch_experiments"),
        help="Directory to store batch results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting batch experiment run")
    logger.info("Experiments to run: %s", args.experiments)
    logger.info("Results directory: %s", results_dir)

    all_results = []

    for exp_name in args.experiments:
        if exp_name not in EXPERIMENTS:
            logger.warning("Unknown experiment: %s, skipping", exp_name)
            continue

        result = run_experiment(exp_name, EXPERIMENTS[exp_name], results_dir)
        all_results.append(result)

    # Save batch summary
    summary_path = results_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Batch summary saved to %s", summary_path)

    # Print comparison
    compare_results(all_results)


if __name__ == "__main__":
    main()
