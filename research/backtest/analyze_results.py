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
from typing import Any, Dict, Optional

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
        freq=getattr(cfg, "BAR_FREQ", "1min"),
    )

    # Overwrite metrics.json if forced OR if it didn't exist
    if force or (existing_metrics is None):
        _write_json(metrics_path, report.get("metrics", report))
        logger.info("✅ Wrote metrics.json: %s", metrics_path)
    else:
        logger.info("Kept existing metrics.json (not overwritten).")

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
