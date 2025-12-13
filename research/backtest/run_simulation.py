# research/backtest/run_simulation.py
"""
Phase 5 Runner — full backtest orchestration.

What this script does
---------------------
1) Creates a reproducible run folder under ./results/run_*/
2) Saves a manifest.json with every knob + environment snapshot
3) Loads raw 1m parquet (pivoted price matrix)
4) Splits into train/test (no look-ahead) + validates continuity
5) Selects cointegrated pairs on TRAIN only
6) Computes warm-start Kalman states on TRAIN and saves them
7) Generates causal signals on TEST (z-score, spread volatility, beta)
8) Applies accountant filter to produce entry/exit masks
9) Runs Numba PnL event loop to produce returns_matrix
10) Writes returns_matrix + metrics.json + plots + optional per-pair diagnosis plots

Run:
  poetry run python research/backtest/run_simulation.py
Optional:
  poetry run python research/backtest/run_simulation.py --run-name "debug_run" --max-pairs 25 --diagnose 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg
from src.backtest import data_segmenter
from src.models.coint_scanner import CointegrationScanner

# These modules are created in Phase 5 steps D-I:
from src.backtest import kalman_state_io
from src.backtest import signal_generation
from src.backtest import accountant_filter
from src.backtest import pnl_engine
from src.backtest.performance_report import generate_performance_report
from src.backtest.diagnostics import plot_pair_diagnosis

logger = logging.getLogger("backtest.runner")


# ------------------------------ utilities -------------------------------- #

def _setup_logging(level: str) -> None:
    level_u = level.upper()
    logging.basicConfig(
        level=getattr(logging, level_u, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_parquet_price_matrix(path: Path) -> pd.DataFrame:
    """
    Load a pivoted price matrix with DatetimeIndex, columns as symbols.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")

    # Try the normal pandas read first. If pyarrow raises on pandas metadata
    # (for example a 'categorical' pandas dtype stored in metadata), fall
    # back to reading via pyarrow and disable pandas metadata to get a clean
    # table we can coerce safely.
    try:
        df = pd.read_parquet(path)
    except Exception as read_exc:
        logger.warning("pd.read_parquet failed (%s). Falling back to pyarrow without pandas metadata.", read_exc)
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            # Build DataFrame manually from pyarrow columns to avoid
            # pyarrow->pandas table metadata handling that may include
            # 'categorical' dtype metadata not supported by the local
            # numpy/pandas/pyarrow combo.
            names = table.schema.names
            # If timestamp is present, use it as index
            if "timestamp" in names:
                ts = table.column("timestamp").to_pandas()
                data = {}
                for name in names:
                    if name == "timestamp":
                        continue
                    try:
                        data[name] = table.column(name).to_pandas()
                    except Exception:
                        # As a last resort, convert to numpy
                        data[name] = table.column(name).to_numpy()
                df = pd.DataFrame(data, index=pd.to_datetime(ts))
            else:
                data = {name: table.column(name).to_pandas() for name in names}
                df = pd.DataFrame(data)
        except Exception as pa_exc:
            logger.error("Failed to read parquet via pyarrow fallback: %s", pa_exc)
            raise

    # If the index was serialized with categorical/complex metadata, coerce to
    # a proper DatetimeIndex or string representation.
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # Fall back to string-based index then raise
            df.index = df.index.astype(str)

    df = df.sort_index()

    # Ensure column names and dtypes are primitive-friendly
    if any(not isinstance(c, (str, int, float)) for c in df.columns):
        df.columns = [str(c) for c in df.columns]

    # Convert categoricals/nonnumeric to numeric where possible
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def _ensure_nonempty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} is empty. Check your data export / ingestion.")


def _coerce_pairs_list(scanner_output: Any) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Your scanner returns a DataFrame of rows with column 'pair', or a list of pair strings.
    We keep naming scheme consistent (e.g., 'ETH-BTC').
    """
    if isinstance(scanner_output, pd.DataFrame):
        if scanner_output.empty:
            return [], scanner_output
        if "pair" not in scanner_output.columns:
            raise ValueError("Scanner DataFrame must contain a 'pair' column.")
        pairs = scanner_output["pair"].astype(str).tolist()
        return pairs, scanner_output
    if isinstance(scanner_output, (list, tuple)):
        return [str(p) for p in scanner_output], None
    raise TypeError(f"Unsupported scanner output type: {type(scanner_output)}")


def _clip_pairs_to_available_columns(pairs: Sequence[str], df: pd.DataFrame) -> List[str]:
    """
    Drop pairs that reference symbols missing from df columns.
    """
    good: List[str] = []
    sep = getattr(cfg, "PAIR_ID_SEPARATOR", "-")

    for p in pairs:
        if sep not in p:
            continue
        y, x = p.split(sep, 1)
        y, x = y.strip(), x.strip()
        if y in df.columns and x in df.columns:
            good.append(p)

    return good


# ------------------------------ main pipeline ----------------------------- #

def run_backtest(
    *,
    run_name: Optional[str],
    parquet_path: Optional[Path],
    max_pairs: Optional[int],
    diagnose_n: int,
    log_level: str,
    btc_symbol: str,
) -> Dict[str, Any]:
    _setup_logging(log_level)
    logger.info("🚀 Starting Phase 5 backtest run...")

    # 1) Create run dir + manifest
    run_id, run_dir = cfg.create_run_dir(run_name=run_name)
    cfg.save_manifest(run_dir=run_dir, extra_metadata={"runner": "research/backtest/run_simulation.py"})
    paths = cfg.get_run_paths(run_dir)
    logger.info("Run folder: %s", run_dir)

    # 2) Load data
    raw_path = Path(parquet_path) if parquet_path is not None else cfg.PATH_RAW_PARQUET
    logger.info("Loading price matrix: %s", raw_path)
    price_df = _read_parquet_price_matrix(raw_path)
    _ensure_nonempty(price_df, "price_df")

    # 3) Split train/test with continuity + no look-ahead (in memory)
    logger.info("Splitting train/test via data_segmenter...")
    train_df, test_df = data_segmenter.load_and_split(raw_path)
    _ensure_nonempty(train_df, "train_df")
    _ensure_nonempty(test_df, "test_df")

    # 4) Pair selection on TRAIN only (reuse your model)
    logger.info("Selecting valid pairs on TRAIN only...")

    # Build cluster map from training data columns
    # For backtest, we create a single cluster containing all available coins
    # This allows the scanner to test all pairwise combinations
    available_coins = [c for c in train_df.columns if c != 'timestamp']
    cluster_map = {0: available_coins}  # Single cluster with all coins
    logger.info(f"Created cluster with {len(available_coins)} coins for pair scanning")

    scanner = CointegrationScanner(
        cluster_map=cluster_map,
        p_value_threshold=0.05,
        min_volatility=0.001,  # Lower threshold for short data periods
        max_drift_z=5.0,
    )

    # Use find_pairs_from_matrix for backtest mode (uses train_df directly)
    scanner_output = scanner.find_pairs_from_matrix(train_df, train_ratio=0.8)
    valid_pairs, pairs_df = _coerce_pairs_list(scanner_output)

    # Enforce consistent format + filter to coins present in train/test
    valid_pairs = [p for p in valid_pairs if isinstance(p, str)]
    valid_pairs = _clip_pairs_to_available_columns(valid_pairs, train_df)
    valid_pairs = _clip_pairs_to_available_columns(valid_pairs, test_df)

    # Optional cap for speed
    if max_pairs is not None and max_pairs > 0:
        valid_pairs = valid_pairs[: max_pairs]
        if pairs_df is not None and not pairs_df.empty:
            pairs_df = pairs_df.iloc[: max_pairs]

    if not valid_pairs:
        raise RuntimeError("Scanner returned 0 valid pairs after filtering. Adjust thresholds or ensure data quality.")

    # Save pairs list and (optional) full DF
    _write_json(paths["valid_pairs"], {"pairs": valid_pairs})
    if pairs_df is not None and not pairs_df.empty:
        pairs_df.to_parquet(run_dir / "valid_pairs_details.parquet")

    logger.info("✅ Valid pairs: %d", len(valid_pairs))

    # 5) Warm-start persistence (TRAIN)
    logger.info("Computing warm states on TRAIN...")
    # Expected API:
    #   warm_states = kalman_state_io.compute_and_save(train_df, valid_pairs, run_dir)
    warm_states = kalman_state_io.compute_and_save(train_df=train_df, valid_pairs=valid_pairs, run_dir=run_dir)
    logger.info("✅ Warm states computed: %d", len(warm_states))

    # 6) Signal generation on TEST (causal)
    logger.info("Generating TEST signals (z/vol/beta)...")
    # Expected API:
    #   z_df, vol_df, beta_df = signal_generation.generate(test_df, valid_pairs, warm_states)
    z_df, vol_df, beta_df = signal_generation.generate(
        test_df=test_df,
        valid_pairs=valid_pairs,
        warm_states=warm_states,
    )

    # Optional debug save (parquet)
    if getattr(cfg, "SAVE_SIGNALS_PARQUET", False):
        signals_out = paths["signals"]
        packed = pd.concat(
            {"z": z_df, "vol": vol_df, "beta": beta_df},
            axis=1,
        )
        packed.to_parquet(signals_out)
        logger.info("Saved signals parquet: %s", signals_out)

    # 7) Accountant masks
    logger.info("Applying accountant filter (entries/exits)...")
    # Expected API:
    #   entries, exits, expected_profit = accountant_filter.compute_masks(z_df, vol_df)
    entries, exits, expected_profit = accountant_filter.compute_masks(
        z_score=z_df,
        spread_volatility=vol_df,
    )

    # 8) PnL engine
    logger.info("Running PnL engine (Numba state machine)...")
    # Expected API:
    #   pnl_result = pnl_engine.run(...)
    # If your run() returns a DataFrame directly, we handle that too.
    pnl_result = pnl_engine.run(
        test_df=test_df,
        valid_pairs=valid_pairs,
        entries=entries,
        exits=exits,
        beta=beta_df,
        z_score=z_df,
        spread_volatility=vol_df,
        fee_rate=cfg.FEE_RATE,
        slippage_model=cfg.SLIPPAGE_MODEL,
        slippage_rate=cfg.SLIPPAGE_RATE,
        capital_per_pair=cfg.CAPITAL_PER_PAIR,
        pnl_mode="price",
    )

    # Normalize output to a returns_matrix DataFrame
    if isinstance(pnl_result, pd.DataFrame):
        returns_matrix = pnl_result
    elif hasattr(pnl_result, "returns_matrix"):
        returns_matrix = pnl_result.returns_matrix
    else:
        raise TypeError("pnl_engine.run must return a DataFrame or an object with .returns_matrix")

    returns_out = paths["returns_matrix"]
    returns_matrix.to_parquet(returns_out)
    logger.info("✅ Saved returns_matrix: %s", returns_out)

    # 9) Performance report + plots
    logger.info("Building performance report...")
    report = generate_performance_report(
        run_dir=run_dir,
        returns_matrix=returns_matrix,
        test_prices=test_df,
        btc_symbol=btc_symbol,
        freq=cfg.BAR_FREQ,
    )

    # 10) Diagnostics (pair-level deep dive plots)
    if diagnose_n > 0:
        logger.info("Generating %d pair diagnosis plots...", diagnose_n)
        for pair_id in valid_pairs[:diagnose_n]:
            try:
                plot_pair_diagnosis(
                    run_dir=run_dir,
                    pair_id=pair_id,
                    test_df=test_df,
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

    logger.info("🏁 Backtest complete. Run ID: %s", run_id)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_pairs": len(valid_pairs),
        "artifacts": {k: str(v) for k, v in paths.items() if isinstance(v, Path)},
        "report_summary": report.get("metrics", {}),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5 backtest runner (research/backtest/run_simulation.py)")
    p.add_argument("--run-name", type=str, default=None, help="Optional run folder name (default: timestamp).")
    p.add_argument("--parquet-path", type=str, default=None, help="Override path to price matrix parquet.")
    p.add_argument("--max-pairs", type=int, default=None, help="Cap number of pairs for faster iteration.")
    p.add_argument("--diagnose", type=int, default=5, help="How many pairs to generate diagnosis plots for.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR).")
    p.add_argument("--btc-symbol", type=str, default="BTC", help="Column name used for BTC in test_df for correlation.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    result = run_backtest(
        run_name=args.run_name,
        parquet_path=Path(args.parquet_path) if args.parquet_path else None,
        max_pairs=args.max_pairs,
        diagnose_n=args.diagnose,
        log_level=args.log_level,
        btc_symbol=args.btc_symbol,
    )

    # Print a short, useful summary
    print(json.dumps(result, indent=2))
