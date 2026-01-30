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

# Feature engineering and clustering for intelligent pair selection
from src.features.clustering import get_cluster_map
from src.features.engineering import FeatureEngineer

# These modules are created in Phase 5 steps D-I:
from src.backtest import kalman_state_io
from src.backtest import signal_generation
from src.backtest import accountant_filter
from src.backtest import pnl_engine
from src.backtest.performance_report import generate_performance_report
from src.backtest.diagnostics import plot_pair_diagnosis
from src.backtest.visualization import generate_all_visualizations

from src.adaptive import config_adaptive as adaptive_cfg
from src.adaptive.online_adaptation import (
    AdaptiveController,
    apply_overrides_to_backtest,
)

# Market regime detection for adaptive parameters
from src.adaptive.market_regime import (
    detect_market_regime,
    MarketRegime,
    VolatilityRegime,
    TrendRegime,
)

# Sector diversification for cluster-based position limits
from src.adaptive.sector_diversification import (
    filter_pairs_by_cluster_limits,
    compute_cluster_penalties_matrix,
    load_cluster_map_from_file,
    invert_cluster_map,
)

# Adaptive Kalman parameters
from src.models.kalman import (
    compute_adaptive_kalman_params,
    get_adaptive_params_from_config,
)

# ML Signal Scorer for learned entry prediction
from src.models.ml_signal_scorer import (
    MLSignalScorer,
    MLScorerConfig,
    create_training_labels_from_returns,
)

# Advanced position sizing
from src.backtest.position_sizing import (
    PositionSizer,
    PositionSizingConfig,
    compute_position_sizes_vectorized,
    apply_risk_prediction_adjustment,
)

# Risk prediction for ML-based position sizing
from src.models.risk_predictor import (
    RiskPredictor,
    RiskPredictorConfig,
    RiskLabelGenerator,
    RiskFeatureExtractor,
    RiskLabels,
)
from src.models.risk_monitor import RiskMonitor

# Phase 1: Stop the bleeding - risk controls
from src.backtest.pair_kill_switch import PairKillSwitch, filter_disabled_pairs
from src.backtest.window_circuit_breaker import WindowCircuitBreaker
from src.backtest.symbol_blacklist import SymbolBlacklist

# Phase 2: Fix the edge - pair scoring and regime filtering
from src.models.pair_scorer import PairScorer, PairScore
from src.adaptive.regime_filter import RegimeFilter, RegimeState
from src.adaptive.live_regime_tracker import LiveRegimeTracker, create_live_regime_tracker
from src.adaptive.regime_parameters import RegimeParameters, create_regime_parameters

# Phase 5B: Window analysis for regime pattern detection
from src.backtest.window_analysis import WindowAnalysis, create_window_analysis

from collections import Counter
from dataclasses import dataclass, asdict, field
logger = logging.getLogger("backtest.runner")


# ------------------------------ Entry Funnel Diagnostics -------------------------------- #

@dataclass
class EntryFunnel:
    """Track entry filtering at each gate for diagnostics."""
    window_idx: int = 0
    raw_z_entries: int = 0           # Z-score condition only (from accountant_filter)
    after_spread_vol: int = 0        # After spread volatility filter
    after_trend_overlay: int = 0     # After trend overlay suppression
    after_regime: int = 0            # After regime filter
    after_cooldown: int = 0          # After cooldown filter
    final_executed: int = 0          # Actually executed (from pnl_engine)

    # Per-condition breakdown from regime filter
    btc_vol_blocked: int = 0
    dispersion_blocked: int = 0
    spread_vol_low_blocked: int = 0
    spread_vol_high_blocked: int = 0

    # Percentage metrics
    btc_vol_ok_pct: float = 0.0
    dispersion_ok_pct: float = 0.0
    spread_vol_ok_pct: float = 0.0

    def compute_conversion_rate(self) -> float:
        """Compute final conversion rate from raw to executed."""
        if self.raw_z_entries == 0:
            return 0.0
        return self.final_executed / self.raw_z_entries * 100

    def get_biggest_blocker(self) -> str:
        """Identify which gate blocked the most entries."""
        blockers = {
            "spread_vol": self.raw_z_entries - self.after_spread_vol,
            "trend_overlay": self.after_spread_vol - self.after_trend_overlay,
            "regime": self.after_trend_overlay - self.after_regime,
            "cooldown": self.after_regime - self.after_cooldown,
        }
        biggest = max(blockers, key=blockers.get)
        blocked_pct = blockers[biggest] / max(self.raw_z_entries, 1) * 100
        return f"{biggest} ({blocked_pct:.1f}% blocked)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert numpy types to native Python types for JSON serialization
        for key, value in d.items():
            if hasattr(value, "item"):  # numpy scalar
                d[key] = value.item()
            elif isinstance(value, (int, float)):
                d[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
        d["conversion_rate_pct"] = f"{self.compute_conversion_rate():.2f}%"
        d["biggest_blocker"] = self.get_biggest_blocker()
        return d


def save_entry_funnel(funnel: EntryFunnel, window_dir: Path) -> None:
    """Save entry funnel diagnostics to JSON."""
    funnel_path = window_dir / "entry_funnel.json"
    with open(funnel_path, "w") as f:
        json.dump(funnel.to_dict(), f, indent=2)
    logger.info(
        "Entry funnel: raw=%d → spread_vol=%d → trend=%d → regime=%d → cooldown=%d → final=%d (%.1f%% conversion)",
        funnel.raw_z_entries,
        funnel.after_spread_vol,
        funnel.after_trend_overlay,
        funnel.after_regime,
        funnel.after_cooldown,
        funnel.final_executed,
        funnel.compute_conversion_rate(),
    )


# ------------------------------ Multi-Timeframe Window -------------------------------- #

@dataclass
class MultiTimeframeWindow:
    """
    Walk-forward window with multi-timeframe training data.

    Structure:
        [Long-term 180d] -> [Short-term 30d subset] -> [Test 21d]

    The short_term_train is the last N days of long_term_train,
    used for regime-aware validation of pairs found in long-term.
    """
    long_term_train: pd.DataFrame      # Full long-term training window (e.g., 180 days)
    short_term_train: pd.DataFrame     # Recent subset for regime validation (e.g., last 30 days)
    test: pd.DataFrame                 # Out-of-sample test window
    train_start: pd.Timestamp          # Start of long-term window
    short_term_start: pd.Timestamp     # Start of short-term subset
    train_end: pd.Timestamp            # End of training (start of test)
    test_end: pd.Timestamp             # End of test window

    @property
    def long_term_days(self) -> int:
        """Number of days in long-term training window."""
        return (self.train_end - self.train_start).days

    @property
    def short_term_days(self) -> int:
        """Number of days in short-term training window."""
        return (self.train_end - self.short_term_start).days

    @property
    def test_days(self) -> int:
        """Number of days in test window."""
        return (self.test_end - self.train_end).days


# ------------------------------ utilities -------------------------------- #

def _filter_pairs_by_features(
    pairs: List[str],
    coin_features: pd.DataFrame
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Filter pairs based on coin quality metrics (alpha, volatility, etc.).

    Args:
        pairs: List of pair strings (e.g., ["ETH-BTC", "SOL-ETH"])
        coin_features: DataFrame with features per coin (index=coin, columns=features)

    Returns:
        Filtered pairs list and feature stats dict
    """
    if coin_features.empty:
        logger.warning("No features available for filtering, skipping feature filter")
        return pairs, {}

    logger.info("Applying feature-based quality filters...")

    # Quality thresholds
    MIN_ALPHA = -0.1  # Allow slightly negative alpha (realistic for crypto)
    MAX_VOLATILITY_Z = 2.5  # Reject extremely volatile coins
    MIN_LIQUIDITY = 0.4  # Volume flow threshold

    valid_pairs = []
    pair_features = {}

    for pair in pairs:
        try:
            # Parse pair (e.g., "ETH-BTC")
            coin1, coin2 = pair.split(cfg.PAIR_ID_SEPARATOR)

            # Check if both coins have features
            if coin1 not in coin_features.index or coin2 not in coin_features.index:
                logger.debug(f"Skipping {pair}: missing features")
                continue

            feat1 = coin_features.loc[coin1]
            feat2 = coin_features.loc[coin2]

            # Filter 1: Reject pairs with both coins having negative alpha
            if feat1.get('alpha', 0) < MIN_ALPHA and feat2.get('alpha', 0) < MIN_ALPHA:
                logger.debug(f"Rejected {pair}: both alphas < {MIN_ALPHA}")
                continue

            # Filter 2: Reject extremely volatile coins
            if feat1.get('volatility_z', 0) > MAX_VOLATILITY_Z or feat2.get('volatility_z', 0) > MAX_VOLATILITY_Z:
                logger.debug(f"Rejected {pair}: volatility too high")
                continue

            # Filter 3: Reject low-liquidity coins
            if feat1.get('volume_flow', 1) < MIN_LIQUIDITY or feat2.get('volume_flow', 1) < MIN_LIQUIDITY:
                logger.debug(f"Rejected {pair}: low liquidity")
                continue

            # Pair passed all filters
            valid_pairs.append(pair)
            pair_features[pair] = {
                'alpha_1': feat1.get('alpha', 0),
                'alpha_2': feat2.get('alpha', 0),
                'beta_1': feat1.get('beta', 1),
                'beta_2': feat2.get('beta', 1),
                'vol_z_1': feat1.get('volatility_z', 0),
                'vol_z_2': feat2.get('volatility_z', 0),
            }

        except Exception as e:
            logger.warning(f"Error filtering {pair}: {e}")
            continue

    logger.info(f"Feature filter: {len(valid_pairs)}/{len(pairs)} pairs passed quality checks")
    return valid_pairs, pair_features


def _filter_pairs_by_historical_performance(
    pairs: List[str],
    historical_pnl_data: Dict[str, Dict],
    max_cost_to_gross_ratio: float = 0.80,
    max_stop_loss_rate: float = 0.40,
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Filter pairs based on historical P&L attribution metrics.

    This is a key ROI optimization: reject pairs that historically lose money
    due to high transaction costs (high churn) or frequent stop-losses
    (regime breaks).

    Args:
        pairs: List of pair strings
        historical_pnl_data: Dict mapping pair -> {gross_pnl, fees, slippage,
                             trade_count, stop_loss_count}
        max_cost_to_gross_ratio: Reject pairs where (fees + slippage) / gross > this
        max_stop_loss_rate: Penalize/reject pairs where stop_loss_count / trade_count > this

    Returns:
        Filtered pairs list and rejection stats dict
    """
    if not historical_pnl_data:
        logger.info("No historical P&L data available, skipping performance filter")
        return pairs, {}

    enable_cost_filter = bool(getattr(cfg, "ENABLE_COST_GROSS_FILTER", True))
    enable_stop_penalty = bool(getattr(cfg, "ENABLE_STOP_LOSS_PENALTY", True))

    valid_pairs = []
    rejection_stats = {
        "rejected_high_cost_ratio": [],
        "rejected_high_stop_rate": [],
        "pair_metrics": {},
    }

    for pair in pairs:
        if pair not in historical_pnl_data:
            # No historical data, include by default
            valid_pairs.append(pair)
            continue

        data = historical_pnl_data[pair]
        gross_pnl = data.get("gross_pnl", 0.0)
        fees = data.get("fees", 0.0)
        slippage = data.get("slippage", 0.0)
        trade_count = data.get("trade_count", 0)
        stop_loss_count = data.get("stop_loss_count", 0)

        total_cost = abs(fees) + abs(slippage)

        # Compute cost-to-gross ratio
        if abs(gross_pnl) > 1e-10:
            cost_to_gross = total_cost / abs(gross_pnl)
        else:
            cost_to_gross = float('inf') if total_cost > 0 else 0.0

        # Compute stop-loss rate
        stop_rate = stop_loss_count / trade_count if trade_count > 0 else 0.0

        rejection_stats["pair_metrics"][pair] = {
            "cost_to_gross": cost_to_gross,
            "stop_rate": stop_rate,
            "gross_pnl": gross_pnl,
            "total_cost": total_cost,
            "trade_count": trade_count,
            "stop_loss_count": stop_loss_count,
        }

        # Filter 1: Reject high cost-to-gross ratio
        if enable_cost_filter and cost_to_gross > max_cost_to_gross_ratio:
            rejection_stats["rejected_high_cost_ratio"].append(pair)
            logger.debug(
                f"Rejected {pair}: cost/gross ratio {cost_to_gross:.2%} > {max_cost_to_gross_ratio:.0%}"
            )
            continue

        # Filter 2: Reject high stop-loss rate
        if enable_stop_penalty and stop_rate > max_stop_loss_rate:
            rejection_stats["rejected_high_stop_rate"].append(pair)
            logger.debug(
                f"Rejected {pair}: stop-loss rate {stop_rate:.2%} > {max_stop_loss_rate:.0%}"
            )
            continue

        valid_pairs.append(pair)

    n_rejected_cost = len(rejection_stats["rejected_high_cost_ratio"])
    n_rejected_stop = len(rejection_stats["rejected_high_stop_rate"])
    logger.info(
        "Historical performance filter: %d/%d pairs kept "
        "(rejected: %d high cost-ratio, %d high stop-rate)",
        len(valid_pairs), len(pairs), n_rejected_cost, n_rejected_stop
    )

    return valid_pairs, rejection_stats


# ------------------------------ utilities -------------------------------- #

def _safe_log_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.where(values > 0)
    return np.log(values)


def _compute_spread_returns(
    test_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    pairs: Sequence[str],
    pnl_mode: str = "log",
) -> pd.DataFrame:
    returns = {}
    pair_sep = getattr(cfg, "PAIR_ID_SEPARATOR", "-")
    use_log = pnl_mode == "log"
    for pair in pairs:
        coin_y, coin_x = pair.split(pair_sep, 1)
        if use_log:
            y_series = _safe_log_series(test_df[coin_y])
            x_series = _safe_log_series(test_df[coin_x])
        else:
            y_series = pd.to_numeric(test_df[coin_y], errors="coerce").astype("float64")
            x_series = pd.to_numeric(test_df[coin_x], errors="coerce").astype("float64")
        beta_series = beta_df[pair]
        spread = y_series - beta_series * x_series
        returns[pair] = spread.diff()
    return pd.DataFrame(returns, index=test_df.index)


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
            # Build DataFrame manually from pyarrow columns using to_pylist()
            # to avoid pandas metadata issues that cause all-NaN results.
            names = table.schema.names
            # If timestamp is present, use it as index
            if "timestamp" in names:
                ts_col = table.column("timestamp")
                ts = pd.to_datetime(ts_col.to_pylist())
                data = {}
                for name in names:
                    if name == "timestamp":
                        continue
                    col = table.column(name)
                    # Use to_pylist() which properly handles the data
                    data[name] = pd.array(col.to_pylist(), dtype="float64")
                df = pd.DataFrame(data, index=ts)
            else:
                data = {}
                for name in names:
                    col = table.column(name)
                    data[name] = pd.array(col.to_pylist(), dtype="float64")
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


def _compute_coin_volatility(train_df: pd.DataFrame) -> pd.Series:
    """
    Liquidity proxy using log-return volatility per coin.
    """
    safe = train_df.replace(0, np.nan).dropna(axis=1, how="all")
    log_prices = np.log(safe)
    returns = log_prices.diff().dropna(how="all")
    vol = returns.std()
    return vol.replace([np.inf, -np.inf], np.nan)


def _compute_funding_stats(
    funding_rates: Optional[pd.DataFrame],
    train_index: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return (mean_funding, coverage) per coin for the train window.
    """
    if funding_rates is None or funding_rates.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    start = train_index.min()
    end = train_index.max()
    window = funding_rates[(funding_rates.index >= start) & (funding_rates.index <= end)]
    if window.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    mean_funding = window.mean()
    coverage = window.notna().mean()
    return mean_funding, coverage


def _compute_holding_hours(entries: pd.DataFrame, exits: pd.DataFrame) -> List[float]:
    """
    Approximate holding hours by pairing each entry with the next exit per pair.
    """
    holds: List[float] = []
    if entries.empty or exits.empty:
        return holds
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    for pair in entries.columns:
        entry_times = entries.index[entries[pair]].to_list()
        exit_times = exits.index[exits[pair]].to_list()
        if not entry_times or not exit_times:
            continue
        exit_iter = iter(exit_times)
        current_exit = next(exit_iter, None)
        for entry_ts in entry_times:
            while current_exit is not None and current_exit <= entry_ts:
                current_exit = next(exit_iter, None)
            if current_exit is None:
                break
            hold_hours = (current_exit - entry_ts).total_seconds() / 3600.0
            if hold_hours >= 0:
                holds.append(hold_hours)
    return holds


# ------------------------------ weekly health check ----------------------- #

def _compute_weekly_health_scores(
    z_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    monthly_beta: pd.DataFrame,
    pairs: List[str],
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> Dict[str, Dict[str, float]]:
    """
    Compute health metrics for each pair during a weekly sub-window.

    Returns dict mapping pair -> health metrics dict.
    """
    week_mask = (z_df.index >= week_start) & (z_df.index < week_end)
    z_week = z_df.loc[week_mask]
    beta_week = beta_df.loc[week_mask]

    health_scores = {}
    for pair in pairs:
        if pair not in z_week.columns:
            continue

        z_series = z_week[pair].dropna()
        if len(z_series) < 10:
            health_scores[pair] = {"healthy": False, "reason": "insufficient_data"}
            continue

        # Metric 1: Zero-crossings (mean-reversion activity)
        z_sign = np.sign(z_series)
        crosses = (z_sign.diff().abs() == 2).sum()

        # Metric 2: Beta drift from monthly estimate
        if pair in beta_week.columns and pair in monthly_beta.columns:
            monthly_beta_val = monthly_beta[pair].iloc[-1] if not monthly_beta[pair].dropna().empty else np.nan
            weekly_beta_mean = beta_week[pair].mean()
            if np.isfinite(monthly_beta_val) and np.isfinite(weekly_beta_mean) and monthly_beta_val != 0:
                beta_drift = abs(weekly_beta_mean - monthly_beta_val) / abs(monthly_beta_val)
            else:
                beta_drift = 0.0
        else:
            beta_drift = 0.0

        # Metric 3: Spread trend (mean z in week - detect drift)
        spread_trend = abs(z_series.mean())

        health_scores[pair] = {
            "zero_crossings": int(crosses),
            "beta_drift": float(beta_drift),
            "spread_trend": float(spread_trend),
            "healthy": True,  # Will be evaluated by caller
        }

    return health_scores


def _filter_pairs_by_weekly_health(
    health_scores: Dict[str, Dict[str, float]],
    min_crosses: int,
    max_beta_drift: float,
    max_spread_trend: float,
) -> Tuple[List[str], List[str]]:
    """
    Filter pairs based on weekly health metrics.

    Returns (healthy_pairs, excluded_pairs).
    """
    healthy = []
    excluded = []

    for pair, metrics in health_scores.items():
        if not metrics.get("healthy", False):
            excluded.append(pair)
            continue

        crosses = metrics.get("zero_crossings", 0)
        drift = metrics.get("beta_drift", 0.0)
        trend = metrics.get("spread_trend", 0.0)

        # Apply health thresholds
        if crosses < min_crosses:
            excluded.append(pair)
        elif drift > max_beta_drift:
            excluded.append(pair)
        elif trend > max_spread_trend:
            excluded.append(pair)
        else:
            healthy.append(pair)

    return healthy, excluded


def _build_weekly_subwindows(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    week_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build weekly sub-windows within a monthly test period.
    """
    windows = []
    cur = test_start
    while cur < test_end:
        week_end = min(cur + pd.Timedelta(days=week_days), test_end)
        windows.append((cur, week_end))
        cur = week_end
    return windows


# ------------------------------ funding utils ---------------------------- #

def _compute_funding_cost_per_bar(
    z_df: pd.DataFrame,
    pairs: List[str],
    funding_rates: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    Build a per-bar funding cost matrix using real funding rates and trade direction.
    """
    f = str(freq).lower().strip()
    if f in ("1min", "1m", "min", "t"):
        mins_per_bar = 1.0
    elif f.endswith("min"):
        mins_per_bar = float(f.replace("min", ""))
    elif f.endswith("m"):
        mins_per_bar = float(f.replace("m", ""))
    elif f.endswith("h"):
        mins_per_bar = float(f.replace("h", "")) * 60.0
    else:
        raise ValueError(f"Unsupported freq '{freq}' for funding conversion.")
    funding_interval_mins = 8.0 * 60.0
    bars_per_funding_period = funding_interval_mins / mins_per_bar

    out = pd.DataFrame(index=z_df.index, columns=z_df.columns, dtype="float64")
    for pair in pairs:
        if pair not in z_df.columns:
            continue
        coin_y, coin_x = pair.split(cfg.PAIR_ID_SEPARATOR, 1)
        if coin_y not in funding_rates.columns or coin_x not in funding_rates.columns:
            out[pair] = 0.0
            continue
        rates = funding_rates[[coin_y, coin_x]].reindex(z_df.index).ffill()
        net_8h = rates[coin_y] - rates[coin_x]
        direction = np.sign(z_df[pair]).fillna(0.0)
        out[pair] = (net_8h / bars_per_funding_period) * direction
    return out.fillna(0.0)


# ------------------------------ detailed logging ----------------------------- #

def _generate_detailed_log(
    run_dir: Path,
    run_id: str,
    window_metrics: List[Dict],
    pair_summaries: List[Dict],
    config_snapshot: Dict,
    attribution_summary: Dict,
    report_metrics: Dict,
) -> Path:
    """
    Generate a comprehensive detailed backtest log file.

    Parameters
    ----------
    run_dir : Path
        Results directory
    run_id : str
        Run identifier
    window_metrics : List[Dict]
        Per-window performance metrics
    pair_summaries : List[Dict]
        Per-pair summary data from attribution
    config_snapshot : Dict
        Configuration parameters used
    attribution_summary : Dict
        P&L attribution summary
    report_metrics : Dict
        Overall performance metrics

    Returns
    -------
    Path
        Path to the detailed log file
    """
    from datetime import datetime

    log_path = run_dir / "detailed_backtest_log.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED BACKTEST LOG")
    lines.append("=" * 80)
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # === CONFIGURATION PARAMETERS ===
    lines.append("=" * 80)
    lines.append("CONFIGURATION PARAMETERS")
    lines.append("=" * 80)
    lines.append("")

    # Group config by category
    config_categories = {
        "Data & Timing": ["TRAIN_DAYS", "TEST_DAYS", "SIGNAL_TIMEFRAME", "BAR_FREQ",
                         "WALK_FORWARD_TRAIN_DAYS", "WALK_FORWARD_TEST_DAYS", "WALK_FORWARD_STEP_DAYS"],
        "Entry/Exit Thresholds": ["ENTRY_Z", "EXIT_Z", "STOP_LOSS_Z", "MAX_ENTRY_Z",
                                 "STOP_LOSS_PCT", "MIN_PROFIT_HURDLE"],
        "Position Sizing": ["MAX_PORTFOLIO_POSITIONS", "MAX_POSITIONS_PER_COIN", "CAPITAL_PER_PAIR",
                          "MAX_SINGLE_POSITION_PCT", "ENABLE_CONTINUOUS_EXPOSURE", "NORMALIZE_NOTIONAL"],
        "Costs": ["FEE_RATE", "SLIPPAGE_BPS", "SLIPPAGE_RATE", "SLIPPAGE_MODEL"],
        "Kalman Settings": ["KALMAN_DELTA", "KALMAN_R", "MIN_HALF_LIFE_BARS"],
        "Risk Controls": ["ENABLE_FDR_CONTROL", "FDR_ALPHA", "ENABLE_SUBWINDOW_STABILITY",
                         "ENABLE_RISK_PARITY", "MAX_HHI"],
    }

    for category, keys in config_categories.items():
        lines.append(f"--- {category} ---")
        for key in keys:
            if key in config_snapshot:
                lines.append(f"  {key}: {config_snapshot[key]}")
        lines.append("")

    # === WALK-FORWARD WINDOW SUMMARY ===
    lines.append("=" * 80)
    lines.append("WALK-FORWARD WINDOW SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    if window_metrics:
        lines.append(f"{'Window':<8} {'Return':>10} {'Trades':>8} {'Win%':>8} {'Gross':>12} {'Net':>12} {'Pairs':>8}")
        lines.append("-" * 80)

        for i, wm in enumerate(window_metrics):
            lines.append(
                f"{i+1:<8} "
                f"{wm.get('total_return', 0)*100:>9.2f}% "
                f"{wm.get('trade_count', 0):>8} "
                f"{wm.get('win_rate', 0)*100:>7.1f}% "
                f"{wm.get('gross_pnl', 0):>+11.4f} "
                f"{wm.get('net_pnl', 0):>+11.4f} "
                f"{wm.get('n_pairs', 0):>8}"
            )

        # Summary statistics
        lines.append("-" * 80)
        total_return = np.prod([1 + wm.get('total_return', 0) for wm in window_metrics]) - 1
        total_trades = sum(wm.get('trade_count', 0) for wm in window_metrics)
        avg_win_rate = np.mean([wm.get('win_rate', 0) for wm in window_metrics if wm.get('trade_count', 0) > 0])
        total_gross = sum(wm.get('gross_pnl', 0) for wm in window_metrics)
        total_net = sum(wm.get('net_pnl', 0) for wm in window_metrics)

        lines.append(f"{'TOTAL':<8} {total_return*100:>9.2f}% {total_trades:>8} {avg_win_rate*100:>7.1f}% {total_gross:>+11.4f} {total_net:>+11.4f}")
    else:
        lines.append("No window metrics available.")

    lines.append("")

    # === P&L ATTRIBUTION SUMMARY ===
    lines.append("=" * 80)
    lines.append("P&L ATTRIBUTION SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    if attribution_summary:
        lines.append(f"Total Trades:      {attribution_summary.get('total_trades', 0):,}")
        lines.append(f"Gross P&L:         {attribution_summary.get('total_gross_pnl', 0):+.4f}")
        lines.append(f"  - Fees:          {attribution_summary.get('total_fees', 0):.4f}")
        lines.append(f"  - Slippage:      {attribution_summary.get('total_slippage', 0):.4f}")
        lines.append(f"  - Funding:       {attribution_summary.get('total_funding_pnl', 0):+.4f}")
        lines.append(f"Net P&L:           {attribution_summary.get('total_net_pnl', 0):+.4f}")
        lines.append(f"Cost/Gross Ratio:  {attribution_summary.get('cost_to_gross_ratio', 0):.2%}")
        lines.append("")

        # Diagnostics
        lines.append("Diagnostic Flags:")
        diag = attribution_summary.get('diagnostics', {})
        if diag.get('gross_positive_net_negative'):
            lines.append("  [!] GROSS POSITIVE, NET NEGATIVE - Friction problem")
        if diag.get('gross_negative'):
            lines.append("  [!] GROSS NEGATIVE - Signal not predictive")
        if diag.get('few_pairs_dominate_losses'):
            lines.append("  [!] FEW PAIRS DOMINATE LOSSES - Universe quality issue")
        if not any(diag.values()):
            lines.append("  [OK] No major issues detected")
    else:
        lines.append("No attribution data available.")

    lines.append("")

    # === PER-PAIR PERFORMANCE BREAKDOWN ===
    lines.append("=" * 80)
    lines.append("PER-PAIR PERFORMANCE BREAKDOWN")
    lines.append("=" * 80)
    lines.append("")

    if pair_summaries:
        # Sort by net P&L
        sorted_summaries = sorted(pair_summaries, key=lambda x: x.get('total_net_pnl', 0), reverse=True)

        lines.append(f"{'Pair':<15} {'Trades':>7} {'Win%':>7} {'AvgHold':>8} {'Gross':>10} {'Fees':>8} {'Net':>10} {'Contrib':>8}")
        lines.append("-" * 90)

        for s in sorted_summaries:
            lines.append(
                f"{s.get('pair', ''):<15} "
                f"{s.get('trade_count', 0):>7} "
                f"{s.get('win_rate', 0)*100:>6.1f}% "
                f"{s.get('avg_hold_bars', 0):>7.0f}b "
                f"{s.get('total_gross_pnl', 0):>+9.4f} "
                f"{s.get('total_fees', 0):>8.4f} "
                f"{s.get('total_net_pnl', 0):>+9.4f} "
                f"{s.get('contribution_pct', 0):>+7.1f}%"
            )
    else:
        lines.append("No pair summaries available.")

    lines.append("")

    # === EXIT REASON ANALYSIS ===
    lines.append("=" * 80)
    lines.append("EXIT REASON ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    if pair_summaries:
        signal_exits = sum(s.get('signal_exits', 0) for s in pair_summaries)
        time_stop_exits = sum(s.get('time_stop_exits', 0) for s in pair_summaries)
        stop_loss_exits = sum(s.get('stop_loss_exits', 0) for s in pair_summaries)
        forced_exits = sum(s.get('forced_exits', 0) for s in pair_summaries)
        total_exits = signal_exits + time_stop_exits + stop_loss_exits + forced_exits

        if total_exits > 0:
            lines.append(f"Signal (Mean Reversion):  {signal_exits:>5} ({signal_exits/total_exits*100:>5.1f}%)")
            lines.append(f"Time Stop:                {time_stop_exits:>5} ({time_stop_exits/total_exits*100:>5.1f}%)")
            lines.append(f"Stop Loss:                {stop_loss_exits:>5} ({stop_loss_exits/total_exits*100:>5.1f}%)")
            lines.append(f"Forced Exit:              {forced_exits:>5} ({forced_exits/total_exits*100:>5.1f}%)")
            lines.append(f"Total:                    {total_exits:>5}")

            lines.append("")
            lines.append("Analysis:")
            if stop_loss_exits / total_exits > 0.4:
                lines.append("  [WARN] High stop-loss rate (>40%) suggests regime breaks or poor pair selection")
            if signal_exits / total_exits < 0.5:
                lines.append("  [WARN] Low signal exit rate (<50%) - trades not reaching mean reversion")
            if time_stop_exits / total_exits > 0.2:
                lines.append("  [INFO] Significant time stops - consider adjusting time stop multiplier")
    else:
        lines.append("No exit reason data available.")

    lines.append("")

    # === PERFORMANCE METRICS ===
    lines.append("=" * 80)
    lines.append("PERFORMANCE METRICS")
    lines.append("=" * 80)
    lines.append("")

    if report_metrics:
        base_case = report_metrics.get('base_case', {})
        lines.append(f"Total Return:      {base_case.get('total_return', 0)*100:+.2f}%")
        lines.append(f"Sharpe Ratio:      {base_case.get('sharpe', 0):.2f}")
        lines.append(f"Calmar Ratio:      {base_case.get('calmar', 0):.2f}")
        lines.append(f"Max Drawdown:      {base_case.get('max_drawdown', 0)*100:.2f}%")
        lines.append(f"Max DD Duration:   {base_case.get('max_dd_duration_minutes', 0)/60:.1f} hours")
        lines.append(f"BTC Correlation:   {base_case.get('corr_to_btc_returns', 0):.3f}")
        lines.append(f"Number of Trades:  {base_case.get('n_trades', 0):,}")
    else:
        lines.append("No performance metrics available.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF DETAILED LOG")
    lines.append("=" * 80)

    # Write to file
    with open(log_path, 'w') as f:
        f.write("\n".join(lines))

    logger.info("Detailed backtest log saved: %s", log_path)
    return log_path


# ------------------------------ walk-forward ----------------------------- #

def _build_walk_forward_windows(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if step_days <= 0 or test_days <= 0 or train_days <= 0:
        raise ValueError("Walk-forward days must be positive.")
    if step_days < test_days:
        # Overlapping test windows are allowed for faster adaptation (expert suggestion)
        # This is common in crypto where regimes change quickly
        logger.warning(
            "step_days (%d) < test_days (%d): test windows will overlap. "
            "This is intentional for faster adaptation.",
            step_days,
            test_days,
        )

    start = df.index.min()
    windows: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cur_start = start
    while True:
        train_start = cur_start
        train_end = train_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > df.index.max():
            break
        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]
        if train_df.empty or test_df.empty:
            break
        windows.append((train_df, test_df, train_start, train_end, test_end))
        cur_start = cur_start + pd.Timedelta(days=step_days)
    return windows


def _build_walk_forward_windows_multi_tf(
    df: pd.DataFrame,
    long_term_days: int,
    short_term_days: int,
    test_days: int,
    step_days: int,
) -> List[MultiTimeframeWindow]:
    """
    Build walk-forward windows with multi-timeframe training data.

    Each window contains:
    - long_term_train: Full training period (e.g., 180 days) for stable pair discovery
    - short_term_train: Recent subset (e.g., last 30 days) for regime validation
    - test: Out-of-sample test period (e.g., 21 days)

    Args:
        df: Full price DataFrame with datetime index
        long_term_days: Length of long-term training window (e.g., 180)
        short_term_days: Length of short-term subset at end of training (e.g., 30)
        test_days: Length of test window (e.g., 21)
        step_days: How much to slide window between iterations (e.g., 14)

    Returns:
        List of MultiTimeframeWindow objects
    """
    if any(d <= 0 for d in [long_term_days, short_term_days, test_days, step_days]):
        raise ValueError("All window parameters must be positive.")
    if short_term_days >= long_term_days:
        raise ValueError(
            f"short_term_days ({short_term_days}) must be less than "
            f"long_term_days ({long_term_days})"
        )

    start = df.index.min()
    windows: List[MultiTimeframeWindow] = []
    cur_start = start

    while True:
        train_start = cur_start
        train_end = train_start + pd.Timedelta(days=long_term_days)
        test_end = train_end + pd.Timedelta(days=test_days)

        # Check if we have enough data
        if test_end > df.index.max():
            break

        # Extract data slices
        long_term_train = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        if long_term_train.empty or test_df.empty:
            break

        # Short-term is the last N days of long-term training
        short_term_start = train_end - pd.Timedelta(days=short_term_days)
        short_term_train = df[(df.index >= short_term_start) & (df.index < train_end)]

        if short_term_train.empty:
            logger.warning(
                "Short-term train window is empty for period %s - %s, skipping",
                short_term_start,
                train_end,
            )
            cur_start = cur_start + pd.Timedelta(days=step_days)
            continue

        window = MultiTimeframeWindow(
            long_term_train=long_term_train,
            short_term_train=short_term_train,
            test=test_df,
            train_start=train_start,
            short_term_start=short_term_start,
            train_end=train_end,
            test_end=test_end,
        )
        windows.append(window)
        cur_start = cur_start + pd.Timedelta(days=step_days)

    logger.info(
        "Built %d multi-timeframe windows (long=%dd, short=%dd, test=%dd, step=%dd)",
        len(windows),
        long_term_days,
        short_term_days,
        test_days,
        step_days,
    )
    return windows


# ------------------------------ main pipeline ----------------------------- #

def run_backtest(
    *,
    run_name: Optional[str],
    parquet_path: Optional[Path],
    max_pairs: Optional[int],
    diagnose_n: int,
    log_level: str,
    btc_symbol: str,
    strict_validation: bool = True,
    pretrained_risk_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    _setup_logging(log_level)
    logger.info("🚀 Starting Phase 5 backtest run...")

    # Feature compatibility check
    if hasattr(cfg, 'check_feature_compatibility'):
        warnings = cfg.check_feature_compatibility()
        for warning in warnings:
            logger.warning(warning)

    # 1) Create run dir + manifest
    run_id, run_dir = cfg.create_run_dir(run_name=run_name)
    cfg.save_manifest(run_dir=run_dir, extra_metadata={"runner": "research/backtest/run_simulation.py"})
    paths = cfg.get_run_paths(run_dir)
    logger.info("Run folder: %s", run_dir)

    # 2) Load data from parquet (produced by step0_export_data_full_year.py)
    raw_path = Path(parquet_path) if parquet_path is not None else cfg.PATH_RAW_PARQUET
    logger.info("Loading price matrix: %s", raw_path)
    price_df = _read_parquet_price_matrix(raw_path)
    _ensure_nonempty(price_df, "price_df")

    # 3) Load/resample full data once for walk-forward or single split
    logger.info("Preparing data via data_segmenter...")
    full_df = data_segmenter.load_and_resample(raw_path, strict_validation=strict_validation)
    _ensure_nonempty(full_df, "full_df")

    funding_rates = None
    if getattr(cfg, "USE_REAL_FUNDING", False) and cfg.PATH_FUNDING_PARQUET.exists():
        funding_rates = pd.read_parquet(cfg.PATH_FUNDING_PARQUET)

    walk_forward = bool(getattr(cfg, "WALK_FORWARD_ENABLED", False))
    if not walk_forward:
        # Standard split
        logger.info("Splitting train/test via data_segmenter...")
        train_df, test_df = data_segmenter.load_and_split(raw_path, strict_validation=strict_validation)
        _ensure_nonempty(train_df, "train_df")
        _ensure_nonempty(test_df, "test_df")

        # Drop coins with insufficient coverage to avoid look-ahead from aggressive fills
        min_coverage = float(getattr(cfg, "MIN_DATA_COVERAGE", 0.95))
        coverage = train_df.notna().mean()
        keep_cols = coverage[coverage >= min_coverage].index.tolist()
        if keep_cols:
            dropped = set(train_df.columns) - set(keep_cols)
            if dropped:
                logger.warning("Dropping %d coins with < %.2f coverage in TRAIN.", len(dropped), min_coverage)
            train_df = train_df[keep_cols]
            test_df = test_df[keep_cols]

        # Persist test prices for later analysis/diagnostics
        test_df.to_parquet(paths["test_prices"])
        windows = [(train_df, test_df, train_df.index.min(), train_df.index.max(), test_df.index.max())]
    else:
        # Resolve walk-forward preset configuration
        wf_preset_name = str(getattr(cfg, "WALK_FORWARD_PRESET", "default"))
        wf_presets = getattr(cfg, "WF_PRESETS", {})
        wf_preset = wf_presets.get(wf_preset_name, {})

        wf_train = int(wf_preset.get("train_days", getattr(cfg, "WALK_FORWARD_TRAIN_DAYS", cfg.TRAIN_DAYS)))
        wf_test = int(wf_preset.get("test_days", getattr(cfg, "WALK_FORWARD_TEST_DAYS", cfg.TEST_DAYS)))
        wf_step = int(wf_preset.get("step_days", getattr(cfg, "WALK_FORWARD_STEP_DAYS", wf_test)))
        use_exp_weighting = wf_preset.get("use_exp_weighting", False)
        exp_weight_hl_days = float(wf_preset.get("exp_weight_half_life_days", 30.0))

        logger.info(
            "Walk-forward preset '%s': train=%dd, test=%dd, step=%dd, exp_weight=%s",
            wf_preset_name,
            wf_train,
            wf_test,
            wf_step,
            use_exp_weighting,
        )

        # Check for multi-timeframe training mode
        use_multi_tf = bool(getattr(cfg, "ENABLE_MULTI_TIMEFRAME_TRAINING", False))

        if use_multi_tf:
            # Multi-timeframe windows with long-term + short-term training
            multi_tf_long_days = int(getattr(cfg, "MULTI_TF_LONG_TERM_DAYS", 180))
            multi_tf_short_days = int(getattr(cfg, "MULTI_TF_SHORT_TERM_DAYS", 30))

            logger.info(
                "Multi-timeframe training ENABLED: long=%dd, short=%dd (last %dd of training)",
                multi_tf_long_days,
                multi_tf_short_days,
                multi_tf_short_days,
            )

            multi_tf_windows = _build_walk_forward_windows_multi_tf(
                full_df,
                long_term_days=multi_tf_long_days,
                short_term_days=multi_tf_short_days,
                test_days=wf_test,
                step_days=wf_step,
            )
            if not multi_tf_windows:
                raise RuntimeError("Walk-forward produced 0 multi-TF windows. Check date span and window sizes.")

            # Convert to standard tuple format for compatibility with existing loop
            # But also store the multi-TF windows for two-stage pair selection
            windows = [
                (w.long_term_train, w.test, w.train_start, w.train_end, w.test_end)
                for w in multi_tf_windows
            ]
            # Store multi-TF windows for two-stage pair selection
            _multi_tf_windows_store = multi_tf_windows
        else:
            windows = _build_walk_forward_windows(full_df, wf_train, wf_test, wf_step)
            _multi_tf_windows_store = None

        if not windows:
            raise RuntimeError("Walk-forward produced 0 windows. Check date span and window sizes.")

    all_returns: List[pd.DataFrame] = []
    all_entries: List[pd.DataFrame] = []
    all_exits: List[pd.DataFrame] = []
    all_position_sizes: List[pd.DataFrame] = []  # Aggregate position sizing for realistic simulation
    all_pairs: List[str] = []
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    adaptive_log_path = run_dir / "adaptive_updates.jsonl"
    adaptive_controller = AdaptiveController(
        enabled=adaptive_cfg.ADAPTIVE_ENABLED,
        window_days=adaptive_cfg.ADAPTIVE_WINDOW_DAYS,
        min_trades=adaptive_cfg.ADAPTIVE_MIN_TRADES,
    )
    window_pairs_history: List[List[str]] = []

    all_test_prices: List[pd.DataFrame] = []
    last_window_data: Optional[Dict[str, Any]] = None

    # Per-window metrics for walk-forward progression tracking
    window_metrics_list: List[Dict[str, Any]] = []

    # Historical P&L data for pair quality filtering (ROI optimization)
    # Accumulates across windows for adaptive filtering
    historical_pnl_data: Dict[str, Dict] = {}

    # Exit reasons from previous window for smart cooldown (Phase 6)
    # Maps pair -> DataFrame of exit reasons from previous pnl_engine run
    previous_window_exit_reasons: Optional[pd.DataFrame] = None

    # ===== PHASE 1: STOP THE BLEEDING - Cross-window risk controls =====
    # Symbol blacklist tracks per-symbol performance across windows
    enable_symbol_blacklist = bool(getattr(cfg, "ENABLE_SYMBOL_BLACKLIST", False))
    symbol_blacklist: Optional[SymbolBlacklist] = None
    if enable_symbol_blacklist:
        symbol_blacklist = SymbolBlacklist.from_config()
        logger.info(
            "Symbol blacklist enabled: min_windows=%d, max_avg_loss=%.2f%%, max_stop_rate=%.1f%%",
            symbol_blacklist.min_windows,
            symbol_blacklist.max_avg_loss_pct * 100,
            symbol_blacklist.max_stop_rate * 100,
        )

    # Regime filter for BTC volatility and cross-sectional dispersion gating
    enable_regime_filter = bool(getattr(cfg, "ENABLE_REGIME_FILTER", False))
    regime_filter: Optional[RegimeFilter] = None
    if enable_regime_filter:
        regime_filter = RegimeFilter.from_config()
        logger.info(
            "Regime filter enabled: btc_vol_max_pctl=%.0f%%, dispersion_max_pctl=%.0f%%, spread_vol_range=[%.0f, %.0f] bps",
            regime_filter.btc_vol_max_percentile * 100,
            regime_filter.dispersion_max_percentile * 100,
            regime_filter.spread_vol_min_bps,
            regime_filter.spread_vol_max_bps,
        )

    # Pair scorer for multi-factor tradability scoring (Phase 2)
    enable_pair_scoring = bool(getattr(cfg, "ENABLE_PAIR_SCORING", False))
    pair_scorer: Optional[PairScorer] = None
    if enable_pair_scoring:
        pair_scorer = PairScorer.from_config()
        logger.info(
            "Pair scoring enabled: min_score=%.2f",
            pair_scorer.min_composite_score,
        )

    # Phase 5B: Window analysis for regime pattern detection
    enable_window_analysis = bool(getattr(cfg, "ENABLE_WINDOW_ANALYSIS", True))
    window_analysis: Optional[WindowAnalysis] = None
    if enable_window_analysis:
        window_analysis = create_window_analysis()
        logger.info("Window analysis enabled: comparing winning vs losing windows")

    # ML Signal Scorer setup for walk-forward training
    use_ml_scoring = bool(getattr(cfg, "USE_ML_SIGNAL_SCORING", False))
    ml_min_train_windows = int(getattr(cfg, "ML_MIN_TRAIN_WINDOWS", 2))
    ml_scorer: Optional[MLSignalScorer] = None
    ml_training_data: List[Dict[str, Any]] = []  # Accumulate training data from windows

    # Risk prediction setup (separate from ML scoring)
    use_risk_prediction = bool(getattr(cfg, "ENABLE_RISK_PREDICTION", False))
    risk_training_data: List[Dict[str, Any]] = []  # Accumulate risk data from windows
    risk_min_train_windows = 2  # Need at least 2 windows to train risk predictor
    pretrained_risk_predictor: Optional[RiskPredictor] = None  # Loaded once at startup

    if use_risk_prediction:
        # Load pretrained risk predictor if provided
        if pretrained_risk_model_path is not None and pretrained_risk_model_path.exists():
            try:
                import pickle
                with open(pretrained_risk_model_path, 'rb') as f:
                    pretrained_risk_predictor = pickle.load(f)
                if pretrained_risk_predictor.is_trained:
                    logger.info("Loaded pretrained risk predictor from %s (applying from window 0)", pretrained_risk_model_path)
                else:
                    logger.warning("Pretrained risk predictor exists but is not trained, ignoring")
                    pretrained_risk_predictor = None
            except Exception as e:
                logger.warning("Failed to load pretrained risk predictor: %s", e)
                pretrained_risk_predictor = None

        if pretrained_risk_predictor is None:
            logger.info("Risk prediction enabled. Will train after %d windows.", risk_min_train_windows)
        else:
            logger.info("Risk prediction enabled with pretrained model (skipping training).")

    if use_ml_scoring:
        logger.info("ML Signal Scoring enabled. Will train after %d windows.", ml_min_train_windows)
        ml_config = MLScorerConfig(
            model_type=str(getattr(cfg, "ML_MODEL_TYPE", "classifier")),
            min_training_samples=int(getattr(cfg, "ML_MIN_TRAINING_SAMPLES", 50)),
            lgb_params={
                "objective": "binary" if getattr(cfg, "ML_MODEL_TYPE", "classifier") == "classifier" else "regression",
                "metric": "auc" if getattr(cfg, "ML_MODEL_TYPE", "classifier") == "classifier" else "mse",
                "boosting_type": "gbdt",
                "num_leaves": int(getattr(cfg, "ML_LGB_NUM_LEAVES", 15)),
                "max_depth": int(getattr(cfg, "ML_LGB_MAX_DEPTH", 4)),
                "learning_rate": float(getattr(cfg, "ML_LGB_LEARNING_RATE", 0.05)),
                "n_estimators": int(getattr(cfg, "ML_LGB_N_ESTIMATORS", 200)),
                "min_child_samples": int(getattr(cfg, "ML_LGB_MIN_CHILD_SAMPLES", 10)),
                "subsample": float(getattr(cfg, "ML_LGB_SUBSAMPLE", 0.8)),
                "colsample_bytree": float(getattr(cfg, "ML_LGB_COLSAMPLE", 0.8)),
                "reg_alpha": float(getattr(cfg, "ML_LGB_REG_ALPHA", 0.1)),
                "reg_lambda": float(getattr(cfg, "ML_LGB_REG_LAMBDA", 0.1)),
                "random_state": 42,
                "verbose": -1,
            },
            velocity_lookback=int(getattr(cfg, "ML_VELOCITY_LOOKBACK", 3)),
            drift_lookback_bars=int(getattr(cfg, "ML_DRIFT_LOOKBACK_BARS", 168)),
            rolling_perf_lookback=int(getattr(cfg, "ML_ROLLING_PERF_LOOKBACK", 240)),
        )
        ml_scorer = MLSignalScorer(config=ml_config)

    for w_idx, (train_df, test_df, train_start, train_end, test_end) in enumerate(windows):
        logger.info("Walk-forward window %d: train=%s->%s test=%s->%s", w_idx, train_start, train_end, train_end, test_end)

        # Drop coins with insufficient coverage per window
        min_coverage = float(getattr(cfg, "MIN_DATA_COVERAGE", 0.95))
        coverage = train_df.notna().mean()
        keep_cols = coverage[coverage >= min_coverage].index.tolist()
        if keep_cols:
            train_df = train_df[keep_cols]
            test_df = test_df[keep_cols]

        if train_df.empty or test_df.empty:
            logger.warning("Skipping window %d due to empty train/test after coverage filter.", w_idx)
            continue

        # Optional universe pruning (e.g., memecoin or blacklist exclusion)
        exclude_symbols = set(getattr(cfg, "EXCLUDE_SYMBOLS", ()))
        if exclude_symbols:
            drop_cols = [c for c in train_df.columns if c in exclude_symbols]
            if drop_cols:
                logger.info("Excluding %d symbols via EXCLUDE_SYMBOLS: %s", len(drop_cols), drop_cols)
                train_df = train_df.drop(columns=drop_cols, errors="ignore")
                test_df = test_df.drop(columns=drop_cols, errors="ignore")

        if train_df.empty or test_df.empty:
            logger.warning("Skipping window %d due to empty train/test after exclusion filter.", w_idx)
            continue

        # Liquidity proxy filter (volatility-based)
        if getattr(cfg, "ENABLE_LIQUIDITY_FILTER", False):
            vol = _compute_coin_volatility(train_df)
            vol = vol.dropna()
            if not vol.empty:
                vol_cap_abs = float(getattr(cfg, "LIQUIDITY_MAX_VOL_ABS", 0.0))
                if vol_cap_abs > 0:
                    drop_coins = vol[vol > vol_cap_abs].index.tolist()
                else:
                    pctl = float(getattr(cfg, "LIQUIDITY_MAX_VOL_PCTL", 0.90))
                    threshold = vol.quantile(pctl)
                    drop_coins = vol[vol > threshold].index.tolist()
                if drop_coins:
                    logger.info("Liquidity proxy filter removed %d coins.", len(drop_coins))
                    train_df = train_df.drop(columns=drop_coins, errors="ignore")
                    test_df = test_df.drop(columns=drop_coins, errors="ignore")

        # Funding filter (per-coin coverage + mean funding)
        if getattr(cfg, "ENABLE_FUNDING_FILTER", False) and funding_rates is not None:
            mean_funding, coverage = _compute_funding_stats(funding_rates, train_df.index)
            min_cov = float(getattr(cfg, "FUNDING_MIN_COVERAGE", 0.7))
            max_abs_mean = float(getattr(cfg, "FUNDING_MAX_ABS_MEAN", 0.0001))
            drop_funding = set()
            for coin in train_df.columns:
                if coin not in mean_funding.index:
                    drop_funding.add(coin)
                    continue
                cov = coverage.get(coin, 0.0)
                mean_val = mean_funding.get(coin, np.nan)
                if cov < min_cov or not np.isfinite(mean_val) or abs(mean_val) > max_abs_mean:
                    drop_funding.add(coin)
            if drop_funding:
                logger.info("Funding filter removed %d coins.", len(drop_funding))
                train_df = train_df.drop(columns=list(drop_funding), errors="ignore")
                test_df = test_df.drop(columns=list(drop_funding), errors="ignore")

        if train_df.empty or test_df.empty:
            logger.warning("Skipping window %d due to empty train/test after liquidity/funding filter.", w_idx)
            continue

        # =============================================================================
        # IMPROVEMENT #1: OUTLIER EVENT EXCLUSION
        # =============================================================================
        # Filter out known outlier events (flash crashes, etc.) from test data
        # These events skew performance and aren't representative of normal alpha
        enable_outlier_exclusion = bool(getattr(cfg, "ENABLE_OUTLIER_EXCLUSION", False))
        outlier_events = getattr(cfg, "OUTLIER_EVENTS", [])
        if enable_outlier_exclusion and outlier_events:
            rows_before = len(test_df)
            for event in outlier_events:
                event_start = pd.Timestamp(event.get("start", "1900-01-01"), tz="UTC")
                event_end = pd.Timestamp(event.get("end", "1900-01-01"), tz="UTC")
                event_name = event.get("name", "Unknown")
                # Make sure test_df index is tz-aware
                if test_df.index.tz is None:
                    test_df.index = test_df.index.tz_localize("UTC")
                # Exclude rows within the event period
                mask = ~((test_df.index >= event_start) & (test_df.index <= event_end))
                excluded_rows = (~mask).sum()
                if excluded_rows > 0:
                    logger.info(
                        "Outlier exclusion: removing %d rows for '%s' (%s to %s)",
                        excluded_rows, event_name, event_start.date(), event_end.date()
                    )
                test_df = test_df[mask]
            rows_after = len(test_df)
            if rows_before > rows_after:
                logger.info(
                    "Outlier exclusion: total %d rows removed (%.1f%% of test data)",
                    rows_before - rows_after,
                    100.0 * (rows_before - rows_after) / rows_before,
                )

        if test_df.empty:
            logger.warning("Skipping window %d due to empty test data after outlier exclusion.", w_idx)
            continue

        all_test_prices.append(test_df)

        # 4) Pair selection on TRAIN only WITH intelligent clustering and feature filtering
        logger.info("Selecting valid pairs on TRAIN only...")

        available_coins = [c for c in train_df.columns if c != 'timestamp']
        logger.info("Available coins: %d", len(available_coins))

        # ===== MARKET REGIME DETECTION =====
        # Detect current market regime for adaptive parameters
        current_regime: Optional[MarketRegime] = None
        regime_half_life_min = int(getattr(cfg, "SCAN_MIN_HALF_LIFE", 60))
        regime_half_life_max = int(getattr(cfg, "SCAN_MAX_HALF_LIFE", 960))

        if getattr(cfg, "ENABLE_REGIME_DETECTION", False):
            try:
                # Build regime config from backtest config
                regime_config = {
                    "lookback_bars": int(getattr(cfg, "REGIME_LOOKBACK_BARS", 336)),
                    "vol_low_percentile": float(getattr(cfg, "VOL_REGIME_LOW_PCTL", 0.25)),
                    "vol_high_percentile": float(getattr(cfg, "VOL_REGIME_HIGH_PCTL", 0.75)),
                    "trend_trending_thresh": float(getattr(cfg, "TREND_REGIME_TRENDING_THRESH", 0.15)),
                    "trend_mean_revert_thresh": float(getattr(cfg, "TREND_REGIME_MEAN_REVERT_THRESH", -0.10)),
                    "half_life_default": getattr(cfg, "HALF_LIFE_RANGE_DEFAULT", (60, 960)),
                    "half_life_high_vol": getattr(cfg, "HALF_LIFE_RANGE_HIGH_VOL", (120, 1200)),
                    "half_life_low_vol": getattr(cfg, "HALF_LIFE_RANGE_LOW_VOL", (40, 720)),
                    "half_life_trending": getattr(cfg, "HALF_LIFE_RANGE_TRENDING", (40, 480)),
                    "half_life_mean_revert": getattr(cfg, "HALF_LIFE_RANGE_MEAN_REVERT", (80, 1440)),
                }

                # Find BTC index for regime detection
                btc_idx = 0
                if btc_symbol in train_df.columns:
                    btc_idx = list(train_df.columns).index(btc_symbol)

                current_regime = detect_market_regime(
                    price_matrix=train_df.values,
                    btc_index=btc_idx,
                    lookback_bars=regime_config["lookback_bars"],
                    config=regime_config,
                )

                # Use regime-based half-life range
                regime_half_life_min = current_regime.recommended_min_half_life
                regime_half_life_max = current_regime.recommended_max_half_life

                logger.info(
                    "Regime detected: vol=%s, trend=%s, btc_vol_pctl=%.2f, autocorr=%.3f",
                    current_regime.volatility.value,
                    current_regime.trend.value,
                    current_regime.btc_vol_percentile,
                    current_regime.market_autocorr,
                )
                logger.info(
                    "Regime-based half-life range: [%d, %d] bars",
                    regime_half_life_min,
                    regime_half_life_max,
                )

            except Exception as e:
                logger.warning("Regime detection failed: %s. Using default half-life range.", e)

        # ===== CLUSTER/SECTOR DIVERSIFICATION SETUP =====
        # Load or create cluster map for diversification
        cluster_map_for_diversification: Dict[int, List[str]] = {}
        coin_to_cluster: Dict[str, int] = {}

        if getattr(cfg, "ENABLE_CLUSTER_DIVERSIFICATION", False):
            cluster_file = str(getattr(cfg, "BACKTEST_CLUSTER_FILE", "data/cluster_assignments.json"))
            use_dynamic = bool(getattr(cfg, "USE_DYNAMIC_CLUSTERING", True))

            # Try to load pre-computed clusters
            if not use_dynamic:
                cluster_map_for_diversification = load_cluster_map_from_file(cluster_file)

            # If no pre-computed or dynamic enabled, use single cluster for now
            # (In production, you'd call get_cluster_map() here with appropriate data)
            if not cluster_map_for_diversification:
                # Fallback: put all coins in one cluster (no diversification effect)
                cluster_map_for_diversification = {0: available_coins}
                logger.info("Cluster diversification: using single cluster (no pre-computed clusters)")
            else:
                logger.info("Cluster diversification: loaded %d clusters", len(cluster_map_for_diversification))

            coin_to_cluster = invert_cluster_map(cluster_map_for_diversification)

        # Use single cluster for pair scanning (clustering disabled due to data source mismatch)
        logger.info("Using single cluster approach for pair scanning")
        coin_features = pd.DataFrame()  # No features for now
        cluster_map = {0: available_coins}
        logger.info("Created cluster with %d coins for pair scanning", len(available_coins))

        scanner = CointegrationScanner(
            cluster_map=cluster_map,
            p_value_threshold=float(getattr(cfg, "SCAN_P_VALUE_THRESHOLD", 0.10)),
            max_drift_z=float(getattr(cfg, "SCAN_MAX_DRIFT_Z", 3.0)),
            min_half_life=float(regime_half_life_min),  # Use regime-based range
            max_half_life=float(regime_half_life_max),  # Use regime-based range
        )
        logger.info("Scanner: p_value<=%s, drift_z<=%s, half_life=%s-%s",
                    scanner.p_value_threshold, scanner.max_drift_z, scanner.min_half_life, scanner.max_half_life)

        # Use find_pairs_from_matrix for backtest mode (uses train_df directly)
        scan_with_rolling = bool(getattr(cfg, "ENABLE_ROLLING_COINT_CHECK", False))
        scan_with_beta = bool(getattr(cfg, "ENABLE_BETA_STABILITY_CHECK", False))

        # Check for multi-timeframe consensus pair selection
        use_multi_tf_consensus = (
            bool(getattr(cfg, "ENABLE_MULTI_TIMEFRAME_TRAINING", False))
            and bool(getattr(cfg, "MULTI_TF_REQUIRE_CONSENSUS", True))
            and _multi_tf_windows_store is not None
            and w_idx < len(_multi_tf_windows_store)
        )

        # Check for two-stage pair selection (expert crypto suggestion)
        use_two_stage = bool(getattr(cfg, "ENABLE_TWO_STAGE_PAIR_SELECTION", False))

        if use_multi_tf_consensus:
            # Multi-timeframe consensus pair selection:
            # 1. Find stable pairs from long-term window (stricter p-value)
            # 2. Validate pairs still work in short-term window (looser p-value)
            # 3. Only keep pairs that pass both (consensus)
            mtf_window = _multi_tf_windows_store[w_idx]
            long_term_pval = float(getattr(cfg, "MULTI_TF_LONG_TERM_PVALUE", 0.02))
            short_term_pval = float(getattr(cfg, "MULTI_TF_SHORT_TERM_PVALUE", 0.05))

            logger.info(
                "Multi-TF CONSENSUS pair selection: long-term p<%.3f, short-term p<%.3f",
                long_term_pval,
                short_term_pval,
            )

            # Stage 1: Find stable pairs from long-term (180d) window
            long_term_scanner = CointegrationScanner(
                cluster_map=cluster_map,
                p_value_threshold=long_term_pval,  # Stricter for stability
                max_drift_z=float(getattr(cfg, "SCAN_MAX_DRIFT_Z", 3.0)),
                min_half_life=float(regime_half_life_min),
                max_half_life=float(regime_half_life_max),
            )
            long_term_output = long_term_scanner.find_pairs_from_matrix(
                mtf_window.long_term_train,
                train_ratio=0.8,
                check_rolling_coint=scan_with_rolling,
                check_beta_stability=scan_with_beta,
            )
            long_term_pairs, long_term_df = _coerce_pairs_list(long_term_output)
            long_term_pairs = [p for p in long_term_pairs if isinstance(p, str)]

            logger.info(
                "Stage 1 (long-term %dd, p<%.3f): found %d candidate pairs",
                mtf_window.long_term_days,
                long_term_pval,
                len(long_term_pairs),
            )

            if not long_term_pairs:
                # Fallback: use standard scanner if no long-term pairs
                logger.warning("No long-term pairs found, falling back to standard scanner")
                scanner_output = scanner.find_pairs_from_matrix(
                    train_df,
                    train_ratio=0.8,
                    check_rolling_coint=scan_with_rolling,
                    check_beta_stability=scan_with_beta,
                )
            else:
                # Stage 2: Validate pairs in short-term (30d) window
                short_term_scanner = CointegrationScanner(
                    cluster_map=cluster_map,
                    p_value_threshold=short_term_pval,  # Looser for validation
                    max_drift_z=float(getattr(cfg, "SCAN_MAX_DRIFT_Z", 3.0)),
                    min_half_life=float(regime_half_life_min),
                    max_half_life=float(regime_half_life_max),
                )
                short_term_output = short_term_scanner.find_pairs_from_matrix(
                    mtf_window.short_term_train,
                    train_ratio=0.8,
                    check_rolling_coint=False,  # Don't need rolling check for validation
                    check_beta_stability=False,
                )
                short_term_pairs, short_term_df = _coerce_pairs_list(short_term_output)
                short_term_pairs_set = set(p for p in short_term_pairs if isinstance(p, str))

                # Consensus: pairs must pass both long-term and short-term
                consensus_pairs = [p for p in long_term_pairs if p in short_term_pairs_set]

                logger.info(
                    "Stage 2 (short-term %dd, p<%.3f): %d pairs valid, consensus: %d/%d (%.1f%%)",
                    mtf_window.short_term_days,
                    short_term_pval,
                    len(short_term_pairs_set),
                    len(consensus_pairs),
                    len(long_term_pairs),
                    100 * len(consensus_pairs) / max(len(long_term_pairs), 1),
                )

                # Use consensus pairs with long-term DataFrame for metadata
                if consensus_pairs:
                    # Filter long_term_df to only consensus pairs
                    if long_term_df is not None and not long_term_df.empty:
                        if 'pair' in long_term_df.columns:
                            scanner_output = (consensus_pairs, long_term_df[long_term_df['pair'].isin(consensus_pairs)])
                        else:
                            scanner_output = (consensus_pairs, long_term_df)
                    else:
                        scanner_output = (consensus_pairs, pd.DataFrame())
                else:
                    # Fallback: if no consensus, use long-term pairs with warning
                    logger.warning(
                        "No consensus pairs found! Using %d long-term pairs instead",
                        len(long_term_pairs),
                    )
                    scanner_output = long_term_output

        elif use_two_stage:
            logger.info("Using TWO-STAGE pair selection...")
            scanner_output = scanner.find_pairs_two_stage(
                price_matrix=train_df,
                discovery_lookback_days=int(getattr(cfg, "TWO_STAGE_DISCOVERY_LOOKBACK_DAYS", 120)),
                validation_lookback_days=int(getattr(cfg, "TWO_STAGE_VALIDATION_LOOKBACK_DAYS", 30)),
                discovery_p_value=float(getattr(cfg, "TWO_STAGE_DISCOVERY_P_VALUE", 0.05)),
                validation_p_value=float(getattr(cfg, "TWO_STAGE_VALIDATION_P_VALUE", 0.02)),
                validation_min_half_life=int(getattr(cfg, "TWO_STAGE_VALIDATION_MIN_HALF_LIFE", 40)),
                validation_max_half_life=int(getattr(cfg, "TWO_STAGE_VALIDATION_MAX_HALF_LIFE", 500)),
                require_both=bool(getattr(cfg, "TWO_STAGE_REQUIRE_BOTH", True)),
            )
        else:
            scanner_output = scanner.find_pairs_from_matrix(
                train_df,
                train_ratio=0.8,
                check_rolling_coint=scan_with_rolling,
                check_beta_stability=scan_with_beta,
            )
        valid_pairs, pairs_df = _coerce_pairs_list(scanner_output)

        # Enforce consistent format + filter to coins present in train/test
        valid_pairs = [p for p in valid_pairs if isinstance(p, str)]
        valid_pairs = _clip_pairs_to_available_columns(valid_pairs, train_df)
        valid_pairs = _clip_pairs_to_available_columns(valid_pairs, test_df)
        valid_pairs_raw = list(valid_pairs)

        # Pair-level funding asymmetry filter
        if getattr(cfg, "ENABLE_FUNDING_FILTER", False) and funding_rates is not None:
            mean_funding, _ = _compute_funding_stats(funding_rates, train_df.index)
            max_diff = float(getattr(cfg, "FUNDING_MAX_MEAN_DIFF", 0.00006))
            filtered_pairs = []
            for pair in valid_pairs:
                coin_y, coin_x = pair.split(cfg.PAIR_ID_SEPARATOR)
                if coin_y not in mean_funding.index or coin_x not in mean_funding.index:
                    continue
                if abs(mean_funding[coin_y] - mean_funding[coin_x]) <= max_diff:
                    filtered_pairs.append(pair)
            if len(filtered_pairs) != len(valid_pairs):
                logger.info(
                    "Funding asymmetry filter: %d/%d pairs kept.",
                    len(filtered_pairs),
                    len(valid_pairs),
                )
            valid_pairs = filtered_pairs

        # ===== CLUSTER DIVERSIFICATION FILTER =====
        # Limit pairs per cluster to prevent sector concentration
        if getattr(cfg, "ENABLE_CLUSTER_DIVERSIFICATION", False) and cluster_map_for_diversification:
            max_per_cluster = int(getattr(cfg, "MAX_POSITIONS_PER_CLUSTER", 2))
            if max_per_cluster > 0 and len(cluster_map_for_diversification) > 1:
                # Build priority scores from pairs_df if available
                priority_scores = {}
                if pairs_df is not None and not pairs_df.empty and "p_value" in pairs_df.columns:
                    for _, row in pairs_df.iterrows():
                        pair = row.get("pair")
                        if pair:
                            # Lower p-value = higher priority
                            priority_scores[pair] = 1.0 - row.get("p_value", 0.5)

                filtered_cluster_pairs, cluster_stats = filter_pairs_by_cluster_limits(
                    pairs=valid_pairs,
                    cluster_map=cluster_map_for_diversification,
                    max_per_cluster=max_per_cluster,
                    existing_positions=[],  # Fresh for each window
                    separator=cfg.PAIR_ID_SEPARATOR,
                    priority_scores=priority_scores if priority_scores else None,
                )

                if len(filtered_cluster_pairs) < len(valid_pairs):
                    logger.info(
                        "Cluster diversification: %d/%d pairs kept (removed %d)",
                        len(filtered_cluster_pairs),
                        len(valid_pairs),
                        cluster_stats.pairs_removed,
                    )
                    logger.info(
                        "Cluster distribution: %s",
                        cluster_stats.cluster_distribution,
                    )
                    valid_pairs = filtered_cluster_pairs
                    if pairs_df is not None and not pairs_df.empty:
                        pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]

        # Multi-window persistence filter (stability across windows)
        if getattr(cfg, "ENABLE_PAIR_PERSISTENCE", False):
            min_count = int(getattr(cfg, "PAIR_PERSISTENCE_MIN_COUNT", 1))
            window_count = int(getattr(cfg, "PAIR_PERSISTENCE_WINDOWS", 1))
            if min_count > 1 and window_count > 1 and len(window_pairs_history) >= (min_count - 1):
                history = window_pairs_history[-(window_count - 1):]
                counts = Counter(p for window in history for p in window)
                persistent = [p for p in valid_pairs if counts.get(p, 0) + 1 >= min_count]
                if len(persistent) != len(valid_pairs):
                    logger.info(
                        "Persistence filter: %d/%d pairs kept.",
                        len(persistent),
                        len(valid_pairs),
                    )
                valid_pairs = persistent

        if pairs_df is not None and not pairs_df.empty and valid_pairs:
            pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]

        window_pairs_history.append(valid_pairs_raw)

        # Apply feature-based quality filtering
        if not coin_features.empty:
            logger.info("Before feature filter: %d pairs", len(valid_pairs))
            valid_pairs, pair_feature_stats = _filter_pairs_by_features(valid_pairs, coin_features)
            logger.info("After feature filter: %d pairs", len(valid_pairs))
            if pair_feature_stats:
                _write_json(run_dir / "pair_feature_stats.json", pair_feature_stats)

        # Apply historical performance filter (ROI optimization)
        # Uses accumulated P&L data from previous windows to filter out underperforming pairs
        enable_hist_filter = bool(getattr(cfg, "ENABLE_COST_GROSS_FILTER", False)) or \
                            bool(getattr(cfg, "ENABLE_STOP_LOSS_PENALTY", False))
        if enable_hist_filter and historical_pnl_data and w_idx > 0:
            logger.info("Before historical performance filter: %d pairs", len(valid_pairs))
            valid_pairs, perf_filter_stats = _filter_pairs_by_historical_performance(
                pairs=valid_pairs,
                historical_pnl_data=historical_pnl_data,
                max_cost_to_gross_ratio=float(getattr(cfg, "MAX_COST_TO_GROSS_RATIO", 0.80)),
                max_stop_loss_rate=float(getattr(cfg, "MAX_STOP_LOSS_RATE", 0.40)),
            )
            logger.info("After historical performance filter: %d pairs", len(valid_pairs))
            if perf_filter_stats and perf_filter_stats.get("pair_metrics"):
                _write_json(window_dir / "pair_performance_filter.json", perf_filter_stats)
            if pairs_df is not None and not pairs_df.empty:
                pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]

        # Diversification: limit per-coin pair concentration
        max_pairs_per_coin = int(getattr(cfg, "MAX_PAIRS_PER_COIN", 0))
        if max_pairs_per_coin > 0 and pairs_df is not None and not pairs_df.empty:
            coin_counts: Dict[str, int] = {}
            selected_pairs: List[str] = []
            pairs_df = pairs_df.sort_values(["p_value", "half_life_bars"])
            for _, row in pairs_df.iterrows():
                pair = row.get("pair")
                coin_y = row.get("coin_y")
                coin_x = row.get("coin_x")
                if not pair or not coin_y or not coin_x:
                    continue
                if coin_counts.get(coin_y, 0) >= max_pairs_per_coin:
                    continue
                if coin_counts.get(coin_x, 0) >= max_pairs_per_coin:
                    continue
                selected_pairs.append(pair)
                coin_counts[coin_y] = coin_counts.get(coin_y, 0) + 1
                coin_counts[coin_x] = coin_counts.get(coin_x, 0) + 1
            if selected_pairs:
                valid_pairs = [p for p in valid_pairs if p in selected_pairs]
                if pairs_df is not None and not pairs_df.empty:
                    pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]

        # Optional cap for speed
        if max_pairs is not None and max_pairs > 0:
            valid_pairs = valid_pairs[: max_pairs]
            if pairs_df is not None and not pairs_df.empty:
                pairs_df = pairs_df.iloc[: max_pairs]

        if not valid_pairs and (scan_with_rolling or scan_with_beta) and getattr(cfg, "ALLOW_SCAN_FALLBACK", False):
            logger.warning("No pairs found under strict stability checks; retrying without rolling/beta checks.")
            scanner_output = scanner.find_pairs_from_matrix(
                train_df,
                train_ratio=0.8,
                check_rolling_coint=False,
                check_beta_stability=False,
            )
            valid_pairs, pairs_df = _coerce_pairs_list(scanner_output)

            # Apply diversification filter to fallback results too
            if valid_pairs and max_pairs_per_coin > 0 and pairs_df is not None and not pairs_df.empty:
                coin_counts_fb: Dict[str, int] = {}
                selected_pairs_fb: List[str] = []
                pairs_df = pairs_df.sort_values(["p_value", "half_life_bars"])
                for _, row in pairs_df.iterrows():
                    pair = row.get("pair")
                    coin_y = row.get("coin_y")
                    coin_x = row.get("coin_x")
                    if not pair or not coin_y or not coin_x:
                        continue
                    if coin_counts_fb.get(coin_y, 0) >= max_pairs_per_coin:
                        continue
                    if coin_counts_fb.get(coin_x, 0) >= max_pairs_per_coin:
                        continue
                    selected_pairs_fb.append(pair)
                    coin_counts_fb[coin_y] = coin_counts_fb.get(coin_y, 0) + 1
                    coin_counts_fb[coin_x] = coin_counts_fb.get(coin_x, 0) + 1
                if len(selected_pairs_fb) < len(valid_pairs):
                    logger.info("Fallback diversification filter: %d/%d pairs kept.", len(selected_pairs_fb), len(valid_pairs))
                    valid_pairs = [p for p in valid_pairs if p in selected_pairs_fb]
                    pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]

        if not valid_pairs:
            logger.warning("Window %d produced 0 pairs; skipping.", w_idx)
            continue

        # ===== PHASE 1: Apply symbol blacklist filter =====
        # Filter out pairs containing blacklisted symbols (from previous windows)
        if enable_symbol_blacklist and symbol_blacklist is not None and len(symbol_blacklist.blacklist) > 0:
            pairs_before_blacklist = len(valid_pairs)
            valid_pairs, blocked_pairs = symbol_blacklist.filter_pairs(valid_pairs, cfg.PAIR_ID_SEPARATOR)
            if pairs_df is not None and not pairs_df.empty:
                pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]
            if len(blocked_pairs) > 0:
                logger.info(
                    "Symbol blacklist filter: %d/%d pairs kept (blocked: %s)",
                    len(valid_pairs),
                    pairs_before_blacklist,
                    blocked_pairs[:5] if len(blocked_pairs) > 5 else blocked_pairs,
                )

        if not valid_pairs:
            logger.warning("Window %d: all pairs blocked by symbol blacklist; skipping.", w_idx)
            continue

        # ===== PHASE 2: Apply pair scoring filter (optional) =====
        # Score pairs on multiple tradability factors and filter by minimum score
        # Phase 6: Top-N + Floor selection for guaranteed diversification
        if enable_pair_scoring and pair_scorer is not None:
            pairs_before_scoring = len(valid_pairs)
            scored_pairs = pair_scorer.score_pairs(
                train_df=train_df,
                pairs=valid_pairs,
                historical_pnl=historical_pnl_data,
            )

            # Sort by score descending
            scored_pairs.sort(key=lambda p: p.composite_score, reverse=True)

            # Top-N + Floor selection (Phase 6)
            enable_topn_selection = bool(getattr(cfg, "ENABLE_TOPN_PAIR_SELECTION", True))
            if enable_topn_selection:
                top_n = int(getattr(cfg, "PAIR_SELECTION_TOP_N", 25))
                min_floor = float(getattr(cfg, "PAIR_SELECTION_MIN_FLOOR", 0.25))

                # Take top N pairs
                top_pairs = scored_pairs[:top_n]

                # Apply safety floor (reject truly bad pairs even if in top N)
                scored_pairs = [p for p in top_pairs if p.composite_score >= min_floor]

                logger.info(
                    "Top-N pair selection: top %d → %d after floor (min=%.2f)",
                    min(top_n, len(scored_pairs)), len(scored_pairs), min_floor
                )
            else:
                # Legacy absolute threshold selection
                scored_pairs = [p for p in scored_pairs if p.composite_score >= pair_scorer.min_composite_score]

            valid_pairs = [p.pair for p in scored_pairs]
            if pairs_df is not None and not pairs_df.empty:
                pairs_df = pairs_df[pairs_df["pair"].isin(valid_pairs)]
            if len(valid_pairs) < pairs_before_scoring:
                logger.info(
                    "Pair scoring filter: %d/%d pairs passed (min_score=%.2f)",
                    len(valid_pairs),
                    pairs_before_scoring,
                    pair_scorer.min_composite_score,
                )
                # Log top 5 pairs by score
                if scored_pairs:
                    logger.info("Top scored pairs: %s", [(p.pair, f"{p.composite_score:.3f}") for p in scored_pairs[:5]])

        if not valid_pairs:
            logger.warning("Window %d: no pairs passed scoring filter; skipping.", w_idx)
            continue

        # Build per-pair half-life map when available
        pair_half_lives: Dict[str, float] = {}
        if pairs_df is not None and not pairs_df.empty and "half_life_bars" in pairs_df.columns:
            hl_series = pairs_df.set_index("pair")["half_life_bars"]
            for pair in valid_pairs:
                if pair in hl_series.index:
                    hl_val = float(hl_series.loc[pair])
                    if np.isfinite(hl_val):
                        pair_half_lives[pair] = hl_val

        # Save pairs list and (optional) full DF per window
        window_id = f"window_{w_idx:02d}"
        window_dir = windows_dir / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        _write_json(window_dir / "valid_pairs.json", {"pairs": valid_pairs})
        if pairs_df is not None and not pairs_df.empty:
            pairs_df.to_parquet(window_dir / "valid_pairs_details.parquet")

        logger.info("✅ Valid pairs: %d", len(valid_pairs))

        # 5) Warm-start persistence (TRAIN)
        # Compute adaptive Kalman parameters if regime detection is enabled
        kalman_delta = None
        kalman_R = None

        if getattr(cfg, "ENABLE_ADAPTIVE_KALMAN", False) and current_regime is not None:
            try:
                kalman_config = {
                    "KALMAN_BASE_DELTA": float(getattr(cfg, "KALMAN_BASE_DELTA", 1e-6)),
                    "KALMAN_BASE_R": float(getattr(cfg, "KALMAN_BASE_R", 1e-2)),
                    "KALMAN_DELTA_MULT_MIN": float(getattr(cfg, "KALMAN_DELTA_MULT_MIN", 0.3)),
                    "KALMAN_DELTA_MULT_MAX": float(getattr(cfg, "KALMAN_DELTA_MULT_MAX", 3.0)),
                    "KALMAN_R_MULT_MIN": float(getattr(cfg, "KALMAN_R_MULT_MIN", 0.5)),
                    "KALMAN_R_MULT_MAX": float(getattr(cfg, "KALMAN_R_MULT_MAX", 3.0)),
                }
                kalman_delta, kalman_R = get_adaptive_params_from_config(
                    regime=current_regime,
                    config=kalman_config,
                )
                logger.info(
                    "Adaptive Kalman params: delta=%.2e, R=%.2e",
                    kalman_delta,
                    kalman_R,
                )
            except Exception as e:
                logger.warning("Adaptive Kalman computation failed: %s. Using defaults.", e)
                kalman_delta = None
                kalman_R = None

        logger.info("Computing warm states on TRAIN...")

        # Determine Kalman warmup window parameters
        kalman_warmup_days = None
        kalman_warmup_hl_mult = None

        if getattr(cfg, "ENABLE_KALMAN_WARMUP_WINDOW", False):
            kalman_warmup_days = int(getattr(cfg, "KALMAN_WARMUP_DAYS", 21))
            kalman_warmup_hl_mult = float(getattr(cfg, "KALMAN_WARMUP_HALF_LIFE_MULT", 3.0))
            logger.info(
                "Kalman warmup window enabled: days=%d, half_life_mult=%.1f",
                kalman_warmup_days,
                kalman_warmup_hl_mult,
            )

        warm_states = kalman_state_io.compute_warm_states(
            train_df=train_df,
            valid_pairs=valid_pairs,
            pair_half_lives=pair_half_lives if pair_half_lives else None,
            delta=kalman_delta,  # Use adaptive delta if computed
            R=kalman_R,          # Use adaptive R if computed
            warmup_days=kalman_warmup_days,
            warmup_half_life_mult=kalman_warmup_hl_mult,
            warmup_min_days=int(getattr(cfg, "KALMAN_WARMUP_MIN_DAYS", 7)),
            warmup_max_days=int(getattr(cfg, "KALMAN_WARMUP_MAX_DAYS", 45)),
        )
        kalman_state_io.save_warm_states(run_dir=window_dir, states=warm_states)

        # Update valid_pairs to only include pairs with valid warm states
        # (some pairs may be rejected by warm-start validation)
        valid_pairs = [p for p in valid_pairs if p in warm_states]
        if not valid_pairs:
            logger.warning("Window %d: all pairs rejected by warm-start validation; skipping.", w_idx)
            continue

        logger.info("✅ Warm states computed: %d", len(warm_states))

        # 6) Signal generation on TEST (causal) with optional trend overlay
        logger.info("Generating TEST signals (z/vol/beta)...")
        enable_trend_overlay = bool(getattr(cfg, "ENABLE_TREND_OVERLAY", False))

        if enable_trend_overlay:
            signals = signal_generation.generate_signals_with_trend_overlay(
                test_df=test_df,
                valid_pairs=valid_pairs,
                warm_states=warm_states,
                enable_trend_overlay=True,
            )
            logger.info("Trend overlay computed for %d pairs", len(valid_pairs))
        else:
            signals = signal_generation.generate_signals(
                test_df=test_df,
                valid_pairs=valid_pairs,
                warm_states=warm_states,
            )

        z_df = signals.z_score
        vol_df = signals.spread_volatility
        beta_df = signals.beta

        # Extract trend overlay if available
        trend_score_df = signals.trend_score if hasattr(signals, 'trend_score') else None
        suppress_long_df = signals.suppress_long if hasattr(signals, 'suppress_long') else None
        suppress_short_df = signals.suppress_short if hasattr(signals, 'suppress_short') else None

        # Optional debug save (parquet)
        if getattr(cfg, "SAVE_SIGNALS_PARQUET", False):
            signals_out = window_dir / "signals.parquet"
            packed = pd.concat({"z": z_df, "vol": vol_df, "beta": beta_df}, axis=1)
            packed.to_parquet(signals_out)
            logger.info("Saved signals parquet: %s", signals_out)

        # 7) Accountant masks
        logger.info("Applying accountant filter (entries/exits)...")
        # Use effective fee rate from fee model (4 legs: entry + exit for both coins)
        effective_fee = pnl_engine.compute_effective_fee_rate(
            fee_model=getattr(cfg, "FEE_MODEL", "taker_only"),
        )
        friction_cost = 4.0 * (effective_fee + cfg.SLIPPAGE_RATE)
        funding_cost_per_bar = None
        if funding_rates is not None:
            funding_cost_per_bar = _compute_funding_cost_per_bar(
                z_df=z_df,
                pairs=valid_pairs,
                funding_rates=funding_rates,
                freq=getattr(cfg, "SIGNAL_TIMEFRAME", None) or getattr(cfg, "BAR_FREQ", "1min"),
            )
        # Option 4: Use OU model for expected profit if enabled and half-lives available
        use_ou = bool(getattr(cfg, "USE_OU_MODEL", False)) and bool(pair_half_lives)
        if use_ou:
            logger.info("Using OU model for expected profit (USE_OU_MODEL=True, %d pairs with half-life)", len(pair_half_lives))
        regime_orig = getattr(cfg, "ENABLE_MEAN_REVERSION_REGIME", False)
        warmup_windows = int(getattr(cfg, "REGIME_WARMUP_WINDOWS", 0))
        if warmup_windows > 0 and w_idx < warmup_windows:
            setattr(cfg, "ENABLE_MEAN_REVERSION_REGIME", False)
        # Pass ML scorer if trained and enabled
        current_ml_scorer = None
        if use_ml_scoring and ml_scorer is not None and ml_scorer.is_trained:
            current_ml_scorer = ml_scorer
            logger.info("Using ML scorer for window %d (trained on %d samples)",
                       w_idx, ml_scorer.training_stats.get("n_samples", 0))

        try:
            trade_masks = accountant_filter.compute_masks(
                z_score=z_df,
                spread_volatility=vol_df,
                beta=beta_df,
                max_entry_z=getattr(cfg, "MAX_ENTRY_Z", None),
                freq=getattr(cfg, "SIGNAL_TIMEFRAME", None) or getattr(cfg, "BAR_FREQ", "1min"),
                use_ou_model=use_ou,
                half_life_bars=pair_half_lives if pair_half_lives else None,
                transaction_cost=friction_cost,
                funding_cost_per_bar=funding_cost_per_bar,
                ml_scorer=current_ml_scorer,
                kalman_gain=signals.kalman_gain if hasattr(signals, 'kalman_gain') else None,
            )
        finally:
            setattr(cfg, "ENABLE_MEAN_REVERSION_REGIME", regime_orig)

        logger.info("Scoring method used: %s", trade_masks.scoring_method)
        entries = trade_masks.entries
        exits = trade_masks.exits
        expected_profit = trade_masks.expected_profit
        signal_score = trade_masks.signal_score  # For conviction-weighted sizing

        # ===== GATING FUNNEL DIAGNOSTICS =====
        # Initialize funnel tracking for this window
        enable_funnel_logging = bool(getattr(cfg, "ENABLE_ENTRY_FUNNEL_LOGGING", True))
        entry_funnel = EntryFunnel(window_idx=w_idx)
        # Raw entries from accountant_filter (includes z-score + spread vol + expected profit filters)
        entry_funnel.raw_z_entries = int(entries.sum().sum())
        entry_funnel.after_spread_vol = entry_funnel.raw_z_entries  # Same at this point

        # Extract continuous sizing if enabled (ROI optimization)
        continuous_size = trade_masks.continuous_size  # May be None if disabled
        if continuous_size is not None:
            logger.info(
                "Continuous exposure enabled: bars_with_nonzero_size=%d (%.1f%%), avg_size=%.3f",
                int((continuous_size.abs() > 0.01).any(axis=1).sum()),
                100.0 * (continuous_size.abs() > 0.01).any(axis=1).mean(),
                continuous_size.abs().mean().mean(),
            )

        # ===== APPLY TREND OVERLAY =====
        # Suppress counter-trend entries and apply score penalties
        if enable_trend_overlay and suppress_long_df is not None and suppress_short_df is not None:
            from src.features.trend_overlay import apply_trend_overlay_to_entries, apply_trend_score_penalty

            # Determine which entries are long vs short based on z-score sign
            # Positive z-score = short entry (spread above mean, expect decline)
            # Negative z-score = long entry (spread below mean, expect rise)
            is_short_signal = z_df > 0
            is_long_signal = z_df < 0

            entries_before = entries.sum().sum()

            # Apply suppression if enabled
            if getattr(cfg, "TREND_SUPPRESS_COUNTER_TREND", True):
                # Suppress long entries when spread is trending up
                # Suppress short entries when spread is trending down
                for col in entries.columns:
                    if col not in suppress_long_df.columns or col not in suppress_short_df.columns:
                        continue

                    # Suppress long entries (z < 0) when suppress_long is True
                    long_mask = is_long_signal[col] & entries[col]
                    suppress_long_mask = suppress_long_df[col].fillna(False).astype(bool)
                    entries.loc[long_mask & suppress_long_mask, col] = False

                    # Suppress short entries (z > 0) when suppress_short is True
                    short_mask = is_short_signal[col] & entries[col]
                    suppress_short_mask = suppress_short_df[col].fillna(False).astype(bool)
                    entries.loc[short_mask & suppress_short_mask, col] = False

                entries_after = entries.sum().sum()
                if entries_before > entries_after:
                    logger.info(
                        "Trend overlay suppressed %d/%d entries (%.1f%%)",
                        int(entries_before - entries_after),
                        int(entries_before),
                        100 * (entries_before - entries_after) / max(entries_before, 1),
                    )

            # Apply trend score penalty to signal scores
            if signal_score is not None and trend_score_df is not None:
                penalty_weight = float(getattr(cfg, "TREND_SCORE_PENALTY", 0.3))
                signal_score = apply_trend_score_penalty(
                    signal_scores=signal_score.values,
                    trend_scores=trend_score_df.reindex_like(signal_score).fillna(1.0).values,
                    penalty_weight=penalty_weight,
                )
                signal_score = pd.DataFrame(
                    signal_score,
                    index=trade_masks.signal_score.index,
                    columns=trade_masks.signal_score.columns,
                )

        # Weekly sub-window health filtering
        # This allows faster detection of pairs losing cointegration
        if getattr(cfg, "ENABLE_WEEKLY_SUBWINDOWS", False):
            weekly_days = int(getattr(cfg, "WEEKLY_SUBWINDOW_DAYS", 7))
            min_crosses = int(getattr(cfg, "WEEKLY_HEALTH_MIN_ZSCORE_CROSSES", 1))
            max_beta_drift = float(getattr(cfg, "WEEKLY_HEALTH_MAX_BETA_DRIFT", 0.3))
            max_spread_trend = float(getattr(cfg, "WEEKLY_HEALTH_MAX_SPREAD_TREND", 1.5))

            weekly_windows = _build_weekly_subwindows(train_end, test_end, weekly_days)
            logger.info("Processing %d weekly sub-windows for health checks...", len(weekly_windows))

            # Get monthly beta reference (from end of training)
            monthly_beta = beta_df.iloc[:24] if len(beta_df) > 24 else beta_df  # First day of test as reference

            excluded_count = 0
            for week_start, week_end in weekly_windows:
                health_scores = _compute_weekly_health_scores(
                    z_df=z_df,
                    beta_df=beta_df,
                    monthly_beta=monthly_beta,
                    pairs=valid_pairs,
                    week_start=week_start,
                    week_end=week_end,
                )

                healthy_pairs, excluded_pairs = _filter_pairs_by_weekly_health(
                    health_scores,
                    min_crosses=min_crosses,
                    max_beta_drift=max_beta_drift,
                    max_spread_trend=max_spread_trend,
                )

                # Mask out entries for excluded pairs in this week
                if excluded_pairs:
                    week_mask = (entries.index >= week_start) & (entries.index < week_end)
                    for pair in excluded_pairs:
                        if pair in entries.columns:
                            entries.loc[week_mask, pair] = False
                            excluded_count += 1

            if excluded_count > 0:
                logger.info("Weekly health checks: excluded %d pair-weeks", excluded_count)

        # ===== ML FALLBACK SAFETY NET =====
        # If ML scoring produced 0 entries but there were potential entries (expected_profit > hurdle),
        # this indicates the ML model is not generalizing well. Fall back to rule-based entries.
        n_entries = int(entries.sum().sum())
        min_profit_hurdle = float(getattr(cfg, "MIN_PROFIT_HURDLE", 0.006))
        n_potential = int((expected_profit > min_profit_hurdle).sum().sum())

        ml_fallback_enabled = bool(getattr(cfg, "ML_MIN_ENTRY_FALLBACK", True))
        is_ml_scoring = trade_masks.scoring_method == "ml_based"

        if ml_fallback_enabled and is_ml_scoring and n_entries == 0 and n_potential > 0:
            logger.warning(
                "ML scoring produced 0 entries from %d potential signals. "
                "Falling back to rule-based entries.",
                n_potential
            )
            # Re-compute entries with rule-based scoring
            fallback_masks = accountant_filter.compute_masks(
                z_score=z_df,
                spread_volatility=vol_df,
                beta=beta_df,
                max_entry_z=getattr(cfg, "MAX_ENTRY_Z", None),
                freq=getattr(cfg, "SIGNAL_TIMEFRAME", None) or getattr(cfg, "BAR_FREQ", "1min"),
                use_ou_model=use_ou,
                half_life_bars=pair_half_lives if pair_half_lives else None,
                transaction_cost=friction_cost,
                funding_cost_per_bar=funding_cost_per_bar,
                ml_scorer=None,  # Force rule-based scoring
                kalman_gain=signals.kalman_gain if hasattr(signals, 'kalman_gain') else None,
            )
            entries = fallback_masks.entries
            signal_score = fallback_masks.signal_score
            logger.info("Fallback entries: %d (was 0 with ML)", int(entries.sum().sum()))

        # Track entries after all pre-regime filters (trend overlay, weekly health, ML fallback)
        entry_funnel.after_trend_overlay = int(entries.sum().sum())

        # ===== PHASE 6: Apply entry quality filters =====
        # Filter bad entry timing to reduce stop-loss hits
        enable_slope_filter = bool(getattr(cfg, "ENABLE_SLOPE_FILTER", False))
        enable_confirmation_filter = bool(getattr(cfg, "ENABLE_CONFIRMATION_FILTER", False))

        if enable_slope_filter or enable_confirmation_filter:
            entries_before_quality = int(entries.sum().sum())
            try:
                entries = accountant_filter.apply_entry_quality_filters(
                    z_scores=z_df,
                    entries=entries,
                    entry_z=float(getattr(cfg, "ENTRY_Z", 2.0)),
                    enable_slope=enable_slope_filter,
                    enable_confirmation=enable_confirmation_filter,
                    confirmation_bars=int(getattr(cfg, "CONFIRMATION_BARS", 2)),
                )
                entries_after_quality = int(entries.sum().sum())
                if entries_before_quality > entries_after_quality:
                    logger.info(
                        "Entry quality filters blocked %d/%d entries (%.1f%%)",
                        entries_before_quality - entries_after_quality,
                        entries_before_quality,
                        100.0 * (entries_before_quality - entries_after_quality) / max(entries_before_quality, 1),
                    )
            except Exception as e:
                logger.warning("Entry quality filters failed: %s. Proceeding without quality filters.", e)

        # ===== PHASE 2a: Live Regime Tracking (Dynamic Updates) =====
        # Updates regime state during test execution rather than freezing at window start
        enable_live_regime = bool(getattr(cfg, "ENABLE_LIVE_REGIME_UPDATES", False))
        live_regime_tracker = None
        live_regime_size_multiplier = None
        live_regime_entry_allowed = None

        if enable_live_regime and btc_symbol in train_df.columns and btc_symbol in test_df.columns:
            try:
                # Initialize tracker from training BTC prices
                btc_prices_train = train_df[btc_symbol].dropna()
                live_regime_tracker = create_live_regime_tracker(btc_prices_train)

                # Get test BTC prices for simulation
                btc_prices_test = test_df[btc_symbol]

                # Simulate the tracker for all test bars to get regime states
                n_test_bars = len(btc_prices_test)
                for bar_idx in range(n_test_bars):
                    btc_price = btc_prices_test.iloc[bar_idx]
                    timestamp = btc_prices_test.index[bar_idx] if hasattr(btc_prices_test.index, '__getitem__') else None
                    live_regime_tracker.update(bar_idx, btc_price, timestamp)

                # Generate state series from tracker
                state_series, size_mult_series, entry_allowed_series = \
                    live_regime_tracker.get_state_series(n_test_bars)

                # Convert to DataFrames aligned with entries
                live_regime_size_multiplier = pd.DataFrame(
                    {pair: size_mult_series.values for pair in entries.columns},
                    index=entries.index[:n_test_bars] if len(entries.index) >= n_test_bars else entries.index,
                )
                live_regime_entry_allowed = pd.DataFrame(
                    {pair: entry_allowed_series.values for pair in entries.columns},
                    index=entries.index[:n_test_bars] if len(entries.index) >= n_test_bars else entries.index,
                )

                # Align with entries index (handle any length mismatch)
                live_regime_size_multiplier = live_regime_size_multiplier.reindex(entries.index).ffill().fillna(1.0)
                live_regime_entry_allowed = live_regime_entry_allowed.reindex(entries.index).ffill().fillna(True)

                # Apply live regime entry mask
                entries_before_live_regime = int(entries.sum().sum())
                entries = entries & live_regime_entry_allowed

                # Get tracker summary for logging
                tracker_summary = live_regime_tracker.get_summary()
                logger.info(
                    "Live regime tracker: %d transitions, GREEN=%.1f%%, YELLOW=%.1f%%, RED=%.1f%% | blocked %d/%d entries",
                    tracker_summary["n_transitions"],
                    tracker_summary["pct_green"],
                    tracker_summary["pct_yellow"],
                    tracker_summary["pct_red"],
                    entries_before_live_regime - int(entries.sum().sum()),
                    entries_before_live_regime,
                )

            except Exception as e:
                logger.warning("Live regime tracker failed: %s. Proceeding without live regime updates.", e)
                live_regime_tracker = None
                live_regime_size_multiplier = None

        # ===== PHASE 2a.2: Apply regime-conditional parameters =====
        # Regime-conditional max_positions limits exposure during adverse regimes
        # Note: Entry_z filtering is NOT applied here since base ENTRY_Z (3.3) is already
        # stricter than regime thresholds. Entry blocking for RED is handled by live_regime_entry_allowed.
        enable_regime_params = bool(getattr(cfg, "ENABLE_REGIME_CONDITIONAL_PARAMS", False))
        regime_params = None

        if enable_regime_params and enable_live_regime and live_regime_tracker is not None:
            try:
                regime_params = create_regime_parameters()
                logger.info(
                    "Regime-conditional params enabled: GREEN(z=%.1f, pos=%d), YELLOW(z=%.1f, pos=%d), RED(z=%.1f, pos=%d)",
                    regime_params.green.entry_z, regime_params.green.max_positions,
                    regime_params.yellow.entry_z, regime_params.yellow.max_positions,
                    regime_params.red.entry_z, regime_params.red.max_positions,
                )
            except Exception as e:
                logger.warning("Regime-conditional parameters failed: %s. Using default parameters.", e)
                regime_params = None

        # ===== PHASE 2b: Apply static regime filter to entries =====
        # Block entries during adverse market regimes (high BTC vol, high dispersion)
        # Supports both hard block (legacy) and soft 3-state gating (Phase 6)
        regime_size_multiplier = None  # For soft regime: position size adjustment
        regime_entry_z_adjustment = None  # For soft regime: entry z-score adjustment
        enable_soft_regime = bool(getattr(cfg, "ENABLE_SOFT_REGIME", False))

        if enable_regime_filter and regime_filter is not None:
            entries_before_regime = int(entries.sum().sum())
            try:
                # Get BTC prices from test data
                btc_prices = test_df[btc_symbol] if btc_symbol in test_df.columns else None
                regime_coin_returns = test_df.pct_change() if not test_df.empty else None

                if btc_prices is not None and regime_coin_returns is not None:
                    if enable_soft_regime:
                        # ===== SOFT REGIME GATING (3-state) =====
                        # Returns entry_allowed mask + size_multiplier + entry_z_adjustment
                        regime_mask, regime_size_multiplier, regime_entry_z_adjustment = \
                            regime_filter.compute_soft_regime_mask(
                                btc_prices=btc_prices,
                                all_returns=regime_coin_returns,
                                spread_vol=vol_df,
                            )
                        # Apply regime mask to entries (GREEN/YELLOW allowed, RED blocked)
                        entries = entries & regime_mask
                        entries_after_regime = int(entries.sum().sum())

                        # Log soft regime stats
                        if hasattr(regime_filter, 'soft_regime_stats'):
                            stats = regime_filter.soft_regime_stats
                            total = sum(stats.values()) or 1
                            logger.info(
                                "Soft regime: GREEN=%.1f%%, YELLOW=%.1f%%, RED=%.1f%% | blocked %d/%d entries",
                                100.0 * stats.get("green", 0) / total,
                                100.0 * stats.get("yellow", 0) / total,
                                100.0 * stats.get("red", 0) / total,
                                entries_before_regime - entries_after_regime,
                                entries_before_regime,
                            )
                    else:
                        # ===== HARD REGIME BLOCK (legacy) =====
                        regime_mask = regime_filter.compute_regime_mask(
                            btc_prices=btc_prices,
                            all_returns=regime_coin_returns,
                            spread_vol=vol_df,
                        )
                        # Apply regime mask to entries (only keep entries where regime is OK)
                        entries = entries & regime_mask
                        entries_after_regime = int(entries.sum().sum())
                        if entries_before_regime > entries_after_regime:
                            logger.info(
                                "Regime filter blocked %d/%d entries (%.1f%%)",
                                entries_before_regime - entries_after_regime,
                                entries_before_regime,
                                100.0 * (entries_before_regime - entries_after_regime) / max(entries_before_regime, 1),
                            )

                    # Capture regime filter stats for funnel
                    if hasattr(regime_filter, 'block_stats'):
                        entry_funnel.btc_vol_blocked = regime_filter.block_stats.get("btc_vol", 0)
                        entry_funnel.dispersion_blocked = regime_filter.block_stats.get("dispersion", 0)
                        entry_funnel.spread_vol_low_blocked = regime_filter.block_stats.get("spread_vol_low", 0)
                        entry_funnel.spread_vol_high_blocked = regime_filter.block_stats.get("spread_vol_high", 0)
                        total_bars = regime_filter.block_stats.get("total_bars", 1)
                        entry_funnel.btc_vol_ok_pct = 100 * (1 - entry_funnel.btc_vol_blocked / max(total_bars, 1))
                        entry_funnel.dispersion_ok_pct = 100 * (1 - entry_funnel.dispersion_blocked / max(total_bars, 1))
                        total_cells = regime_filter.block_stats.get("total_cells", 1)
                        spread_blocked = entry_funnel.spread_vol_low_blocked + entry_funnel.spread_vol_high_blocked
                        entry_funnel.spread_vol_ok_pct = 100 * (1 - spread_blocked / max(total_cells, 1))
            except Exception as e:
                logger.warning("Regime filter failed: %s. Proceeding without regime filtering.", e)

        # Track entries after regime filter
        entry_funnel.after_regime = int(entries.sum().sum())

        # ===== PHASE 3: Apply entry cooldown =====
        # Prevent re-entry within cooldown period after exit (reduce churn)
        # Supports smart cooldown (Phase 6) with different cooldowns by exit type
        enable_entry_cooldown = bool(getattr(cfg, "ENABLE_ENTRY_COOLDOWN", False))
        enable_smart_cooldown = bool(getattr(cfg, "ENABLE_SMART_COOLDOWN", False))

        if enable_entry_cooldown:
            entries_before_cooldown = int(entries.sum().sum())
            try:
                if enable_smart_cooldown and previous_window_exit_reasons is not None:
                    # ===== SMART COOLDOWN (Phase 6) =====
                    # Use exit reasons from previous window for different cooldowns
                    cooldown_signal_bars = int(getattr(cfg, "COOLDOWN_AFTER_SIGNAL_BARS", 12))
                    cooldown_stop_loss_bars = int(getattr(cfg, "COOLDOWN_AFTER_STOP_LOSS_BARS", 48))

                    # Align previous exit reasons with current window's exits
                    # (exits from accountant_filter are signal-based exits for current window)
                    entries = accountant_filter.apply_smart_cooldown(
                        entries=entries,
                        exits=exits,
                        exit_reasons=previous_window_exit_reasons,
                        cooldown_signal_bars=cooldown_signal_bars,
                        cooldown_stop_loss_bars=cooldown_stop_loss_bars,
                        freq=getattr(cfg, "SIGNAL_TIMEFRAME", "15min"),
                    )
                    logger.info("Using smart cooldown with exit-type awareness")
                else:
                    # Standard cooldown (same for all exit types)
                    cooldown_bars = int(getattr(cfg, "ENTRY_COOLDOWN_BARS", 24))
                    entries = accountant_filter.apply_entry_cooldown(
                        entries=entries,
                        exits=exits,
                        cooldown_bars=cooldown_bars,
                    )

                entries_after_cooldown = int(entries.sum().sum())
                if entries_before_cooldown > entries_after_cooldown:
                    logger.info(
                        "Entry cooldown blocked %d/%d entries",
                        entries_before_cooldown - entries_after_cooldown,
                        entries_before_cooldown,
                    )
            except Exception as e:
                logger.warning("Entry cooldown failed: %s. Proceeding without cooldown.", e)

        # Track entries after cooldown
        entry_funnel.after_cooldown = int(entries.sum().sum())

        # Persist masks per window
        entries.to_parquet(window_dir / "entries.parquet")
        exits.to_parquet(window_dir / "exits.parquet")
        expected_profit.to_parquet(window_dir / "expected_profit.parquet")
        logger.info("✅ Saved entry/exit masks and expected profit.")

        # 8) Advanced Position Sizing (compute before PnL engine)
        enable_advanced_sizing = bool(getattr(cfg, "ENABLE_ADVANCED_SIZING", False))
        position_size_multiplier = None

        if enable_advanced_sizing and signal_score is not None:
            logger.info("Computing advanced position sizes...")

            # Build config from backtest config
            sizing_config = PositionSizingConfig(
                base_capital_per_pair=1.0,  # We scale separately in pnl_engine
                conviction_method=str(getattr(cfg, "CONVICTION_METHOD", "power")),
                conviction_min=float(getattr(cfg, "CONVICTION_MIN", 0.3)),
                conviction_max=float(getattr(cfg, "CONVICTION_MAX", 1.2)),
                conviction_power=float(getattr(cfg, "CONVICTION_POWER", 1.5)),
                sigmoid_k=float(getattr(cfg, "SIGMOID_K", 10.0)),
                sigmoid_midpoint=float(getattr(cfg, "SIGMOID_MIDPOINT", 0.65)),
                kelly_fraction=float(getattr(cfg, "KELLY_FRACTION", 0.25)),
                kelly_win_rate_floor=float(getattr(cfg, "KELLY_WIN_RATE_FLOOR", 0.45)),
                kelly_payoff_ratio=float(getattr(cfg, "KELLY_PAYOFF_RATIO", 1.5)),
                enable_correlation_adjustment=bool(getattr(cfg, "ENABLE_CORRELATION_ADJUSTMENT", True)),
                correlation_lookback_bars=int(getattr(cfg, "CORRELATION_LOOKBACK_BARS", 96)),
                correlation_penalty_threshold=float(getattr(cfg, "CORRELATION_PENALTY_THRESHOLD", 0.5)),
                correlation_max_penalty=float(getattr(cfg, "CORRELATION_MAX_PENALTY", 0.5)),
                enable_volatility_targeting=bool(getattr(cfg, "ENABLE_VOLATILITY_TARGETING", True)),
                target_annual_volatility=float(getattr(cfg, "TARGET_ANNUAL_VOLATILITY", 0.15)),
                volatility_lookback_bars=int(getattr(cfg, "VOLATILITY_LOOKBACK_BARS", 48)),
                vol_adjustment_min=float(getattr(cfg, "VOL_ADJUSTMENT_MIN", 0.5)),
                vol_adjustment_max=float(getattr(cfg, "VOL_ADJUSTMENT_MAX", 2.0)),
                vol_floor=float(getattr(cfg, "VOL_FLOOR", 0.001)),
                max_total_exposure=float(getattr(cfg, "MAX_TOTAL_EXPOSURE", 1.0)),
                max_single_position=float(getattr(cfg, "MAX_SINGLE_POSITION_PCT", 0.10)),
            )

            spread_returns = None
            if sizing_config.enable_volatility_targeting or sizing_config.enable_correlation_adjustment:
                spread_returns = _compute_spread_returns(
                    test_df=test_df,
                    beta_df=beta_df,
                    pairs=valid_pairs,
                    pnl_mode="log",
                )

            # Compute correlation matrix from spread returns
            correlation_matrix = None
            if sizing_config.enable_correlation_adjustment and spread_returns is not None:
                recent = spread_returns.dropna(how="all")
                if len(recent) >= sizing_config.correlation_lookback_bars:
                    correlation_matrix = recent.tail(sizing_config.correlation_lookback_bars).corr()

            # Determine bars_per_year based on frequency
            freq_str = str(getattr(cfg, "SIGNAL_TIMEFRAME", "15min")).lower()
            if freq_str.endswith("min"):
                mins = float(freq_str.replace("min", ""))
                bars_per_year = (365.25 * 24 * 60) / mins
            elif freq_str.endswith("h"):
                hours = float(freq_str.replace("h", ""))
                bars_per_year = (365.25 * 24) / hours
            else:
                bars_per_year = 8760.0  # Default to hourly

            # Compute position sizes
            position_size_multiplier = compute_position_sizes_vectorized(
                signal_scores=signal_score,
                spread_returns=spread_returns,
                config=sizing_config,
                correlation_matrix=correlation_matrix,
                bars_per_year=bars_per_year,
            )

            logger.info(
                "Position sizes computed: min=%.3f, max=%.3f, mean=%.3f",
                position_size_multiplier.min().min(),
                position_size_multiplier.max().max(),
                position_size_multiplier.mean().mean(),
            )

        # 8.5) ML-Based Risk Prediction Position Adjustment
        # Apply risk-based sizing if enabled and risk predictor is trained
        risk_predictor: Optional[RiskPredictor] = None

        # Create default position size multiplier if advanced sizing is disabled
        if position_size_multiplier is None and use_risk_prediction:
            position_size_multiplier = pd.DataFrame(
                1.0, index=z_df.index, columns=valid_pairs
            )
            logger.info("Created default position size multiplier for risk prediction")

        if use_risk_prediction and position_size_multiplier is not None:
            # Use pretrained model if available (applies from window 0)
            # Otherwise, use model trained during this run (after 2 windows)
            if pretrained_risk_predictor is not None:
                risk_predictor = pretrained_risk_predictor
                logger.info("Using pretrained risk predictor for window %d", w_idx)
            elif w_idx >= risk_min_train_windows:
                # Try to load or use existing risk predictor trained during this run
                risk_model_path = run_dir / "risk_predictor.pkl"
                if risk_model_path.exists():
                    try:
                        import pickle
                        with open(risk_model_path, 'rb') as f:
                            risk_predictor = pickle.load(f)
                        logger.info("Loaded trained risk predictor from %s", risk_model_path)
                    except Exception as e:
                        logger.warning("Failed to load risk predictor: %s", e)

            # If we have a trained predictor, apply risk adjustments
            if risk_predictor is not None and risk_predictor.is_trained:
                    logger.info("Applying ML-based risk prediction adjustments...")

                    # Extract features for risk prediction
                    risk_feature_extractor = RiskFeatureExtractor(RiskPredictorConfig())
                    risk_features = risk_feature_extractor.extract_features(
                        z_score=z_df,
                        spread_volatility=vol_df,
                        kalman_gain=signals.kalman_gain if hasattr(signals, 'kalman_gain') else pd.DataFrame(0.1, index=z_df.index, columns=z_df.columns),
                        beta=beta_df,
                        half_life=pd.DataFrame({p: pair_half_lives.get(p, 500) for p in valid_pairs}, index=z_df.index),
                        price_matrix=test_df,
                        entry_z=float(getattr(cfg, "ENTRY_Z", 2.0)),
                        max_entry_z=float(getattr(cfg, "MAX_ENTRY_Z", 4.0)),
                        btc_column=btc_symbol,
                    )

                    # Flatten features for prediction at entry points only
                    entry_mask = entries.astype(bool)
                    predictions_dict = {}

                    for pair in valid_pairs:
                        pair_entries = entry_mask[pair]
                        if pair_entries.sum() == 0:
                            continue

                        # Build feature matrix for this pair's entries
                        entry_indices = pair_entries[pair_entries].index
                        X_rows = []
                        for fn in RiskFeatureExtractor.FEATURE_NAMES:
                            if fn in risk_features and pair in risk_features[fn].columns:
                                X_rows.append(risk_features[fn].loc[entry_indices, pair].values)
                            else:
                                X_rows.append(np.zeros(len(entry_indices)))

                        if X_rows:
                            X = np.array(X_rows).T
                            preds = risk_predictor.predict(X)

                            # Store predictions indexed by entry time
                            for i, idx in enumerate(entry_indices):
                                if pair not in predictions_dict:
                                    predictions_dict[pair] = {}
                                predictions_dict[pair][idx] = {
                                    'predicted_mae': preds['predicted_mae'][i],
                                    'predicted_vol': preds['predicted_vol'][i],
                                    'stopout_prob': preds['stopout_prob'][i],
                                    'confidence': preds['confidence'][i],
                                }

                    # Build risk predictions DataFrames
                    if predictions_dict:
                        risk_preds = {
                            'predicted_mae': pd.DataFrame(np.nan, index=z_df.index, columns=valid_pairs),
                            'predicted_vol': pd.DataFrame(np.nan, index=z_df.index, columns=valid_pairs),
                            'stopout_prob': pd.DataFrame(np.nan, index=z_df.index, columns=valid_pairs),
                            'confidence': pd.DataFrame(np.nan, index=z_df.index, columns=valid_pairs),
                        }

                        for pair, pair_preds in predictions_dict.items():
                            for idx, preds in pair_preds.items():
                                for key in risk_preds:
                                    risk_preds[key].loc[idx, pair] = preds[key]

                        # Apply asymmetric risk adjustment (reward low-risk, punish high-risk)
                        position_size_multiplier = apply_risk_prediction_adjustment(
                            position_sizes=position_size_multiplier,
                            risk_predictions=risk_preds,
                            historical_mae_median=float(getattr(cfg, "RISK_HISTORICAL_MAE_MEDIAN", 0.01)),
                            historical_vol_median=float(getattr(cfg, "RISK_HISTORICAL_VOL_MEDIAN", 0.005)),
                            size_reduction_max=float(getattr(cfg, "RISK_SIZE_REDUCTION_MAX", 0.5)),
                            size_increase_max=float(getattr(cfg, "RISK_SIZE_INCREASE_MAX", 0.5)),
                            low_risk_threshold=float(getattr(cfg, "RISK_LOW_THRESHOLD", 0.3)),
                            high_risk_threshold=float(getattr(cfg, "RISK_HIGH_THRESHOLD", 0.5)),
                            confidence_threshold=float(getattr(cfg, "RISK_CONFIDENCE_THRESHOLD", 0.3)),
                        )

                        logger.info(
                            "Risk-adjusted position sizes: min=%.3f, max=%.3f, mean=%.3f",
                            position_size_multiplier.min().min(),
                            position_size_multiplier.max().max(),
                            position_size_multiplier.mean().mean(),
                        )

        # 8.7) Integrate continuous exposure sizing (ROI optimization)
        # If continuous sizing is enabled, use it as a position multiplier
        enable_continuous = bool(getattr(cfg, "ENABLE_CONTINUOUS_EXPOSURE", False))
        if enable_continuous and continuous_size is not None:
            # Convert continuous_size to absolute values (direction is handled by z-score in PnL engine)
            continuous_mult = continuous_size.abs()

            if position_size_multiplier is not None:
                # Multiply existing size multiplier by continuous factor
                position_size_multiplier = position_size_multiplier * continuous_mult
                logger.info(
                    "Combined continuous sizing with advanced sizing: "
                    "min=%.3f, max=%.3f, mean=%.3f",
                    position_size_multiplier.min().min(),
                    position_size_multiplier.max().max(),
                    position_size_multiplier.mean().mean(),
                )
            else:
                # Use continuous size directly as position multiplier
                position_size_multiplier = continuous_mult
                enable_advanced_sizing = True  # Enable advanced sizing to use the multiplier
                logger.info(
                    "Using continuous sizing as position multiplier: "
                    "min=%.3f, max=%.3f, mean=%.3f",
                    position_size_multiplier.min().min(),
                    position_size_multiplier.max().max(),
                    position_size_multiplier.mean().mean(),
                )

        # 8.8) Integrate soft regime size multiplier (Phase 6)
        # Apply regime-based position size adjustment (YELLOW state = 0.5x)
        if enable_soft_regime and regime_size_multiplier is not None:
            if position_size_multiplier is not None:
                # Multiply existing size multiplier by regime factor
                position_size_multiplier = position_size_multiplier * regime_size_multiplier
                logger.info(
                    "Applied soft regime size adjustment: min=%.3f, max=%.3f, mean=%.3f",
                    position_size_multiplier.min().min(),
                    position_size_multiplier.max().max(),
                    position_size_multiplier.mean().mean(),
                )
            else:
                # Use regime size directly as position multiplier
                position_size_multiplier = regime_size_multiplier
                enable_advanced_sizing = True  # Enable to use the multiplier
                logger.info(
                    "Using soft regime sizing as position multiplier: min=%.3f, max=%.3f, mean=%.3f",
                    position_size_multiplier.min().min(),
                    position_size_multiplier.max().max(),
                    position_size_multiplier.mean().mean(),
                )

        # 8.9) Integrate live regime size multiplier (dynamic regime updates)
        # Apply live regime-based position size adjustment (YELLOW state = 0.5x, RED state = 0.0x)
        if enable_live_regime and live_regime_size_multiplier is not None:
            if position_size_multiplier is not None:
                # Multiply existing size multiplier by live regime factor
                position_size_multiplier = position_size_multiplier * live_regime_size_multiplier
                logger.info(
                    "Applied live regime size adjustment: min=%.3f, max=%.3f, mean=%.3f",
                    position_size_multiplier.min().min(),
                    position_size_multiplier.max().max(),
                    position_size_multiplier.mean().mean(),
                )
            else:
                # Use live regime size directly as position multiplier
                position_size_multiplier = live_regime_size_multiplier
                enable_advanced_sizing = True  # Enable to use the multiplier
                logger.info(
                    "Using live regime sizing as position multiplier: min=%.3f, max=%.3f, mean=%.3f",
                    position_size_multiplier.min().min(),
                    position_size_multiplier.max().max(),
                    position_size_multiplier.mean().mean(),
                )

        # 9) PnL engine
        logger.info("Running PnL engine (Numba state machine)...")
        max_hold_bars = pnl_engine.compute_time_stops_from_half_life(
            valid_pairs,
            pair_half_lives,
        )

        # Determine position limits - use higher concurrency if enabled (ROI optimization)
        use_higher_concurrency = bool(getattr(cfg, "ENABLE_HIGHER_CONCURRENCY", False))
        if use_higher_concurrency:
            max_positions_total = int(getattr(cfg, "MAX_PORTFOLIO_POSITIONS_HIGH", 15))
            max_positions_per_coin = int(getattr(cfg, "MAX_POSITIONS_PER_COIN_HIGH", 3))
            logger.info(
                "Higher concurrency mode: max_positions=%d, max_per_coin=%d",
                max_positions_total, max_positions_per_coin
            )
        else:
            max_positions_total = int(getattr(cfg, "MAX_PORTFOLIO_POSITIONS", 8))
            max_positions_per_coin = int(getattr(cfg, "MAX_POSITIONS_PER_COIN", 2))

        # Apply regime-conditional max_positions if enabled
        # Uses the minimum max_positions from regime distribution to be conservative
        if enable_regime_params and regime_params is not None and live_regime_tracker is not None:
            try:
                tracker_summary = live_regime_tracker.get_summary()
                pct_green = tracker_summary["pct_green"] / 100.0
                pct_yellow = tracker_summary["pct_yellow"] / 100.0
                pct_red = tracker_summary["pct_red"] / 100.0

                # Weighted average max_positions based on regime distribution
                regime_max_pos = (
                    pct_green * regime_params.green.max_positions +
                    pct_yellow * regime_params.yellow.max_positions +
                    pct_red * regime_params.red.max_positions
                )

                # Use floor to be conservative (round down)
                regime_adjusted_max_positions = max(1, int(regime_max_pos))

                # Only reduce, never increase beyond config
                if regime_adjusted_max_positions < max_positions_total:
                    logger.info(
                        "Regime-conditional max_positions: %d -> %d (GREEN=%.0f%%, YELLOW=%.0f%%, RED=%.0f%%)",
                        max_positions_total,
                        regime_adjusted_max_positions,
                        pct_green * 100,
                        pct_yellow * 100,
                        pct_red * 100,
                    )
                    max_positions_total = regime_adjusted_max_positions

            except Exception as e:
                logger.warning("Failed to apply regime-conditional max_positions: %s", e)

        capital_per_pair = cfg.CAPITAL_PER_PAIR
        # CRITICAL FIX: Always scale capital_per_pair by max_positions to prevent
        # single positions from dominating the portfolio.
        # Advanced sizing multipliers scale FROM this base (0.3x to 1.2x).
        # Without this, enable_advanced_sizing + capital_per_pair=1.0 allows
        # single trades to lose ~40% of portfolio (catastrophic).
        if getattr(cfg, "SCALE_CAPITAL_BY_MAX_POSITIONS", False):
            if max_positions_total > 0:
                capital_per_pair = 1.0 / max_positions_total
                logger.info(
                    "Capital per pair scaled: %.4f (1/%d positions)",
                    capital_per_pair, max_positions_total
                )
        # Compute effective fee rate based on fee model
        fee_model = getattr(cfg, "FEE_MODEL", "taker_only")
        effective_fee_rate = pnl_engine.compute_effective_fee_rate(
            fee_model=fee_model,
            taker_rate=getattr(cfg, "TAKER_FEE_RATE", cfg.FEE_RATE),
            maker_rate=getattr(cfg, "MAKER_FEE_RATE", 0.0002),
            maker_fill_prob=getattr(cfg, "MAKER_FILL_PROBABILITY", 0.70),
        )
        if fee_model != "taker_only":
            logger.info("Using fee model '%s': effective rate = %.4f bps", fee_model, effective_fee_rate * 10000)

        # Prepare funding rates for PnL engine (align to test period)
        test_funding_rates = None
        if funding_rates is not None and getattr(cfg, "USE_REAL_FUNDING", False):
            try:
                # Extract funding rates for the test period and relevant coins
                test_funding_rates = funding_rates.reindex(index=test_df.index).ffill()
            except Exception as e:
                logger.warning("Failed to align funding rates: %s", e)
                test_funding_rates = None

        pnl_result = pnl_engine.run_pnl_engine(
            test_df=test_df,
            pairs=valid_pairs,
            entries=entries,
            exits=exits,
            beta=beta_df,
            z_score=z_df,
            spread_volatility=vol_df,
            signal_score=signal_score,  # For conviction-weighted sizing
            position_size_multiplier=position_size_multiplier,  # Advanced sizing
            funding_rates=test_funding_rates,  # NEW: pass funding rates for cost tracking
            fee_rate=effective_fee_rate,  # Use computed effective fee rate
            slippage_model=cfg.SLIPPAGE_MODEL,
            slippage_rate=cfg.SLIPPAGE_RATE,
            slippage_vol_mult=cfg.SLIPPAGE_VOL_MULT,
            capital_per_pair=capital_per_pair,
            pnl_mode="log",
            max_positions_total=max_positions_total,
            max_positions_per_coin=max_positions_per_coin,
            stop_loss_pct=getattr(cfg, "STOP_LOSS_PCT", None),
            max_hold_bars=max_hold_bars,
            enable_advanced_sizing=enable_advanced_sizing,
        )

        # Normalize output to a returns_matrix DataFrame
        if isinstance(pnl_result, pd.DataFrame):
            returns_matrix = pnl_result
        elif hasattr(pnl_result, "returns_matrix"):
            returns_matrix = pnl_result.returns_matrix
        else:
            raise TypeError("pnl_engine.run must return a DataFrame or an object with .returns_matrix")

        returns_out = window_dir / "returns_matrix.parquet"
        returns_matrix.to_parquet(returns_out)
        logger.info("✅ Saved returns_matrix: %s", returns_out)

        # Save P&L attribution components (Phase 0)
        if hasattr(pnl_result, "gross_pnl_matrix"):
            pnl_result.gross_pnl_matrix.to_parquet(window_dir / "gross_pnl.parquet")
            pnl_result.fees_matrix.to_parquet(window_dir / "fees.parquet")
            pnl_result.slippage_matrix.to_parquet(window_dir / "slippage.parquet")
            pnl_result.hold_bars_matrix.to_parquet(window_dir / "hold_bars.parquet")
            pnl_result.exit_reason_matrix.to_parquet(window_dir / "exit_reasons.parquet")

            # Save funding costs matrix if available
            if hasattr(pnl_result, "funding_costs_matrix") and pnl_result.funding_costs_matrix is not None:
                pnl_result.funding_costs_matrix.to_parquet(window_dir / "funding_costs.parquet")
                logger.info(
                    "Funding costs saved: total=%.6f",
                    pnl_result.total_funding if hasattr(pnl_result, "total_funding") else 0.0
                )

            # ===== PHASE 5D: Hold time calibration logging =====
            # Compare expected hold time (from OU model) vs actual hold time for calibration
            enable_hold_time_logging = bool(getattr(cfg, "ENABLE_HOLD_TIME_LOGGING", True))
            expected_hold_bars = trade_masks.expected_hold_bars if hasattr(trade_masks, "expected_hold_bars") else None

            if enable_hold_time_logging and expected_hold_bars is not None and hasattr(pnl_result, "hold_bars_matrix"):
                actual_hold = pnl_result.hold_bars_matrix
                hold_comparison = []

                for pair in valid_pairs:
                    if pair not in actual_hold.columns or pair not in expected_hold_bars.columns:
                        continue
                    # Only look at bars where we actually had trades (hold_bars > 0)
                    actual_vals = actual_hold[pair][actual_hold[pair] > 0]
                    if len(actual_vals) > 0:
                        # Get expected hold time at entry points (where entries == True)
                        entry_times = entries[pair][entries[pair]].index
                        expected_at_entry = expected_hold_bars.loc[entry_times, pair].dropna()
                        expected_mean = expected_at_entry.mean() if len(expected_at_entry) > 0 else np.nan
                        actual_mean = actual_vals.mean()

                        hold_comparison.append({
                            "pair": pair,
                            "expected_hold_bars": expected_mean,
                            "actual_hold_bars": actual_mean,
                            "ratio": actual_mean / expected_mean if expected_mean > 0 and not np.isnan(expected_mean) else np.nan,
                            "n_trades": len(actual_vals),
                        })

                if hold_comparison:
                    hold_df = pd.DataFrame(hold_comparison)
                    valid_ratios = hold_df["ratio"].dropna()
                    avg_ratio = valid_ratios.mean() if len(valid_ratios) > 0 else np.nan

                    logger.info(
                        "Hold time calibration: expected=%.1f bars, actual=%.1f bars, ratio=%.2f (n=%d pairs)",
                        hold_df["expected_hold_bars"].mean(),
                        hold_df["actual_hold_bars"].mean(),
                        avg_ratio if not np.isnan(avg_ratio) else 0.0,
                        len(hold_comparison),
                    )

                    # Save for analysis
                    hold_df.to_csv(window_dir / "hold_time_calibration.csv", index=False)

                    # Warn if OU model is severely overestimating hold time
                    hold_time_warning_ratio = float(getattr(cfg, "HOLD_TIME_WARNING_RATIO", 0.3))
                    if not np.isnan(avg_ratio) and avg_ratio < hold_time_warning_ratio:
                        logger.warning(
                            "OU model severely over-estimates hold time (ratio=%.2f < %.2f). "
                            "Consider increasing calibration_discount or reducing max_expected_hold_bars.",
                            avg_ratio, hold_time_warning_ratio,
                        )

        # ===== GATING FUNNEL: Capture final executed trades and save =====
        # Count total executed trades from pnl_result
        if hasattr(pnl_result, "trades_count"):
            entry_funnel.final_executed = int(pnl_result.trades_count.sum())
        elif hasattr(pnl_result, "hold_bars_matrix"):
            # Fallback: count non-zero hold bars as trades
            entry_funnel.final_executed = int((pnl_result.hold_bars_matrix > 0).any(axis=0).sum())
        else:
            entry_funnel.final_executed = 0

        # Save entry funnel diagnostics
        if enable_funnel_logging:
            save_entry_funnel(entry_funnel, window_dir)

            # Accumulate historical P&L data for pair quality filtering (ROI optimization)
            for pair in valid_pairs:
                if pair not in pnl_result.gross_pnl_matrix.columns:
                    continue
                gross_pnl = pnl_result.gross_pnl_matrix[pair].sum()
                fees = pnl_result.fees_matrix[pair].sum()
                slippage = pnl_result.slippage_matrix[pair].sum()
                trade_count = int(pnl_result.trades_count.get(pair, 0))
                stop_loss_count = int(pnl_result.stop_loss_count.get(pair, 0)) if hasattr(pnl_result, "stop_loss_count") else 0

                if pair not in historical_pnl_data:
                    historical_pnl_data[pair] = {
                        "gross_pnl": 0.0,
                        "fees": 0.0,
                        "slippage": 0.0,
                        "trade_count": 0,
                        "stop_loss_count": 0,
                    }
                # Accumulate
                historical_pnl_data[pair]["gross_pnl"] += gross_pnl
                historical_pnl_data[pair]["fees"] += fees
                historical_pnl_data[pair]["slippage"] += slippage
                historical_pnl_data[pair]["trade_count"] += trade_count
                historical_pnl_data[pair]["stop_loss_count"] += stop_loss_count

            # ===== PHASE 1: Update symbol blacklist with window results =====
            # Record per-pair performance for symbol tracking across windows
            if enable_symbol_blacklist and symbol_blacklist is not None:
                pair_results_for_blacklist = {}
                for pair in valid_pairs:
                    if pair not in pnl_result.gross_pnl_matrix.columns:
                        continue
                    net_pnl = pnl_result.gross_pnl_matrix[pair].sum() - pnl_result.fees_matrix[pair].sum() - pnl_result.slippage_matrix[pair].sum()
                    trade_count = int(pnl_result.trades_count.get(pair, 0))
                    stop_count = int(pnl_result.stop_loss_count.get(pair, 0)) if hasattr(pnl_result, "stop_loss_count") else 0
                    pair_results_for_blacklist[pair] = {
                        "net_pnl": net_pnl,
                        "trade_count": trade_count,
                        "stop_count": stop_count,
                    }
                # Record window results and check for new blacklistings
                blacklist_before = len(symbol_blacklist.blacklist)
                symbol_blacklist.record_window(w_idx, pair_results_for_blacklist, cfg.PAIR_ID_SEPARATOR)
                blacklist_after = len(symbol_blacklist.blacklist)
                if blacklist_after > blacklist_before:
                    new_blacklisted = blacklist_after - blacklist_before
                    logger.info(
                        "Symbol blacklist updated: %d new symbols blacklisted (total: %d)",
                        new_blacklisted,
                        blacklist_after,
                    )

        # ===== Store exit reasons for smart cooldown in next window (Phase 6) =====
        if hasattr(pnl_result, "exit_reason_matrix"):
            previous_window_exit_reasons = pnl_result.exit_reason_matrix.copy()
            logger.debug("Stored exit reasons for smart cooldown: %d exit events",
                        int((previous_window_exit_reasons > 0).sum().sum()))

        all_returns.append(returns_matrix)
        all_entries.append(entries)
        all_exits.append(exits)
        all_pairs.extend(valid_pairs)
        # Append position sizes for realistic simulation
        if position_size_multiplier is not None:
            all_position_sizes.append(position_size_multiplier)
        last_window_data = {
            "test_df": test_df,
            "z_df": z_df,
            "beta_df": beta_df,
            "entries": entries,
            "exits": exits,
            "expected_profit": expected_profit,
            "vol_df": vol_df,
        }

        # Track per-window metrics for walk-forward analysis
        window_net_pnl = float(returns_matrix.values.sum())
        window_trade_count = int((returns_matrix != 0).values.sum())
        window_winners = int((returns_matrix > 0).values.sum())
        window_win_rate = window_winners / window_trade_count if window_trade_count > 0 else 0.0
        window_gross_pnl = 0.0
        if hasattr(pnl_result, "gross_pnl_matrix"):
            window_gross_pnl = float(pnl_result.gross_pnl_matrix.values.sum())

        window_metrics_list.append({
            "window_idx": w_idx,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_end": str(test_end),
            "n_pairs": len(valid_pairs),
            "trade_count": window_trade_count,
            "win_rate": window_win_rate,
            "gross_pnl": window_gross_pnl,
            "net_pnl": window_net_pnl,
            "total_return": window_net_pnl,  # Normalized returns
        })

        # ===== PHASE 5B: Record window metrics for analysis =====
        if enable_window_analysis and window_analysis is not None:
            # Get stop-loss count from pnl_result
            window_stop_loss_count = 0
            if hasattr(pnl_result, "stop_loss_count") and pnl_result.stop_loss_count is not None:
                try:
                    if isinstance(pnl_result.stop_loss_count, dict):
                        window_stop_loss_count = sum(pnl_result.stop_loss_count.values())
                    else:
                        window_stop_loss_count = int(pnl_result.stop_loss_count.sum())
                except Exception:
                    window_stop_loss_count = 0

            # Get BTC prices for regime analysis
            btc_symbol = getattr(cfg, "BTC_SYMBOL", "BTC")
            btc_prices = test_df[btc_symbol] if btc_symbol in test_df.columns else test_df.iloc[:, 0]

            # Get returns for all coins (use different name to avoid shadowing the list)
            window_coin_returns = test_df.pct_change().dropna()

            window_analysis.record_window(
                window_idx=w_idx,
                pnl=window_net_pnl,
                trade_count=window_trade_count,
                stop_loss_count=window_stop_loss_count,
                win_rate=window_win_rate,
                btc_prices=btc_prices,
                all_returns=window_coin_returns,
            )

        # === ML TRAINING: Collect data and train when ready ===
        if use_ml_scoring and ml_scorer is not None:
            # Store window data for ML training
            ml_training_data.append({
                "z_score": z_df.copy(),
                "spread_volatility": vol_df.copy(),
                "expected_profit": expected_profit.copy(),
                "expected_hold_bars": trade_masks.expected_hold_bars.copy() if trade_masks.expected_hold_bars is not None else None,
                "half_life_bars": pair_half_lives.copy() if pair_half_lives else {},
                "beta": beta_df.copy(),
                "entries": entries.copy(),
                "returns_matrix": returns_matrix.copy(),
                "kalman_gain": signals.kalman_gain.copy() if hasattr(signals, 'kalman_gain') and signals.kalman_gain is not None else None,
            })

            # Train ML scorer after accumulating enough windows
            if len(ml_training_data) >= ml_min_train_windows and not ml_scorer.is_trained:
                logger.info("Training ML scorer on %d windows of data...", len(ml_training_data))

                # Combine all window data for training
                all_X = []
                all_y = []

                for window_data in ml_training_data:
                    # Create training labels from trade returns
                    trade_labels = create_training_labels_from_returns(
                        returns_matrix=window_data["returns_matrix"],
                        entry_mask=window_data["entries"],
                    )

                    # Extract features
                    feature_set = ml_scorer.extract_features(
                        z_score=window_data["z_score"],
                        spread_volatility=window_data["spread_volatility"],
                        expected_profit=window_data["expected_profit"],
                        expected_hold_bars=window_data["expected_hold_bars"] if window_data["expected_hold_bars"] is not None else pd.DataFrame(500, index=window_data["z_score"].index, columns=window_data["z_score"].columns),
                        half_life_bars=window_data["half_life_bars"] if window_data["half_life_bars"] else 500,
                        kalman_gain=window_data["kalman_gain"],
                        beta=window_data["beta"],
                        entry_z=float(getattr(cfg, "ENTRY_Z", 2.0)),
                        max_entry_z=float(getattr(cfg, "MAX_ENTRY_Z", 4.0)),
                    )

                    # Prepare training data
                    X, y, _ = ml_scorer.prepare_training_data(
                        feature_set=feature_set,
                        trade_returns=trade_labels,
                        entry_mask=window_data["entries"],
                    )

                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)

                if all_X:
                    X_combined = np.vstack(all_X)
                    y_combined = np.concatenate(all_y)

                    logger.info("Combined training data: %d samples", len(X_combined))

                    # Train the model
                    training_stats = ml_scorer.train(X_combined, y_combined)

                    if ml_scorer.is_trained:
                        logger.info("ML Scorer trained successfully!")
                        logger.info("Training stats: %s", {
                            k: v for k, v in training_stats.items()
                            if k not in ["feature_importance", "feature_names"]
                        })

                        # Log feature importance
                        if "feature_importance" in training_stats:
                            importance = training_stats["feature_importance"]
                            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                            logger.info("Top feature importance:")
                            for fname, imp in sorted_importance[:5]:
                                logger.info("  %s: %.1f%%", fname, imp * 100)

                        # Save the trained model
                        ml_model_path = run_dir / "ml_scorer.pkl"
                        ml_scorer.save(ml_model_path)
                        logger.info("ML scorer saved to %s", ml_model_path)
                    else:
                        logger.warning("ML scorer training failed or insufficient samples")
                else:
                    logger.warning("No valid training samples found for ML scorer")

        # === RISK PREDICTION: Collect data and train when ready ===
        if use_risk_prediction:
            # Collect MAE/MFE data from PnL result if available
            if hasattr(pnl_result, "mae_matrix") and hasattr(pnl_result, "exit_reason_matrix"):
                risk_training_data.append({
                    "entries": entries.copy(),
                    "exits": exits.copy(),
                    "returns_matrix": returns_matrix.copy(),
                    "mae_matrix": pnl_result.mae_matrix.copy(),
                    "exit_reason_matrix": pnl_result.exit_reason_matrix.copy(),
                    "spread_returns": _compute_spread_returns(test_df, beta_df, valid_pairs, "log"),
                    "z_score": z_df.copy(),
                    "spread_volatility": vol_df.copy(),
                    "beta": beta_df.copy(),
                    "half_life_bars": pair_half_lives.copy() if pair_half_lives else {},
                    "kalman_gain": signals.kalman_gain.copy() if hasattr(signals, 'kalman_gain') and signals.kalman_gain is not None else None,
                    "price_matrix": test_df.copy(),
                })
                logger.info("Collected risk training data from window %d (%d entries)",
                           w_idx, int(entries.sum().sum()))

            # Train risk predictor after accumulating enough windows
            if len(risk_training_data) >= risk_min_train_windows:
                logger.info("Training risk predictor on %d windows of data...", len(risk_training_data))

                try:
                    risk_config = RiskPredictorConfig(
                        predict_mae=bool(getattr(cfg, "RISK_PREDICT_MAE", True)),
                        predict_forward_vol=bool(getattr(cfg, "RISK_PREDICT_FORWARD_VOL", True)),
                        predict_stopout=bool(getattr(cfg, "RISK_PREDICT_STOPOUT", True)),
                        min_training_samples=int(getattr(cfg, "RISK_MIN_TRAINING_SAMPLES", 100)),
                    )

                    risk_label_gen = RiskLabelGenerator(risk_config)
                    risk_feature_ext = RiskFeatureExtractor(risk_config)

                    all_X_risk = []
                    all_y_mae = []
                    all_y_vol = []
                    all_y_stopout = []

                    for wd in risk_training_data:
                        # Generate labels from MAE/exit data
                        labels = risk_label_gen.generate_labels(
                            entries=wd["entries"],
                            exits=wd["exits"],
                            returns_matrix=wd["returns_matrix"],
                            mae_matrix=wd["mae_matrix"],
                            exit_reason_matrix=wd["exit_reason_matrix"],
                            spread_returns=wd["spread_returns"],
                        )

                        # Extract features
                        features = risk_feature_ext.extract_features(
                            z_score=wd["z_score"],
                            spread_volatility=wd["spread_volatility"],
                            kalman_gain=wd["kalman_gain"] if wd["kalman_gain"] is not None else pd.DataFrame(0.1, index=wd["z_score"].index, columns=wd["z_score"].columns),
                            beta=wd["beta"],
                            half_life=pd.DataFrame({p: wd["half_life_bars"].get(p, 500) for p in wd["z_score"].columns}, index=wd["z_score"].index),
                            price_matrix=wd["price_matrix"],
                            entry_z=float(getattr(cfg, "ENTRY_Z", 2.0)),
                            max_entry_z=float(getattr(cfg, "MAX_ENTRY_Z", 4.0)),
                            btc_column=btc_symbol,
                        )

                        # Prepare training data
                        X, y_dict, _ = risk_feature_ext.prepare_training_data(
                            features=features,
                            labels=labels,
                            entry_mask=wd["entries"].astype(bool),
                        )

                        if len(X) > 0:
                            all_X_risk.append(X)
                            all_y_mae.extend(y_dict["mae"])
                            all_y_vol.extend(y_dict["forward_vol"])
                            all_y_stopout.extend(y_dict["stopout"])

                    if all_X_risk:
                        X_combined = np.vstack(all_X_risk)
                        y_combined = {
                            "mae": np.array(all_y_mae),
                            "forward_vol": np.array(all_y_vol),
                            "stopout": np.array(all_y_stopout),
                        }

                        logger.info("Risk training data: %d samples", len(X_combined))

                        # Train risk predictor
                        new_risk_predictor = RiskPredictor(risk_config)
                        training_stats = new_risk_predictor.train(
                            X_combined,
                            y_combined,
                            RiskFeatureExtractor.FEATURE_NAMES,
                        )

                        if new_risk_predictor.is_trained:
                            logger.info("Risk predictor trained: %s", training_stats)

                            # Save the trained model
                            import pickle
                            risk_model_path = run_dir / "risk_predictor.pkl"
                            with open(risk_model_path, 'wb') as f:
                                pickle.dump(new_risk_predictor, f)
                            logger.info("Risk predictor saved to %s", risk_model_path)

                except Exception as e:
                    logger.warning("Risk predictor training failed: %s", e)
                    import traceback
                    traceback.print_exc()

        if adaptive_cfg.ADAPTIVE_ENABLED:
            trade_returns = [float(r) for r in returns_matrix.to_numpy().ravel() if r != 0]
            hold_hours = _compute_holding_hours(entries, exits)
            stop_loss_rate = 0.0
            if hasattr(pnl_result, "stop_loss_count") and hasattr(pnl_result, "trades_count"):
                try:
                    total_stop = float(pnl_result.stop_loss_count.sum())
                    total_trades = float(pnl_result.trades_count.sum())
                    stop_loss_rate = total_stop / total_trades if total_trades else 0.0
                except Exception:
                    stop_loss_rate = 0.0
            overrides, _ = adaptive_controller.maybe_update_backtest(
                current_params={
                    "ENTRY_Z": cfg.ENTRY_Z,
                    "EXIT_Z": cfg.EXIT_Z,
                    "MIN_PROFIT_HURDLE": cfg.MIN_PROFIT_HURDLE,
                    "MAX_PORTFOLIO_POSITIONS": cfg.MAX_PORTFOLIO_POSITIONS,
                    "MAX_POSITIONS_PER_COIN": cfg.MAX_POSITIONS_PER_COIN,
                    "STOP_LOSS_PCT": cfg.STOP_LOSS_PCT,
                },
                window_id=window_id,
                trades=trade_returns,
                stop_loss_rate=stop_loss_rate,
                hold_hours=hold_hours,
                log_path=adaptive_log_path,
            )
            if overrides:
                apply_overrides_to_backtest(cfg, overrides)
                logger.info("Adaptive overrides applied for next window: %s", overrides)

    if not all_returns:
        raise RuntimeError("No windows produced returns; check coverage and filters.")

    returns_matrix = pd.concat(all_returns, axis=0).sort_index().fillna(0.0)
    entries_all = pd.concat(all_entries, axis=0).sort_index().fillna(False)
    exits_all = pd.concat(all_exits, axis=0).sort_index().fillna(False)

    # Aggregate position sizes for realistic simulation
    position_sizes_all = None
    if all_position_sizes:
        position_sizes_all = pd.concat(all_position_sizes, axis=0).sort_index().fillna(1.0)
        logger.info("Aggregated position sizes for realistic simulation: %d rows", len(position_sizes_all))

    returns_out = paths["returns_matrix"]
    returns_matrix.to_parquet(returns_out)
    entries_all.to_parquet(paths["entries"])
    exits_all.to_parquet(paths["exits"])
    logger.info("✅ Saved aggregated returns_matrix and masks.")

    # Save combined valid pairs
    all_pairs_unique = sorted(set(all_pairs))
    _write_json(paths["valid_pairs"], {"pairs": all_pairs_unique})

    # Save combined test prices for analysis
    test_prices_all = None
    if all_test_prices:
        test_prices_all = pd.concat(all_test_prices, axis=0).sort_index()
        test_prices_all.to_parquet(paths["test_prices"])

    # 9) Performance report + plots
    logger.info("Building performance report...")
    report_freq = getattr(cfg, "SIGNAL_TIMEFRAME", None) or getattr(cfg, "BAR_FREQ", "1min")
    initial_capital = float(getattr(cfg, "INITIAL_CAPITAL", 100_000.0))
    report = generate_performance_report(
        run_dir=run_dir,
        returns_matrix=returns_matrix,
        test_prices=test_prices_all,
        btc_symbol=btc_symbol,
        freq=report_freq,
        entries=entries_all,
        exits=exits_all,
        max_hold_bars=None,
        position_size_multiplier=position_sizes_all,
        initial_capital=initial_capital,
    )

    # 9.5) P&L Attribution Report (Phase 0)
    try:
        from src.backtest.pnl_attribution import (
            generate_attribution_from_pnl_result,
            generate_attribution_json,
            format_attribution_report,
            save_attribution_report,
        )

        # Aggregate PnL component matrices from all windows
        # These have the ACTUAL costs computed by the PnL engine (scaled by position_weight)
        fees_all = pd.DataFrame(0.0, index=returns_matrix.index, columns=returns_matrix.columns)
        slippage_all = pd.DataFrame(0.0, index=returns_matrix.index, columns=returns_matrix.columns)
        gross_pnl_all = pd.DataFrame(0.0, index=returns_matrix.index, columns=returns_matrix.columns)
        hold_bars_all = pd.DataFrame(0, index=returns_matrix.index, columns=returns_matrix.columns, dtype=np.int64)
        exit_reason_all = pd.DataFrame(0, index=returns_matrix.index, columns=returns_matrix.columns)
        # NOTE: Do NOT default to signal exit - keep 0 (unknown) and load actual values from parquet
        # The old line `exit_reason_all[returns_matrix != 0] = 1` was a bug that masked actual exit reasons
        funding_costs_all = pd.DataFrame(0.0, index=returns_matrix.index, columns=returns_matrix.columns)

        # Load per-window data
        for wd in sorted(Path(run_dir / "windows").iterdir()):
            try:
                # Load all component matrices
                for fname, target_df in [
                    ("fees.parquet", fees_all),
                    ("slippage.parquet", slippage_all),
                    ("gross_pnl.parquet", gross_pnl_all),
                    ("hold_bars.parquet", hold_bars_all),
                    ("exit_reasons.parquet", exit_reason_all),
                    ("funding_costs.parquet", funding_costs_all),
                ]:
                    fpath = wd / fname
                    if fpath.exists():
                        window_df = pd.read_parquet(fpath)
                        # Merge into aggregate (align by index and columns)
                        for col in window_df.columns:
                            if col in target_df.columns:
                                mask = window_df[col] != 0
                                target_df.loc[window_df.index[mask], col] = window_df.loc[mask, col]
            except Exception as e:
                logger.warning("Failed to load window data from %s: %s", wd, e)

        # Use the actual PnL engine outputs (with correct position-scaled costs)
        attribution_report = generate_attribution_from_pnl_result(
            returns_matrix=returns_matrix,
            gross_pnl_matrix=gross_pnl_all,
            fees_matrix=fees_all,
            slippage_matrix=slippage_all,
            hold_bars_matrix=hold_bars_all,
            exit_reason_matrix=exit_reason_all,
            funding_costs_matrix=funding_costs_all,  # NEW: use funding from PnL engine
            freq=report_freq,
        )

        # Save attribution report
        attribution_paths = save_attribution_report(
            report=attribution_report,
            output_dir=run_dir,
            filename_prefix="pnl_attribution",
        )

        # Also save JSON version alongside metrics.json
        attribution_json = generate_attribution_json(attribution_report, freq=report_freq)
        with open(run_dir / "pnl_attribution.json", "w") as f:
            json.dump(attribution_json, f, indent=2)

        logger.info("✅ P&L Attribution report saved")

    except Exception as attr_exc:
        logger.warning("Failed to generate attribution report: %s", attr_exc)

    # 10) Diagnostics (pair-level deep dive plots)
    if diagnose_n > 0:
        if last_window_data is None:
            logger.warning("Diagnostics requested but no window data available. Skipping.")
        else:
            test_df = last_window_data["test_df"]
            z_df = last_window_data["z_df"]
            beta_df = last_window_data["beta_df"]
            entries = last_window_data["entries"]
            exits = last_window_data["exits"]
            expected_profit = last_window_data["expected_profit"]
            vol_df = last_window_data["vol_df"]
            valid_pairs = list(z_df.columns)
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

    # 11) Generate advanced visualizations
    logger.info("Generating advanced diagnostic visualizations...")
    try:
        # Prepare pair summaries from attribution JSON (if available)
        pair_summaries_for_viz = []
        attribution_summary_for_log = {}
        if "attribution_json" in dir() and attribution_json is not None:
            pair_summaries_for_viz = attribution_json.get("pair_details", [])
            attribution_summary_for_log = attribution_json.get("portfolio_summary", {})
            attribution_summary_for_log["diagnostics"] = attribution_json.get("diagnostics", {})
        elif "attribution_report" in dir() and attribution_report is not None:
            # Convert from dataclass
            from dataclasses import asdict
            for s in attribution_report.pair_summaries:
                pair_summaries_for_viz.append({
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
            attribution_summary_for_log = {
                "total_trades": attribution_report.total_trades,
                "total_gross_pnl": attribution_report.total_gross_pnl,
                "total_fees": attribution_report.total_fees,
                "total_slippage": attribution_report.total_slippage,
                "total_funding_pnl": attribution_report.total_funding_pnl,
                "total_net_pnl": attribution_report.total_net_pnl,
                "cost_to_gross_ratio": attribution_report.cost_to_gross_ratio,
                "diagnostics": {
                    "gross_positive_net_negative": attribution_report.gross_positive_net_negative,
                    "gross_negative": attribution_report.gross_negative,
                    "few_pairs_dominate_losses": attribution_report.few_pairs_dominate_losses,
                },
            }

        # Generate all visualizations
        viz_paths = generate_all_visualizations(
            run_dir=run_dir,
            returns_matrix=returns_matrix,
            pair_summaries=pair_summaries_for_viz,
            window_metrics=window_metrics_list,
            attribution_summary=attribution_summary_for_log,
            freq=report_freq,
        )
        logger.info("✅ Generated %d diagnostic visualizations", len(viz_paths))
    except Exception as viz_exc:
        logger.warning("Failed to generate some visualizations: %s", viz_exc)

    # 12) Generate detailed backtest log
    logger.info("Generating detailed backtest log...")
    try:
        # Collect config parameters
        config_snapshot = {}
        for name in dir(cfg):
            if name.isupper() and not name.startswith("_"):
                val = getattr(cfg, name)
                if not callable(val):
                    try:
                        json.dumps(val)  # Check if serializable
                        config_snapshot[name] = val
                    except (TypeError, ValueError):
                        config_snapshot[name] = str(val)

        _generate_detailed_log(
            run_dir=run_dir,
            run_id=run_id,
            window_metrics=window_metrics_list,
            pair_summaries=pair_summaries_for_viz,
            config_snapshot=config_snapshot,
            attribution_summary=attribution_summary_for_log,
            report_metrics=report.get("metrics", {}),
        )
        logger.info("✅ Detailed backtest log saved")

        # Also save window metrics as JSON for programmatic access
        window_metrics_path = run_dir / "window_metrics.json"
        with open(window_metrics_path, "w") as f:
            json.dump(window_metrics_list, f, indent=2)
        logger.info("✅ Window metrics saved: %s", window_metrics_path)

        # ===== PHASE 1: Save symbol blacklist summary =====
        if enable_symbol_blacklist and symbol_blacklist is not None:
            blacklist_summary = symbol_blacklist.get_summary()
            blacklist_path = run_dir / "symbol_blacklist.json"
            _write_json(blacklist_path, blacklist_summary)
            logger.info(
                "✅ Symbol blacklist saved: %d symbols tracked, %d blacklisted",
                blacklist_summary.get("total_symbols_tracked", 0),
                blacklist_summary.get("blacklisted_count", 0),
            )
            # Also save full blacklist state for potential reuse
            symbol_blacklist.save(run_dir / "symbol_blacklist_state.json")

        # ===== PHASE 5B: Save window analysis report =====
        if enable_window_analysis and window_analysis is not None:
            # Log summary
            window_analysis.log_summary()

            # Save detailed report
            window_analysis.save_to_csv(str(run_dir / "window_analysis.csv"))

            # Save comparison report
            comparison = window_analysis.get_comparison_report()
            if not comparison.empty:
                comparison.to_csv(run_dir / "window_analysis_comparison.csv")

            # Log regime recommendations
            recommendations = window_analysis.get_regime_recommendations()
            if recommendations:
                logger.info("Regime recommendations based on window analysis:")
                for key, rec in recommendations.items():
                    logger.info("  %s: %s -> %s", key, rec.get("observation", ""), rec.get("suggestion", ""))

    except Exception as log_exc:
        logger.warning("Failed to generate detailed log: %s", log_exc)

    logger.info("🏁 Backtest complete. Run ID: %s", run_id)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_pairs": len(all_pairs_unique),
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
    p.add_argument("--no-strict", action="store_true", help="Disable strict data validation (warn on gaps instead of error).")
    p.add_argument("--pretrained-risk-model", type=str, default=None, help="Path to pre-trained risk predictor pickle (applies from window 0).")
    p.add_argument("--config-overrides", type=str, default=None, help="Path to JSON file with config overrides (for batch experiments).")

    # Phase 1 parameter overrides (for parameter sweep)
    p.add_argument("--entry-z", type=float, default=None, help="Override ENTRY_Z threshold.")
    p.add_argument("--exit-z", type=float, default=None, help="Override EXIT_Z threshold.")
    p.add_argument("--coint-pvalue", type=float, default=None, help="Override COINT_PVALUE_THRESHOLD.")

    return p


def apply_config_overrides(overrides_path: Path) -> None:
    """Apply config overrides from a JSON file."""
    if not overrides_path.exists():
        logger.warning("Config overrides file not found: %s", overrides_path)
        return

    with open(overrides_path) as f:
        overrides = json.load(f)

    for key, value in overrides.items():
        if hasattr(cfg, key):
            old_value = getattr(cfg, key)
            setattr(cfg, key, value)
            logger.info("Config override: %s = %s (was %s)", key, value, old_value)
        else:
            logger.warning("Unknown config key: %s", key)


if __name__ == "__main__":
    args = build_argparser().parse_args()

    # Apply config overrides if provided (for batch experiments)
    if args.config_overrides:
        apply_config_overrides(Path(args.config_overrides))

    # Apply direct parameter overrides (for parameter sweep)
    if args.entry_z is not None:
        old_val = cfg.ENTRY_Z
        cfg.ENTRY_Z = args.entry_z
        logger.info("Parameter override: ENTRY_Z = %s (was %s)", args.entry_z, old_val)

    if args.exit_z is not None:
        old_val = cfg.EXIT_Z
        cfg.EXIT_Z = args.exit_z
        logger.info("Parameter override: EXIT_Z = %s (was %s)", args.exit_z, old_val)

    if args.coint_pvalue is not None:
        old_val = cfg.COINT_PVALUE_THRESHOLD
        cfg.COINT_PVALUE_THRESHOLD = args.coint_pvalue
        logger.info("Parameter override: COINT_PVALUE_THRESHOLD = %s (was %s)", args.coint_pvalue, old_val)

    result = run_backtest(
        run_name=args.run_name,
        parquet_path=Path(args.parquet_path) if args.parquet_path else None,
        max_pairs=args.max_pairs,
        diagnose_n=args.diagnose,
        log_level=args.log_level,
        btc_symbol=args.btc_symbol,
        strict_validation=not args.no_strict,
        pretrained_risk_model_path=Path(args.pretrained_risk_model) if args.pretrained_risk_model else None,
    )

    # Print a short, useful summary
    print(json.dumps(result, indent=2))
