"""
src/backtest/config_backtest.py

Institutional-grade backtest configuration + run/manifest utilities.

Design goals:
- Single source of truth for all Phase 5 knobs.
- Reproducible runs via per-run folders under ./results/run_*/...
- Consistent with:
    - src/backtest/data_segmenter.py
    - src/backtest/accountant_filter.py
    - src/backtest/pnl_engine.py
    - src/backtest/performance_report.py
    - src/backtest/diagnostics.py
- Backwards-compatible aliases for older scripts that referenced lowercase path names.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numba
import numpy as np

# =============================================================================
# 1. METADATA / VERSIONING
# =============================================================================

BACKTEST_VERSION: str = "1.2.0"
RANDOM_SEED: int = 42

# Update this whenever you change the data snapshot (or table version)
DATA_SNAPSHOT_ID: str = "hyperliquid_2025_01_snapshot"

# =============================================================================
# 2. FILE PATHS (CENTRALIZED)
# =============================================================================

# This file lives at: crypto_quant_bot/src/backtest/config_backtest.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_downloads"
READY_DATA_DIR: Path = DATA_DIR / "backtest_ready"  # optional legacy staging
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Raw input file (pivoted price matrix) produced by export step
RAW_PARQUET_FILE: str = "crypto_prices_1m.parquet"
PATH_RAW_PARQUET: Path = RAW_DATA_DIR / RAW_PARQUET_FILE

# Legacy/staging filenames (optional—kept for compatibility)
TEST_DATA_FILE: str = "test_market_data.parquet"
STATE_FILE: str = "warm_start_states.pkl"
PATH_TEST_DATA: Path = READY_DATA_DIR / TEST_DATA_FILE
PATH_STATE: Path = READY_DATA_DIR / STATE_FILE

# Backwards-compatible aliases (older scripts may reference these)
path_raw_parquet = PATH_RAW_PARQUET
path_test_data = PATH_TEST_DATA
path_state = PATH_STATE

# =============================================================================
# 3. BACKTEST RUNTIME / SPLIT PARAMS
# =============================================================================

# Bar frequency used by vectorbt and by duration conversions
BAR_FREQ: str = "1min"
TIMEZONE: str = "UTC"

# Split mode:
# - "ratio": split by TEST_RATIO
# - "days": split by TRAIN_DAYS/TEST_DAYS (recommended for exact "half-year total")
SPLIT_MODE: str = "days"

# If SPLIT_MODE == "ratio"
TEST_RATIO: float = 0.20

# If SPLIT_MODE == "days"
# Production values: TRAIN_DAYS=150, TEST_DAYS=32 (182 days total)
# Current values adjusted for limited data availability.
# TODO: Restore to 150/32 when more historical data is ingested.
TRAIN_DAYS: int = 2
TEST_DAYS: int = 1

# Data safety
MAX_DATA_GAP_MINS: int = 5   # crash if any timestamp gap exceeds this
LOOKBACK_WINDOW: int = 60    # generic rolling window length (mins)

# Pair naming
PAIR_ID_SEPARATOR: str = "-"  # e.g., "ETH-BTC"

# =============================================================================
# 4. STRATEGY PARAMS ("THE KNOBS")
# =============================================================================

# Kalman Settings
KALMAN_DELTA: float = 1e-6
KALMAN_R: float = 1e-2

# Accountant thresholds
ENTRY_Z: float = 2.0
EXIT_Z: float = 0.5
STOP_LOSS_Z: float = 4.5

# Expected reversion / profitability filter
EXPECTED_REVERT_MULT: float = 0.75
MIN_PROFIT_HURDLE: float = 0.002

# Fees: 0.05% per leg; 4 legs total
FEE_RATE: float = 0.0005

# Microstructure / Slippage hook (used in pnl_engine)
SLIPPAGE_MODEL: str = "fixed"   # {"fixed", "vol_adjusted"}
SLIPPAGE_BPS: float = 2.0       # base slippage in bps per leg
SLIPPAGE_RATE: float = SLIPPAGE_BPS / 10_000.0
SLIPPAGE_VOL_MULT: float = 1.0  # multiplier for vol_adjusted
SLIPPAGE_CAP_BPS: float = 25.0  # safety cap
SLIPPAGE_CAP_RATE: float = SLIPPAGE_CAP_BPS / 10_000.0

# Capital assumptions
# pnl_engine returns (net_pnl / capital_per_pair).
# If you model each pair as having an equal capital budget, keep 1.0 for normalized.
CAPITAL_PER_PAIR: float = 1.0
INIT_CASH: float = 1.0  # used for equity curves (normalized)

# Performance report scenarios (daily rates)
FUNDING_DRAG_OPT_DAILY: float = 0.0
FUNDING_DRAG_BASE_DAILY: float = 0.0001   # 0.01% daily
FUNDING_DRAG_STRESS_DAILY: float = 0.0003 # 0.03% daily

# Stress scenario: extra slippage applied per exit event (performance_report)
STRESS_EXTRA_SLIPPAGE_PER_EXIT: float = 0.0005  # 5 bps per completed trade (exit timestamp)

# Pass/fail thresholds (reporting gate)
SUCCESS_SHARPE_MIN: float = 1.5
SUCCESS_MAX_DD_DURATION_DAYS: int = 5
SUCCESS_BTC_CORR_MAX: float = 0.5

# =============================================================================
# 5. MANIFEST DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class EnvironmentInfo:
    timestamp: str
    run_id: str
    python_version: str
    platform: str
    numba_version: str
    numpy_version: str
    git_commit: str
    data_snapshot_id: str
    random_seed: int
    backtest_version: str


@dataclass(frozen=True)
class Manifest:
    parameters: Dict[str, Any]
    environment: EnvironmentInfo
    extra_metadata: Dict[str, Any]


# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def get_git_revision_hash() -> str:
    """Retrieve the current git commit hash for traceability."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
    except Exception:
        return "NOT_A_GIT_REPO"


def _serialize_param(value: Any) -> Any:
    """Make sure all config parameters are JSON-serializable."""
    if isinstance(value, Path):
        return str(value)
    return value


def _collect_config_parameters() -> Dict[str, Any]:
    """
    Collect all uppercase module-level variables as 'parameters'.

    Notes:
    - Ignores callables and private names.
    - Captures all knobs automatically; keep constants uppercase.
    """
    params: Dict[str, Any] = {}
    for name, val in globals().items():
        if not name.isupper():
            continue
        if name.startswith("_"):
            continue
        if callable(val):
            continue
        params[name] = _serialize_param(val)
    return params


# =============================================================================
# 7. RUN FOLDER + ARTIFACT PATHS (PHASE 5 CANON)
# =============================================================================

def create_run_dir(run_name: Optional[str] = None) -> Tuple[str, Path]:
    """
    Create a per-run results folder:
      results/run_YYYYMMDD_HHMMSS/

    Returns (run_id, run_dir).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"run_{timestamp}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def get_run_paths(run_dir: Path) -> Dict[str, Path]:
    """
    Standardized artifact locations under a given run_dir.
    """
    run_dir = Path(run_dir)
    return {
        "manifest": run_dir / "manifest.json",
        "valid_pairs": run_dir / "valid_pairs.json",
        "warm_states": run_dir / "warm_states.pkl",
        "signals": run_dir / "signals.parquet",       # optional
        "returns_matrix": run_dir / "returns_matrix.parquet",
        "metrics": run_dir / "metrics.json",
        "plots_dir": run_dir / "plots",
    }


# =============================================================================
# 8. PUBLIC API: save_manifest
# =============================================================================

def save_manifest(
    run_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
) -> Tuple[str, Path]:
    """
    Save a 'flight recorder' snapshot of the current experiment.

    You may either:
    - call save_manifest() directly (it will create a run dir), or
    - call create_run_dir() first and pass run_dir here.

    Returns (run_id, run_dir).
    """
    if run_dir is None:
        run_id, run_dir = create_run_dir(run_name=run_name)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name

    parameters = _collect_config_parameters()
    env_info = EnvironmentInfo(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        run_id=run_id,
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        numba_version=numba.__version__,
        numpy_version=np.__version__,
        git_commit=get_git_revision_hash(),
        data_snapshot_id=DATA_SNAPSHOT_ID,
        random_seed=RANDOM_SEED,
        backtest_version=BACKTEST_VERSION,
    )

    manifest = Manifest(
        parameters=parameters,
        environment=env_info,
        extra_metadata=extra_metadata or {},
    )

    manifest_path = get_run_paths(run_dir)["manifest"]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=4)

    print(f"Manifest saved: {manifest_path}")
    return run_id, run_dir


__all__ = [
    # metadata
    "BACKTEST_VERSION",
    "RANDOM_SEED",
    "DATA_SNAPSHOT_ID",
    # paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "READY_DATA_DIR",
    "RESULTS_DIR",
    "RAW_PARQUET_FILE",
    "PATH_RAW_PARQUET",
    "PATH_TEST_DATA",
    "PATH_STATE",
    # legacy aliases
    "path_raw_parquet",
    "path_test_data",
    "path_state",
    # runtime/split
    "BAR_FREQ",
    "TIMEZONE",
    "SPLIT_MODE",
    "TEST_RATIO",
    "TRAIN_DAYS",
    "TEST_DAYS",
    "MAX_DATA_GAP_MINS",
    "LOOKBACK_WINDOW",
    "PAIR_ID_SEPARATOR",
    # strategy knobs
    "KALMAN_DELTA",
    "KALMAN_R",
    "ENTRY_Z",
    "EXIT_Z",
    "STOP_LOSS_Z",
    "EXPECTED_REVERT_MULT",
    "MIN_PROFIT_HURDLE",
    "FEE_RATE",
    "SLIPPAGE_MODEL",
    "SLIPPAGE_BPS",
    "SLIPPAGE_RATE",
    "SLIPPAGE_VOL_MULT",
    "SLIPPAGE_CAP_BPS",
    "SLIPPAGE_CAP_RATE",
    "CAPITAL_PER_PAIR",
    "INIT_CASH",
    "FUNDING_DRAG_OPT_DAILY",
    "FUNDING_DRAG_BASE_DAILY",
    "FUNDING_DRAG_STRESS_DAILY",
    "STRESS_EXTRA_SLIPPAGE_PER_EXIT",
    "SUCCESS_SHARPE_MIN",
    "SUCCESS_MAX_DD_DURATION_DAYS",
    "SUCCESS_BTC_CORR_MAX",
    # run/manifest
    "create_run_dir",
    "get_run_paths",
    "save_manifest",
]
