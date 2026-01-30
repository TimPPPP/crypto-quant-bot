"""
src/backtest/config_backtest.py

Institutional-grade backtest configuration and manifest utilities.

Design goals:
- Single source of truth for all configuration parameters.
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
BACKTEST_LOOKBACK_DAYS: int = 365

# =============================================================================
# 2. FILE PATHS (CENTRALIZED)
# =============================================================================

# This file lives at: crypto_quant_bot/src/backtest/config_backtest.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_downloads"
READY_DATA_DIR: Path = DATA_DIR / "backtest_ready"  # optional legacy staging
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# =============================================================================
# DATA FILES (produced by step0_export_data_full_year.py)
# =============================================================================
# Run step0 to export data from QuestDB with coverage filtering:
#   poetry run python research/pipeline/step0_export_data_full_year.py
# Environment variables control export: EXPORT_MIN_COVERAGE (default 0.90)
RAW_PARQUET_FILE: str = "crypto_prices_1m_2025_full_year_90p.parquet"
PATH_RAW_PARQUET: Path = RAW_DATA_DIR / RAW_PARQUET_FILE
FUNDING_PARQUET_FILE: str = "funding_rates_2025_full_year.parquet"
PATH_FUNDING_PARQUET: Path = RAW_DATA_DIR / FUNDING_PARQUET_FILE

# Use real funding rates from ingested data (vs constant assumptions)
USE_REAL_FUNDING: bool = True

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
# LOW-FREQUENCY REDESIGN: Changed from 15min to 1h for longer-term mean reversion
BAR_FREQ: str = "1h"  # 1-hour candles for multi-day holding periods
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
TRAIN_DAYS: int = 180
TEST_DAYS: int = 32

# Data safety
# Tighter data quality requirements
MAX_DATA_GAP_MINS: int = 60  # crash if any timestamp gap exceeds this (reduced from 360)
WARN_DATA_GAP_MINS: int = 10  # warn if any gap exceeds this
MAX_INTERPOLATE_MINS: int = 5  # interpolate gaps up to this size
LOOKBACK_WINDOW: int = 60    # generic rolling window length (mins)
MIN_DATA_COVERAGE: float = 0.90  # minimum non-missing ratio per coin in train/test

# Multi-timeframe support # Aggregate 1-min bars to higher timeframes where edge > friction.
# SIGNAL_TIMEFRAME: Target timeframe for signal generation ("1min", "5min", "15min", "1h")
# When SIGNAL_TIMEFRAME != "1min", data will be resampled before Kalman/signal computation.
SIGNAL_TIMEFRAME: str = "15min"  # Optimal for intraday mean-reversion
SUPPORTED_TIMEFRAMES: tuple = ("1min", "5min", "15min", "30min", "1h", "4h")

# Walk-forward backtest settings
WALK_FORWARD_ENABLED: bool = True
WALK_FORWARD_TRAIN_DAYS: int = 180
WALK_FORWARD_TEST_DAYS: int = 30
WALK_FORWARD_STEP_DAYS: int = 30

# =============================================================================
# WALK-FORWARD PRESETS (Expert Crypto Suggestions)
# =============================================================================
# Preset selection: "default", "option_a", "option_b", "option_c"
# - default: Current behavior (180/30/30)
# - option_a: Balanced faster adaptation (90/21/14)
# - option_b: Aggressive fast adaptation (60/14/7)
# - option_c: Long window with exponential weighting (180/30/14 + exp decay)
# Expert suggestion: option_c (14d step + exp weighting) adapts faster to crypto regime changes
WALK_FORWARD_PRESET: str = "option_a"  # 90/21/14 for faster adaptation and more trades

WF_PRESETS: dict = {
    "default": {"train_days": 180, "test_days": 30, "step_days": 30},
    "option_a": {"train_days": 90, "test_days": 21, "step_days": 14},
    "option_b": {"train_days": 60, "test_days": 14, "step_days": 7},
    "option_c": {
        "train_days": 180,
        "test_days": 30,
        "step_days": 14,
        "use_exp_weighting": True,
        "exp_weight_half_life_days": 30.0,
    },
}

# =============================================================================
# TWO-STAGE PAIR SELECTION
# =============================================================================
# Stage 1 (Discovery): Longer lookback, looser thresholds to find candidates
# Stage 2 (Validation): Recent window with tighter thresholds to confirm
ENABLE_TWO_STAGE_PAIR_SELECTION: bool = True  # Expert suggestion: use discovery + validation
TWO_STAGE_DISCOVERY_LOOKBACK_DAYS: int = 120
TWO_STAGE_DISCOVERY_P_VALUE: float = 0.05

# Relax validation p-value to accept more pairs
# Previous: 0.03 filtered out ~99% of pairs → Only 80-100 pairs available
# New: 0.10 should double or triple pair universe → Target 200-300 pairs
TWO_STAGE_VALIDATION_LOOKBACK_DAYS: int = 45  # Keep existing (sufficient statistical power)
TWO_STAGE_VALIDATION_P_VALUE: float = 0.10  # Relaxed from 0.03 (more pairs)
TWO_STAGE_VALIDATION_MIN_HALF_LIFE: int = 40
TWO_STAGE_VALIDATION_MAX_HALF_LIFE: int = 500
TWO_STAGE_REQUIRE_BOTH: bool = True

# =============================================================================
# MULTI-TIMEFRAME TRAINING # =============================================================================
# Uses dual training windows: long-term for stable pairs, short-term for regime
# Long-term window: 180 days for discovering stable cointegrated relationships
# Short-term window: Last 30 days of long-term for regime-aware validation
#
# Structure: [Long-term 180d] -> [Short-term 30d subset] -> [Test 21d]
#
# DISABLED: Multi-TF consensus selection not finding pairs due to combined constraints
# (p-value + half-life range + internal 80/20 split makes 180d windows too restrictive)
# Requires scanner modifications to work properly (e.g., looser half-life for discovery)
ENABLE_MULTI_TIMEFRAME_TRAINING: bool = False

# Long-term window for stable pair discovery
MULTI_TF_LONG_TERM_DAYS: int = 180          # Look for pairs stable over 180 days
MULTI_TF_LONG_TERM_PVALUE: float = 0.05     # Find pairs cointegrated over long period

# Short-term window (subset of long-term) for regime validation
MULTI_TF_SHORT_TERM_DAYS: int = 30          # Last 30 days of long-term window
MULTI_TF_SHORT_TERM_PVALUE: float = 0.03    # Stricter for short-term (must be strongly valid NOW)

# Require pairs to pass both windows (consensus pairs only)
MULTI_TF_REQUIRE_CONSENSUS: bool = True

# =============================================================================
# LIVE REGIME UPDATES # =============================================================================
# Instead of freezing regime at window start, update it dynamically during test.
# This allows the strategy to react to changing market conditions in real-time.
#
# Key features:
# - Recompute BTC volatility and cross-sectional dispersion every N bars
# - Immediate transition to RED regime on volatility spikes
# - Hysteresis to prevent whipsawing between regimes

ENABLE_LIVE_REGIME_UPDATES: bool = False  # Disabled (blocks 26% of entries)

# How often to recompute regime (in bars)
# 96 bars = 1 day at 15-min frequency
REGIME_UPDATE_FREQUENCY_BARS: int = 96

# Minimum bars between regime transitions (hysteresis)
# Prevents rapid switching between GREEN/YELLOW/RED
REGIME_TRANSITION_COOLDOWN_BARS: int = 24  # 6 hours at 15-min

# Volatility spike detection (immediate RED trigger)
# If BTC volatility z-score exceeds this, immediately transition to RED
BTC_VOL_SPIKE_Z_THRESHOLD: float = 2.0

# Lookback window for computing rolling BTC volatility (in bars)
REGIME_BTC_VOL_LOOKBACK_BARS: int = 96  # 1 day

# Percentile thresholds for regime classification
REGIME_VOL_GREEN_MAX_PCTL: float = 0.50   # GREEN if vol < 50th percentile
REGIME_VOL_YELLOW_MAX_PCTL: float = 0.70  # YELLOW if vol < 70th percentile, else RED
REGIME_DISPERSION_RED_PCTL: float = 0.80  # RED if dispersion > 80th percentile

# -----------------------------------------------------------------------------
# REGIME-CONDITIONAL TRADING PARAMETERS # -----------------------------------------------------------------------------
# Enable dynamic parameter adjustment based on current regime state
# When enabled, entry thresholds, position limits, and cooldowns adjust per regime
ENABLE_REGIME_CONDITIONAL_PARAMS: bool = False  # Disabled - reduces Sharpe without benefit

# Entry Z-score thresholds per regime
# GREEN: Normal selectivity, YELLOW: More selective, RED: Block entries
REGIME_GREEN_ENTRY_Z: float = 2.5       # Standard entry threshold in GREEN
REGIME_YELLOW_ENTRY_Z: float = 3.0      # Stricter threshold in YELLOW (require stronger signal)
REGIME_RED_ENTRY_Z: float = 999.0       # Effectively block entries in RED

# Position size multipliers per regime (already used by LiveRegimeTracker)
REGIME_GREEN_SIZE_MULT: float = 1.0     # Full position size in GREEN
REGIME_YELLOW_SIZE_MULT: float = 0.5    # Half position size in YELLOW
REGIME_RED_SIZE_MULT: float = 0.0       # No new positions in RED

# Maximum portfolio positions per regime
# Reduces exposure during adverse regimes
REGIME_GREEN_MAX_POSITIONS: int = 8     # Full capacity in GREEN
REGIME_YELLOW_MAX_POSITIONS: int = 4    # Half capacity in YELLOW
REGIME_RED_MAX_POSITIONS: int = 0       # No new positions in RED (exits only)

# Entry cooldown periods per regime (bars after exit before re-entry)
# Longer cooldowns during volatile regimes prevent overtrading
REGIME_GREEN_COOLDOWN_BARS: int = 4     # 1 hour cooldown in GREEN
REGIME_YELLOW_COOLDOWN_BARS: int = 12   # 3 hour cooldown in YELLOW
REGIME_RED_COOLDOWN_BARS: int = 96      # 1 day cooldown in RED (if somehow entered)

# =============================================================================
# KALMAN WARMUP WINDOW
# =============================================================================
# Use only recent data for Kalman initialization (avoid stale beta pollution)
ENABLE_KALMAN_WARMUP_WINDOW: bool = True  # Expert suggestion: avoid stale beta from old regimes
KALMAN_WARMUP_DAYS: int = 21
KALMAN_WARMUP_HALF_LIFE_MULT: float = 3.0  # Alternative: warmup = 3 * half_life
KALMAN_WARMUP_MIN_DAYS: int = 7
KALMAN_WARMUP_MAX_DAYS: int = 45

# Pair naming
PAIR_ID_SEPARATOR: str = "-"  # e.g., "ETH-BTC"

# =============================================================================
# 4. STRATEGY PARAMS ("THE KNOBS")
# =============================================================================

# Kalman Settings
KALMAN_DELTA: float = 1e-6
KALMAN_R: float = 1e-2

# Volatility window scaling # The vol window should be proportional to half-life for consistent z-score scaling
VOL_WINDOW_HALF_LIFE_MULT: float = 1.0  # vol_window = half_life * this multiplier
# LOW-FREQ: Adjusted for 1h bars instead of 15min
MIN_VOL_WINDOW: int = 24    # minimum 24 bars (1 day at 1h)
MAX_VOL_WINDOW: int = 360   # maximum 360 bars (15 days at 1h)

# Robust volatility estimation
VOL_METHOD: str = "ewma"       # {"std", "ewma", "mad"}
VOL_EWMA_ALPHA: float = 0.2    # EWMA smoothing for spread volatility
VOL_MAD_SCALE: float = 1.4826  # Convert MAD to sigma

# Signal quality filters (Priority 4)
# Regime change detection: suppress signals when Kalman gain spikes
ENABLE_REGIME_FILTER: bool = False  # Disabled (blocks entries on beta instability)
KALMAN_GAIN_THRESHOLD: float = 0.3  # Suppress signals when K[0] > this (beta unstable)
# Beta uncertainty: scale z-score down when P matrix indicates high uncertainty
ENABLE_BETA_UNCERTAINTY_SCALING: bool = True
MAX_BETA_UNCERTAINTY: float = 0.5  # P[0,0] above this triggers scaling

# Entry behavior
CROSS_REVERT_ENTRY: bool = False  # disabled - was filtering too many entries

# Entry/Exit Thresholds - Optimized for 15min mean-reversion
ENTRY_Z: float = 2.55         # Entry threshold (z-score standard deviations)
EXIT_Z: float = 0.5           # Exit threshold (reversion to mean)
STOP_LOSS_Z: float = 4.0      # Stop-loss threshold
MAX_ENTRY_Z: float = 6.0      # Maximum entry z-score
STOP_LOSS_PCT: float = 0.025  # 2.5% maximum loss per trade

# Profitability filter
EXPECTED_REVERT_MULT: float = 0.75
MIN_PROFIT_HURDLE: float = 0.012  # 1.2% minimum expected profit

# =============================================================================
# INFLECTION POINT DETECTION
# =============================================================================
# Wait for z-score to peak and start reverting before entering
# Prevents early entries while spread is still diverging
ENABLE_INFLECTION_FILTER: bool = True

# Minimum bars to wait after z crosses threshold before allowing entry
# LOW-FREQ: At 1h bars, wait 3 hours for confirmation
INFLECTION_MIN_BARS_SINCE_EXTREME: int = 3

# Maximum bars to wait (signal expires after this)
# LOW-FREQ: At 1h bars, allow up to 24 hours (1 day) for signal
INFLECTION_MAX_BARS_SINCE_EXTREME: int = 24

# Velocity must change by this much to confirm reversal
INFLECTION_VELOCITY_THRESHOLD: float = -0.06  # Moderate for quality

# Minimum confidence score to trigger entry (0-1)
INFLECTION_MIN_CONFIDENCE: float = 0.50  # Relaxed for more trades

# Use acceleration (2nd derivative) for stronger confirmation (experimental)
INFLECTION_USE_ACCELERATION: bool = False
INFLECTION_MIN_ACCELERATION: float = -0.02  # Must be decelerating

# =============================================================================
# IMPROVEMENT #5: VOLATILITY FILTER FOR ENTRIES
# =============================================================================
# Only enter trades when spread volatility is sufficient (mean-reversion opportunity)
# Low volatility = no opportunity to capture spread movements
ENABLE_VOLATILITY_FILTER: bool = True
MIN_SPREAD_VOLATILITY_BPS: float = 15.0  # Minimum spread vol in bps to enter
# Suppress entries during extreme volatility (regime breaks / flash crashes)
MAX_SPREAD_VOLATILITY_BPS: float = 500.0  # Maximum spread vol in bps to enter

# Option 4: Use OU (Ornstein-Uhlenbeck) model for expected profit
# This accounts for time-to-revert, transaction costs, and funding costs
USE_OU_MODEL: bool = True

# Pair selection filters
MIN_HALF_LIFE_BARS: int = 80       # Minimum 80 bars (20 hours at 15min)
MAX_TRADES_PER_PAIR: int = 15     # Allow more trades per pair
MAX_PAIRS_PER_COIN: int = 3       # Allow more pairs per coin
EXCLUDE_SYMBOLS: tuple = ()  # Optional symbol blacklist for universe pruning
ENABLE_ROLLING_COINT_CHECK: bool = True   # Enabled: validate cointegration stability
ENABLE_BETA_STABILITY_CHECK: bool = True  # Enabled: validate hedge ratio stability
ALLOW_SCAN_FALLBACK: bool = True
SCAN_P_VALUE_THRESHOLD: float = 0.03      # Moderately tightened (was 0.05)
SCAN_MAX_DRIFT_Z: float = 2.0             # Moderately tightened (was 2.5)
SCAN_MIN_HALF_LIFE: int = 80              # Minimum half-life (aligned with MIN_HALF_LIFE_BARS)
SCAN_MAX_HALF_LIFE: int = 720             # Moderately lowered (was 2000)

# Optional: validate universe against QuestDB data availability
VALIDATE_UNIVERSE: bool = False

# Time-based stop # Exit if trade hasn't reverted within TIME_STOP_HALF_LIFE_MULT × half_life bars.
# This prevents holding "dead" trades through regime changes.
# Set to 0 to disable time stops.
# Option 2 cont'd: Allow positions more time to work (reduce frequency)
TIME_STOP_HALF_LIFE_MULT: float = 4.0  # Raised from 3.0 - more time to mean-revert
DEFAULT_TIME_STOP_BARS: int = 2880     # Keep baseline 30-day stop

# =============================================================================
# FEE MODEL
# =============================================================================
# Fee model options:
# - "taker_only": All trades at taker rate (conservative, current default)
# - "maker_taker_mix": Blend of maker and taker based on fill probability
# - "maker_only": All trades at maker rate (optimistic)
# IMPROVEMENT #3: Use maker_taker_mix for more realistic fee assumptions
FEE_MODEL: str = "taker_only"  # Conservative: assume all taker fills
TAKER_FEE_RATE: float = 0.0005   # 5 bps (current exchange taker rate)
MAKER_FEE_RATE: float = 0.0002   # 2 bps (current exchange maker rate)
MAKER_FILL_PROBABILITY: float = 0.70  # 70% of limit orders fill at maker rate
# Effective fee with maker_taker_mix: 0.70 * 0.0002 + 0.30 * 0.0005 = 0.00029 = 2.9 bps

# DEPRECATED: Legacy FEE_RATE kept for backward compatibility only.
# New code should use FEE_MODEL system above (TAKER_FEE_RATE, MAKER_FEE_RATE).
FEE_RATE: float = TAKER_FEE_RATE  # Alias to TAKER_FEE_RATE

# Microstructure / Slippage hook (used in pnl_engine)
SLIPPAGE_MODEL: str = "fixed"   # {"fixed", "vol_adjusted"}
SLIPPAGE_BPS: float = 5.0       # Conservative slippage assumption
SLIPPAGE_RATE: float = SLIPPAGE_BPS / 10_000.0
SLIPPAGE_VOL_MULT: float = 1.0  # Baseline volatility multiplier
SLIPPAGE_CAP_BPS: float = 25.0  # safety cap
SLIPPAGE_CAP_RATE: float = SLIPPAGE_CAP_BPS / 10_000.0

# Adverse selection / market impact buffer
# Added to expected costs in OU profit calculation to account for:
# - Partial fills and legging risk
# - Unfavorable fill prices during volatile z-score spikes
# - Market impact of entering/exiting pairs
ADVERSE_SELECTION_BPS: float = 5.0  # Conservative buffer for market impact

# Capital assumptions
# pnl_engine returns (net_pnl / capital_per_pair).
CAPITAL_PER_PAIR: float = 1.0  # Capital allocation per pair
INIT_CASH: float = 1.0  # used for equity curves (normalized)
SCALE_CAPITAL_BY_MAX_POSITIONS: bool = False  # FIXED: Was dividing positions by 200x!

# Portfolio-level risk limits (align with execution)
MAX_PORTFOLIO_POSITIONS: int = 20  # Up from 8 for higher utilization
MAX_POSITIONS_PER_COIN: int = 4   # Up from 2 for more pairs per coin

# Position sizing normalization # When True, positions are sized so gross exposure = 1.0 regardless of beta:
#   w_Y = 1 / (1 + |beta|), w_X = |beta| / (1 + |beta|)
# This ensures consistent risk and scale-invariant returns across all pairs.
NORMALIZE_NOTIONAL: bool = True

# Performance report scenarios (daily rates)
# NOTE: These are applied per-bar when in a position. With pairs trading where
# positions are typically held for hours to days, the actual impact is much smaller.
FUNDING_DRAG_OPT_DAILY: float = 0.0
FUNDING_DRAG_BASE_DAILY: float = 0.0001   # 0.01% daily (realistic for perps)
FUNDING_DRAG_STRESS_DAILY: float = 0.0002 # 0.02% daily (reduced from 0.03% - too aggressive)

# Stress scenario: extra slippage applied per exit event (performance_report)
# This models additional market impact and adverse selection during volatile exits
STRESS_EXTRA_SLIPPAGE_PER_EXIT: float = 0.0003  # 3 bps per exit (reduced from 5 bps)

# Pass/fail thresholds (reporting gate)
SUCCESS_SHARPE_MIN: float = 1.5
SUCCESS_MAX_DD_DURATION_DAYS: int = 5
SUCCESS_BTC_CORR_MAX: float = 0.5

# =============================================================================
# P&L ATTRIBUTION # =============================================================================
# Enable detailed P&L component tracking for diagnostics
ENABLE_TRADE_ATTRIBUTION: bool = True
# Track funding costs per trade (requires funding rate data)
TRACK_FUNDING_PER_TRADE: bool = True
# Threshold for detecting loss concentration in attribution report
LOSS_CONCENTRATION_THRESHOLD: float = 0.5  # If top 2 pairs > 50% of losses

# =============================================================================
# CAPITAL UTILIZATION IMPROVEMENTS (ROI Optimization)
# =============================================================================

# --- Pair Quality Filtering ---
# Reject pairs with high cost-to-gross ratio (inefficient churning)
ENABLE_COST_GROSS_FILTER: bool = True
MAX_COST_TO_GROSS_RATIO: float = 0.80  # Reject pairs with cost > 80% of gross

# Penalize pairs with high stop-loss rate (regime breaks)
ENABLE_STOP_LOSS_PENALTY: bool = True
MAX_STOP_LOSS_RATE: float = 0.40  # Penalize pairs with >40% stop-loss exits

# Rolling pair quality tracking (ban pairs that perform poorly)
ENABLE_ROLLING_PAIR_QUALITY: bool = True
PAIR_QUALITY_LOOKBACK_TRADES: int = 10  # Look at last N trades for quality
MIN_PAIR_QUALITY_SCORE: float = 0.3  # Ban pairs below this score

# =============================================================================
# IMPROVEMENT #1: OUTLIER EVENT EXCLUSION
# =============================================================================
# Exclude known outlier events (flash crashes, black swans) from backtest
# These events skew performance and aren't representative of normal alpha
ENABLE_OUTLIER_EXCLUSION: bool = True
OUTLIER_EVENTS: list = [
    # Oct 2025 Flash Crash (Trump tariff announcement)
    {"start": "2025-10-10", "end": "2025-10-13", "name": "Oct 2025 Flash Crash"},
]

# =============================================================================
# PAIR QUALITY KILL SWITCH (Mid-Window Retirement)
# =============================================================================
# Aggressively retire pairs that show poor performance during a test window.
# Example: MERL-SOL with 8/15 stop losses, -115% contribution should be killed early.
ENABLE_PAIR_KILL_SWITCH: bool = True  # Expert suggestion: retire poor performers mid-window
KILL_SWITCH_MIN_TRADES: int = 6        # Minimum trades before evaluation
KILL_SWITCH_MAX_STOP_RATE: float = 0.50   # Kill if stop_loss_rate > 50%
KILL_SWITCH_MIN_EXPECTANCY: float = 0.0   # Kill if expectancy < 0
KILL_SWITCH_MIN_WIN_RATE: float = 0.25    # Kill if win_rate < 25%
KILL_SWITCH_EVAL_INTERVAL_BARS: int = 96  # Evaluate every N bars (96 = 1 day @ 15min)
KILL_SWITCH_MAX_CONSECUTIVE_STOPS: int = 3  # Kill after 3 consecutive stop-losses
KILL_SWITCH_MIN_AVG_RETURN_BPS: float = -50.0  # Kill if avg return < -0.5%
KILL_SWITCH_MIN_CUMULATIVE_PNL: float = -0.02  # Kill if cumulative P&L < -2%

# =============================================================================
# WINDOW CIRCUIT BREAKER # =============================================================================
# Auto-reduce risk when window performance deteriorates.
# This prevents windows like Window 5 & 11 from contributing excessive losses.
ENABLE_WINDOW_CIRCUIT_BREAKER: bool = True
CIRCUIT_BREAKER_MAX_WINDOW_LOSS: float = 0.005  # -0.5% triggers de-risk
CIRCUIT_BREAKER_MAX_STOP_RATE: float = 0.50     # 50% stop-loss rate triggers de-risk
CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES: int = 5  # 5 consecutive losses triggers de-risk
CIRCUIT_BREAKER_DE_RISK_POSITION_MULT: float = 0.5  # Reduce position sizes by 50%
CIRCUIT_BREAKER_DE_RISK_ENTRY_Z_ADD: float = 0.5    # Increase entry threshold by 0.5
CIRCUIT_BREAKER_DE_RISK_PROFIT_HURDLE_ADD: float = 0.01  # Add 1% to profit hurdle

# =============================================================================
# SYMBOL BLACKLIST # =============================================================================
# Track per-symbol performance across windows and blacklist repeat offenders.
# Example: ACE appeared in 3 of bottom 5 pairs - should be blacklisted.
# Disabled (redundant with cointegration filter)
ENABLE_SYMBOL_BLACKLIST: bool = False  # Disabled (redundant filter)
BLACKLIST_MIN_WINDOWS: int = 2               # Minimum windows before blacklisting
BLACKLIST_MIN_TRADES_PER_WINDOW: int = 5     # Minimum trades per window to count
BLACKLIST_MAX_AVG_LOSS_PCT: float = -0.003   # -0.3% avg loss per window triggers blacklist
BLACKLIST_MAX_STOP_RATE: float = 0.45        # 45% stop-loss rate triggers blacklist

# =============================================================================
# REGIME FILTER # =============================================================================
# Block entries during adverse market regimes (high BTC vol, high dispersion)
ENABLE_REGIME_FILTER: bool = False  # Disabled - slope filter does the work
REGIME_BTC_VOL_MAX_PERCENTILE: float = 0.85   # Block when BTC vol in top 15% (relaxed from 0.70)
REGIME_BTC_VOL_LOOKBACK_DAYS: int = 30
REGIME_DISPERSION_MAX_PERCENTILE: float = 0.90  # Block when dispersion in top 10% (relaxed from 0.80)
REGIME_DISPERSION_LOOKBACK_DAYS: int = 7
REGIME_SPREAD_VOL_MIN_BPS: float = 15.0       # Min spread vol for entry
REGIME_SPREAD_VOL_MAX_BPS: float = 200.0      # Max spread vol for entry (tighter than 500)

# =============================================================================
# SOFT REGIME GATING (3-state system)
# =============================================================================
# Replace hard regime block with 3-state soft gating:
#   GREEN: Trade normally (mult=1.0)
#   YELLOW: Trade smaller + stricter entry (mult=0.5, entry_z += 0.2)
#   RED: No new entries
ENABLE_SOFT_REGIME: bool = False  # Disabled for max capital deployment
# GREEN thresholds: Trade normally when below these (widened for more trades)
REGIME_GREEN_BTC_VOL_MAX: float = 0.92     # Widened from 0.85
REGIME_GREEN_DISPERSION_MAX: float = 0.95  # Widened from 0.90
# YELLOW thresholds: De-risked trading when between GREEN and YELLOW thresholds
REGIME_YELLOW_BTC_VOL_MAX: float = 0.97    # Widened from 0.92
REGIME_YELLOW_DISPERSION_MAX: float = 0.99 # Widened from 0.95
# YELLOW adjustments
REGIME_YELLOW_SIZE_MULT: float = 0.5      # Position size multiplier in YELLOW state
REGIME_YELLOW_ENTRY_Z_ADD: float = 0.2    # Entry z-score increase in YELLOW state

# =============================================================================
# PAIR SCORING # =============================================================================
# Multi-factor tradability scoring for pair selection
ENABLE_PAIR_SCORING: bool = True
PAIR_SCORE_MIN_COMPOSITE: float = 0.35        # Minimum composite score for selection (relaxed from 0.40)
PAIR_SCORE_RECENT_LOOKBACK_DAYS: int = 30     # Recent period for MR strength
PAIR_SCORE_ROLLING_WINDOW_BARS: int = 480     # Rolling window for stability metrics

# =============================================================================
# TOP-N PAIR SELECTION # =============================================================================
# Instead of absolute threshold, select top N pairs with safety floor
ENABLE_TOPN_PAIR_SELECTION: bool = True
PAIR_SELECTION_TOP_N: int = 25                # Take top N pairs by score
PAIR_SELECTION_MIN_FLOOR: float = 0.25        # Safety floor: reject pairs below this even in top N
MAX_TRADES_PER_PAIR_PER_WINDOW: int = 10      # Prevent single-pair churn

# =============================================================================
# OU MODEL V2 # =============================================================================
# Calibrated OU model with sanity clamping
# Disabled (dead code, never called)
USE_OU_MODEL_V2: bool = False  # Disabled (dead code)
OU_CALIBRATION_DISCOUNT: float = 0.5          # 50% haircut on raw OU estimates
OU_USE_ROLLING_HALF_LIFE: bool = True         # Use rolling half-life estimation
OU_ROLLING_WINDOW_DAYS: int = 14
OU_MAX_EXPECTED_HOLD_BARS: int = 168          # LOW-FREQ: 168h = 7 days max hold
OU_MIN_EXPECTED_PROFIT_PCT: float = 0.001     # Min expected profit (0.1%)
OU_MAX_EXPECTED_PROFIT_PCT: float = 0.05      # Max expected profit (5%)

# =============================================================================
# ENTRY COOLDOWN # =============================================================================
# Prevent re-entry within cooldown period after exit
ENABLE_ENTRY_COOLDOWN: bool = False  # Disabled (blocked 81% of entries)
ENTRY_COOLDOWN_BARS: int = 12                 # 3 hours at 15-min bars (reduced from 6h)

# =============================================================================
# SMART COOLDOWN # =============================================================================
# Different cooldowns based on exit type:
# - Signal exit: Shorter cooldown (relationship worked)
# - Stop-loss exit: Longer cooldown (relationship may be broken)
# Disabled (no significant impact)
ENABLE_SMART_COOLDOWN: bool = False  # Disabled (zero impact)
COOLDOWN_AFTER_SIGNAL_BARS: int = 4           # 1 hour at 15-min bars (minimal)
COOLDOWN_AFTER_STOP_LOSS_BARS: int = 12       # 3 hours at 15-min bars (reduced)

# =============================================================================
# ENTRY QUALITY FILTERS # =============================================================================
# Filter bad entry timing without widening stops
ENABLE_SLOPE_FILTER: bool = True              # Require z to be turning (not still expanding)
ENABLE_CONFIRMATION_FILTER: bool = False      # Disabled: conflicts with slope filter at high ENTRY_Z
CONFIRMATION_BARS: int = 2                    # Bars for confirmation filter

# =============================================================================
# SIGNAL QUALITY IMPROVEMENTS
# =============================================================================
# NOTE: Inflection point detection moved to line 328 (Option C Redesign)

# Stale signal rejection: reject entries if z has been extreme too long
# NOTE: Disabled - conflicts with high ENTRY_Z (3.3). By the time z reaches 3.3, it's often been high for a while.
ENABLE_STALE_SIGNAL_FILTER: bool = False
MAX_STALE_SIGNAL_BARS: int = 12               # Reject if z > entry_z for > 12 bars (3h at 15min)

# Expected profit threshold: only trade when OU predicts meaningful profit
MIN_OU_EXPECTED_PROFIT_PCT: float = 0.002     # 0.2% minimum expected profit (relaxed)

# =============================================================================
# EXPOSURE CONTROLLER # =============================================================================
# Systematic capital deployment to maintain target exposure
ENABLE_EXPOSURE_CONTROLLER: bool = True
TARGET_GROSS_EXPOSURE: float = 0.80           # Target 80% notional deployed (up from 60%)
MIN_PAIRS_FOR_FULL_SIZE: int = 3              # Reduced from 8 - full size with fewer pairs
UNDER_DIVERSIFIED_SCALE: float = 0.8          # Up from 0.5 - less penalty for few pairs
EXPOSURE_MAX_SCALE_UP: float = 3.0            # Maximum scale-up when below target (up from 1.5)
EXPOSURE_MIN_SCALE: float = 0.3               # Minimum scale (floor)

# =============================================================================
# GATING FUNNEL DIAGNOSTICS
# =============================================================================
# Enable entry funnel logging to identify which gate blocks most trades
ENABLE_ENTRY_FUNNEL_LOGGING: bool = True

# =============================================================================
# DIAGNOSTICS & EFFICIENCY IMPROVEMENTS
# =============================================================================

# --- 5A: Scanner Efficiency (Liquidity Pre-filtering) ---
# Filter coins by minimum daily volume before correlation/cointegration tests
ENABLE_LIQUIDITY_PREFILTER: bool = True
SCANNER_MIN_DAILY_VOLUME_USD: float = 1_000_000  # $1M minimum daily volume

# --- 5B: Window Analysis ---
# Compare winning vs losing windows to identify regime patterns
ENABLE_WINDOW_ANALYSIS: bool = True

# --- 5D: Hold Time Calibration Logging ---
# Compare expected hold time (from OU model) vs actual hold time for calibration
ENABLE_HOLD_TIME_LOGGING: bool = True
HOLD_TIME_WARNING_RATIO: float = 0.3  # Warn if actual/expected < 30%

# --- Continuous Exposure (Gradual Position Building) ---
# Instead of binary entry at ENTRY_Z, build position gradually
ENABLE_CONTINUOUS_EXPOSURE: bool = True
# Start building position at this z-score (lower than ENTRY_Z)
CONTINUOUS_ENTRY_START_Z: float = 1.5
# Full position at this z-score
CONTINUOUS_ENTRY_FULL_Z: float = 2.5
# Scaling function: size = clip(alpha * (|z| - z_start), 0, 1)
# where alpha = 1 / (z_full - z_start)

# --- Dynamic Entry Threshold ---
# Adjust entry threshold based on spread volatility and costs
ENABLE_DYNAMIC_ENTRY_Z: bool = True
# Base entry z-score (minimum)
DYNAMIC_ENTRY_Z_MIN: float = 1.3
# Scaling factor: entry_z = max(min, base + k * cost_bps / spread_vol_bps)
DYNAMIC_ENTRY_Z_BASE: float = 1.5
DYNAMIC_ENTRY_Z_COST_MULT: float = 0.5

# --- Higher Concurrency ---
# When ENABLE_HIGHER_CONCURRENCY is True, use the _HIGH values
ENABLE_HIGHER_CONCURRENCY: bool = True
# Raise max positions to maximize capital utilization
MAX_PORTFOLIO_POSITIONS_HIGH: int = 40  # Doubled for aggressive sizing
MAX_POSITIONS_PER_COIN_HIGH: int = 8    # Doubled for more pairs per coin
# Lower per-position cap to avoid concentration
MAX_SINGLE_POSITION_PCT: float = 0.25  # 25% max per position (up from 10%)

# Cluster/correlation cap to avoid loading on same theme
ENABLE_CLUSTER_CAP: bool = True
MAX_POSITIONS_PER_CLUSTER: int = 3  # Max 3 positions in same cluster
# Clusters are defined by shared coins (e.g., all SOL pairs in one cluster)

# =============================================================================
# FDR CONTROL & SUBWINDOW STABILITY # =============================================================================
# Enable Benjamini-Hochberg FDR correction on cointegration p-values
ENABLE_FDR_CONTROL: bool = True
# Target FDR level (expect this fraction of significant results to be false)
FDR_ALPHA: float = 0.05
# FDR method: "benjamini_hochberg" (less conservative) or "bonferroni" (very conservative)
FDR_METHOD: str = "benjamini_hochberg"

# Enable subwindow stability filter (cointegration must hold in ALL subwindows)
ENABLE_SUBWINDOW_STABILITY: bool = True
# Number of subwindows to test (e.g., 3 = thirds of training data)
N_SUBWINDOWS: int = 3
# Minimum fraction of subwindows that must pass (1.0 = all must pass)
# Expert suggestion: 0.67 (2 of 3) is often better for crypto - 1.0 too strict
SUBWINDOW_MIN_PASS_RATE: float = 0.67
# Maximum coefficient of variation for half-life across subwindows
SUBWINDOW_MAX_HALF_LIFE_CV: float = 0.5

# =============================================================================
# TRAINING WINDOW CONFIGURATION # =============================================================================
# Enable exponentially weighted cointegration estimation (recent data weighted more)
ENABLE_EXP_WEIGHTED_COINT: bool = True  # Expert suggestion: recent data should dominate in crypto
# Half-life for exponential weighting (in days) - recent 30 days weighted most
EXP_WEIGHT_HALF_LIFE_DAYS: float = 30.0

# =============================================================================
# KALMAN BETA VALIDATION # =============================================================================
# Validate that Kalman-implied spread is also stationary (not just OLS spread)
VALIDATE_KALMAN_SPREAD: bool = True
# Maximum allowed divergence between Kalman and OLS beta
MAX_BETA_DIVERGENCE: float = 0.3  # |kalman_beta - ols_beta| / |ols_beta|
# ADF threshold for Kalman spread stationarity check
KALMAN_SPREAD_ADF_THRESHOLD: float = 0.05

# =============================================================================
# CARRY FILTER # =============================================================================
# Enable carry-aware entry filter (reject if funding cost > expected profit)
# Disabled (no significant impact)
ENABLE_CARRY_FILTER: bool = False  # Disabled (zero impact)
# Required edge multiplier: expected_profit > this * expected_funding_cost
CARRY_FILTER_MULT: float = 1.5
# Prefer positive-carry direction when both directions are valid
PREFER_POSITIVE_CARRY: bool = True
# Days of funding history to use for estimation
FUNDING_LOOKBACK_DAYS: int = 7

# =============================================================================
# STRUCTURAL BREAK EXITS # =============================================================================
# Enable structural break detection for early exit
ENABLE_STRUCTURAL_BREAK_EXIT: bool = True
# Lookback period for structural break detection (bars)
STRUCTURAL_BREAK_LOOKBACK: int = 50
# ADF p-value threshold for rolling stationarity check (exit if p > this)
ROLLING_ADF_EXIT_THRESHOLD: float = 0.10
# Beta jump threshold (exit if |beta_change| / |beta| > this)
BETA_JUMP_THRESHOLD: float = 0.20
# Variance spike threshold (exit if recent_var / old_var > this)
VARIANCE_SPIKE_THRESHOLD: float = 2.0

# Enable two-layer stops (soft stop reduces position, hard stop exits)
ENABLE_TWO_LAYER_STOPS: bool = False  # Disabled by default - adds complexity
# Z-score for soft stop (reduce position size)
SOFT_STOP_Z: float = 2.5
# Position reduction factor at soft stop (0.5 = reduce to 50%)
SOFT_STOP_REDUCTION: float = 0.5
# Z-score for hard stop (full exit)
HARD_STOP_Z: float = 3.5

# =============================================================================
# POSITION SIZING HARDENING # =============================================================================
# Hard clamp on final position size (after all multipliers)
FINAL_MAX_SINGLE_POSITION: float = 0.40  # No position > 40% of capital (up from 25%)
FINAL_MAX_TOTAL_EXPOSURE: float = 1.5  # Allow 150% gross exposure (levered)

# Enable spread volatility-based risk parity
ENABLE_RISK_PARITY: bool = True
# Target volatility contribution per position (1/N for N positions)
TARGET_VOL_CONTRIBUTION: float = 0.125  # 1/8 for 8 max positions

# Concentration limits
MAX_HHI: float = 0.50  # Relaxed from 0.25 - allow more concentration
MIN_EFFECTIVE_POSITIONS: int = 2  # Reduced from 4 - allow fewer positions

# =============================================================================
# CONVICTION SIZING (Signal-strength based position scaling)
# =============================================================================
# When enabled, position size scales with signal_score (from pair selection).
# Higher conviction trades get larger allocations.
ENABLE_CONVICTION_SIZING: bool = False  # Set True to enable signal-based scaling
MIN_CONVICTION_SIZE: float = 0.5        # Minimum position size multiplier (low signal)
MAX_CONVICTION_SIZE: float = 1.0        # Maximum position size multiplier (high signal)
MIN_SIGNAL_SCORE: float = 0.65          # Signals below this get MIN_CONVICTION_SIZE

# Advanced sizing uses pre-computed position_size_multiplier from position_sizing module
# (includes correlation adjustment, non-linear conviction, volatility targeting)
ENABLE_ADVANCED_SIZING: bool = False

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
# 7. RUN FOLDER + ARTIFACT PATHS # =============================================================================

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
        "entries": run_dir / "entries.parquet",
        "exits": run_dir / "exits.parquet",
        "expected_profit": run_dir / "expected_profit.parquet",
        "test_prices": run_dir / "test_prices.parquet",
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
    "PATH_FUNDING_PARQUET",
    "PATH_TEST_DATA",
    "PATH_STATE",
    "USE_REAL_FUNDING",
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
    "WARN_DATA_GAP_MINS",
    "MAX_INTERPOLATE_MINS",
    "LOOKBACK_WINDOW",
    "SIGNAL_TIMEFRAME",
    "SUPPORTED_TIMEFRAMES",
    "PAIR_ID_SEPARATOR",
    # strategy knobs
    "KALMAN_DELTA",
    "KALMAN_R",
    "VOL_WINDOW_HALF_LIFE_MULT",
    "MIN_VOL_WINDOW",
    "MAX_VOL_WINDOW",
    "ENTRY_Z",
    "EXIT_Z",
    "STOP_LOSS_Z",
    "EXPECTED_REVERT_MULT",
    "MIN_PROFIT_HURDLE",
    "MIN_HALF_LIFE_BARS",
    "MAX_TRADES_PER_PAIR",
    "TIME_STOP_HALF_LIFE_MULT",
    "DEFAULT_TIME_STOP_BARS",
    "FEE_RATE",
    "SLIPPAGE_MODEL",
    "SLIPPAGE_BPS",
    "SLIPPAGE_RATE",
    "SLIPPAGE_VOL_MULT",
    "SLIPPAGE_CAP_BPS",
    "SLIPPAGE_CAP_RATE",
    "CAPITAL_PER_PAIR",
    "INIT_CASH",
    "NORMALIZE_NOTIONAL",
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
