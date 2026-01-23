"""
Configuration endpoints.
"""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter

from src.backtest import config_backtest as cfg

logger = logging.getLogger("api.config")
router = APIRouter()


@router.get("/config")
async def get_config() -> Dict:
    """
    Get current backtest configuration parameters.

    Returns all uppercase constants from config_backtest.
    """
    config_dict = {}

    # Extract all uppercase module-level variables
    for name in dir(cfg):
        if name.isupper() and not name.startswith("_"):
            value = getattr(cfg, name)
            # Skip non-serializable values
            if callable(value):
                continue
            # Convert Path to string
            if hasattr(value, "__fspath__"):
                value = str(value)
            config_dict[name] = value

    return config_dict


@router.get("/config/strategy")
async def get_strategy_params() -> Dict:
    """
    Get strategy-specific parameters (entry/exit thresholds, etc.).
    """
    return {
        "kalman": {
            "delta": cfg.KALMAN_DELTA,
            "r": cfg.KALMAN_R,
        },
        "thresholds": {
            "entry_z": cfg.ENTRY_Z,
            "exit_z": cfg.EXIT_Z,
            "stop_loss_z": cfg.STOP_LOSS_Z,
        },
        "filters": {
            "expected_revert_mult": cfg.EXPECTED_REVERT_MULT,
            "min_profit_hurdle": cfg.MIN_PROFIT_HURDLE,
            "min_half_life_bars": cfg.MIN_HALF_LIFE_BARS,
            "max_trades_per_pair": cfg.MAX_TRADES_PER_PAIR,
        },
        "costs": {
            "fee_model": cfg.FEE_MODEL,
            "taker_fee_rate": cfg.TAKER_FEE_RATE,
            "maker_fee_rate": cfg.MAKER_FEE_RATE,
            "maker_fill_probability": cfg.MAKER_FILL_PROBABILITY,
            "slippage_model": cfg.SLIPPAGE_MODEL,
            "slippage_bps": cfg.SLIPPAGE_BPS,
        },
        "funding": {
            "use_real_funding": cfg.USE_REAL_FUNDING,
            "funding_drag_base_daily": cfg.FUNDING_DRAG_BASE_DAILY,
            "funding_drag_stress_daily": cfg.FUNDING_DRAG_STRESS_DAILY,
        },
    }


@router.get("/config/paths")
async def get_paths() -> Dict:
    """
    Get configured file paths.
    """
    return {
        "project_root": str(cfg.PROJECT_ROOT),
        "data_dir": str(cfg.DATA_DIR),
        "results_dir": str(cfg.RESULTS_DIR),
        "raw_parquet": str(cfg.PATH_RAW_PARQUET),
        "funding_parquet": str(cfg.PATH_FUNDING_PARQUET),
    }
