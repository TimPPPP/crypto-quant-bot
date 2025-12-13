# src/backtest/accountant_filter.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.accountant_filter")


@dataclass(frozen=True)
class TradeMasks:
    """
    Outputs are aligned as:
      index  -> time (t)
      columns -> pair_id (e.g., "ETH-BTC")
    """
    entries: pd.DataFrame          # bool
    exits: pd.DataFrame            # bool
    expected_profit: pd.DataFrame  # float


def _cfg_get(name: str, default):
    """Get config value with a safe fallback, raise clean error if missing and no default."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if default is not None:
        return default
    raise AttributeError(f"Missing required config: cfg.{name}")


def compute_masks(
    z_score: pd.DataFrame,
    spread_volatility: pd.DataFrame,
    *,
    entry_z: Optional[float] = None,
    exit_z: Optional[float] = None,
    stop_loss_z: Optional[float] = None,
    min_profit_hurdle: Optional[float] = None,
    expected_revert_mult: Optional[float] = None,
) -> TradeMasks:
    """
    Step F: Accountant Filter (vectorized masks).

    expected_profit = spread_volatility * EXPECTED_REVERT_MULT

    entries = (abs(z) > ENTRY_Z) & (expected_profit > MIN_PROFIT_HURDLE)
    exits   = (abs(z) < EXIT_Z) | (abs(z) > STOP_LOSS_Z)

    Notes
    -----
    - This module does NOT enforce blocking. Blocking belongs to pnl_engine (Step G).
    - NaN policy:
        - entries NaN -> False (never enter on unknown signal)
        - exits NaN   -> False (exit logic is handled by pnl_engine’s “end of data” rule)
      If you want “risk-off” NaN exits, do it explicitly in pnl_engine when IN_TRADE.

    Returns
    -------
    TradeMasks with DataFrames aligned to z_score index/columns.
    """
    if z_score is None or spread_volatility is None:
        raise ValueError("z_score and spread_volatility must not be None.")
    if z_score.empty or spread_volatility.empty:
        raise ValueError("z_score and spread_volatility must be non-empty DataFrames.")

    # Alignment checks (critical for correctness)
    if not z_score.index.equals(spread_volatility.index):
        raise ValueError("Index mismatch: z_score and spread_volatility must share the same time index.")
    if not z_score.columns.equals(spread_volatility.columns):
        raise ValueError("Column mismatch: z_score and spread_volatility must share the same pair columns.")

    # Load knobs (with safe fallbacks where reasonable)
    entry_z = float(entry_z if entry_z is not None else _cfg_get("ENTRY_Z", None))
    exit_z = float(exit_z if exit_z is not None else _cfg_get("EXIT_Z", None))
    stop_loss_z = float(stop_loss_z if stop_loss_z is not None else _cfg_get("STOP_LOSS_Z", None))

    # These two may not exist yet in your config; provide defaults that match your plan.
    expected_revert_mult = float(
        expected_revert_mult if expected_revert_mult is not None else _cfg_get("EXPECTED_REVERT_MULT", 0.75)
    )
    min_profit_hurdle = float(
        min_profit_hurdle if min_profit_hurdle is not None else _cfg_get("MIN_PROFIT_HURDLE", 0.002)
    )

    expected_profit = spread_volatility * expected_revert_mult

    abs_z = z_score.abs()

    entries = (abs_z > entry_z) & (expected_profit > min_profit_hurdle)
    exits = (abs_z < exit_z) | (abs_z > stop_loss_z)

    # Strict boolean masks; never enter/exit on NaNs
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    logger.info(
        "Accountant masks computed: entries_true=%d exits_true=%d (pairs=%d, rows=%d)",
        int(entries.to_numpy().sum()),
        int(exits.to_numpy().sum()),
        entries.shape[1],
        entries.shape[0],
    )

    return TradeMasks(entries=entries, exits=exits, expected_profit=expected_profit)
