"""
src/adaptive/regime_filter.py

Regime Gating Filter - Block entries during adverse market regimes.

This module implements a regime filter that blocks new entries when market
conditions are unfavorable for pairs trading:
- High BTC volatility periods (market chaos)
- High cross-sectional dispersion (pairs breaking down)
- Spread volatility outside sweet spot

Key insight from analysis: Strategy has 0.759 correlation to BTC equity curve
despite 0 return correlation. Drawdowns align with crypto volatility regimes.

Example usage:
    regime_filter = RegimeFilter.from_config()
    regime_mask = regime_filter.compute_regime_mask(
        btc_prices=test_df["BTC"],
        all_returns=test_df.pct_change(),
        spread_vol=spread_volatility_df,
    )
    entries = entries & regime_mask
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg

logger = logging.getLogger("adaptive.regime_filter")


# =============================================================================
# SOFT REGIME GATING (3-state system)
# =============================================================================

class RegimeState(Enum):
    """3-state regime classification for soft gating."""
    GREEN = "green"    # Trade normally (mult=1.0)
    YELLOW = "yellow"  # Trade smaller + stricter entry (mult=0.5, entry_z += 0.2)
    RED = "red"        # No new entries (but allow managed exits)


@dataclass
class SoftRegimeResult:
    """Result from soft regime gating computation."""
    state: RegimeState
    size_multiplier: float
    entry_z_adjustment: float
    btc_vol_pct: float
    dispersion_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "size_multiplier": self.size_multiplier,
            "entry_z_adjustment": self.entry_z_adjustment,
            "btc_vol_pct": self.btc_vol_pct,
            "dispersion_pct": self.dispersion_pct,
        }


@dataclass
class RegimeFilter:
    """
    Filter entries based on market regime indicators.

    Filter entries during market regimes where pairs trading fails.

    Regime indicators:
    1. BTC realized volatility (blocks when in top 30%)
    2. Cross-sectional dispersion (blocks when in top 20%)
    3. Spread volatility sweet spot (blocks outside 15-200 bps range)

    Soft Regime Gating (3-state):
    - GREEN: Trade normally (mult=1.0)
    - YELLOW: Trade smaller + stricter entry (mult=0.5, entry_z += 0.2)
    - RED: No new entries
    """

    # BTC volatility thresholds (hard block mode)
    btc_vol_max_percentile: float = 0.70  # Block when BTC vol in top 30%
    btc_vol_lookback_days: int = 30

    # Cross-sectional dispersion (market chaos indicator)
    dispersion_max_percentile: float = 0.80  # Block when dispersion in top 20%
    dispersion_lookback_days: int = 7

    # Spread volatility sweet spot
    spread_vol_min_bps: float = 15.0
    spread_vol_max_bps: float = 200.0  # Tighter than original 500 bps

    # Bars per day (for lookback calculations)
    bars_per_day: int = 96

    # ===== SOFT REGIME GATING (3-state) =====
    enable_soft_regime: bool = False
    # GREEN thresholds: Trade normally
    green_btc_vol_max: float = 0.85
    green_dispersion_max: float = 0.90
    # YELLOW thresholds: Trade smaller, stricter entry
    yellow_btc_vol_max: float = 0.92
    yellow_dispersion_max: float = 0.95
    # YELLOW adjustments
    yellow_size_mult: float = 0.5
    yellow_entry_z_add: float = 0.2

    # Diagnostics
    block_stats: Dict[str, int] = field(default_factory=dict)
    soft_regime_stats: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> "RegimeFilter":
        """Create a RegimeFilter using configuration parameters."""
        return cls(
            btc_vol_max_percentile=getattr(cfg, "REGIME_BTC_VOL_MAX_PERCENTILE", 0.70),
            btc_vol_lookback_days=getattr(cfg, "REGIME_BTC_VOL_LOOKBACK_DAYS", 30),
            dispersion_max_percentile=getattr(cfg, "REGIME_DISPERSION_MAX_PERCENTILE", 0.80),
            dispersion_lookback_days=getattr(cfg, "REGIME_DISPERSION_LOOKBACK_DAYS", 7),
            spread_vol_min_bps=getattr(cfg, "REGIME_SPREAD_VOL_MIN_BPS", 15.0),
            spread_vol_max_bps=getattr(cfg, "REGIME_SPREAD_VOL_MAX_BPS", 200.0),
            bars_per_day=96 if getattr(cfg, "BAR_FREQ", "15min") == "15min" else 1440,
            # Soft regime gating config
            enable_soft_regime=getattr(cfg, "ENABLE_SOFT_REGIME", False),
            green_btc_vol_max=getattr(cfg, "REGIME_GREEN_BTC_VOL_MAX", 0.85),
            green_dispersion_max=getattr(cfg, "REGIME_GREEN_DISPERSION_MAX", 0.90),
            yellow_btc_vol_max=getattr(cfg, "REGIME_YELLOW_BTC_VOL_MAX", 0.92),
            yellow_dispersion_max=getattr(cfg, "REGIME_YELLOW_DISPERSION_MAX", 0.95),
            yellow_size_mult=getattr(cfg, "REGIME_YELLOW_SIZE_MULT", 0.5),
            yellow_entry_z_add=getattr(cfg, "REGIME_YELLOW_ENTRY_Z_ADD", 0.2),
        )

    def compute_regime_mask(
        self,
        btc_prices: pd.Series,
        all_returns: pd.DataFrame,
        spread_vol: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute regime mask: True = OK to trade, False = regime block.

        Parameters
        ----------
        btc_prices : pd.Series
            BTC price series (same index as spread_vol)
        all_returns : pd.DataFrame
            Returns for all coins (for dispersion calculation)
        spread_vol : pd.DataFrame
            Spread volatility per pair (index=time, columns=pairs)

        Returns
        -------
        pd.DataFrame
            Boolean mask with same shape as spread_vol
        """
        # Reset block stats
        self.block_stats = {
            "btc_vol": 0,
            "dispersion": 0,
            "spread_vol_low": 0,
            "spread_vol_high": 0,
            "total_bars": len(spread_vol),
            "total_cells": spread_vol.size,
        }

        # 1. BTC realized volatility filter
        btc_ok = self._compute_btc_vol_mask(btc_prices)

        # 2. Cross-sectional dispersion filter
        dispersion_ok = self._compute_dispersion_mask(all_returns)

        # 3. Spread volatility sweet spot (per-pair)
        spread_vol_ok = self._compute_spread_vol_mask(spread_vol)

        # Combine: need all conditions to pass
        # Broadcast btc_ok and dispersion_ok to match spread_vol shape
        regime_mask = spread_vol_ok.copy()
        for col in regime_mask.columns:
            # Align indices
            col_mask = spread_vol_ok[col]
            btc_aligned = btc_ok.reindex(col_mask.index, fill_value=True)
            disp_aligned = dispersion_ok.reindex(col_mask.index, fill_value=True)
            regime_mask[col] = col_mask & btc_aligned & disp_aligned

        # Log summary
        total_blocked = (~regime_mask).sum().sum()
        block_pct = total_blocked / regime_mask.size * 100 if regime_mask.size > 0 else 0

        logger.info(
            "Regime filter: blocked %.1f%% of entries | btc_vol=%d, dispersion=%d, "
            "spread_low=%d, spread_high=%d",
            block_pct,
            self.block_stats["btc_vol"],
            self.block_stats["dispersion"],
            self.block_stats["spread_vol_low"],
            self.block_stats["spread_vol_high"],
        )

        return regime_mask

    def _compute_btc_vol_mask(self, btc_prices: pd.Series) -> pd.Series:
        """
        Compute BTC volatility mask.

        Blocks entries when BTC realized volatility is in top percentile.
        """
        # Compute returns
        btc_returns = btc_prices.pct_change()

        # Compute rolling realized volatility (annualized)
        lookback_bars = self.btc_vol_lookback_days * self.bars_per_day
        btc_vol = btc_returns.rolling(lookback_bars, min_periods=lookback_bars // 2).std()
        btc_vol_annualized = btc_vol * np.sqrt(self.bars_per_day * 365)

        # Compute percentile rank within the series
        btc_vol_pct = btc_vol_annualized.rank(pct=True)

        # Block when volatility is in top percentile
        btc_ok = btc_vol_pct <= self.btc_vol_max_percentile

        # Track blocked bars
        self.block_stats["btc_vol"] = (~btc_ok).sum()

        return btc_ok

    def _compute_dispersion_mask(self, all_returns: pd.DataFrame) -> pd.Series:
        """
        Compute cross-sectional dispersion mask.

        High dispersion = coins moving independently = pairs breaking down.
        """
        # Cross-sectional standard deviation at each timestamp
        cross_dispersion = all_returns.std(axis=1)

        # Rolling normalized dispersion (compare to recent history)
        lookback_bars = self.dispersion_lookback_days * self.bars_per_day

        def normalize_dispersion(x):
            if len(x) < 2:
                return 0.5
            min_val = x.min()
            max_val = x.max()
            if max_val - min_val < 1e-10:
                return 0.5
            return (x.iloc[-1] - min_val) / (max_val - min_val)

        dispersion_pct = cross_dispersion.rolling(lookback_bars, min_periods=lookback_bars // 4).apply(
            normalize_dispersion, raw=False
        )

        # Block when dispersion is in top percentile
        dispersion_ok = dispersion_pct <= self.dispersion_max_percentile

        # Fill NaN with True (allow trading during warmup)
        dispersion_ok = dispersion_ok.fillna(True)

        # Track blocked bars
        self.block_stats["dispersion"] = (~dispersion_ok).sum()

        return dispersion_ok

    def _compute_spread_vol_mask(self, spread_vol: pd.DataFrame) -> pd.DataFrame:
        """
        Compute spread volatility sweet spot mask.

        Blocks when spread volatility is outside the tradable range:
        - Too low: no opportunity to capture spread movements
        - Too high: regime breaks, flash crashes
        """
        # Convert to bps
        spread_vol_bps = spread_vol * 10000

        # Check sweet spot
        vol_ok_low = spread_vol_bps >= self.spread_vol_min_bps
        vol_ok_high = spread_vol_bps <= self.spread_vol_max_bps

        spread_vol_ok = vol_ok_low & vol_ok_high

        # Track blocked cells
        self.block_stats["spread_vol_low"] = (~vol_ok_low).sum().sum()
        self.block_stats["spread_vol_high"] = (~vol_ok_high).sum().sum()

        return spread_vol_ok

    # =========================================================================
    # SOFT REGIME GATING METHODS (3-state)
    # =========================================================================

    def compute_soft_regime_state(
        self,
        btc_vol_pct: float,
        dispersion_pct: float,
    ) -> SoftRegimeResult:
        """
        Compute regime state for a single bar using 3-state soft gating.

        Parameters
        ----------
        btc_vol_pct : float
            BTC volatility percentile (0-1)
        dispersion_pct : float
            Cross-sectional dispersion percentile (0-1)

        Returns
        -------
        SoftRegimeResult
            Contains state, size_multiplier, entry_z_adjustment
        """
        # GREEN: Normal trading
        if btc_vol_pct <= self.green_btc_vol_max and dispersion_pct <= self.green_dispersion_max:
            return SoftRegimeResult(
                state=RegimeState.GREEN,
                size_multiplier=1.0,
                entry_z_adjustment=0.0,
                btc_vol_pct=btc_vol_pct,
                dispersion_pct=dispersion_pct,
            )

        # YELLOW: De-risked trading
        if btc_vol_pct <= self.yellow_btc_vol_max and dispersion_pct <= self.yellow_dispersion_max:
            return SoftRegimeResult(
                state=RegimeState.YELLOW,
                size_multiplier=self.yellow_size_mult,
                entry_z_adjustment=self.yellow_entry_z_add,
                btc_vol_pct=btc_vol_pct,
                dispersion_pct=dispersion_pct,
            )

        # RED: No new entries
        return SoftRegimeResult(
            state=RegimeState.RED,
            size_multiplier=0.0,
            entry_z_adjustment=0.0,  # N/A when blocked
            btc_vol_pct=btc_vol_pct,
            dispersion_pct=dispersion_pct,
        )

    def compute_soft_regime_series(
        self,
        btc_prices: pd.Series,
        all_returns: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute soft regime series for all bars.

        Parameters
        ----------
        btc_prices : pd.Series
            BTC price series
        all_returns : pd.DataFrame
            Returns for all coins

        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (state_series, size_mult_series, entry_z_adj_series)
        """
        # Reset soft regime stats
        self.soft_regime_stats = {"green": 0, "yellow": 0, "red": 0}

        # Compute BTC vol percentile series
        btc_returns = btc_prices.pct_change()
        lookback_bars = self.btc_vol_lookback_days * self.bars_per_day
        btc_vol = btc_returns.rolling(lookback_bars, min_periods=lookback_bars // 2).std()
        btc_vol_pct = btc_vol.rank(pct=True)

        # Compute dispersion percentile series
        cross_dispersion = all_returns.std(axis=1)
        disp_lookback = self.dispersion_lookback_days * self.bars_per_day

        def normalize_dispersion(x):
            if len(x) < 2:
                return 0.5
            min_val, max_val = x.min(), x.max()
            if max_val - min_val < 1e-10:
                return 0.5
            return (x.iloc[-1] - min_val) / (max_val - min_val)

        dispersion_pct = cross_dispersion.rolling(
            disp_lookback, min_periods=disp_lookback // 4
        ).apply(normalize_dispersion, raw=False)
        dispersion_pct = dispersion_pct.fillna(0.5)  # Default to middle

        # Compute state for each bar
        state_list = []
        size_mult_list = []
        entry_z_adj_list = []

        for idx in btc_vol_pct.index:
            bv = btc_vol_pct.loc[idx] if idx in btc_vol_pct.index else 0.5
            dp = dispersion_pct.loc[idx] if idx in dispersion_pct.index else 0.5

            if pd.isna(bv):
                bv = 0.5
            if pd.isna(dp):
                dp = 0.5

            result = self.compute_soft_regime_state(bv, dp)
            state_list.append(result.state.value)
            size_mult_list.append(result.size_multiplier)
            entry_z_adj_list.append(result.entry_z_adjustment)

            # Track stats
            self.soft_regime_stats[result.state.value] += 1

        state_series = pd.Series(state_list, index=btc_vol_pct.index, name="regime_state")
        size_mult_series = pd.Series(size_mult_list, index=btc_vol_pct.index, name="size_multiplier")
        entry_z_adj_series = pd.Series(entry_z_adj_list, index=btc_vol_pct.index, name="entry_z_adjustment")

        # Log summary
        total_bars = len(state_series)
        green_pct = self.soft_regime_stats["green"] / max(total_bars, 1) * 100
        yellow_pct = self.soft_regime_stats["yellow"] / max(total_bars, 1) * 100
        red_pct = self.soft_regime_stats["red"] / max(total_bars, 1) * 100
        logger.info(
            "Soft regime: GREEN=%.1f%%, YELLOW=%.1f%%, RED=%.1f%% of bars",
            green_pct, yellow_pct, red_pct,
        )

        return state_series, size_mult_series, entry_z_adj_series

    def compute_soft_regime_mask(
        self,
        btc_prices: pd.Series,
        all_returns: pd.DataFrame,
        spread_vol: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute soft regime mask with size multipliers and entry_z adjustments.

        This is an alternative to compute_regime_mask that provides:
        - entry_allowed: Boolean mask (False for RED, True for GREEN/YELLOW)
        - size_multiplier: Per-bar size multiplier (1.0 for GREEN, 0.5 for YELLOW, 0.0 for RED)
        - entry_z_adjustment: Per-bar entry_z adjustment (0.0 for GREEN, 0.2 for YELLOW)

        Parameters
        ----------
        btc_prices : pd.Series
            BTC price series
        all_returns : pd.DataFrame
            Returns for all coins
        spread_vol : pd.DataFrame
            Spread volatility per pair

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (entry_allowed, size_multiplier, entry_z_adjustment) DataFrames
        """
        # Get soft regime series
        state_series, size_mult_series, entry_z_adj_series = self.compute_soft_regime_series(
            btc_prices, all_returns
        )

        # Also apply spread volatility filter
        spread_vol_ok = self._compute_spread_vol_mask(spread_vol)

        # Create masks matching spread_vol shape
        entry_allowed = spread_vol_ok.copy()
        size_multiplier = spread_vol.copy()
        entry_z_adjustment = spread_vol.copy()

        for col in spread_vol.columns:
            col_spread_ok = spread_vol_ok[col]

            # Align soft regime series to column index
            state_aligned = state_series.reindex(col_spread_ok.index, fill_value="green")
            size_aligned = size_mult_series.reindex(col_spread_ok.index, fill_value=1.0)
            entry_z_aligned = entry_z_adj_series.reindex(col_spread_ok.index, fill_value=0.0)

            # Entry allowed = spread_vol OK AND not RED
            is_red = state_aligned == "red"
            entry_allowed[col] = col_spread_ok & ~is_red

            # Size multiplier (0 for spread_vol blocked or RED)
            size_multiplier[col] = np.where(col_spread_ok & ~is_red, size_aligned, 0.0)

            # Entry Z adjustment (only matters when allowed)
            entry_z_adjustment[col] = entry_z_aligned

        return entry_allowed, size_multiplier, entry_z_adjustment

    def get_regime_summary(
        self,
        btc_prices: pd.Series,
        all_returns: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get summary of current regime indicators.

        Useful for diagnostics and understanding when regime blocked.
        """
        # BTC volatility
        btc_returns = btc_prices.pct_change()
        lookback_bars = self.btc_vol_lookback_days * self.bars_per_day
        btc_vol = btc_returns.rolling(lookback_bars).std() * np.sqrt(self.bars_per_day * 365)
        btc_vol_current = btc_vol.iloc[-1] if len(btc_vol) > 0 else np.nan
        btc_vol_pct = btc_vol.rank(pct=True).iloc[-1] if len(btc_vol) > 0 else np.nan

        # Dispersion
        cross_disp = all_returns.std(axis=1)
        disp_current = cross_disp.iloc[-1] if len(cross_disp) > 0 else np.nan

        return {
            "btc_vol_annualized": btc_vol_current,
            "btc_vol_percentile": btc_vol_pct,
            "btc_vol_blocked": btc_vol_pct > self.btc_vol_max_percentile if not np.isnan(btc_vol_pct) else False,
            "cross_dispersion": disp_current,
            "regime_assessment": self._assess_regime(btc_vol_pct, disp_current),
        }

    def _assess_regime(
        self,
        btc_vol_pct: float,
        dispersion: float,
    ) -> str:
        """Assess overall regime quality."""
        if np.isnan(btc_vol_pct):
            return "unknown"

        if btc_vol_pct > 0.9:
            return "high_risk"
        elif btc_vol_pct > self.btc_vol_max_percentile:
            return "elevated_risk"
        elif btc_vol_pct < 0.3:
            return "low_vol"
        else:
            return "normal"


@dataclass
class AdaptiveRegimeFilter(RegimeFilter):
    """
    Extended regime filter with adaptive thresholds.

    Adjusts thresholds based on historical performance during different regimes.
    """

    # Historical regime performance tracking
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    def update_regime_performance(
        self,
        regime_type: str,
        window_pnl: float,
        n_trades: int,
    ) -> None:
        """
        Update performance tracking for a regime type.

        Parameters
        ----------
        regime_type : str
            One of "high_risk", "elevated_risk", "low_vol", "normal"
        window_pnl : float
            P&L from the window
        n_trades : int
            Number of trades in the window
        """
        if regime_type not in self.regime_performance:
            self.regime_performance[regime_type] = {
                "total_pnl": 0.0,
                "n_windows": 0,
                "n_trades": 0,
            }

        self.regime_performance[regime_type]["total_pnl"] += window_pnl
        self.regime_performance[regime_type]["n_windows"] += 1
        self.regime_performance[regime_type]["n_trades"] += n_trades

    def adapt_thresholds(self) -> None:
        """
        Adapt thresholds based on historical performance.

        Tightens thresholds for regime types that historically underperformed.
        """
        for regime_type, perf in self.regime_performance.items():
            if perf["n_windows"] < 3:
                continue  # Not enough data

            avg_pnl = perf["total_pnl"] / perf["n_windows"]

            if regime_type == "elevated_risk" and avg_pnl < -0.005:
                # Tighten BTC vol threshold
                self.btc_vol_max_percentile = max(0.5, self.btc_vol_max_percentile - 0.05)
                logger.info(
                    "Adapted BTC vol threshold to %.2f based on elevated_risk performance",
                    self.btc_vol_max_percentile,
                )

            if regime_type == "high_risk" and avg_pnl < -0.01:
                # Much tighter threshold
                self.btc_vol_max_percentile = max(0.4, self.btc_vol_max_percentile - 0.10)
                logger.info(
                    "Adapted BTC vol threshold to %.2f based on high_risk performance",
                    self.btc_vol_max_percentile,
                )


def create_regime_filter() -> RegimeFilter:
    """Factory function to create appropriate regime filter."""
    return RegimeFilter.from_config()
