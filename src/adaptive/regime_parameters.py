"""
src/adaptive/regime_parameters.py

Regime-Conditional Trading Parameters - Dynamic parameter adjustment by regime state.

This module provides a lookup interface for trading parameters that vary based on
the current market regime (GREEN/YELLOW/RED). Key features:

1. Entry Z-score thresholds: Stricter during adverse regimes
2. Position size multipliers: Reduced exposure in volatile periods
3. Max position limits: Fewer concurrent positions when risky
4. Cooldown periods: Longer waits between trades in adverse regimes

Example usage:
    params = RegimeParameters.from_config()

    # Get parameters for current regime
    entry_z = params.get_entry_z(RegimeState.YELLOW)  # Returns 3.0
    max_pos = params.get_max_positions(RegimeState.GREEN)  # Returns 8

    # Or use the unified getter
    regime_params = params.get_params(RegimeState.YELLOW)
    # Returns: {'entry_z': 3.0, 'size_mult': 0.5, 'max_positions': 4, 'cooldown_bars': 12}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.backtest import config_backtest as cfg
from src.adaptive.regime_filter import RegimeState

logger = logging.getLogger("adaptive.regime_parameters")


@dataclass
class RegimeParameterSet:
    """Parameters for a single regime state."""
    entry_z: float
    size_mult: float
    max_positions: int
    cooldown_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_z": self.entry_z,
            "size_mult": self.size_mult,
            "max_positions": self.max_positions,
            "cooldown_bars": self.cooldown_bars,
        }


@dataclass
class RegimeParameters:
    """
    Lookup table for regime-conditional trading parameters.

    Provides different parameter values based on current regime state:
    - GREEN: Normal trading with standard parameters
    - YELLOW: De-risked trading with stricter entry and smaller positions
    - RED: No new entries (exits only)

    Attributes
    ----------
    green : RegimeParameterSet
        Parameters for GREEN (normal) regime
    yellow : RegimeParameterSet
        Parameters for YELLOW (caution) regime
    red : RegimeParameterSet
        Parameters for RED (adverse) regime
    enabled : bool
        Whether regime-conditional parameters are enabled
    """

    green: RegimeParameterSet
    yellow: RegimeParameterSet
    red: RegimeParameterSet
    enabled: bool = True

    # Default parameters (used when regime-conditional is disabled)
    default_entry_z: float = 2.5
    default_size_mult: float = 1.0
    default_max_positions: int = 8
    default_cooldown_bars: int = 4

    @classmethod
    def from_config(cls) -> "RegimeParameters":
        """
        Create RegimeParameters from config_backtest settings.

        Reads REGIME_*_ENTRY_Z, REGIME_*_SIZE_MULT, REGIME_*_MAX_POSITIONS,
        and REGIME_*_COOLDOWN_BARS for each regime state.
        """
        enabled = bool(getattr(cfg, "ENABLE_REGIME_CONDITIONAL_PARAMS", False))

        # GREEN regime parameters
        green = RegimeParameterSet(
            entry_z=float(getattr(cfg, "REGIME_GREEN_ENTRY_Z", 2.5)),
            size_mult=float(getattr(cfg, "REGIME_GREEN_SIZE_MULT", 1.0)),
            max_positions=int(getattr(cfg, "REGIME_GREEN_MAX_POSITIONS", 8)),
            cooldown_bars=int(getattr(cfg, "REGIME_GREEN_COOLDOWN_BARS", 4)),
        )

        # YELLOW regime parameters
        yellow = RegimeParameterSet(
            entry_z=float(getattr(cfg, "REGIME_YELLOW_ENTRY_Z", 3.0)),
            size_mult=float(getattr(cfg, "REGIME_YELLOW_SIZE_MULT", 0.5)),
            max_positions=int(getattr(cfg, "REGIME_YELLOW_MAX_POSITIONS", 4)),
            cooldown_bars=int(getattr(cfg, "REGIME_YELLOW_COOLDOWN_BARS", 12)),
        )

        # RED regime parameters
        red = RegimeParameterSet(
            entry_z=float(getattr(cfg, "REGIME_RED_ENTRY_Z", 999.0)),
            size_mult=float(getattr(cfg, "REGIME_RED_SIZE_MULT", 0.0)),
            max_positions=int(getattr(cfg, "REGIME_RED_MAX_POSITIONS", 0)),
            cooldown_bars=int(getattr(cfg, "REGIME_RED_COOLDOWN_BARS", 96)),
        )

        # Default parameters (when regime-conditional is disabled)
        default_entry_z = float(getattr(cfg, "ENTRY_Z", 2.5))
        default_max_positions = int(getattr(cfg, "MAX_PORTFOLIO_POSITIONS", 8))
        default_cooldown_bars = int(getattr(cfg, "ENTRY_COOLDOWN_BARS", 4))

        instance = cls(
            green=green,
            yellow=yellow,
            red=red,
            enabled=enabled,
            default_entry_z=default_entry_z,
            default_max_positions=default_max_positions,
            default_cooldown_bars=default_cooldown_bars,
        )

        if enabled:
            logger.info(
                "Regime-conditional parameters enabled: "
                "GREEN(z=%.1f, pos=%d), YELLOW(z=%.1f, pos=%d), RED(z=%.1f, pos=%d)",
                green.entry_z, green.max_positions,
                yellow.entry_z, yellow.max_positions,
                red.entry_z, red.max_positions,
            )

        return instance

    def get_params(self, state: RegimeState) -> RegimeParameterSet:
        """Get full parameter set for a regime state."""
        if not self.enabled:
            return RegimeParameterSet(
                entry_z=self.default_entry_z,
                size_mult=self.default_size_mult,
                max_positions=self.default_max_positions,
                cooldown_bars=self.default_cooldown_bars,
            )

        if state == RegimeState.GREEN:
            return self.green
        elif state == RegimeState.YELLOW:
            return self.yellow
        else:  # RED
            return self.red

    def get_entry_z(self, state: RegimeState) -> float:
        """Get entry z-score threshold for a regime state."""
        return self.get_params(state).entry_z

    def get_size_mult(self, state: RegimeState) -> float:
        """Get position size multiplier for a regime state."""
        return self.get_params(state).size_mult

    def get_max_positions(self, state: RegimeState) -> int:
        """Get maximum portfolio positions for a regime state."""
        return self.get_params(state).max_positions

    def get_cooldown_bars(self, state: RegimeState) -> int:
        """Get entry cooldown (bars) for a regime state."""
        return self.get_params(state).cooldown_bars

    def get_entry_z_series(self, regime_states: "pd.Series") -> "pd.Series":
        """
        Convert a series of regime states to entry_z thresholds.

        Parameters
        ----------
        regime_states : pd.Series
            Series with RegimeState values or state strings ('green', 'yellow', 'red')

        Returns
        -------
        pd.Series
            Series of entry_z thresholds aligned with input index
        """
        import pandas as pd

        def state_to_entry_z(state):
            if isinstance(state, str):
                state = RegimeState(state)
            return self.get_entry_z(state)

        return regime_states.apply(state_to_entry_z)

    def get_max_positions_series(self, regime_states: "pd.Series") -> "pd.Series":
        """
        Convert a series of regime states to max_positions limits.

        Parameters
        ----------
        regime_states : pd.Series
            Series with RegimeState values or state strings

        Returns
        -------
        pd.Series
            Series of max_positions limits aligned with input index
        """
        import pandas as pd

        def state_to_max_pos(state):
            if isinstance(state, str):
                state = RegimeState(state)
            return self.get_max_positions(state)

        return regime_states.apply(state_to_max_pos)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters to dictionary."""
        return {
            "enabled": self.enabled,
            "green": self.green.to_dict(),
            "yellow": self.yellow.to_dict(),
            "red": self.red.to_dict(),
            "defaults": {
                "entry_z": self.default_entry_z,
                "size_mult": self.default_size_mult,
                "max_positions": self.default_max_positions,
                "cooldown_bars": self.default_cooldown_bars,
            },
        }


def create_regime_parameters() -> RegimeParameters:
    """Factory function to create RegimeParameters from config."""
    return RegimeParameters.from_config()
