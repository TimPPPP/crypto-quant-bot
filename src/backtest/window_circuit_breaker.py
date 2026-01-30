"""
src/backtest/window_circuit_breaker.py

Per-Window Risk Circuit Breaker - Auto-reduce risk when window performance deteriorates.

This module implements a circuit breaker that monitors performance during a test window
and automatically reduces risk exposure when losses accumulate or stop-loss rate spikes.

Key features:
- Tracks cumulative P&L within a window
- Monitors stop-loss rate
- Auto-triggers de-risk mode when thresholds breached
- Returns parameter overrides to reduce position sizes and tighten entry criteria

Example usage:
    circuit_breaker = WindowCircuitBreaker.from_config()

    # After each trade:
    overrides = circuit_breaker.update(trade_return, was_stop_loss)
    if overrides:
        # Apply de-risk parameters
        entry_z += overrides.get("entry_z_add", 0)
        position_mult *= overrides.get("position_mult", 1.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.circuit_breaker")


@dataclass
class WindowCircuitBreaker:
    """
    Auto-reduce risk when window performance deteriorates.

    Prevents underperforming windows from contributing excessive losses.

    De-risk triggers (any condition triggers):
    1. Window P&L < -max_window_loss_pct
    2. Stop-loss rate > max_stop_loss_rate
    3. Consecutive losing trades >= max_consecutive_losses

    De-risk actions:
    1. Reduce position sizes by de_risk_position_mult
    2. Increase entry z-score threshold by de_risk_entry_z_add
    3. Increase profit hurdle by de_risk_profit_hurdle_add
    """

    # Thresholds
    max_window_loss_pct: float = 0.005  # -0.5% triggers de-risk
    max_stop_loss_rate: float = 0.50    # 50% stop-loss rate triggers de-risk
    max_consecutive_losses: int = 5      # 5 consecutive losses triggers de-risk

    # De-risk parameters
    de_risk_position_mult: float = 0.5   # Reduce position sizes by 50%
    de_risk_entry_z_add: float = 0.5     # Increase entry threshold by 0.5
    de_risk_profit_hurdle_add: float = 0.01  # Add 1% to profit hurdle

    # State
    window_pnl: float = 0.0
    window_trades: int = 0
    window_stops: int = 0
    consecutive_losses: int = 0
    is_de_risked: bool = False
    de_risk_trigger: Optional[str] = None

    # History for diagnostics
    trade_history: list = field(default_factory=list)

    @classmethod
    def from_config(cls) -> "WindowCircuitBreaker":
        """Create a WindowCircuitBreaker using configuration parameters."""
        return cls(
            max_window_loss_pct=getattr(cfg, "CIRCUIT_BREAKER_MAX_WINDOW_LOSS", 0.005),
            max_stop_loss_rate=getattr(cfg, "CIRCUIT_BREAKER_MAX_STOP_RATE", 0.50),
            max_consecutive_losses=getattr(cfg, "CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES", 5),
            de_risk_position_mult=getattr(cfg, "CIRCUIT_BREAKER_DE_RISK_POSITION_MULT", 0.5),
            de_risk_entry_z_add=getattr(cfg, "CIRCUIT_BREAKER_DE_RISK_ENTRY_Z_ADD", 0.5),
            de_risk_profit_hurdle_add=getattr(cfg, "CIRCUIT_BREAKER_DE_RISK_PROFIT_HURDLE_ADD", 0.01),
        )

    def update(
        self,
        trade_return: float,
        was_stop_loss: bool,
        pair: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update state after a trade and return any parameter overrides.

        Parameters
        ----------
        trade_return : float
            Net return from the trade (as a decimal)
        was_stop_loss : bool
            Whether the trade was exited via stop-loss
        pair : str, optional
            Pair identifier (for logging)

        Returns
        -------
        Dict[str, Any]
            Parameter overrides if de-risked, empty dict otherwise.
            Keys: "position_mult", "entry_z_add", "profit_hurdle_add"
        """
        # Update state
        self.window_pnl += trade_return
        self.window_trades += 1

        if was_stop_loss:
            self.window_stops += 1

        if trade_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Track for diagnostics
        self.trade_history.append({
            "pair": pair,
            "return": trade_return,
            "was_stop": was_stop_loss,
            "cumulative_pnl": self.window_pnl,
        })

        # Check for de-risk triggers (only trigger once)
        if not self.is_de_risked:
            trigger = self._check_triggers()
            if trigger:
                self._activate_de_risk(trigger)
                return self._get_overrides()

        # Return overrides if already de-risked
        if self.is_de_risked:
            return self._get_overrides()

        return {}

    def _check_triggers(self) -> Optional[str]:
        """Check if any de-risk trigger has been hit."""
        # Check cumulative P&L
        if self.window_pnl < -self.max_window_loss_pct:
            return f"window_pnl ({self.window_pnl:.2%} < -{self.max_window_loss_pct:.2%})"

        # Check stop-loss rate (only after some trades)
        if self.window_trades >= 5:
            stop_rate = self.window_stops / self.window_trades
            if stop_rate > self.max_stop_loss_rate:
                return f"stop_rate ({stop_rate:.1%} > {self.max_stop_loss_rate:.1%})"

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return f"consecutive_losses ({self.consecutive_losses} >= {self.max_consecutive_losses})"

        return None

    def _activate_de_risk(self, trigger: str) -> None:
        """Activate de-risk mode."""
        self.is_de_risked = True
        self.de_risk_trigger = trigger

        logger.warning(
            "Circuit breaker TRIGGERED: %s | window_pnl=%.2f%%, trades=%d, stops=%d, stop_rate=%.1f%%",
            trigger,
            self.window_pnl * 100,
            self.window_trades,
            self.window_stops,
            (self.window_stops / self.window_trades * 100) if self.window_trades > 0 else 0,
        )

    def _get_overrides(self) -> Dict[str, Any]:
        """Get parameter overrides for de-risk mode."""
        return {
            "position_mult": self.de_risk_position_mult,
            "entry_z_add": self.de_risk_entry_z_add,
            "profit_hurdle_add": self.de_risk_profit_hurdle_add,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics."""
        stop_rate = self.window_stops / self.window_trades if self.window_trades > 0 else 0.0
        return {
            "window_pnl": self.window_pnl,
            "window_trades": self.window_trades,
            "window_stops": self.window_stops,
            "stop_rate": stop_rate,
            "consecutive_losses": self.consecutive_losses,
            "is_de_risked": self.is_de_risked,
            "de_risk_trigger": self.de_risk_trigger,
        }

    def reset(self) -> None:
        """Reset state for a new window."""
        self.window_pnl = 0.0
        self.window_trades = 0
        self.window_stops = 0
        self.consecutive_losses = 0
        self.is_de_risked = False
        self.de_risk_trigger = None
        self.trade_history.clear()


@dataclass
class AdaptiveCircuitBreaker(WindowCircuitBreaker):
    """
    Extended circuit breaker with adaptive thresholds based on market conditions.

    This version adjusts thresholds based on:
    - Recent market volatility
    - Number of active pairs
    - Historical window performance
    """

    # Adaptive factors
    vol_adjustment_factor: float = 1.0  # Multiply thresholds by this when vol is high
    pair_count_threshold: int = 10      # Below this, tighten thresholds

    def adjust_for_conditions(
        self,
        current_volatility: float,
        baseline_volatility: float,
        active_pair_count: int,
    ) -> None:
        """
        Adjust thresholds based on current market conditions.

        Parameters
        ----------
        current_volatility : float
            Current market volatility (e.g., BTC 30-day realized vol)
        baseline_volatility : float
            Baseline volatility for comparison
        active_pair_count : int
            Number of active pairs in the window
        """
        # Adjust for volatility: tighter thresholds when vol is high
        vol_ratio = current_volatility / baseline_volatility if baseline_volatility > 0 else 1.0
        if vol_ratio > 1.5:
            self.vol_adjustment_factor = 0.7  # Tighter by 30%
        elif vol_ratio > 1.2:
            self.vol_adjustment_factor = 0.85  # Tighter by 15%
        else:
            self.vol_adjustment_factor = 1.0

        # Adjust for pair count: tighter when fewer pairs (less diversification)
        if active_pair_count < self.pair_count_threshold:
            pair_factor = active_pair_count / self.pair_count_threshold
            self.max_window_loss_pct *= pair_factor

        logger.info(
            "Circuit breaker adjusted: vol_factor=%.2f, max_loss=%.2f%%, pairs=%d",
            self.vol_adjustment_factor,
            self.max_window_loss_pct * 100,
            active_pair_count,
        )
