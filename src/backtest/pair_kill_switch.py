"""
src/backtest/pair_kill_switch.py

Enhanced Pair Kill Switch - Real-time pair disabling during test windows.

This module provides an enhanced kill switch that tracks per-pair performance
in real-time and disables underperforming pairs before they cause excessive losses.

Key improvements over basic PairQualityTracker:
- Consecutive stop-loss tracking (3+ consecutive stops = kill)
- Average return threshold (not just win rate)
- Real-time integration with PnL engine
- Detailed logging and diagnostics

Example usage:
    kill_switch = PairKillSwitch.from_config()

    # After each trade:
    if kill_switch.record_trade_and_evaluate(pair, trade_return, was_stop_loss):
        # Pair has been disabled
        logger.warning(f"Pair {pair} disabled by kill switch")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.pair_kill_switch")


@dataclass
class PairKillSwitch:
    """
    Track per-pair performance and disable underperformers in real-time.

    Prevents underperforming pairs from continuing to trade.

    Kill conditions (any triggers retirement):
    1. Win rate < min_win_rate after min_trades
    2. Consecutive stop-losses >= max_consecutive_stops
    3. Average return < min_avg_return_bps
    4. Cumulative P&L < min_cumulative_pnl
    """

    # Configuration
    min_trades_to_evaluate: int = 6
    max_consecutive_stops: int = 3
    min_win_rate: float = 0.35
    min_avg_return_bps: float = -50.0  # -0.5%
    min_cumulative_pnl: float = -0.02  # -2% cumulative

    # State tracking (per pair)
    trade_count: Dict[str, int] = field(default_factory=dict)
    win_count: Dict[str, int] = field(default_factory=dict)
    stop_loss_count: Dict[str, int] = field(default_factory=dict)
    consecutive_stops: Dict[str, int] = field(default_factory=dict)
    cumulative_return: Dict[str, float] = field(default_factory=dict)
    disabled_pairs: Set[str] = field(default_factory=set)

    # Diagnostics
    kill_reasons: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> "PairKillSwitch":
        """Create a PairKillSwitch using configuration parameters."""
        return cls(
            min_trades_to_evaluate=getattr(cfg, "KILL_SWITCH_MIN_TRADES", 6),
            max_consecutive_stops=getattr(cfg, "KILL_SWITCH_MAX_CONSECUTIVE_STOPS", 3),
            min_win_rate=getattr(cfg, "KILL_SWITCH_MIN_WIN_RATE", 0.25),
            min_avg_return_bps=getattr(cfg, "KILL_SWITCH_MIN_AVG_RETURN_BPS", -50.0),
            min_cumulative_pnl=getattr(cfg, "KILL_SWITCH_MIN_CUMULATIVE_PNL", -0.02),
        )

    @classmethod
    def create_for_pairs(cls, pairs: List[str], **kwargs) -> "PairKillSwitch":
        """Create a PairKillSwitch initialized for specific pairs."""
        instance = cls.from_config() if not kwargs else cls(**kwargs)
        for pair in pairs:
            instance.trade_count[pair] = 0
            instance.win_count[pair] = 0
            instance.stop_loss_count[pair] = 0
            instance.consecutive_stops[pair] = 0
            instance.cumulative_return[pair] = 0.0
        return instance

    def is_disabled(self, pair: str) -> bool:
        """Check if a pair is disabled."""
        return pair in self.disabled_pairs

    def record_trade(
        self,
        pair: str,
        trade_return: float,
        was_stop_loss: bool,
    ) -> None:
        """
        Record a completed trade for tracking.

        Parameters
        ----------
        pair : str
            Pair identifier (e.g., "ETH-BTC")
        trade_return : float
            Net return from the trade (as a decimal, e.g., 0.01 = 1%)
        was_stop_loss : bool
            Whether the trade was exited via stop-loss
        """
        if pair in self.disabled_pairs:
            return

        # Initialize if new pair
        if pair not in self.trade_count:
            self.trade_count[pair] = 0
            self.win_count[pair] = 0
            self.stop_loss_count[pair] = 0
            self.consecutive_stops[pair] = 0
            self.cumulative_return[pair] = 0.0

        # Update trade count and cumulative return
        self.trade_count[pair] += 1
        self.cumulative_return[pair] += trade_return

        # Update win count
        if trade_return > 0:
            self.win_count[pair] += 1
            self.consecutive_stops[pair] = 0  # Reset consecutive stops on any win

        # Update stop-loss tracking
        if was_stop_loss:
            self.stop_loss_count[pair] += 1
            self.consecutive_stops[pair] += 1
        elif trade_return <= 0:
            # Loss but not stop-loss - don't count towards consecutive stops
            # but do reset if it was a winning trade (handled above)
            pass

    def should_disable(self, pair: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a pair should be disabled.

        Returns
        -------
        Tuple[bool, Optional[str]]
            (should_disable, reason) - reason is None if not disabled
        """
        if pair in self.disabled_pairs:
            return False, None  # Already disabled

        n = self.trade_count.get(pair, 0)
        if n < self.min_trades_to_evaluate:
            return False, None  # Not enough data

        # Check consecutive stop-losses
        if self.consecutive_stops.get(pair, 0) >= self.max_consecutive_stops:
            return True, f"consecutive_stops ({self.consecutive_stops[pair]} >= {self.max_consecutive_stops})"

        # Check win rate
        win_rate = self.win_count.get(pair, 0) / n
        if win_rate < self.min_win_rate:
            return True, f"win_rate ({win_rate:.1%} < {self.min_win_rate:.1%})"

        # Check average return
        avg_return_bps = (self.cumulative_return.get(pair, 0) / n) * 10000
        if avg_return_bps < self.min_avg_return_bps:
            return True, f"avg_return ({avg_return_bps:.1f}bps < {self.min_avg_return_bps:.1f}bps)"

        # Check cumulative P&L
        if self.cumulative_return.get(pair, 0) < self.min_cumulative_pnl:
            return True, f"cumulative_pnl ({self.cumulative_return[pair]:.2%} < {self.min_cumulative_pnl:.2%})"

        return False, None

    def record_trade_and_evaluate(
        self,
        pair: str,
        trade_return: float,
        was_stop_loss: bool,
    ) -> bool:
        """
        Record a trade and check if the pair should be disabled.

        Convenience method that combines record_trade() and should_disable().

        Returns
        -------
        bool
            True if the pair was disabled after this trade
        """
        if pair in self.disabled_pairs:
            return False  # Already disabled

        self.record_trade(pair, trade_return, was_stop_loss)

        should_kill, reason = self.should_disable(pair)
        if should_kill:
            self.disable_pair(pair, reason)
            return True

        return False

    def disable_pair(self, pair: str, reason: Optional[str] = None) -> None:
        """
        Disable a pair from further trading.

        Parameters
        ----------
        pair : str
            Pair identifier
        reason : str, optional
            Reason for disabling (for diagnostics)
        """
        self.disabled_pairs.add(pair)
        if reason:
            self.kill_reasons[pair] = reason

        stats = self.get_pair_stats(pair)
        logger.warning(
            "Kill switch DISABLED pair %s: reason=%s, trades=%d, win_rate=%.1f%%, "
            "avg_return=%.2f%%, cumulative=%.2f%%, consecutive_stops=%d",
            pair,
            reason or "unknown",
            stats["trade_count"],
            stats["win_rate"] * 100,
            stats["avg_return_bps"] / 100,
            stats["cumulative_return"] * 100,
            stats["consecutive_stops"],
        )

    def get_pair_stats(self, pair: str) -> Dict[str, float]:
        """Get current statistics for a pair."""
        n = self.trade_count.get(pair, 0)
        if n == 0:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "stop_rate": 0.0,
                "avg_return_bps": 0.0,
                "cumulative_return": 0.0,
                "consecutive_stops": 0,
                "is_disabled": pair in self.disabled_pairs,
            }

        return {
            "trade_count": n,
            "win_rate": self.win_count.get(pair, 0) / n,
            "stop_rate": self.stop_loss_count.get(pair, 0) / n,
            "avg_return_bps": (self.cumulative_return.get(pair, 0) / n) * 10000,
            "cumulative_return": self.cumulative_return.get(pair, 0),
            "consecutive_stops": self.consecutive_stops.get(pair, 0),
            "is_disabled": pair in self.disabled_pairs,
        }

    def get_summary(self) -> Dict[str, any]:
        """Get summary of kill switch activity."""
        active_pairs = [p for p in self.trade_count if p not in self.disabled_pairs]
        return {
            "total_pairs_tracked": len(self.trade_count),
            "disabled_pairs": len(self.disabled_pairs),
            "active_pairs": len(active_pairs),
            "disabled_pair_list": list(self.disabled_pairs),
            "kill_reasons": self.kill_reasons.copy(),
        }

    def reset(self) -> None:
        """Reset all tracking state (for new window)."""
        self.trade_count.clear()
        self.win_count.clear()
        self.stop_loss_count.clear()
        self.consecutive_stops.clear()
        self.cumulative_return.clear()
        self.disabled_pairs.clear()
        self.kill_reasons.clear()


def filter_disabled_pairs(
    pairs: List[str],
    kill_switch: PairKillSwitch,
) -> List[str]:
    """
    Filter out disabled pairs from a list.

    Parameters
    ----------
    pairs : List[str]
        List of pair identifiers
    kill_switch : PairKillSwitch
        Kill switch instance

    Returns
    -------
    List[str]
        Pairs that are not disabled
    """
    return [p for p in pairs if not kill_switch.is_disabled(p)]
