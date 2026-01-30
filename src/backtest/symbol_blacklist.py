"""
src/backtest/symbol_blacklist.py

Cross-Window Symbol Blacklist - Track and blacklist consistently underperforming symbols.

This module tracks per-symbol performance across multiple walk-forward windows
and automatically blacklists symbols that consistently underperform.

Key features:
- Tracks per-symbol P&L, trades, and stop-loss rates across windows
- Identifies repeat offenders (e.g., ACE appearing in 3 of bottom 5 pairs)
- Maintains blacklist that persists across windows
- Exports blacklist data for analysis

Example usage:
    blacklist = SymbolBlacklist.from_config()

    # After each window:
    blacklist.record_window(window_idx, pair_results)

    # Before pair selection:
    if blacklist.is_blacklisted(pair):
        continue  # Skip this pair
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.symbol_blacklist")


@dataclass
class SymbolStats:
    """Statistics for a single symbol across windows."""
    symbol: str
    windows_participated: int = 0
    total_pnl: float = 0.0
    total_trades: float = 0.0
    total_stops: float = 0.0
    window_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def avg_pnl_per_window(self) -> float:
        """Average P&L per window participated."""
        if self.windows_participated == 0:
            return 0.0
        return self.total_pnl / self.windows_participated

    @property
    def stop_rate(self) -> float:
        """Overall stop-loss rate."""
        if self.total_trades == 0:
            return 0.0
        return self.total_stops / self.total_trades

    @property
    def avg_trades_per_window(self) -> float:
        """Average trades per window."""
        if self.windows_participated == 0:
            return 0.0
        return self.total_trades / self.windows_participated


@dataclass
class SymbolBlacklist:
    """
    Track and blacklist consistently underperforming symbols.

    Identifies symbols that appear in multiple bottom-performing pairs and
    excludes them from future windows.

    Blacklist criteria (any triggers blacklisting):
    1. Average P&L per window < max_avg_loss_pct
    2. Stop-loss rate > max_stop_rate
    3. Appeared in bottom performers in multiple windows
    """

    # Configuration
    min_windows: int = 2              # Minimum windows before blacklisting
    min_trades_per_window: int = 5    # Minimum trades per window to count
    max_avg_loss_pct: float = -0.003  # -0.3% avg loss per window
    max_stop_rate: float = 0.45       # 45% stop-loss rate

    # State
    symbol_stats: Dict[str, SymbolStats] = field(default_factory=dict)
    blacklist: Set[str] = field(default_factory=set)
    blacklist_reasons: Dict[str, str] = field(default_factory=dict)

    # Track which pairs a symbol contributed to losses
    symbol_loss_pairs: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_config(cls) -> "SymbolBlacklist":
        """Create a SymbolBlacklist using configuration parameters."""
        return cls(
            min_windows=getattr(cfg, "BLACKLIST_MIN_WINDOWS", 2),
            min_trades_per_window=getattr(cfg, "BLACKLIST_MIN_TRADES_PER_WINDOW", 5),
            max_avg_loss_pct=getattr(cfg, "BLACKLIST_MAX_AVG_LOSS_PCT", -0.003),
            max_stop_rate=getattr(cfg, "BLACKLIST_MAX_STOP_RATE", 0.45),
        )

    def record_window(
        self,
        window_idx: int,
        pair_results: Dict[str, Dict[str, Any]],
        pair_separator: str = "-",
    ) -> None:
        """
        Record per-symbol performance after each window.

        Parameters
        ----------
        window_idx : int
            Index of the walk-forward window
        pair_results : Dict[str, Dict]
            Per-pair results with keys like:
            - "net_pnl": float
            - "trade_count": int
            - "stop_count": int
        pair_separator : str
            Separator used in pair names (default "-")
        """
        # Aggregate per-symbol statistics
        symbol_window_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"pnl": 0.0, "trades": 0.0, "stops": 0.0, "pairs": []}
        )

        for pair, stats in pair_results.items():
            try:
                parts = pair.split(pair_separator)
                if len(parts) != 2:
                    continue
                coin1, coin2 = parts
            except (ValueError, AttributeError):
                continue

            net_pnl = stats.get("net_pnl", 0.0)
            trade_count = stats.get("trade_count", 0)
            stop_count = stats.get("stop_count", 0)

            # Split attribution between both symbols
            for symbol in [coin1, coin2]:
                symbol_window_stats[symbol]["pnl"] += net_pnl / 2
                symbol_window_stats[symbol]["trades"] += trade_count / 2
                symbol_window_stats[symbol]["stops"] += stop_count / 2
                symbol_window_stats[symbol]["pairs"].append(pair)

                # Track loss pairs
                if net_pnl < 0:
                    self.symbol_loss_pairs[symbol].append(pair)

        # Update symbol statistics
        for symbol, window_data in symbol_window_stats.items():
            if window_data["trades"] < self.min_trades_per_window:
                continue  # Not enough trades to count

            if symbol not in self.symbol_stats:
                self.symbol_stats[symbol] = SymbolStats(symbol=symbol)

            stats = self.symbol_stats[symbol]
            stats.windows_participated += 1
            stats.total_pnl += window_data["pnl"]
            stats.total_trades += window_data["trades"]
            stats.total_stops += window_data["stops"]
            stats.window_history.append({
                "window_idx": window_idx,
                "pnl": window_data["pnl"],
                "trades": window_data["trades"],
                "stops": window_data["stops"],
                "pairs": window_data["pairs"],
            })

            # Evaluate for blacklisting
            self._evaluate_symbol(symbol)

    def _evaluate_symbol(self, symbol: str) -> None:
        """Check if a symbol should be blacklisted."""
        if symbol in self.blacklist:
            return  # Already blacklisted

        stats = self.symbol_stats.get(symbol)
        if stats is None:
            return

        if stats.windows_participated < self.min_windows:
            return  # Not enough history

        # Check average P&L per window
        if stats.avg_pnl_per_window < self.max_avg_loss_pct:
            self._blacklist_symbol(
                symbol,
                f"avg_pnl ({stats.avg_pnl_per_window:.2%} < {self.max_avg_loss_pct:.2%})"
            )
            return

        # Check stop-loss rate
        if stats.stop_rate > self.max_stop_rate:
            self._blacklist_symbol(
                symbol,
                f"stop_rate ({stats.stop_rate:.1%} > {self.max_stop_rate:.1%})"
            )
            return

    def _blacklist_symbol(self, symbol: str, reason: str) -> None:
        """Add a symbol to the blacklist."""
        self.blacklist.add(symbol)
        self.blacklist_reasons[symbol] = reason

        stats = self.symbol_stats.get(symbol)
        logger.warning(
            "Symbol BLACKLISTED: %s | reason=%s | windows=%d, total_pnl=%.2f%%, "
            "trades=%.0f, stop_rate=%.1f%%, loss_pairs=%s",
            symbol,
            reason,
            stats.windows_participated if stats else 0,
            (stats.total_pnl * 100) if stats else 0,
            stats.total_trades if stats else 0,
            (stats.stop_rate * 100) if stats else 0,
            self.symbol_loss_pairs.get(symbol, [])[:5],  # Show first 5 loss pairs
        )

    def is_blacklisted(self, pair_or_symbol: str, pair_separator: str = "-") -> bool:
        """
        Check if a symbol or pair is blacklisted.

        Parameters
        ----------
        pair_or_symbol : str
            Either a symbol (e.g., "ACE") or pair (e.g., "ACE-BTC")
        pair_separator : str
            Separator used in pair names

        Returns
        -------
        bool
            True if the symbol or any symbol in the pair is blacklisted
        """
        # Check if it's a pair
        if pair_separator in pair_or_symbol:
            try:
                parts = pair_or_symbol.split(pair_separator)
                if len(parts) == 2:
                    return parts[0] in self.blacklist or parts[1] in self.blacklist
            except (ValueError, AttributeError):
                pass

        # Check as symbol directly
        return pair_or_symbol in self.blacklist

    def filter_pairs(
        self,
        pairs: List[str],
        pair_separator: str = "-",
    ) -> Tuple[List[str], List[str]]:
        """
        Filter out pairs containing blacklisted symbols.

        Returns
        -------
        Tuple[List[str], List[str]]
            (allowed_pairs, blocked_pairs)
        """
        allowed = []
        blocked = []

        for pair in pairs:
            if self.is_blacklisted(pair, pair_separator):
                blocked.append(pair)
            else:
                allowed.append(pair)

        if blocked:
            logger.info(
                "Symbol blacklist filtered %d pairs: %s",
                len(blocked),
                blocked[:5] if len(blocked) > 5 else blocked,
            )

        return allowed, blocked

    def get_symbol_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific symbol."""
        stats = self.symbol_stats.get(symbol)
        if stats is None:
            return None

        return {
            "symbol": symbol,
            "windows_participated": stats.windows_participated,
            "total_pnl": stats.total_pnl,
            "avg_pnl_per_window": stats.avg_pnl_per_window,
            "total_trades": stats.total_trades,
            "stop_rate": stats.stop_rate,
            "is_blacklisted": symbol in self.blacklist,
            "blacklist_reason": self.blacklist_reasons.get(symbol),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of blacklist status."""
        # Find worst performing symbols
        symbol_rankings = sorted(
            [
                (sym, stats.avg_pnl_per_window, stats.stop_rate)
                for sym, stats in self.symbol_stats.items()
                if stats.windows_participated >= self.min_windows
            ],
            key=lambda x: x[1],  # Sort by avg P&L
        )

        return {
            "total_symbols_tracked": len(self.symbol_stats),
            "blacklisted_count": len(self.blacklist),
            "blacklisted_symbols": list(self.blacklist),
            "blacklist_reasons": self.blacklist_reasons.copy(),
            "worst_5_symbols": [
                {"symbol": s, "avg_pnl": p, "stop_rate": r}
                for s, p, r in symbol_rankings[:5]
            ],
            "best_5_symbols": [
                {"symbol": s, "avg_pnl": p, "stop_rate": r}
                for s, p, r in symbol_rankings[-5:][::-1]
            ],
        }

    def save(self, path: Path) -> None:
        """Save blacklist state to JSON file."""
        data = {
            "blacklist": list(self.blacklist),
            "blacklist_reasons": self.blacklist_reasons,
            "symbol_stats": {
                sym: {
                    "windows_participated": stats.windows_participated,
                    "total_pnl": stats.total_pnl,
                    "total_trades": stats.total_trades,
                    "total_stops": stats.total_stops,
                }
                for sym, stats in self.symbol_stats.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Symbol blacklist saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SymbolBlacklist":
        """Load blacklist state from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        instance = cls()
        instance.blacklist = set(data.get("blacklist", []))
        instance.blacklist_reasons = data.get("blacklist_reasons", {})

        for sym, stats_data in data.get("symbol_stats", {}).items():
            instance.symbol_stats[sym] = SymbolStats(
                symbol=sym,
                windows_participated=stats_data.get("windows_participated", 0),
                total_pnl=stats_data.get("total_pnl", 0.0),
                total_trades=stats_data.get("total_trades", 0.0),
                total_stops=stats_data.get("total_stops", 0.0),
            )

        logger.info(
            "Symbol blacklist loaded from %s: %d symbols tracked, %d blacklisted",
            path,
            len(instance.symbol_stats),
            len(instance.blacklist),
        )

        return instance

    def reset(self) -> None:
        """Reset all state (for fresh backtest)."""
        self.symbol_stats.clear()
        self.blacklist.clear()
        self.blacklist_reasons.clear()
        self.symbol_loss_pairs.clear()
