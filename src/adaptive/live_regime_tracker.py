"""
src/adaptive/live_regime_tracker.py

Live Regime Tracker - Dynamic regime updates during test execution.

This module implements real-time regime detection that updates during the test
period rather than being frozen at window start. Key features:

1. Periodic regime recomputation (every N bars, default 96 = 1 day)
2. Immediate volatility spike detection for fast regime transitions
3. Hysteresis to prevent whipsawing between states
4. State machine with GREEN -> YELLOW -> RED transitions

Example usage:
    tracker = LiveRegimeTracker.from_config(btc_prices_train)

    for bar_idx in range(len(test_data)):
        btc_price = test_data["BTC"].iloc[bar_idx]
        regime_update = tracker.update(bar_idx, btc_price)

        if regime_update is not None:
            logger.info("Regime changed to %s", regime_update.state.value)

        current_state = tracker.current_state
        size_mult = tracker.get_size_multiplier()
        entry_z_adj = tracker.get_entry_z_adjustment()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg
from src.adaptive.regime_filter import RegimeState, SoftRegimeResult

logger = logging.getLogger("adaptive.live_regime_tracker")


@dataclass
class RegimeTransition:
    """Record of a regime state transition."""
    bar_idx: int
    timestamp: Optional[pd.Timestamp]
    from_state: RegimeState
    to_state: RegimeState
    trigger: str  # "periodic", "spike", "cooldown_expired"
    btc_vol: float
    btc_vol_z: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_idx": self.bar_idx,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger,
            "btc_vol": self.btc_vol,
            "btc_vol_z": self.btc_vol_z,
        }


@dataclass
class LiveRegimeTracker:
    """
    Tracks regime state dynamically during test execution.

    Unlike static regime filters that compute state once at window start,
    this tracker updates regime state periodically and on volatility spikes.

    State Machine:
    - GREEN: Normal trading (size_mult=1.0, entry_z_adj=0.0)
    - YELLOW: De-risked trading (size_mult=0.5, entry_z_adj=+0.5)
    - RED: No new entries (size_mult=0.0)

    Transitions:
    - GREEN -> YELLOW: BTC vol crosses 70th percentile (confirmed)
    - YELLOW -> RED: BTC vol crosses 85th percentile OR spike detected
    - RED -> YELLOW: BTC vol drops below 70th percentile (confirmed)
    - YELLOW -> GREEN: BTC vol drops below 50th percentile (confirmed)
    - Any -> RED: Immediate on volatility spike (z > threshold)
    """

    # Update frequency
    update_frequency_bars: int = 96  # 1 day at 15-min bars

    # Cooldown between transitions (hysteresis)
    transition_cooldown_bars: int = 24  # 6 hours at 15-min bars

    # Volatility spike detection
    spike_z_threshold: float = 2.0  # Immediate RED if BTC vol z-score > this

    # Lookback for rolling volatility
    vol_lookback_bars: int = 96  # 1 day

    # Percentile thresholds for regime classification
    green_max_vol_pctl: float = 0.50   # GREEN if vol < 50th percentile
    yellow_max_vol_pctl: float = 0.70  # YELLOW if vol < 70th percentile
    red_vol_pctl: float = 0.85         # RED if vol > 85th percentile

    # Regime parameter adjustments
    green_size_mult: float = 1.0
    green_entry_z_adj: float = 0.0
    yellow_size_mult: float = 0.5
    yellow_entry_z_adj: float = 0.5
    red_size_mult: float = 0.0
    red_entry_z_adj: float = 0.0  # N/A when blocked

    # State tracking
    current_state: RegimeState = field(default=RegimeState.GREEN)
    last_update_bar: int = 0
    last_transition_bar: int = 0

    # Historical data for percentile computation
    historical_vol: List[float] = field(default_factory=list)
    recent_btc_returns: List[float] = field(default_factory=list)

    # Transition history for diagnostics
    transitions: List[RegimeTransition] = field(default_factory=list)

    # Statistics
    bars_in_state: Dict[str, int] = field(default_factory=lambda: {"green": 0, "yellow": 0, "red": 0})

    @classmethod
    def from_config(
        cls,
        btc_prices_train: pd.Series,
        bars_per_day: int = 96,
    ) -> "LiveRegimeTracker":
        """
        Create a LiveRegimeTracker using config parameters and training data.

        Parameters
        ----------
        btc_prices_train : pd.Series
            BTC prices from training period (for percentile baseline)
        bars_per_day : int
            Number of bars per day (96 for 15-min bars)
        """
        # Compute historical volatility from training data
        btc_returns = btc_prices_train.pct_change().dropna()
        vol_lookback = int(getattr(cfg, "REGIME_BTC_VOL_LOOKBACK_BARS", 96))
        rolling_vol = btc_returns.rolling(vol_lookback, min_periods=vol_lookback // 2).std()
        historical_vol = rolling_vol.dropna().tolist()

        # Initialize recent returns buffer with training data tail
        recent_returns = btc_returns.tail(vol_lookback).tolist()

        # Compute initial regime state based on last training data
        if len(historical_vol) > 0:
            current_vol = historical_vol[-1]
            vol_pctl = sum(1 for v in historical_vol if v <= current_vol) / len(historical_vol)
        else:
            vol_pctl = 0.5  # Default to middle

        # Determine initial state
        green_max = float(getattr(cfg, "REGIME_VOL_GREEN_MAX_PCTL", 0.50))
        yellow_max = float(getattr(cfg, "REGIME_VOL_YELLOW_MAX_PCTL", 0.70))

        if vol_pctl <= green_max:
            initial_state = RegimeState.GREEN
        elif vol_pctl <= yellow_max:
            initial_state = RegimeState.YELLOW
        else:
            initial_state = RegimeState.RED

        logger.info(
            "LiveRegimeTracker initialized: state=%s, vol_pctl=%.2f, historical_vol_samples=%d",
            initial_state.value,
            vol_pctl,
            len(historical_vol),
        )

        return cls(
            update_frequency_bars=int(getattr(cfg, "REGIME_UPDATE_FREQUENCY_BARS", 96)),
            transition_cooldown_bars=int(getattr(cfg, "REGIME_TRANSITION_COOLDOWN_BARS", 24)),
            spike_z_threshold=float(getattr(cfg, "BTC_VOL_SPIKE_Z_THRESHOLD", 2.0)),
            vol_lookback_bars=vol_lookback,
            green_max_vol_pctl=green_max,
            yellow_max_vol_pctl=yellow_max,
            red_vol_pctl=float(getattr(cfg, "REGIME_DISPERSION_RED_PCTL", 0.85)),
            current_state=initial_state,
            historical_vol=historical_vol,
            recent_btc_returns=recent_returns,
        )

    def update(
        self,
        bar_idx: int,
        btc_price: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Optional[RegimeTransition]:
        """
        Update regime state based on new BTC price.

        Called for each bar during test execution. Returns a RegimeTransition
        if the state changed, otherwise None.

        Parameters
        ----------
        bar_idx : int
            Current bar index in test data
        btc_price : float
            Current BTC price
        timestamp : Optional[pd.Timestamp]
            Current timestamp (for logging)

        Returns
        -------
        Optional[RegimeTransition]
            Transition record if state changed, None otherwise
        """
        # Track bars in current state
        self.bars_in_state[self.current_state.value] += 1

        # Compute current BTC return
        if len(self.recent_btc_returns) > 0:
            # Use price change as proxy for return
            prev_price = self._get_prev_price()
            if prev_price is not None and prev_price > 0:
                btc_return = (btc_price - prev_price) / prev_price
                self.recent_btc_returns.append(btc_return)

                # Keep buffer at lookback size
                if len(self.recent_btc_returns) > self.vol_lookback_bars:
                    self.recent_btc_returns.pop(0)

        # Store price for next iteration
        self._store_price(btc_price)

        # Compute current volatility
        current_vol = self._compute_current_vol()
        vol_z = self._compute_vol_z_score(current_vol)

        # Check for volatility spike (immediate RED trigger)
        if vol_z > self.spike_z_threshold:
            if self.current_state != RegimeState.RED:
                transition = self._transition_to(
                    RegimeState.RED,
                    bar_idx,
                    timestamp,
                    "spike",
                    current_vol,
                    vol_z,
                )
                return transition

        # Check if periodic update is due
        bars_since_update = bar_idx - self.last_update_bar
        if bars_since_update < self.update_frequency_bars:
            return None  # Not time for periodic update yet

        # Check cooldown
        bars_since_transition = bar_idx - self.last_transition_bar
        if bars_since_transition < self.transition_cooldown_bars:
            # Update the last_update_bar but don't transition
            self.last_update_bar = bar_idx
            return None

        # Periodic update: compute volatility percentile
        vol_pctl = self._compute_vol_percentile(current_vol)

        # Determine target state based on percentile
        target_state = self._determine_target_state(vol_pctl)

        # Apply state machine rules (no direct GREEN <-> RED transitions)
        new_state = self._apply_state_machine(target_state)

        self.last_update_bar = bar_idx

        if new_state != self.current_state:
            transition = self._transition_to(
                new_state,
                bar_idx,
                timestamp,
                "periodic",
                current_vol,
                vol_z,
            )
            return transition

        return None

    def _transition_to(
        self,
        new_state: RegimeState,
        bar_idx: int,
        timestamp: Optional[pd.Timestamp],
        trigger: str,
        btc_vol: float,
        btc_vol_z: float,
    ) -> RegimeTransition:
        """Execute a state transition and record it."""
        transition = RegimeTransition(
            bar_idx=bar_idx,
            timestamp=timestamp,
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            btc_vol=btc_vol,
            btc_vol_z=btc_vol_z,
        )

        logger.info(
            "Regime transition: %s -> %s (trigger=%s, bar=%d, vol_z=%.2f)",
            self.current_state.value,
            new_state.value,
            trigger,
            bar_idx,
            btc_vol_z,
        )

        self.current_state = new_state
        self.last_transition_bar = bar_idx
        self.transitions.append(transition)

        return transition

    def _determine_target_state(self, vol_pctl: float) -> RegimeState:
        """Determine target state based on volatility percentile."""
        if vol_pctl <= self.green_max_vol_pctl:
            return RegimeState.GREEN
        elif vol_pctl <= self.yellow_max_vol_pctl:
            return RegimeState.YELLOW
        else:
            return RegimeState.RED

    def _apply_state_machine(self, target_state: RegimeState) -> RegimeState:
        """
        Apply state machine rules to prevent invalid transitions.

        Rules:
        - No direct GREEN <-> RED transitions (must go through YELLOW)
        - Can always transition to adjacent state
        """
        if self.current_state == RegimeState.GREEN:
            if target_state == RegimeState.RED:
                return RegimeState.YELLOW  # Must go through YELLOW first
            return target_state

        elif self.current_state == RegimeState.YELLOW:
            return target_state  # Can go to either GREEN or RED

        else:  # RED
            if target_state == RegimeState.GREEN:
                return RegimeState.YELLOW  # Must go through YELLOW first
            return target_state

    def _compute_current_vol(self) -> float:
        """Compute current rolling volatility from recent returns."""
        if len(self.recent_btc_returns) < 10:
            return 0.0
        return np.std(self.recent_btc_returns)

    def _compute_vol_z_score(self, current_vol: float) -> float:
        """Compute z-score of current volatility vs historical."""
        if len(self.historical_vol) < 10 or current_vol == 0:
            return 0.0

        hist_mean = np.mean(self.historical_vol)
        hist_std = np.std(self.historical_vol)

        if hist_std < 1e-10:
            return 0.0

        return (current_vol - hist_mean) / hist_std

    def _compute_vol_percentile(self, current_vol: float) -> float:
        """Compute percentile rank of current volatility."""
        if len(self.historical_vol) == 0:
            return 0.5

        # Add current vol to historical for fair comparison
        all_vol = self.historical_vol + [current_vol]
        rank = sum(1 for v in all_vol if v <= current_vol)
        return rank / len(all_vol)

    def _get_prev_price(self) -> Optional[float]:
        """Get previous BTC price from internal storage."""
        return getattr(self, "_prev_btc_price", None)

    def _store_price(self, price: float) -> None:
        """Store current BTC price for next iteration."""
        self._prev_btc_price = price

    def get_size_multiplier(self) -> float:
        """Get position size multiplier for current regime state."""
        if self.current_state == RegimeState.GREEN:
            return self.green_size_mult
        elif self.current_state == RegimeState.YELLOW:
            return self.yellow_size_mult
        else:
            return self.red_size_mult

    def get_entry_z_adjustment(self) -> float:
        """Get entry z-score adjustment for current regime state."""
        if self.current_state == RegimeState.GREEN:
            return self.green_entry_z_adj
        elif self.current_state == RegimeState.YELLOW:
            return self.yellow_entry_z_adj
        else:
            return self.red_entry_z_adj

    def is_entry_allowed(self) -> bool:
        """Check if new entries are allowed in current regime."""
        return self.current_state != RegimeState.RED

    def get_state_series(self, n_bars: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate state series for all bars (reconstructed from transitions).

        Used for post-hoc analysis and logging.

        Parameters
        ----------
        n_bars : int
            Total number of bars in test period

        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (state_series, size_mult_series, entry_allowed_series)
        """
        states = []
        size_mults = []
        entry_allowed = []

        current = RegimeState.GREEN  # Default start
        transition_idx = 0

        for bar in range(n_bars):
            # Check if there's a transition at this bar
            while (
                transition_idx < len(self.transitions)
                and self.transitions[transition_idx].bar_idx <= bar
            ):
                current = self.transitions[transition_idx].to_state
                transition_idx += 1

            states.append(current.value)

            if current == RegimeState.GREEN:
                size_mults.append(self.green_size_mult)
                entry_allowed.append(True)
            elif current == RegimeState.YELLOW:
                size_mults.append(self.yellow_size_mult)
                entry_allowed.append(True)
            else:
                size_mults.append(self.red_size_mult)
                entry_allowed.append(False)

        return (
            pd.Series(states, name="regime_state"),
            pd.Series(size_mults, name="size_multiplier"),
            pd.Series(entry_allowed, name="entry_allowed"),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the tracking period."""
        total_bars = sum(self.bars_in_state.values())

        return {
            "n_transitions": len(self.transitions),
            "bars_in_green": self.bars_in_state.get("green", 0),
            "bars_in_yellow": self.bars_in_state.get("yellow", 0),
            "bars_in_red": self.bars_in_state.get("red", 0),
            "pct_green": self.bars_in_state.get("green", 0) / max(total_bars, 1) * 100,
            "pct_yellow": self.bars_in_state.get("yellow", 0) / max(total_bars, 1) * 100,
            "pct_red": self.bars_in_state.get("red", 0) / max(total_bars, 1) * 100,
            "transitions": [t.to_dict() for t in self.transitions],
        }


def create_live_regime_tracker(btc_prices_train: pd.Series) -> LiveRegimeTracker:
    """Factory function to create a LiveRegimeTracker from config."""
    return LiveRegimeTracker.from_config(btc_prices_train)
