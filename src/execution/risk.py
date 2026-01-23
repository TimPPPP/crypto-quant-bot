import os
import numpy as np
import logging
from typing import Optional, Set
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskEngine")


class RiskEngine:
    """
    Active Risk Manager for Pairs Trading.

    Features:
    - Configurable risk limits via constructor or environment variables
    - Peak equity tracking for accurate max drawdown
    - Position count tracking
    - Kelly-criterion inspired position sizing
    """

    # Default values (can be overridden)
    DEFAULT_EQUITY = 10000.0
    DEFAULT_RISK_PER_TRADE = 0.01  # 1%
    DEFAULT_MAX_LEVERAGE = 2.0
    DEFAULT_MAX_Z_SCORE = 4.0      # Stop loss threshold
    DEFAULT_MAX_ENTRY_Z = 3.5      # Don't enter if too stretched
    DEFAULT_DRAWDOWN_LIMIT = 0.05  # 5% kill switch
    DEFAULT_MAX_POSITIONS = 8      # Max concurrent positions
    DEFAULT_MAX_PER_COIN = 2       # Max positions per coin
    DEFAULT_STOP_LOSS_PCT = 0.03   # Hard PnL stop per position

    def __init__(
        self,
        total_equity: float = None,
        risk_per_trade: float = None,
        max_leverage: float = None,
        max_z_score: float = None,
        max_entry_z: float = None,
        drawdown_limit: float = None,
        max_positions: int = None,
        max_positions_per_coin: int = None
    ):
        """
        Initialize Risk Engine with configurable parameters.

        All parameters can be set via:
        1. Constructor arguments
        2. Environment variables (RISK_EQUITY, RISK_PER_TRADE, etc.)
        3. Default values
        """
        # Load from env or use defaults
        self.start_equity = float(
            total_equity or
            os.getenv('RISK_EQUITY', self.DEFAULT_EQUITY)
        )
        self.risk_per_trade = float(
            risk_per_trade or
            os.getenv('RISK_PER_TRADE', self.DEFAULT_RISK_PER_TRADE)
        )
        self.max_leverage = float(
            max_leverage or
            os.getenv('RISK_MAX_LEVERAGE', self.DEFAULT_MAX_LEVERAGE)
        )
        self.MAX_Z_SCORE = float(
            max_z_score or
            os.getenv('RISK_MAX_Z', self.DEFAULT_MAX_Z_SCORE)
        )
        self.MAX_ENTRY_Z = float(
            max_entry_z or
            os.getenv('RISK_MAX_ENTRY_Z', self.DEFAULT_MAX_ENTRY_Z)
        )
        self.DRAWDOWN_LIMIT = float(
            drawdown_limit or
            os.getenv('RISK_DRAWDOWN_LIMIT', self.DEFAULT_DRAWDOWN_LIMIT)
        )
        self.max_positions = int(
            max_positions or
            os.getenv('RISK_MAX_POSITIONS', self.DEFAULT_MAX_POSITIONS)
        )
        self.max_positions_per_coin = int(
            max_positions_per_coin or
            os.getenv('RISK_MAX_PER_COIN', self.DEFAULT_MAX_PER_COIN)
        )
        self.stop_loss_pct = float(
            os.getenv('RISK_STOP_LOSS_PCT', self.DEFAULT_STOP_LOSS_PCT)
        )

        # Current state
        self.current_equity = float(self.start_equity)
        self.peak_equity = float(self.start_equity)  # Track peak for drawdown

        # Position tracking
        self.active_positions: Set[str] = set()  # Set of pair_ids
        self.coin_exposure: dict = {}  # coin -> count of positions

        logger.info(
            f"Risk Engine Initialized | "
            f"Equity: ${self.start_equity:,.0f} | "
            f"Risk: {self.risk_per_trade*100:.1f}% | "
            f"Max Leverage: {self.max_leverage}x | "
            f"Max Positions: {self.max_positions}"
        )

    def update_equity(self, new_equity: float):
        """Update current equity and track peak for drawdown calculation."""
        self.current_equity = float(new_equity)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def get_drawdown_from_start(self) -> float:
        """Calculate drawdown from starting equity."""
        if self.start_equity <= 0:
            return 0.0
        return (self.start_equity - self.current_equity) / self.start_equity

    def register_position(self, pair_id: str) -> bool:
        """
        Register a new position. Returns False if limits exceeded.

        Args:
            pair_id: The pair identifier (e.g., "ETH-BTC")

        Returns:
            True if position can be opened, False otherwise
        """
        # Check total position limit
        if len(self.active_positions) >= self.max_positions:
            logger.warning(f"Position limit reached ({self.max_positions})")
            return False

        # Check per-coin limit
        coins = pair_id.split('-')
        for coin in coins:
            current_count = self.coin_exposure.get(coin, 0)
            if current_count >= self.max_positions_per_coin:
                logger.warning(f"Coin exposure limit reached for {coin}")
                return False

        # Register position
        self.active_positions.add(pair_id)
        for coin in coins:
            self.coin_exposure[coin] = self.coin_exposure.get(coin, 0) + 1

        return True

    def unregister_position(self, pair_id: str):
        """Remove a closed position from tracking."""
        if pair_id in self.active_positions:
            self.active_positions.discard(pair_id)

            coins = pair_id.split('-')
            for coin in coins:
                if coin in self.coin_exposure:
                    self.coin_exposure[coin] = max(0, self.coin_exposure[coin] - 1)
                    if self.coin_exposure[coin] == 0:
                        del self.coin_exposure[coin]

    def check_entry_signal(self, pair: str, z_score: float) -> bool:
        """
        Check if entry is allowed based on risk parameters.

        Args:
            pair: Pair identifier
            z_score: Current z-score of the spread

        Returns:
            True if entry is allowed, False otherwise
        """
        # Check if Z-score is too close to stop loss
        if abs(z_score) >= self.MAX_ENTRY_Z:
            logger.warning(
                f"SKIP ENTRY: {pair} Z={z_score:.2f} too close to stop ({self.MAX_Z_SCORE})"
            )
            return False

        # Check position limits
        if len(self.active_positions) >= self.max_positions:
            logger.warning(f"SKIP ENTRY: {pair} - max positions reached")
            return False

        # Check per-coin limits
        coins = pair.split('-')
        for coin in coins:
            if self.coin_exposure.get(coin, 0) >= self.max_positions_per_coin:
                logger.warning(f"SKIP ENTRY: {pair} - {coin} exposure limit reached")
                return False

        return True

    def calculate_size(
        self,
        price: float,
        volatility_pct: float,
        is_pair_trade: bool = True
    ) -> float:
        """
        Calculate position size using Kelly-inspired formula.

        Args:
            price: Current price of the asset
            volatility_pct: Volatility as decimal (e.g., 0.02 for 2%)
            is_pair_trade: If True, accounts for 2-leg notional

        Returns:
            Position size in units
        """
        if volatility_pct <= 0 or price <= 0:
            return 0.0

        # Risk budget in USD
        risk_budget_usd = self.current_equity * self.risk_per_trade

        # Position value = risk budget / volatility
        position_value_usd = risk_budget_usd / volatility_pct

        # For pair trades, each leg is half the total risk
        # (since we have 2 legs with offsetting risk)
        if is_pair_trade:
            position_value_usd = position_value_usd / 2

        # Apply leverage cap
        max_position_usd = self.current_equity * self.max_leverage
        if is_pair_trade:
            max_position_usd = max_position_usd / 2  # Per leg

        if position_value_usd > max_position_usd:
            position_value_usd = max_position_usd

        # Convert to units
        size = position_value_usd / price

        return round(float(size), 8)

    def check_stop_loss(
        self,
        pair: str,
        z_score: float,
        is_open_position: bool = False,
        pnl_return: Optional[float] = None
    ) -> bool:
        """
        Check if stop loss should be triggered.

        Args:
            pair: Pair identifier
            z_score: Current z-score
            is_open_position: Whether we have an open position

        Returns:
            True if stop loss triggered, False otherwise
        """
        # Only check if we have a position
        if not is_open_position:
            return False

        if pnl_return is not None and pnl_return <= -self.stop_loss_pct:
            logger.warning(
                f"STOP LOSS: {pair} PnL={pnl_return:.2%} exceeds {self.stop_loss_pct:.2%}"
            )
            return True

        if abs(z_score) > self.MAX_Z_SCORE:
            logger.warning(
                f"STOP LOSS: {pair} Z={z_score:.2f} exceeds {self.MAX_Z_SCORE}"
            )
            return True

        return False

    def is_kill_switch_triggered(self) -> bool:
        """
        Check if kill switch would be triggered (no side effects).

        Returns:
            True if drawdown exceeds limit
        """
        return self.get_current_drawdown() > self.DRAWDOWN_LIMIT

    def check_kill_switch(self) -> bool:
        """
        Check if emergency shutdown should be triggered and log if so.

        Uses peak drawdown (more conservative than drawdown from start).

        Returns:
            True if kill switch should be activated
        """
        drawdown_pct = self.get_current_drawdown()

        if drawdown_pct > self.DRAWDOWN_LIMIT:
            logger.critical(
                f"KILL SWITCH: Drawdown {drawdown_pct*100:.2f}% > {self.DRAWDOWN_LIMIT*100:.1f}%"
            )
            return True

        return False

    def get_status(self) -> dict:
        """
        Get current risk status summary.

        Note: Uses is_kill_switch_triggered() to avoid logging side effects.
        """
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'start_equity': self.start_equity,
            'drawdown_from_peak': self.get_current_drawdown(),
            'drawdown_from_start': self.get_drawdown_from_start(),
            'active_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'coin_exposure': dict(self.coin_exposure),
            'kill_switch_triggered': self.is_kill_switch_triggered()
        }


if __name__ == "__main__":
    # Test Risk Engine
    print("Testing Risk Engine...")

    engine = RiskEngine(total_equity=10000.0)

    # Test position registration
    print("\n1. Position Registration Test:")
    print(f"   Register ETH-BTC: {engine.register_position('ETH-BTC')}")
    print(f"   Register SOL-ETH: {engine.register_position('SOL-ETH')}")
    print(f"   Active positions: {engine.active_positions}")
    print(f"   Coin exposure: {engine.coin_exposure}")

    # Test per-coin limit
    print(f"   Register ETH-SOL (should fail - ETH limit): {engine.register_position('ETH-SOL')}")

    # Test size calculation
    print("\n2. Size Calculation Test:")
    size = engine.calculate_size(price=2000.0, volatility_pct=0.02, is_pair_trade=True)
    print(f"   Size for $2000 asset, 2% vol, pair trade: {size:.4f} units")
    print(f"   Notional: ${size * 2000:.2f}")

    # Test drawdown tracking
    print("\n3. Drawdown Tracking Test:")
    engine.update_equity(11000)  # Profit
    print(f"   After profit: equity=${engine.current_equity}, peak=${engine.peak_equity}")
    engine.update_equity(9500)   # Loss
    print(f"   After loss: equity=${engine.current_equity}, peak=${engine.peak_equity}")
    print(f"   Drawdown from peak: {engine.get_current_drawdown()*100:.2f}%")
    print(f"   Drawdown from start: {engine.get_drawdown_from_start()*100:.2f}%")

    # Test kill switch
    print("\n4. Kill Switch Test:")
    engine.update_equity(10000)
    engine.peak_equity = 10000
    engine.update_equity(9400)  # 6% drawdown
    print(f"   Kill switch (6% DD): {engine.check_kill_switch()}")

    # Test entry signal check
    print("\n5. Entry Signal Check:")
    engine.unregister_position('ETH-BTC')
    print(f"   Check entry Z=2.5: {engine.check_entry_signal('NEW-PAIR', 2.5)}")
    print(f"   Check entry Z=4.2: {engine.check_entry_signal('NEW-PAIR', 4.2)}")

    print("\n6. Status Summary:")
    print(f"   {engine.get_status()}")
