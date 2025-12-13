import os
import re
import logging
import requests
from typing import Dict, Optional, List, Set
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeExecutor")


@dataclass
class TradeRecord:
    """Record of an executed trade for analysis."""
    pair: str
    direction: str
    timestamp: str
    entry_z: float
    expected_profit_pct: float
    estimated_duration_hours: float
    funding_cost_pct: float
    friction_cost_pct: float
    net_ev_pct: float
    approved: bool
    rejection_reason: Optional[str] = None


class TradeExecutor:
    """
    The Accountant & Gatekeeper for trade execution.

    Responsibilities:
    - Calculate expected value (EV) of trades
    - Apply friction costs (fees + slippage)
    - Estimate funding costs based on direction
    - Track executed trades for analysis
    """

    # Default cost constants
    DEFAULT_TAKER_FEE = 0.0005      # 0.05% per leg
    DEFAULT_SLIPPAGE = 0.0005       # 0.05% per leg estimate
    NUM_LEGS = 4                     # Entry (2) + Exit (2)

    # Configurable via environment
    DEFAULT_MIN_NET_PROFIT = 0.01   # 1% minimum for swing trades
    DEFAULT_DURATION_MULTIPLIER = 2.0  # Expected hold = half_life * multiplier
    DEFAULT_PROFIT_CAPTURE = 0.75   # Expect to capture 75% of spread reversion

    def __init__(
        self,
        taker_fee: float = None,
        slippage: float = None,
        min_net_profit: float = None,
        duration_multiplier: float = None,
        profit_capture: float = None
    ):
        """
        Initialize Trade Executor.

        Args:
            taker_fee: Fee per leg (decimal)
            slippage: Estimated slippage per leg (decimal)
            min_net_profit: Minimum net profit threshold (decimal)
            duration_multiplier: Multiplier for half-life to estimate hold duration
            profit_capture: Expected fraction of spread reversion captured
        """
        # QuestDB Connection
        self.DB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
        self.QUESTDB_URL = f"http://{self.DB_HOST}:9000/exec"
        self.DB_TIMEOUT = 5  # Increased from 2 to 5 seconds

        # Cost parameters
        self.taker_fee = taker_fee or float(os.getenv('EXEC_TAKER_FEE', self.DEFAULT_TAKER_FEE))
        self.slippage = slippage or float(os.getenv('EXEC_SLIPPAGE', self.DEFAULT_SLIPPAGE))

        # Total friction per round-trip trade
        self.TOTAL_FRICTION = (self.taker_fee + self.slippage) * self.NUM_LEGS

        # EV thresholds
        self.MIN_NET_PROFIT = min_net_profit or float(
            os.getenv('EXEC_MIN_PROFIT', self.DEFAULT_MIN_NET_PROFIT)
        )
        self.duration_multiplier = duration_multiplier or float(
            os.getenv('EXEC_DURATION_MULT', self.DEFAULT_DURATION_MULTIPLIER)
        )
        self.profit_capture = profit_capture or float(
            os.getenv('EXEC_PROFIT_CAPTURE', self.DEFAULT_PROFIT_CAPTURE)
        )

        # Position tracking (Set for O(1) lookups)
        self.MAX_POSITIONS_PER_COIN = 1
        self.active_positions: Set[str] = set()

        # Trade history for analysis
        self.trade_history: List[TradeRecord] = []

        logger.info(
            f"Trade Executor Initialized | "
            f"Friction: {self.TOTAL_FRICTION*100:.2f}% | "
            f"Min Profit: {self.MIN_NET_PROFIT*100:.1f}%"
        )

    def _sanitize_symbol(self, symbol: str) -> str:
        """
        Sanitize symbol input to prevent SQL injection.
        Only allows alphanumeric characters and common symbol suffixes.
        """
        # Only allow alphanumeric, dash, underscore (common in trading symbols)
        if not re.match(r'^[A-Za-z0-9_-]+$', symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        # Escape single quotes as additional safety
        return symbol.replace("'", "''")

    def get_24h_funding_avg(self, symbol: str) -> Optional[float]:
        """
        Query QuestDB for 24-hour average funding rate.

        Returns:
            Average hourly funding rate, or None if data unavailable
        """
        # Sanitize input to prevent SQL injection
        safe_symbol = self._sanitize_symbol(symbol)
        query = f"""
        SELECT avg(funding_rate)
        FROM funding_history
        WHERE symbol = '{safe_symbol}'
        AND timestamp >= dateadd('d', -1, now());
        """
        try:
            r = requests.get(self.QUESTDB_URL, params={'query': query}, timeout=self.DB_TIMEOUT)
            data = r.json()

            if 'dataset' in data and data['dataset']:
                avg_rate = data['dataset'][0][0]
                if avg_rate is not None:
                    return float(avg_rate)

            logger.warning(f"No funding data for {symbol}")
            return None

        except requests.Timeout:
            logger.error(f"DB timeout fetching funding for {symbol}")
            return None
        except Exception as e:
            logger.error(f"DB error for {symbol}: {e}")
            return None

    def check_position_limit(self, coin_a: str, coin_b: str) -> bool:
        """
        Check if position limit allows new trade.

        Args:
            coin_a: First coin in pair
            coin_b: Second coin in pair

        Returns:
            True if trade allowed, False otherwise
        """
        count_a = 0
        count_b = 0

        for pos_pair in self.active_positions:
            held_coins = pos_pair.split('-')
            if coin_a in held_coins:
                count_a += 1
            if coin_b in held_coins:
                count_b += 1

        if count_a >= self.MAX_POSITIONS_PER_COIN:
            logger.info(f"Position limit reached for {coin_a}")
            return False
        if count_b >= self.MAX_POSITIONS_PER_COIN:
            logger.info(f"Position limit reached for {coin_b}")
            return False

        return True

    def calculate_ev(self, signal: Dict) -> bool:
        """
        Calculate Expected Value and decide if trade should proceed.

        Args:
            signal: Dictionary containing:
                - pair: str (e.g., "ETH-BTC")
                - z_score: float
                - spread_std: float (spread volatility as decimal)
                - half_life_hours: float

        Returns:
            True if trade approved (positive EV), False otherwise
        """
        pair = signal['pair']
        coin_y, coin_x = pair.split('-')
        z_score = signal['z_score']

        # 1. Position Limit Check
        if not self.check_position_limit(coin_y, coin_x):
            self._record_trade(signal, False, 0, 0, 0, "Position limit reached")
            return False

        # 2. Fetch Funding Rates
        rate_y = self.get_24h_funding_avg(coin_y)
        rate_x = self.get_24h_funding_avg(coin_x)

        if rate_y is None or rate_x is None:
            self._record_trade(signal, False, 0, 0, 0, "Funding data unavailable")
            logger.warning(f"NO-GO: Missing funding data for {pair}")
            return False

        # 3. Determine Direction and Funding Cost
        # Positive Z = Spread too HIGH = SHORT the spread (Short Y, Long X)
        # Negative Z = Spread too LOW = LONG the spread (Long Y, Short X)
        #
        # Funding cost calculation:
        # - When SHORT an asset: PAY funding if rate > 0, RECEIVE if rate < 0
        # - When LONG an asset: RECEIVE funding if rate > 0, PAY if rate < 0
        #
        # Net hourly funding = (funding on shorts) - (funding on longs)

        if z_score > 0:
            direction = "SHORT_SPREAD"
            # Short Y (pay rate_y if positive), Long X (receive rate_x if positive)
            net_hourly_funding = rate_y - rate_x
        else:
            direction = "LONG_SPREAD"
            # Long Y (receive rate_y if positive), Short X (pay rate_x if positive)
            net_hourly_funding = rate_x - rate_y

        # 4. Calculate Expected Profit
        # spread_std is the realized volatility of the spread
        # Expected move = spread_std * profit_capture (we don't capture 100%)
        spread_std = signal.get('spread_std', 0.01)  # Default 1% if not provided

        # Convert to percentage profit expectation
        # For log prices, spread_std is already in log-return units
        expected_profit_pct = spread_std * self.profit_capture

        # 5. Estimate Duration and Funding Cost
        half_life = signal.get('half_life_hours', 24)
        est_duration = half_life * self.duration_multiplier
        total_funding_cost = net_hourly_funding * est_duration

        # 6. Calculate Net EV
        net_ev = expected_profit_pct - self.TOTAL_FRICTION - total_funding_cost

        # 7. Log Decision
        funding_label = "Cost" if total_funding_cost > 0 else "Rebate"

        logger.info(f"\n{'='*50}")
        logger.info(f"EVALUATING: {pair} | Direction: {direction}")
        logger.info(f"  Z-Score: {z_score:.2f}")
        logger.info(f"  Est Duration: {est_duration:.1f} hours")
        logger.info(f"  ---------------------------------")
        logger.info(f"  Expected Profit:  +{expected_profit_pct*100:.3f}%")
        logger.info(f"  Friction (fees):  -{self.TOTAL_FRICTION*100:.3f}%")
        logger.info(f"  Funding ({funding_label}): {'-' if total_funding_cost > 0 else '+'}{abs(total_funding_cost)*100:.3f}%")
        logger.info(f"  ---------------------------------")
        logger.info(f"  NET EV:           {net_ev*100:.3f}%")
        logger.info(f"{'='*50}")

        # 8. Decision
        if net_ev > self.MIN_NET_PROFIT:
            logger.info(f"GO: Positive EV trade approved for {pair}")
            self._record_trade(
                signal, True, expected_profit_pct, total_funding_cost, net_ev
            )
            return True
        else:
            reason = f"Net EV {net_ev*100:.2f}% < {self.MIN_NET_PROFIT*100:.1f}%"
            logger.info(f"NO-GO: {reason}")
            self._record_trade(
                signal, False, expected_profit_pct, total_funding_cost, net_ev, reason
            )
            return False

    def _record_trade(
        self,
        signal: Dict,
        approved: bool,
        expected_profit: float,
        funding_cost: float,
        net_ev: float,
        rejection_reason: str = None
    ):
        """Record trade decision for analysis."""
        record = TradeRecord(
            pair=signal['pair'],
            direction="SHORT_SPREAD" if signal['z_score'] > 0 else "LONG_SPREAD",
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_z=signal['z_score'],
            expected_profit_pct=expected_profit,
            estimated_duration_hours=signal.get('half_life_hours', 24) * self.duration_multiplier,
            funding_cost_pct=funding_cost,
            friction_cost_pct=self.TOTAL_FRICTION,
            net_ev_pct=net_ev,
            approved=approved,
            rejection_reason=rejection_reason
        )
        self.trade_history.append(record)

    def get_trade_history(self) -> List[Dict]:
        """Get trade history as list of dicts."""
        return [asdict(t) for t in self.trade_history]

    def add_active_position(self, pair: str):
        """Register an active position."""
        self.active_positions.add(pair)

    def remove_active_position(self, pair: str):
        """Remove a closed position."""
        self.active_positions.discard(pair)


if __name__ == "__main__":
    print("Testing Trade Executor...")

    executor = TradeExecutor()

    # Test signal
    test_signal = {
        'pair': 'ETH-BTC',
        'z_score': 2.5,
        'spread_std': 0.015,  # 1.5% spread volatility
        'half_life_hours': 24
    }

    print("\n1. EV Calculation Test (without DB):")
    print(f"   Signal: {test_signal}")

    # Since we don't have DB, let's test the math directly
    print("\n2. Cost breakdown:")
    print(f"   Taker fee per leg: {executor.taker_fee*100:.3f}%")
    print(f"   Slippage per leg: {executor.slippage*100:.3f}%")
    print(f"   Total friction (4 legs): {executor.TOTAL_FRICTION*100:.3f}%")

    print("\n3. Expected profit calculation:")
    spread_std = test_signal['spread_std']
    expected_profit = spread_std * executor.profit_capture
    print(f"   Spread volatility: {spread_std*100:.2f}%")
    print(f"   Profit capture ratio: {executor.profit_capture}")
    print(f"   Expected profit: {expected_profit*100:.3f}%")

    print("\n4. Trade history:")
    print(f"   Recorded trades: {len(executor.trade_history)}")
