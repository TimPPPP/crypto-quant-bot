import os
import time
import logging
from typing import Dict, Optional, Tuple, List
from decimal import Decimal, ROUND_DOWN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExchangeClient")

# Hyperliquid imports
try:
    from eth_account.account import Account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    logger.warning("Hyperliquid SDK not installed. Running in mock mode only.")
    HYPERLIQUID_AVAILABLE = False


# Size precision per asset (Hyperliquid specific)
# These are the maximum decimal places allowed for order sizes
SIZE_DECIMALS = {
    'BTC': 4,
    'ETH': 3,
    'SOL': 2,
    'DOGE': 0,
    'XRP': 1,
    'AVAX': 2,
    'MATIC': 1,
    'LINK': 2,
    'DOT': 2,
    'UNI': 2,
    'ATOM': 2,
    'LTC': 3,
    'ARB': 1,
    'OP': 1,
    'APT': 2,
    'SUI': 1,
    'SEI': 0,
    'INJ': 2,
    'TIA': 2,
    'JUP': 0,
    'WIF': 0,
    'PEPE': 0,
    'BONK': 0,
    'SHIB': 0,
    'FLOKI': 0,
}
DEFAULT_SIZE_DECIMALS = 2


class ExchangeClient:
    """
    Hyperliquid Exchange Client with production-grade error handling.

    Features:
    - Configurable slippage tolerance
    - Proper size rounding per asset
    - Retry logic for transient failures
    - Order tracking and verification
    - Emergency position closing
    """

    DEFAULT_SLIPPAGE = 0.02          # 2% default slippage tolerance
    DEFAULT_MAX_RETRIES = 3          # Retry attempts for failed orders
    DEFAULT_RETRY_DELAY = 1.0        # Seconds between retries

    def __init__(
        self,
        private_key: str = None,
        account_address: str = None,
        base_url: str = None,
        slippage: float = None,
        max_retries: int = None
    ):
        """
        Initialize Exchange Client.

        Args:
            private_key: Hyperliquid private key (or from HL_PRIVATE_KEY env)
            account_address: Account address (or from HL_ACCOUNT_ADDRESS env)
            base_url: API URL (defaults to mainnet)
            slippage: Slippage tolerance as decimal (e.g., 0.02 for 2%)
            max_retries: Max retry attempts for failed orders
        """
        self.private_key = private_key or os.getenv("HL_PRIVATE_KEY")
        self.account_address = account_address or os.getenv("HL_ACCOUNT_ADDRESS")
        self.base_url = base_url or (constants.MAINNET_API_URL if HYPERLIQUID_AVAILABLE else "")

        # Configuration
        self.slippage = slippage or float(os.getenv('EXCHANGE_SLIPPAGE', self.DEFAULT_SLIPPAGE))
        self.max_retries = max_retries or int(os.getenv('EXCHANGE_MAX_RETRIES', self.DEFAULT_MAX_RETRIES))
        self.retry_delay = self.DEFAULT_RETRY_DELAY

        # Order tracking
        self.last_order_ids: List[str] = []
        self.execution_log: List[Dict] = []

        # Initialize connections
        self._init_connections()

    def _init_connections(self):
        """Initialize API connections."""
        if not HYPERLIQUID_AVAILABLE:
            logger.warning("Hyperliquid SDK not available. Mock mode active.")
            self.info = None
            self.account = None
            self.exchange = None
            return

        # Always initialize Info (public data)
        try:
            self.info = Info(self.base_url, skip_ws=True)
            logger.info("Connected to Hyperliquid Info API")
        except Exception as e:
            logger.error(f"Failed to connect to Hyperliquid Info: {e}")
            raise

        # Initialize Exchange (private data) if key provided
        if not self.private_key:
            logger.warning("No Private Key found. ExchangeClient is in READ-ONLY mode.")
            self.account = None
            self.exchange = None
        else:
            try:
                self.account = Account.from_key(self.private_key)
                self.exchange = Exchange(self.account, self.base_url, self.account_address)
                logger.info(f"Connected to Hyperliquid Exchange: {self.account.address[:8]}...")
            except Exception as e:
                logger.error(f"Failed to connect to Exchange: {e}")
                raise

    def round_size(self, coin: str, size: float) -> float:
        """
        Round size to exchange-accepted precision.

        Args:
            coin: Asset symbol (e.g., 'BTC', 'ETH')
            size: Raw size value

        Returns:
            Rounded size value
        """
        decimals = SIZE_DECIMALS.get(coin, DEFAULT_SIZE_DECIMALS)

        # Use Decimal for precise rounding
        d = Decimal(str(size))
        factor = Decimal(10) ** -decimals
        rounded = float(d.quantize(factor, rounding=ROUND_DOWN))

        return rounded

    def get_slippage_price(self, price: float, is_buy: bool) -> float:
        """
        Calculate limit price with slippage tolerance.

        Args:
            price: Current market price
            is_buy: True for buy orders, False for sell

        Returns:
            Limit price adjusted for slippage
        """
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def execute_pair_batch(
        self,
        pair: str,
        direction: str,
        size_a: float,
        size_b: float,
        price_a: float,
        price_b: float
    ) -> bool:
        """
        Execute a pair trade as an atomic batch.

        Args:
            pair: Pair identifier (e.g., "ETH-BTC")
            direction: "SHORT_SPREAD" or "LONG_SPREAD"
            size_a: Size for first leg (coin Y)
            size_b: Size for second leg (coin X)
            price_a: Current price of coin Y
            price_b: Current price of coin X

        Returns:
            True if both legs filled successfully
        """
        if not self.exchange:
            logger.info(f"Mock Execution: {pair} {direction} (No keys)")
            self._log_execution(pair, direction, size_a, size_b, True, "MOCK")
            return True

        coin_a, coin_b = pair.split('-')

        # Determine order directions
        if direction == "SHORT_SPREAD":
            is_buy_a = False  # Short Y
            is_buy_b = True   # Long X
        else:
            is_buy_a = True   # Long Y
            is_buy_b = False  # Short X

        # Round sizes to exchange precision
        size_a = self.round_size(coin_a, size_a)
        size_b = self.round_size(coin_b, size_b)

        # Check for zero sizes after rounding
        if size_a <= 0 or size_b <= 0:
            logger.error(f"Size too small after rounding: {coin_a}={size_a}, {coin_b}={size_b}")
            return False

        logger.info(f"EXECUTING BATCH: {pair} | {direction}")
        logger.info(f"  {coin_a}: {'BUY' if is_buy_a else 'SELL'} {size_a} @ ~{price_a:.2f}")
        logger.info(f"  {coin_b}: {'BUY' if is_buy_b else 'SELL'} {size_b} @ ~{price_b:.2f}")

        # Calculate limit prices with slippage
        limit_a = self.get_slippage_price(price_a, is_buy_a)
        limit_b = self.get_slippage_price(price_b, is_buy_b)

        # Build orders
        order_a = {
            'coin': coin_a,
            'is_buy': is_buy_a,
            'sz': size_a,
            'limit_px': limit_a,
            'order_type': {'limit': {'tif': 'Ioc'}},  # Immediate-or-Cancel
            'reduce_only': False
        }
        order_b = {
            'coin': coin_b,
            'is_buy': is_buy_b,
            'sz': size_b,
            'limit_px': limit_b,
            'order_type': {'limit': {'tif': 'Ioc'}},
            'reduce_only': False
        }

        # Execute with retry
        for attempt in range(self.max_retries):
            try:
                responses = self.exchange.bulk_orders([order_a, order_b])
                success = self._verify_batch_fill(responses, pair, order_a, order_b)

                if success:
                    self._log_execution(pair, direction, size_a, size_b, True, "FILLED")
                    return True

                # If partial fill, don't retry
                if self._had_partial_fill(responses):
                    return False

            except Exception as e:
                logger.error(f"Execution error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        self._log_execution(pair, direction, size_a, size_b, False, "FAILED")
        return False

    def _verify_batch_fill(
        self,
        responses: Dict,
        pair: str,
        order_a: Dict,
        order_b: Dict
    ) -> bool:
        """
        Verify batch order execution and handle partial fills.

        Returns:
            True if both legs fully filled
        """
        try:
            statuses = responses['response']['data']['statuses']

            # For IOC orders, only 'filled' counts as success
            # 'resting' means the order is sitting unfilled
            def is_filled(status) -> Tuple[bool, float]:
                """Returns (is_filled, filled_size)"""
                if isinstance(status, dict):
                    if 'filled' in status:
                        filled_info = status['filled']
                        filled_sz = float(filled_info.get('totalSz', 0))
                        return True, filled_sz
                    return False, 0.0
                return status == 'filled', 0.0

            filled_a, sz_a = is_filled(statuses[0])
            filled_b, sz_b = is_filled(statuses[1])

            if filled_a and filled_b:
                logger.info(f"BATCH COMPLETE: Both legs filled ({sz_a}, {sz_b})")
                return True

            if not filled_a and not filled_b:
                logger.warning(f"BATCH REJECTED: Neither leg filled - {statuses}")
                return False

            # Partial fill - one leg succeeded, other failed
            logger.critical(f"LEGGING ERROR: Partial fill detected!")
            logger.critical(f"  Leg A ({order_a['coin']}): {'FILLED' if filled_a else 'FAILED'}")
            logger.critical(f"  Leg B ({order_b['coin']}): {'FILLED' if filled_b else 'FAILED'}")

            # Emergency close the filled leg
            if filled_a and not filled_b:
                self._emergency_close_leg(order_a['coin'], sz_a, not order_a['is_buy'])
            elif filled_b and not filled_a:
                self._emergency_close_leg(order_b['coin'], sz_b, not order_b['is_buy'])

            return False

        except Exception as e:
            logger.error(f"Error verifying batch response: {e}")
            logger.error(f"Raw response: {responses}")
            return False

    def _had_partial_fill(self, responses: Dict) -> bool:
        """Check if response indicates a partial fill situation."""
        try:
            statuses = responses['response']['data']['statuses']
            filled_count = sum(1 for s in statuses if isinstance(s, dict) and 'filled' in s)
            return filled_count == 1  # Exactly one leg filled
        except Exception:
            return False

    def _emergency_close_leg(self, coin: str, size: float, close_is_buy: bool):
        """
        Emergency close a single leg position.

        Args:
            coin: Asset to close
            size: Size to close
            close_is_buy: True to buy back (close short), False to sell (close long)
        """
        logger.info(f"EMERGENCY CLOSE: {coin} size={size} buy={close_is_buy}")

        # Get current price for limit calculation
        limit_px = self._get_emergency_price(coin, close_is_buy)

        # Round size
        size = self.round_size(coin, size)

        order = {
            'coin': coin,
            'is_buy': close_is_buy,
            'sz': size,
            'limit_px': limit_px,
            'order_type': {'limit': {'tif': 'Ioc'}},
            'reduce_only': True  # Important: reduce only
        }

        try:
            res = self.exchange.order(
                order['coin'],
                order['is_buy'],
                order['sz'],
                order['limit_px'],
                order['order_type'],
                order['reduce_only']
            )

            status = res['response']['data']['statuses'][0]
            if isinstance(status, dict) and 'filled' in status:
                logger.info(f"Emergency close successful: {coin}")
            else:
                logger.critical(f"Emergency close FAILED: {coin} - {status}")

        except Exception as e:
            logger.critical(f"Emergency close EXCEPTION: {coin} - {e}")

    def _get_emergency_price(self, coin: str, is_buy: bool) -> float:
        """
        Get emergency limit price with extra slippage.

        Args:
            coin: Asset symbol
            is_buy: True for buy, False for sell

        Returns:
            Aggressive limit price
        """
        try:
            if self.info:
                all_mids = self.info.all_mids()
                price = float(all_mids.get(coin, 0.0))
                if price > 0:
                    # Use 5% slippage for emergency
                    emergency_slippage = 0.05
                    if is_buy:
                        return price * (1 + emergency_slippage)
                    else:
                        return price * (1 - emergency_slippage)
        except Exception as e:
            logger.warning(f"Price fetch failed for emergency close: {e}")

        # Fallback to extreme but reasonable prices
        logger.warning(f"Using fallback emergency price for {coin}")
        return 999999.0 if is_buy else 0.01

    def _log_execution(
        self,
        pair: str,
        direction: str,
        size_a: float,
        size_b: float,
        success: bool,
        status: str
    ):
        """Log execution for analysis."""
        self.execution_log.append({
            'timestamp': time.time(),
            'pair': pair,
            'direction': direction,
            'size_a': size_a,
            'size_b': size_b,
            'success': success,
            'status': status
        })

    def get_open_positions(self) -> List[Dict]:
        """
        Fetch current open positions from exchange.

        Returns:
            List of position dictionaries
        """
        if not self.info or not self.account_address:
            return []

        try:
            user_state = self.info.user_state(self.account_address)
            positions = user_state.get('assetPositions', [])
            return [
                {
                    'coin': p['position']['coin'],
                    'size': float(p['position']['szi']),
                    'entry_px': float(p['position']['entryPx']),
                    'unrealized_pnl': float(p['position']['unrealizedPnl']),
                    'leverage': float(p['position']['leverage']['value'])
                }
                for p in positions
                if float(p['position']['szi']) != 0
            ]
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_account_value(self) -> float:
        """Get current account equity value."""
        if not self.info or not self.account_address:
            return 0.0

        try:
            user_state = self.info.user_state(self.account_address)
            return float(user_state.get('marginSummary', {}).get('accountValue', 0))
        except Exception as e:
            logger.error(f"Failed to fetch account value: {e}")
            return 0.0


if __name__ == "__main__":
    print("Testing Exchange Client...")

    client = ExchangeClient()

    # Test size rounding
    print("\n1. Size Rounding Test:")
    test_cases = [
        ('BTC', 0.12345678),
        ('ETH', 1.23456),
        ('SOL', 12.345),
        ('DOGE', 1234.567),
    ]
    for coin, size in test_cases:
        rounded = client.round_size(coin, size)
        print(f"   {coin}: {size} -> {rounded}")

    # Test slippage calculation
    print("\n2. Slippage Price Test:")
    price = 100.0
    print(f"   Base price: ${price}")
    print(f"   Buy limit (2% slippage): ${client.get_slippage_price(price, True):.2f}")
    print(f"   Sell limit (2% slippage): ${client.get_slippage_price(price, False):.2f}")

    # Test mock execution
    print("\n3. Mock Execution Test:")
    success = client.execute_pair_batch(
        pair="ETH-BTC",
        direction="SHORT_SPREAD",
        size_a=0.5,
        size_b=0.01,
        price_a=2000.0,
        price_b=45000.0
    )
    print(f"   Execution result: {success}")
    print(f"   Execution log: {client.execution_log}")
