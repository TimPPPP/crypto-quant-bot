import asyncio
import os
import signal
import logging
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Set

# Import Modules
from src.utils.universe import get_liquid_universe
from src.execution.store import StateManager, StateCorruptionError
from src.execution.exchange_client import ExchangeClient
from src.execution.risk import RiskEngine
from src.execution.executor import TradeExecutor
from src.models.kalman import KalmanFilterRegime
from src.models.coint_scanner import CointegrationScanner
from src.features.clustering import get_cluster_map
from src.adaptive.online_adaptation import AdaptiveController, apply_overrides_to_live

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Orchestrator")

# Configuration from environment
DEFAULT_EQUITY = float(os.getenv('RISK_EQUITY', 10000.0))
DEFAULT_RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
DEFAULT_MAX_LEVERAGE = float(os.getenv('RISK_MAX_LEVERAGE', 2.0))
MAX_CONSECUTIVE_ERRORS = int(os.getenv('MAX_CONSECUTIVE_ERRORS', 5))

# Timing constants (seconds)
TICK_LOOP_SLEEP_INTERVAL = int(os.getenv('TICK_LOOP_INTERVAL', 30))
TICK_LOOP_POST_PROCESS_SLEEP = int(os.getenv('TICK_LOOP_POST_SLEEP', 65))
TICK_LOOP_ERROR_SLEEP = int(os.getenv('TICK_LOOP_ERROR_SLEEP', 60))
RISK_LOOP_INTERVAL = int(os.getenv('RISK_LOOP_INTERVAL', 10))
SCANNER_LOOP_INTERVAL = int(os.getenv('SCANNER_LOOP_INTERVAL', 86400))

# Exit thresholds
EXIT_Z_THRESHOLD_LONG = float(os.getenv('EXIT_Z_THRESHOLD_LONG', -0.5))
EXIT_Z_THRESHOLD_SHORT = float(os.getenv('EXIT_Z_THRESHOLD_SHORT', 0.5))
ADAPTIVE_LOOP_INTERVAL = int(os.getenv('ADAPTIVE_LOOP_INTERVAL', 3600))

# Top level function for multiprocessing (Pickle requirement)
def _run_heavy_scan_job():
    logger.info("   [Bg] Fetching Universe & Clustering...")
    cluster_map = get_cluster_map(lookback_days=60)
    if not cluster_map: return None
    logger.info(f"   [Bg] Scanning {len(cluster_map)} clusters...")
    scanner = CointegrationScanner(cluster_map=cluster_map)
    return scanner.find_pairs() 

class TradingBot:
    def __init__(self):
        # 1. Initialize Components
        self.state_manager = StateManager()
        self.exchange = ExchangeClient()
        self.risk_engine = RiskEngine(
            total_equity=DEFAULT_EQUITY,
            risk_per_trade=DEFAULT_RISK_PER_TRADE,
            max_leverage=DEFAULT_MAX_LEVERAGE
        )
        self.trade_executor = TradeExecutor()
        self.exit_z_short = EXIT_Z_THRESHOLD_SHORT
        self.exit_z_long = EXIT_Z_THRESHOLD_LONG
        self._adaptive_entry_z = float(KalmanFilterRegime.DEFAULT_ENTRY_Z)
        self.adaptive = AdaptiveController()
        saved_overrides = self.adaptive.load_saved_overrides()
        if saved_overrides:
            self._apply_adaptive_overrides(saved_overrides)

        # 2. Strategy Memory
        self.active_pairs = {}
        self.retiring_pairs = {}
        self._pending_orders: Set[str] = set()
        self._pending_orders_lock = asyncio.Lock()

        # 3. Execution State
        self.is_running = True
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.consecutive_errors = 0  # Circuit breaker counter
        self._shutdown_event = asyncio.Event()

        # 4. Setup signal handlers
        self._setup_signal_handlers()

        # 5. Load State and Reconcile
        self._restore_state()
        self._reconcile_with_exchange()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        loop = asyncio.get_event_loop()

        def signal_handler(sig):
            logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
            self._shutdown_event.set()
            self.is_running = False

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    def _restore_state(self):
        """Restore positions and Kalman states from disk."""
        saved_data = self.state_manager.load_all()
        logger.info(f"Found {len(saved_data)} saved positions.")
        for pair_id, data in saved_data.items():
            if data.get('is_active'):
                kf = KalmanFilterRegime(entry_z_threshold=self._current_entry_z())
                success = self.state_manager.hydrate_kalman_model(kf, pair_id)
                if success:
                    self.active_pairs[pair_id] = kf
                    # Register with risk engine
                    self.risk_engine.register_position(pair_id)
                    logger.info(f"Restored active strategy: {pair_id}")

    def _reconcile_with_exchange(self):
        """
        Verify saved positions match exchange state on startup.
        Prevents 'zombie positions' from corrupted state files.
        """
        if not self.exchange.info or not self.exchange.account_address:
            logger.warning("Exchange in read-only mode, skipping reconciliation")
            return

        try:
            exchange_positions = self.exchange.get_open_positions()
            exchange_coins = {p['coin'] for p in exchange_positions}

            saved_pairs = self.state_manager.get_all_active_pairs()

            # Check for orphaned positions (in saved state but not on exchange)
            for pair_id in saved_pairs:
                coin_a, coin_b = pair_id.split('-')
                # If neither coin has exchange position, state might be stale
                if coin_a not in exchange_coins and coin_b not in exchange_coins:
                    logger.warning(
                        f"RECONCILIATION WARNING: {pair_id} in saved state "
                        f"but no matching exchange positions found"
                    )

            # Update equity from exchange
            account_value = self.exchange.get_account_value()
            if account_value > 0:
                self.risk_engine.update_equity(account_value)
                logger.info(f"Synced equity from exchange: ${account_value:,.2f}")

        except Exception as e:
            logger.error(f"Reconciliation error: {e}")

    async def run_scanner_process(self):
        logger.info("STARTING DAILY SCAN (Background Process)...")
        loop = asyncio.get_running_loop()
        try:
            new_pairs_df = await loop.run_in_executor(self.executor, _run_heavy_scan_job)
            if new_pairs_df is None or new_pairs_df.empty:
                logger.warning("Scanner found no pairs. Keeping existing strategy.")
                return

            new_pair_ids = set(new_pairs_df['pair'].tolist())
            current_ids = set(self.active_pairs.keys())

            for pair in new_pair_ids:
                if pair not in current_ids:
                    logger.info(f"NEW PAIR FOUND: {pair}")
                    kf = KalmanFilterRegime(entry_z_threshold=self._current_entry_z())
                    kf.latest_z = 0.0
                    self.active_pairs[pair] = kf

            for pair in current_ids:
                if pair not in new_pair_ids:
                    pos = self.state_manager.get_position(pair)
                    if pos and pos.get('is_active'):
                        logger.info(f"RETIRING PAIR: {pair}")
                        self.retiring_pairs[pair] = self.active_pairs[pair]
                    else:
                        logger.info(f"REMOVING PAIR: {pair}")
                    if pair in self.active_pairs:
                        del self.active_pairs[pair]
        except Exception as e:
            logger.error(f"Scanner Process Failed: {e}")

    async def scanner_loop(self):
        logger.info("Scanner Loop Started (daily scan).")
        while self.is_running:
            try:
                await self.run_scanner_process()
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
            await asyncio.sleep(SCANNER_LOOP_INTERVAL)

    async def tick_loop(self):
        logger.info("Tick Loop Started (Swing Mode: Hourly).")
        while self.is_running:
            try:
                # Use UTC time for consistency across timezones
                now_utc = datetime.now(timezone.utc)
                if now_utc.minute != 0:
                    await asyncio.sleep(TICK_LOOP_SLEEP_INTERVAL)
                    continue

                all_prices = self.exchange.info.all_mids()
                all_strategy_pairs = {**self.active_pairs, **self.retiring_pairs}

                for pair_id, kf in all_strategy_pairs.items():
                    coin_y, coin_x = pair_id.split('-')
                    py = float(all_prices.get(coin_y, 0))
                    px = float(all_prices.get(coin_x, 0))

                    # Validate prices before log transform
                    if py <= 0 or px <= 0:
                        logger.warning(f"Invalid price for {pair_id}: {coin_y}={py}, {coin_x}={px}")
                        continue

                    res = kf.update(np.log(py), np.log(px))

                    # Entry (Active Only, warmed up filter only)
                    if res['is_signal'] and res['is_warmed_up'] and pair_id in self.active_pairs:
                        await self.execute_entry(pair_id, res, py, px)

                    # Exit (Active + Retiring)
                    position_data = self.state_manager.get_position(pair_id)
                    if position_data:
                        entry_z = position_data['trade'].get('entry_z', 0)
                        current_z = res['z_score']
                        should_exit = False

                        # Exit when spread reverts past the zero line
                        # SHORT_SPREAD (entry_z > 0): exit when z drops below threshold
                        # LONG_SPREAD (entry_z < 0): exit when z rises above threshold
                        if entry_z > 0 and current_z < self.exit_z_short:
                            should_exit = True
                        if entry_z < 0 and current_z > self.exit_z_long:
                            should_exit = True

                        if should_exit:
                            await self.execute_exit(pair_id, position_data['trade'], py, px, "Take Profit")

                # Reset error counter on success
                self.consecutive_errors = 0

                # Sleep past the minute boundary to avoid double processing
                await asyncio.sleep(TICK_LOOP_POST_PROCESS_SLEEP)

            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Tick Loop Error ({self.consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")

                # Circuit breaker
                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical("Too many consecutive errors, initiating shutdown")
                    self.shutdown()
                    break

                await asyncio.sleep(TICK_LOOP_ERROR_SLEEP)
                
    async def risk_loop(self):
        logger.info("Risk Loop Started.")
        while self.is_running:
            try:
                # Update equity from exchange periodically
                account_value = self.exchange.get_account_value()
                if account_value > 0:
                    self.risk_engine.update_equity(account_value)

                if self.risk_engine.check_kill_switch():
                    logger.critical("KILL SWITCH TRIGGERED - INITIATING EMERGENCY SHUTDOWN")
                    self.shutdown()
                    break

                all_prices = self.exchange.info.all_mids()
                all_strategy_pairs = {**self.active_pairs, **self.retiring_pairs}

                for pair, kf in all_strategy_pairs.items():
                    current_z = getattr(kf, 'latest_z', 0.0)
                    pos = self.state_manager.get_position(pair)
                    has_position = pos is not None

                    pnl_return = None
                    if pos:
                        trade = pos.get('trade', {})
                        direction = trade.get('direction')
                        entry_py = float(trade.get('price_a', 0))
                        entry_px = float(trade.get('price_b', 0))
                        size_a = float(trade.get('size_a', 0))
                        size_b = float(trade.get('size_b', 0))
                        coin_y, coin_x = pair.split('-')
                        py = float(all_prices.get(coin_y, 0))
                        px = float(all_prices.get(coin_x, 0))
                        if entry_py > 0 and entry_px > 0 and py > 0 and px > 0:
                            notional = entry_py * size_a + entry_px * size_b
                            if notional > 0:
                                if direction == "LONG_SPREAD":
                                    pnl_y = (py - entry_py) * size_a
                                    pnl_x = (entry_px - px) * size_b
                                else:  # SHORT_SPREAD
                                    pnl_y = (entry_py - py) * size_a
                                    pnl_x = (px - entry_px) * size_b
                                pnl_return = (pnl_y + pnl_x) / notional

                    # Pass is_open_position to only check stop loss when we have a position
                    if self.risk_engine.check_stop_loss(
                        pair,
                        current_z,
                        is_open_position=has_position,
                        pnl_return=pnl_return,
                    ):
                        if pos:
                            coin_y, coin_x = pair.split('-')
                            py = float(all_prices.get(coin_y, 0))
                            px = float(all_prices.get(coin_x, 0))
                            if py > 0 and px > 0:
                                await self.execute_exit(pair, pos['trade'], py, px, "STOP LOSS")
            except Exception as e:
                logger.error(f"Risk Loop Error: {e}")
            await asyncio.sleep(RISK_LOOP_INTERVAL)

    async def execute_entry(self, pair, signal_data, price_a, price_b):
        """Execute entry for a pair trade."""
        # Thread-safe race condition check using async lock
        async with self._pending_orders_lock:
            if pair in self._pending_orders:
                return
            if self.state_manager.get_position(pair):
                return
            # Mark as pending
            self._pending_orders.add(pair)

        try:
            z_score = signal_data['z_score']

            # Risk Gate: Don't enter if stretched or limits exceeded
            if not self.risk_engine.check_entry_signal(pair, z_score):
                return

            # Accountant Gate: Calculate EV
            ev_signal = {
                'pair': pair,
                'z_score': z_score,
                'spread_std': signal_data.get('spread_std', 0.01),
                'half_life_hours': signal_data.get('half_life_hours', 24)
            }

            if not self.trade_executor.calculate_ev(ev_signal):
                logger.info(f"REJECTED: {pair} (Low EV)")
                return

            logger.info(f"GO SIGNAL: {pair} Z={z_score:.2f} | EV Approved")

            # Dynamic Volatility Sizing using spread std
            volatility_pct = signal_data.get('spread_std', 0.02)
            if volatility_pct < 0.001:
                volatility_pct = 0.02

            size_a = self.risk_engine.calculate_size(price_a, volatility_pct)
            beta = signal_data['hedge_ratio']
            size_b = size_a * abs(beta) * (price_a / price_b)

            direction = "SHORT_SPREAD" if z_score > 0 else "LONG_SPREAD"

            success = self.exchange.execute_pair_batch(pair, direction, size_a, size_b, price_a, price_b)

            if success:
                trade_record = {
                    'size_a': size_a, 'size_b': size_b,
                    'price_a': price_a, 'price_b': price_b,
                    'entry_z': z_score,
                    'direction': direction,
                    'entry_ts': datetime.now(timezone.utc).isoformat(),
                }
                self.state_manager.save_position(pair, trade_record, self.active_pairs[pair])
                # Register position with risk engine
                self.risk_engine.register_position(pair)
                # Track in executor
                self.trade_executor.add_active_position(pair)

        finally:
            # Always release lock
            async with self._pending_orders_lock:
                self._pending_orders.discard(pair)

    async def execute_exit(self, pair, trade_data, price_a, price_b, reason="Exit"):
        """Execute exit for a pair trade."""
        # Thread-safe race condition check using async lock
        async with self._pending_orders_lock:
            if pair in self._pending_orders:
                return
            self._pending_orders.add(pair)

        try:
            logger.info(f"{reason}: Closing {pair}...")
            entry_dir = trade_data['direction']
            exit_dir = "LONG_SPREAD" if entry_dir == "SHORT_SPREAD" else "SHORT_SPREAD"

            success = self.exchange.execute_pair_batch(
                pair, exit_dir,
                trade_data['size_a'], trade_data['size_b'],
                price_a, price_b
            )

            if success:
                self.state_manager.close_position(pair)
                # Unregister from risk engine
                self.risk_engine.unregister_position(pair)
                # Remove from executor tracking
                self.trade_executor.remove_active_position(pair)
                self._record_exit_trade(pair, trade_data, price_a, price_b, reason)

                if pair in self.retiring_pairs:
                    del self.retiring_pairs[pair]
                    logger.info(f"Retired pair {pair} removed.")
        finally:
            async with self._pending_orders_lock:
                self._pending_orders.discard(pair)

    def _record_exit_trade(self, pair, trade_data, price_a, price_b, reason) -> None:
        entry_py = float(trade_data.get('price_a', 0))
        entry_px = float(trade_data.get('price_b', 0))
        size_a = float(trade_data.get('size_a', 0))
        size_b = float(trade_data.get('size_b', 0))
        direction = trade_data.get('direction')
        entry_ts = trade_data.get("entry_ts")
        exit_ts = datetime.now(timezone.utc).isoformat()

        pnl_return = None
        if entry_py > 0 and entry_px > 0 and price_a > 0 and price_b > 0:
            notional = entry_py * size_a + entry_px * size_b
            if notional > 0:
                if direction == "LONG_SPREAD":
                    pnl_y = (price_a - entry_py) * size_a
                    pnl_x = (entry_px - price_b) * size_b
                else:
                    pnl_y = (entry_py - price_a) * size_a
                    pnl_x = (price_b - entry_px) * size_b
                pnl_return = (pnl_y + pnl_x) / notional

        hold_hours = None
        if entry_ts:
            try:
                entry_dt = datetime.fromisoformat(str(entry_ts))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                exit_dt = datetime.fromisoformat(exit_ts)
                if exit_dt.tzinfo is None:
                    exit_dt = exit_dt.replace(tzinfo=timezone.utc)
                hold_hours = (exit_dt - entry_dt).total_seconds() / 3600.0
            except Exception:
                hold_hours = None

        record = {
            "pair": pair,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "entry_z": trade_data.get("entry_z"),
            "exit_reason": reason,
            "pnl_return": pnl_return,
            "hold_hours": hold_hours,
        }
        self.adaptive.record_trade(record)

    def _current_entry_z(self) -> float:
        for kf in self.active_pairs.values():
            return float(getattr(kf, "entry_z_threshold", KalmanFilterRegime.DEFAULT_ENTRY_Z))
        return float(self._adaptive_entry_z)

    def _current_adaptive_params(self) -> dict:
        return {
            "ENTRY_Z": self._current_entry_z(),
            "EXIT_Z": abs(float(self.exit_z_short)),
            "MIN_PROFIT_HURDLE": float(self.trade_executor.MIN_NET_PROFIT),
            "MAX_PORTFOLIO_POSITIONS": int(self.risk_engine.max_positions),
            "MAX_POSITIONS_PER_COIN": int(self.risk_engine.max_positions_per_coin),
            "STOP_LOSS_PCT": float(self.risk_engine.stop_loss_pct),
        }

    def _apply_adaptive_overrides(self, overrides: dict) -> None:
        if "ENTRY_Z" in overrides:
            self._adaptive_entry_z = float(overrides["ENTRY_Z"])
        apply_overrides_to_live(self, overrides)
        logger.info("Applied adaptive overrides: %s", overrides)

    async def start(self):
        logger.info("SYSTEM STARTUP INITIATED")
        logger.info(f"Configuration: Equity=${DEFAULT_EQUITY:,.0f}, Risk={DEFAULT_RISK_PER_TRADE*100:.1f}%, Leverage={DEFAULT_MAX_LEVERAGE}x")
        logger.info(f"Active pairs: {list(self.active_pairs.keys())}")

        try:
            # Run all loops concurrently
            await asyncio.gather(
                self.tick_loop(),
                self.risk_loop(),
                self.scanner_loop(),
                self.adaptive_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Main loop cancelled. Cleaning up...")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the trading bot."""
        if not self.is_running:
            return  # Already shutting down

        logger.info("SHUTDOWN SIGNAL RECEIVED.")
        self.is_running = False

        # Log final state
        try:
            status = self.risk_engine.get_status()
            logger.info(f"Final equity: ${status['current_equity']:,.2f}")
            logger.info(f"Drawdown from peak: {status['drawdown_from_peak']*100:.2f}%")
            logger.info(f"Active positions at shutdown: {status['active_positions']}")
        except Exception as e:
            logger.error(f"Error getting final status: {e}")

        # Kill the background process pool
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Process pool killed.")
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")

        logger.info("Shutdown complete.")

    async def adaptive_loop(self):
        logger.info("Adaptive Loop Started.")
        while self.is_running:
            try:
                overrides, stats = self.adaptive.maybe_update_live(self._current_adaptive_params())
                if overrides:
                    self._apply_adaptive_overrides(overrides)
                    logger.info("Adaptive update applied. Stats=%s", stats.__dict__ if stats else None)
            except Exception as e:
                logger.error(f"Adaptive Loop Error: {e}")
            await asyncio.sleep(ADAPTIVE_LOOP_INTERVAL)


if __name__ == "__main__":
    try:
        bot = TradingBot()
        asyncio.run(bot.start())
    except StateCorruptionError as e:
        logger.critical(f"State corruption detected: {e}")
        logger.critical("Please verify exchange positions manually before restarting.")
        sys.exit(1)
    except KeyboardInterrupt:
        # This catches the physical Ctrl+C in the terminal
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
