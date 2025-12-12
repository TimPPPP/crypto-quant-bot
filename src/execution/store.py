import json
import os
import shutil
import sys
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StateManager")


class StateManager:
    """
    State persistence manager for the trading bot.

    Features:
    - Atomic writes to prevent corruption
    - In-memory caching to reduce disk reads
    - Automatic backup before modifications
    - Kalman filter state serialization/deserialization
    """

    def __init__(self, filepath: str = "data/state/positions.json"):
        """
        Initialize State Manager.

        Args:
            filepath: Path to the state file
        """
        self.filepath = filepath
        self.backup_filepath = filepath + ".backup"
        self._cache: Optional[Dict] = None
        self._cache_dirty = False

        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure the state directory exists."""
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created state directory: {directory}")

    def _create_backup(self):
        """Create backup of current state file."""
        if os.path.exists(self.filepath):
            try:
                shutil.copy2(self.filepath, self.backup_filepath)
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

    def save_position(self, pair_id: str, trade_data: Dict, kf_model: Any) -> bool:
        """
        Save a position with its Kalman filter state.

        Args:
            pair_id: Pair identifier (e.g., "ETH-BTC")
            trade_data: Trade details (size, price, direction, etc.)
            kf_model: KalmanFilterRegime instance

        Returns:
            True if saved successfully
        """
        try:
            # Load current state (uses cache if available)
            current_state = self.load_all()

            # Create backup before modification
            self._create_backup()

            # Extract Kalman state using the model's built-in method if available
            if hasattr(kf_model, 'get_state_dict'):
                kf_state = kf_model.get_state_dict()
            else:
                # Fallback for backwards compatibility
                raw_delta = kf_model.Q[1, 1] if hasattr(kf_model, 'Q') else 1e-6

                # Handle error_history which could be deque or list
                if hasattr(kf_model, 'error_history'):
                    error_hist = list(kf_model.error_history)
                else:
                    error_hist = []

                kf_state = {
                    'x': kf_model.x.tolist() if hasattr(kf_model.x, 'tolist') else list(kf_model.x),
                    'P': kf_model.P.tolist() if hasattr(kf_model.P, 'tolist') else [list(row) for row in kf_model.P],
                    'R': getattr(kf_model, 'R', 0.01),
                    'delta': raw_delta,
                    'error_history': error_hist
                }

            record = {
                'last_updated': datetime.utcnow().isoformat(),
                'pair_id': pair_id,
                'trade': trade_data,
                'kalman_state': kf_state,
                'is_active': True
            }

            current_state[pair_id] = record
            self._atomic_write(current_state)

            # Update cache
            self._cache = current_state

            beta = kf_model.x[0] if hasattr(kf_model, 'x') else 0
            logger.info(f"State saved: {pair_id} (Beta: {beta:.4f})")
            return True

        except Exception as e:
            logger.error(f"Failed to save position {pair_id}: {e}")
            return False

    def load_all(self) -> Dict:
        """
        Load all saved positions.

        Uses in-memory cache if available.

        Returns:
            Dictionary of all positions
        """
        # Return cached data if available
        if self._cache is not None:
            return self._cache.copy()

        if not os.path.exists(self.filepath):
            self._cache = {}
            return {}

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self._cache = data
                return data.copy()

        except json.JSONDecodeError:
            # Corruption detected - critical error
            self._handle_corruption()
            return {}

        except Exception as e:
            logger.error(f"Unknown error loading state: {e}")
            return {}

    def _handle_corruption(self):
        """Handle corrupted state file."""
        corrupt_path = self.filepath + ".CORRUPT"

        if os.path.exists(self.filepath):
            os.rename(self.filepath, corrupt_path)

        # Try to restore from backup
        if os.path.exists(self.backup_filepath):
            try:
                with open(self.backup_filepath, 'r') as f:
                    data = json.load(f)
                    self._atomic_write(data)
                    self._cache = data
                    logger.warning(f"Restored state from backup after corruption")
                    return
            except Exception:
                pass

        error_msg = (
            f"\n{'='*50}\n"
            f"CRITICAL ERROR: State file is CORRUPT!\n"
            f"Corrupted file moved to: {corrupt_path}\n"
            f"The bot has stopped to prevent 'Zombie Positions'.\n"
            f"Please manually verify your exchange positions.\n"
            f"{'='*50}\n"
        )
        logger.critical(error_msg)
        sys.exit(1)

    def get_position(self, pair_id: str) -> Optional[Dict]:
        """
        Get a specific position.

        Args:
            pair_id: Pair identifier

        Returns:
            Position data or None if not found
        """
        data = self.load_all()
        return data.get(pair_id)

    def close_position(self, pair_id: str) -> bool:
        """
        Remove a closed position from state.

        Args:
            pair_id: Pair identifier

        Returns:
            True if removed successfully
        """
        current_state = self.load_all()

        if pair_id not in current_state:
            logger.warning(f"Position not found: {pair_id}")
            return False

        self._create_backup()
        del current_state[pair_id]
        self._atomic_write(current_state)

        # Update cache
        self._cache = current_state

        logger.info(f"Position closed: {pair_id}")
        return True

    def update_position(self, pair_id: str, updates: Dict) -> bool:
        """
        Update specific fields of a position.

        Args:
            pair_id: Pair identifier
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully
        """
        current_state = self.load_all()

        if pair_id not in current_state:
            logger.warning(f"Position not found for update: {pair_id}")
            return False

        self._create_backup()
        current_state[pair_id].update(updates)
        current_state[pair_id]['last_updated'] = datetime.utcnow().isoformat()
        self._atomic_write(current_state)

        # Update cache
        self._cache = current_state

        return True

    def _atomic_write(self, data: Dict):
        """
        Write data atomically using temp file + rename.

        Args:
            data: Data to write
        """
        temp_path = self.filepath + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            os.replace(temp_path, self.filepath)

        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def hydrate_kalman_model(self, kf_instance: Any, pair_id: str) -> bool:
        """
        Restore Kalman filter state from saved data.

        Args:
            kf_instance: KalmanFilterRegime instance to hydrate
            pair_id: Pair identifier

        Returns:
            True if hydrated successfully
        """
        data = self.get_position(pair_id)
        if not data:
            logger.warning(f"No saved state for {pair_id}")
            return False

        try:
            saved_state = data['kalman_state']

            # Use built-in method if available
            if hasattr(kf_instance, 'load_state_dict'):
                success = kf_instance.load_state_dict(saved_state)
                if success:
                    logger.info(f"Kalman state restored for {pair_id}")
                return success

            # Fallback for backwards compatibility
            kf_instance.x = np.array(saved_state['x'])
            kf_instance.P = np.array(saved_state['P'])

            # Restore R
            if 'R' in saved_state:
                kf_instance.R = saved_state['R']

            # Restore Q from delta
            if 'delta' in saved_state:
                delta = saved_state['delta']
                kf_instance.n_dim = 2
                kf_instance.Q = np.eye(kf_instance.n_dim) * delta
                kf_instance.Q[0, 0] = delta / 100

            # Restore error history
            if hasattr(kf_instance, 'error_history'):
                kf_instance.error_history.clear()
                for err in saved_state.get('error_history', []):
                    kf_instance.error_history.append(err)

            logger.info(f"Kalman state restored for {pair_id} (Beta: {kf_instance.x[0]:.4f})")
            return True

        except Exception as e:
            logger.error(f"Failed to hydrate model for {pair_id}: {e}")
            return False

    def get_all_active_pairs(self) -> list:
        """Get list of all active pair IDs."""
        data = self.load_all()
        return [
            pair_id for pair_id, pos in data.items()
            if pos.get('is_active', False)
        ]

    def clear_cache(self):
        """Clear the in-memory cache to force reload from disk."""
        self._cache = None

    def get_state_summary(self) -> Dict:
        """Get summary of current state for logging."""
        data = self.load_all()
        return {
            'total_positions': len(data),
            'active_positions': len([p for p in data.values() if p.get('is_active')]),
            'pairs': list(data.keys()),
            'last_backup': os.path.getmtime(self.backup_filepath) if os.path.exists(self.backup_filepath) else None
        }


if __name__ == "__main__":
    print("Testing State Manager...")

    # Clean up test files
    test_path = "data/state/test_positions.json"
    for f in [test_path, test_path + ".backup", test_path + ".tmp"]:
        if os.path.exists(f):
            os.remove(f)

    mgr = StateManager(test_path)

    # Mock Kalman model
    class MockKF:
        def __init__(self):
            self.x = np.array([1.2345, 0.5])
            self.P = np.eye(2) * 0.1
            self.Q = np.eye(2) * 1e-5
            self.Q[0, 0] = 1e-7
            self.R = 0.01
            self.error_history = deque(maxlen=30)
            for i in range(10):
                self.error_history.append(0.001 * i)

    # Test save
    print("\n1. Save Test:")
    kf = MockKF()
    trade = {'size_a': 1.0, 'price_a': 2000.0, 'direction': 'LONG_SPREAD'}
    success = mgr.save_position("ETH-BTC", trade, kf)
    print(f"   Save result: {success}")

    # Test load
    print("\n2. Load Test:")
    loaded = mgr.get_position("ETH-BTC")
    print(f"   Loaded pair: {loaded['pair_id']}")
    print(f"   Beta: {loaded['kalman_state']['x'][0]:.4f}")

    # Test hydration
    print("\n3. Hydration Test:")
    kf_new = MockKF()
    kf_new.x = np.zeros(2)  # Reset
    success = mgr.hydrate_kalman_model(kf_new, "ETH-BTC")
    print(f"   Hydration result: {success}")
    print(f"   Restored beta: {kf_new.x[0]:.4f}")

    # Test backup
    print("\n4. Backup Test:")
    print(f"   Backup exists: {os.path.exists(test_path + '.backup')}")

    # Test summary
    print("\n5. Summary Test:")
    print(f"   {mgr.get_state_summary()}")

    # Clean up
    print("\n6. Cleanup:")
    mgr.close_position("ETH-BTC")
    print(f"   Positions after close: {mgr.get_state_summary()}")

    # Remove test files
    for f in [test_path, test_path + ".backup"]:
        if os.path.exists(f):
            os.remove(f)
    print("   Test files cleaned up")
