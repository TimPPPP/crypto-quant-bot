import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from tqdm import tqdm
from numba import njit

# --- IMPORTS ---
from src.models.kalman import KalmanFilterRegime
from src.models.coint_scanner import CointegrationScanner

# --- CONFIGURATION ---
INPUT_FILE = os.getenv('BACKTEST_INPUT_FILE', "data/raw_downloads/crypto_prices_1m.parquet")
OUTPUT_DIR = os.getenv('BACKTEST_OUTPUT_DIR', "data/backtest_ready")
TEST_RATIO = float(os.getenv('BACKTEST_TEST_RATIO', 0.20))

# Match Kalman parameters with production
KALMAN_DELTA = float(os.getenv('KALMAN_DELTA', 1e-6))
KALMAN_R = float(os.getenv('KALMAN_R', 1e-2))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Step1_Prep")

# --- NUMBA ACCELERATION ---
@njit
def fast_warmup_loop(y_arr, x_arr, delta=1e-6, R=1e-2):
    """
    Runs the Kalman logic in compiled C-speed.
    Returns the final State Vector (x) and Covariance (P).

    Note: delta and R should match production KalmanFilterRegime defaults.
    """
    # State [Beta, Alpha]
    x = np.zeros(2)
    P = np.eye(2)

    # Process Noise (matches KalmanFilterRegime)
    Q = np.eye(2) * delta
    Q[0, 0] = delta / 100  # Beta changes 100x slower than alpha

    n = len(y_arr)

    for i in range(n):
        # Skip NaNs (Data gaps)
        if np.isnan(y_arr[i]) or np.isnan(x_arr[i]):
            continue

        # Observation Matrix H = [x, 1]
        H = np.array([x_arr[i], 1.0])

        # Predict
        P_pred = P + Q

        # Bound covariance for stability
        for j in range(2):
            for k in range(2):
                if P_pred[j, k] > 1e6:
                    P_pred[j, k] = 1e6
                if P_pred[j, k] < 1e-10:
                    P_pred[j, k] = 1e-10

        y_pred = np.dot(H, x)
        error = y_arr[i] - y_pred

        # Update
        S = np.dot(H, np.dot(P_pred, H.T)) + R
        if S < 1e-10:
            continue

        K = np.dot(P_pred, H.T) / S

        # Bound Kalman gain
        for j in range(2):
            if K[j] > 1.0:
                K[j] = 1.0
            if K[j] < -1.0:
                K[j] = -1.0

        x = x + (K * error)
        P = P_pred - np.outer(K, H) @ P_pred

    return x, P

def load_and_split():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        sys.exit(1)

    logger.info("1️⃣ Loading 1-Minute Data...")
    df = pd.read_parquet(INPUT_FILE)
    df.sort_index(inplace=True)
    
    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"   ✂️ Split: {len(train_df):,} Train rows | {len(test_df):,} Test rows")
    return train_df, test_df

def run_discovery_on_train(train_df):
    logger.info("2️⃣ Running Cointegration Scanner (Training Phase)...")
    try:
        # Pass data directly or via init, depending on your implementation
        scanner = CointegrationScanner(cluster_map=None)
        scanner.price_data = train_df 
        
        candidates = scanner.find_pairs()
        
        if isinstance(candidates, pd.DataFrame):
            valid_pairs = candidates['pair'].tolist()
        else:
            valid_pairs = candidates
            
        if not valid_pairs:
            logger.warning("⚠️ Scanner found 0 pairs. Using fallback.")
            valid_pairs = ['ETH-BTC', 'SOL-ETH']
            
        logger.info(f"   ✅ Discovered {len(valid_pairs)} valid pairs.")
        return valid_pairs
        
    except Exception as e:
        logger.error(f"❌ Scanner failed: {e}")
        sys.exit(1)

def warm_up_kalman(train_df, valid_pairs):
    """Warm up Kalman filters using training data."""
    logger.info("Warming up Kalman Filters (Numba Accelerated)...")

    initial_states = {}

    # Pre-fetch numpy arrays to avoid overhead in loop
    df_values = train_df.to_dict('series')

    for pair in tqdm(valid_pairs, desc="Fast Warmup"):
        try:
            coin_y, coin_x = pair.split('-')

            if coin_y not in df_values or coin_x not in df_values:
                logger.warning(f"Missing data for {pair}, skipping")
                continue

            # Get price series
            y_prices = df_values[coin_y].values
            x_prices = df_values[coin_x].values

            # Validate positive prices before log transform
            if (y_prices <= 0).any() or (x_prices <= 0).any():
                logger.warning(f"Non-positive prices in {pair}, skipping")
                continue

            # Convert to Log Prices
            y = np.log(y_prices).astype(np.float64)
            x = np.log(x_prices).astype(np.float64)

            # Run fast warmup with production parameters
            final_x, final_P = fast_warmup_loop(y, x, delta=KALMAN_DELTA, R=KALMAN_R)

            initial_states[pair] = {
                'x': final_x.tolist(),
                'P': final_P.tolist(),
                'delta': KALMAN_DELTA,
                'R': KALMAN_R,
                'latest_beta': float(final_x[0])
            }

        except Exception as e:
            logger.warning(f"Failed to warm {pair}: {e}")

    if not initial_states:
        logger.error("No valid pairs warmed up! Cannot proceed.")
        sys.exit(1)

    logger.info(f"Warmed up {len(initial_states)} pairs")
    return initial_states

def save_artifacts(test_df, initial_states, valid_pairs):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # --- FIX: SAFE FILTERING ---
    relevant_coins = set()
    for p in valid_pairs:
        c1, c2 = p.split('-')
        relevant_coins.add(c1)
        relevant_coins.add(c2)
    
    # Only keep columns that actually exist in test_df
    # This prevents the KeyError crash
    available_cols = [c for c in relevant_coins if c in test_df.columns]
    missing_cols = relevant_coins - set(available_cols)
    
    if missing_cols:
        logger.warning(f"   ⚠️ Dropping {len(missing_cols)} coins missing from Test Data.")
        
    final_test_df = test_df[available_cols].copy()
    
    test_path = os.path.join(OUTPUT_DIR, "test_market_data.parquet")
    final_test_df.to_parquet(test_path)
    
    state_path = os.path.join(OUTPUT_DIR, "warm_start_states.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(initial_states, f)
        
    logger.info("-" * 40)
    logger.info(f"🏁 DONE. Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    train_df, test_df = load_and_split()
    pairs = run_discovery_on_train(train_df)
    states = warm_up_kalman(train_df, pairs)
    save_artifacts(test_df, states, pairs)