from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from src.backtest import config_backtest as cfg

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("step1_prep")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# -----------------------------------------------------------------------------
# Numba Kalman Warmup (per pair)
# -----------------------------------------------------------------------------
@njit
def fast_warmup_loop(
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    delta: float = 1e-6,
    R: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simple Kalman filter for a linear hedge ratio model:

        y_t = beta * x_t + alpha + noise

    Parameters
    ----------
    y_arr : np.ndarray
        Log prices of the dependent asset (Y).
    x_arr : np.ndarray
        Log prices of the independent asset (X).
    delta : float
        Process noise scalar.
    R : float
        Measurement noise scalar.

    Returns
    -------
    x : np.ndarray
        Final state vector [beta, alpha].
    P : np.ndarray
        Final 2x2 covariance matrix.
    """
    # State [beta, alpha]
    x = np.zeros(2)
    P = np.eye(2)

    # Process noise (with smaller variance for beta)
    Q = np.eye(2) * delta
    Q[0, 0] = delta / 100.0

    n = len(y_arr)

    for i in range(n):
        if np.isnan(y_arr[i]) or np.isnan(x_arr[i]):
            continue

        H = np.array([x_arr[i], 1.0])

        # Predict
        P_pred = P + Q

        # Stability clip on covariance
        for j in range(2):
            for k in range(2):
                if P_pred[j, k] > 1e6:
                    P_pred[j, k] = 1e6
                if P_pred[j, k] < 1e-10:
                    P_pred[j, k] = 1e-10

        y_pred = H[0] * x[0] + H[1] * x[1]
        error = y_arr[i] - y_pred

        # Innovation covariance
        S = H[0] * (P_pred[0, 0] * H[0] + P_pred[0, 1] * H[1]) + \
            H[1] * (P_pred[1, 0] * H[0] + P_pred[1, 1] * H[1]) + R
        if S < 1e-10:
            continue

        # Kalman gain K = P_pred H^T / S
        K0 = (P_pred[0, 0] * H[0] + P_pred[0, 1] * H[1]) / S
        K1 = (P_pred[1, 0] * H[0] + P_pred[1, 1] * H[1]) / S

        # Gain clip
        if K0 > 1.0:
            K0 = 1.0
        if K0 < -1.0:
            K0 = -1.0
        if K1 > 1.0:
            K1 = 1.0
        if K1 < -1.0:
            K1 = -1.0

        # Update state
        x[0] = x[0] + K0 * error
        x[1] = x[1] + K1 * error

        # Update covariance: P = (I - K H) P_pred
        KH00 = K0 * H[0]
        KH01 = K0 * H[1]
        KH10 = K1 * H[0]
        KH11 = K1 * H[1]

        I_KH00 = 1.0 - KH00
        I_KH01 = -KH01
        I_KH10 = -KH10
        I_KH11 = 1.0 - KH11

        P00 = I_KH00 * P_pred[0, 0] + I_KH01 * P_pred[1, 0]
        P01 = I_KH00 * P_pred[0, 1] + I_KH01 * P_pred[1, 1]
        P10 = I_KH10 * P_pred[0, 0] + I_KH11 * P_pred[1, 0]
        P11 = I_KH10 * P_pred[0, 1] + I_KH11 * P_pred[1, 1]

        P[0, 0] = P00
        P[0, 1] = P01
        P[1, 0] = P10
        P[1, 1] = P11

    return x, P


# -----------------------------------------------------------------------------
# Data loading & splitting
# -----------------------------------------------------------------------------
def load_and_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw 1-minute parquet file and split into train/test
    according to cfg.TEST_RATIO.
    """
    raw_path: Path = cfg.PATH_RAW_PARQUET  # from config_backtest
    if not raw_path.exists():
        msg = (
            f"Input file not found: {raw_path}\n"
            "Hint: run 'research/pipeline/step0_ingest.py' first."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("1️⃣ Loading 1-minute price data...")
    df = pd.read_parquet(raw_path)
    df = df.sort_index()

    split_idx = int(len(df) * (1.0 - cfg.TEST_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(
        "   ✂️ Split: %s train rows | %s test rows",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
    )
    return train_df, test_df


# -----------------------------------------------------------------------------
# Pair discovery (cointegration scanner)
# -----------------------------------------------------------------------------
def run_discovery_on_train(train_df: pd.DataFrame) -> List[str]:
    """
    Run the cointegration scanner on the training data to obtain valid pairs.

    Returns
    -------
    valid_pairs : list of str
        List of pair identifiers, e.g. ['ETH-BTC', 'SOL-ETH'].
    """
    logger.info("2️⃣ Running Cointegration Scanner (training phase)...")

    # Lazy import to keep this pipeline step decoupled from models package
    from src.models.coint_scanner import CointegrationScanner  # type: ignore

    scanner = CointegrationScanner(cluster_map=None)
    scanner.price_data = train_df

    candidates = scanner.find_pairs()
    if isinstance(candidates, pd.DataFrame) and "pair" in candidates.columns:
        valid_pairs = candidates["pair"].tolist()
    else:
        valid_pairs = list(candidates)

    if not valid_pairs:
        logger.warning("⚠️ Scanner found 0 pairs. Using fallback universe.")
        valid_pairs = ["ETH-BTC", "SOL-ETH"]

    logger.info("   ✅ Discovered %d valid pairs.", len(valid_pairs))
    return valid_pairs


# -----------------------------------------------------------------------------
# Kalman warm start
# -----------------------------------------------------------------------------
def warm_up_kalman(
    train_df: pd.DataFrame,
    valid_pairs: Iterable[str],
) -> Dict[str, Dict[str, object]]:
    """
    Run the Numba-accelerated warmup for each valid pair on the training set.

    Returns
    -------
    initial_states : dict
        Mapping pair -> {'x': [...], 'P': [[...]], 'delta': ..., 'R': ..., 'latest_beta': ...}
    """
    logger.info("3️⃣ Warming up Kalman filters (Numba accelerated)...")

    initial_states: Dict[str, Dict[str, object]] = {}
    # Ensure we only use columns that actually exist
    available_cols = set(train_df.columns)

    for pair in tqdm(list(valid_pairs), desc="Fast Warmup"):
        try:
            coin_y, coin_x = pair.split("-")

            if coin_y not in available_cols or coin_x not in available_cols:
                logger.warning("   Missing data for pair %s, skipping.", pair)
                continue

            y = np.log(train_df[coin_y].to_numpy(dtype=np.float64))
            x = np.log(train_df[coin_x].to_numpy(dtype=np.float64))

            final_x, final_P = fast_warmup_loop(
                y,
                x,
                delta=cfg.KALMAN_DELTA,
                R=cfg.KALMAN_R,
            )

            initial_states[pair] = {
                "x": final_x.tolist(),
                "P": final_P.tolist(),
                "delta": cfg.KALMAN_DELTA,
                "R": cfg.KALMAN_R,
                "latest_beta": float(final_x[0]),
            }

        except Exception as exc:  # pragma: no cover (defensive)
            logger.warning("Failed to warm pair %s: %s", pair, exc)

    if not initial_states:
        raise RuntimeError("No valid pairs warmed up; cannot proceed.")

    logger.info("   🔥 Warmed up %d pairs.", len(initial_states))
    return initial_states


# -----------------------------------------------------------------------------
# Artifact saving
# -----------------------------------------------------------------------------
def save_artifacts(
    test_df: pd.DataFrame,
    initial_states: Dict[str, Dict[str, object]],
    valid_pairs: Iterable[str],
) -> None:
    """
    Save test data (restricted to relevant coins) and warm-start states
    to the locations defined in config_backtest.
    """
    ready_dir: Path = cfg.READY_DATA_DIR
    ready_dir.mkdir(parents=True, exist_ok=True)

    # Only keep coins that actually appear in valid_pairs and in test_df
    relevant_coins = set()
    for p in valid_pairs:
        c1, c2 = p.split("-")
        relevant_coins.add(c1)
        relevant_coins.add(c2)

    available_cols = [c for c in relevant_coins if c in test_df.columns]

    dropped = len(relevant_coins) - len(available_cols)
    if dropped > 0:
        logger.warning("   ⚠️ Dropped %d coins missing from test data.", dropped)

    final_test_df = test_df[available_cols].copy()

    final_test_df.to_parquet(cfg.PATH_TEST_DATA)

    with cfg.PATH_STATE.open("wb") as f:
        pickle.dump(initial_states, f)

    logger.info("-" * 40)
    logger.info("🏁 Saved test data to %s", cfg.PATH_TEST_DATA)
    logger.info("💾 Saved warm-start states to %s", cfg.PATH_STATE)


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    train_df, test_df = load_and_split()
    pairs = run_discovery_on_train(train_df)
    states = warm_up_kalman(train_df, pairs)
    save_artifacts(test_df, states, pairs)


if __name__ == "__main__":
    main()
