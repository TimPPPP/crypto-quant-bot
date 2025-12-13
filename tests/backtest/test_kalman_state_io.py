from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.kalman import KalmanFilterRegime


def _run_kf_stream(y: np.ndarray, x: np.ndarray, kf: KalmanFilterRegime) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Kalman on a full stream and return arrays:
      - z_scores[t]
      - betas[t]
    """
    z = np.zeros(len(y), dtype=float)
    b = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        out = kf.update(float(y[i]), float(x[i]))
        z[i] = float(out["z_score"])
        b[i] = float(out["hedge_ratio"])
    return z, b


def test_warm_start_matches_continuous_at_test_start():
    n = 300
    rng = np.random.default_rng(42)

    # Synthetic "cointegrated-ish" series in *log space* to match your Kalman design
    x = np.cumsum(rng.normal(0.0, 0.002, size=n)) + 5.0
    y = 1.3 * x + rng.normal(0.0, 0.002, size=n)

    split = 200  # train = 0..199, test starts at 200

    # Continuous run
    kf_cont = KalmanFilterRegime(delta=1e-6, R=1e-2, rolling_window=30)
    z_cont, _ = _run_kf_stream(y, x, kf_cont)

    # Warm start:
    kf_train = KalmanFilterRegime(delta=1e-6, R=1e-2, rolling_window=30)
    _ = _run_kf_stream(y[:split], x[:split], kf_train)
    state = kf_train.get_state_dict()

    kf_warm = KalmanFilterRegime(delta=1e-6, R=1e-2, rolling_window=30)
    ok = kf_warm.load_state_dict(state)
    assert ok is True

    z_warm, _ = _run_kf_stream(y[split:], x[split:], kf_warm)

    # Compare the first few z values at start of test period
    # (allow tiny numerical differences)
    k = 10
    np.testing.assert_allclose(z_warm[:k], z_cont[split:split + k], rtol=1e-8, atol=1e-8)
