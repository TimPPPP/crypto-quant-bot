from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.kalman import KalmanFilterRegime
from src.backtest.signal_generation import generate_signals  # <-- match your implementation


def test_signal_generation_is_causal():
    idx = pd.date_range("2025-01-01", periods=200, freq="1min", tz="UTC")

    # Base series: smooth relationship
    x = np.linspace(5.0, 5.2, len(idx))
    y = 1.2 * x + 0.001 * np.sin(np.linspace(0, 20, len(idx)))

    df_no_jump = pd.DataFrame({"ETH": np.exp(y), "BTC": np.exp(x)}, index=idx)

    # Inject future jump at t_jump+1 only
    t_jump = 120
    y2 = y.copy()
    y2[t_jump + 1] += 0.10  # big jump in log-price at t+1
    df_with_jump = pd.DataFrame({"ETH": np.exp(y2), "BTC": np.exp(x)}, index=idx)

    pairs = ["ETH-BTC"]

    # Build warm state using first 80 points so signal generator starts from same state
    train_n = 80
    kf = KalmanFilterRegime(delta=1e-6, R=1e-2, rolling_window=30)
    for i in range(train_n):
        kf.update(float(y[i]), float(x[i]))
    warm_states = {"ETH-BTC": kf.get_state_dict()}

    z1, vol1, beta1 = generate_signals(df_no_jump, pairs, warm_states=warm_states)
    z2, vol2, beta2 = generate_signals(df_with_jump, pairs, warm_states=warm_states)

    # Causality requirement: values at times <= t_jump must match exactly (or very tight tolerance)
    np.testing.assert_allclose(
        z1["ETH-BTC"].iloc[: t_jump + 1].to_numpy(),
        z2["ETH-BTC"].iloc[: t_jump + 1].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        beta1["ETH-BTC"].iloc[: t_jump + 1].to_numpy(),
        beta2["ETH-BTC"].iloc[: t_jump + 1].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    # vol is allowed tiny differences only if you compute it from past errors;
    # still should match because jump is in the future
    np.testing.assert_allclose(
        vol1["ETH-BTC"].iloc[: t_jump + 1].to_numpy(),
        vol2["ETH-BTC"].iloc[: t_jump + 1].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
