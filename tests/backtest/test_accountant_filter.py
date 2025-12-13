from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.accountant_filter import compute_accountant_masks  # <-- match your implementation


def test_accountant_masks_match_hand_computation():
    idx = pd.date_range("2025-01-01", periods=5, freq="1min", tz="UTC")
    pairs = ["ETH-BTC"]

    z = pd.DataFrame({"ETH-BTC": [0.0, 2.1, 1.5, 0.4, 4.6]}, index=idx)
    vol = pd.DataFrame({"ETH-BTC": [0.001, 0.004, 0.004, 0.003, 0.004]}, index=idx)

    # expected_profit = vol * 0.75
    # profit_hurdle = 0.002
    # entries: abs(z)>2 & expected_profit>0.002 => at t=1 only: 0.004*0.75=0.003>0.002 and 2.1>2
    # exits: abs(z)<0.5 OR abs(z)>4.5 => t=3 (0.4), t=4 (4.6)
    entries, exits, expected_profit = compute_accountant_masks(
        z_score=z,
        spread_volatility=vol,
        expected_revert_mult=0.75,
        entry_z=2.0,
        exit_z=0.5,
        stop_loss_z=4.5,
        min_profit_hurdle=0.002,
    )

    expected_entries = np.array([False, True, False, False, False])
    expected_exits = np.array([True, False, False, True, True])  # note: t=0 abs(z)<0.5 => exit mask is True (harmless)

    assert entries["ETH-BTC"].to_numpy().tolist() == expected_entries.tolist()
    assert exits["ETH-BTC"].to_numpy().tolist() == expected_exits.tolist()

    # sanity: expected_profit values
    np.testing.assert_allclose(expected_profit["ETH-BTC"].to_numpy(), (vol["ETH-BTC"] * 0.75).to_numpy())
