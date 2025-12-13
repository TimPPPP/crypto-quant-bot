from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.pnl_engine import run_pnl_engine


def test_pnl_engine_flat_prices_no_trades_returns_zero(toy_prices_flat, toy_pair):
    idx = toy_prices_flat.index
    pairs = [toy_pair]

    # No entries/exits => no trades => all zeros
    entries = pd.DataFrame(False, index=idx, columns=pairs)
    exits = pd.DataFrame(False, index=idx, columns=pairs)
    beta = pd.DataFrame(2.0, index=idx, columns=pairs)
    z = pd.DataFrame(0.0, index=idx, columns=pairs)
    vol = pd.DataFrame(0.01, index=idx, columns=pairs)

    res = run_pnl_engine(
        test_df=toy_prices_flat,
        pairs=pairs,
        entries=entries,
        exits=exits,
        beta=beta,
        z_score=z,
        spread_volatility=vol,
        fee_rate=0.0005,
        slippage_rate=0.0,
        slippage_model="fixed",
        capital_per_pair=1.0,
        pnl_mode="price",
    )
    assert float(res.returns_matrix.to_numpy().sum()) == 0.0
    assert int(res.trades_count.iloc[0]) == 0


def test_pnl_engine_round_trip_net_is_minus_fees():
    # Construct prices so spread is constant when beta=2:
    # y = [100, 110, 100], x = [50, 55, 50] => y - 2x == 0 always
    idx = pd.date_range("2025-01-01", periods=3, freq="1min", tz="UTC")
    test_df = pd.DataFrame({"ETH": [100.0, 110.0, 100.0], "BTC": [50.0, 55.0, 50.0]}, index=idx)

    pairs = ["ETH-BTC"]
    entries = pd.DataFrame(False, index=idx, columns=pairs)
    exits = pd.DataFrame(False, index=idx, columns=pairs)
    entries.iloc[1, 0] = True
    exits.iloc[2, 0] = True

    beta = pd.DataFrame(2.0, index=idx, columns=pairs)

    # z sign sets direction; any sign is fine since gross spread pnl is 0
    z = pd.DataFrame(-2.1, index=idx, columns=pairs)

    # vol is unused by fixed slippage but required by API
    vol = pd.DataFrame(0.01, index=idx, columns=pairs)

    fee_rate = 0.0005
    # Capital scaling: use notional at entry as capital to make return meaningful
    entry_y = 110.0
    entry_x = 55.0
    notional = abs(entry_y) + abs(2.0) * abs(entry_x)  # matches pnl_engine
    capital = notional

    res = run_pnl_engine(
        test_df=test_df,
        pairs=pairs,
        entries=entries,
        exits=exits,
        beta=beta,
        z_score=z,
        spread_volatility=vol,
        fee_rate=fee_rate,
        slippage_rate=0.0,
        slippage_model="fixed",
        capital_per_pair=capital,
        pnl_mode="price",
    )

    # One trade closes at t=2 with gross pnl 0 => net = -fees
    fees = notional * 4.0 * fee_rate
    expected_return = -fees / capital

    got = float(res.returns_matrix.iloc[2, 0])
    np.testing.assert_allclose(got, expected_return, rtol=0.0, atol=1e-12)
    assert int(res.trades_count.iloc[0]) == 1


def test_pnl_engine_scale_invariance_when_capital_scales():
    # Same scenario as round trip, but scale prices by 100
    idx = pd.date_range("2025-01-01", periods=3, freq="1min", tz="UTC")
    base_df = pd.DataFrame({"ETH": [100.0, 110.0, 100.0], "BTC": [50.0, 55.0, 50.0]}, index=idx)
    scaled_df = base_df * 100.0

    pairs = ["ETH-BTC"]

    def run(df: pd.DataFrame) -> float:
        entries = pd.DataFrame(False, index=df.index, columns=pairs)
        exits = pd.DataFrame(False, index=df.index, columns=pairs)
        entries.iloc[1, 0] = True
        exits.iloc[2, 0] = True

        beta = pd.DataFrame(2.0, index=df.index, columns=pairs)
        z = pd.DataFrame(-2.1, index=df.index, columns=pairs)
        vol = pd.DataFrame(0.01, index=df.index, columns=pairs)

        fee_rate = 0.0005

        entry_y = float(df["ETH"].iloc[1])
        entry_x = float(df["BTC"].iloc[1])
        notional = abs(entry_y) + abs(2.0) * abs(entry_x)
        capital = notional  # scale capital with notional -> invariant return

        res = run_pnl_engine(
            test_df=df,
            pairs=pairs,
            entries=entries,
            exits=exits,
            beta=beta,
            z_score=z,
            spread_volatility=vol,
            fee_rate=fee_rate,
            slippage_rate=0.0,
            slippage_model="fixed",
            capital_per_pair=capital,
            pnl_mode="price",
        )
        return float(res.returns_matrix.iloc[2, 0])

    r1 = run(base_df)
    r2 = run(scaled_df)

    np.testing.assert_allclose(r1, r2, rtol=0.0, atol=1e-12)
