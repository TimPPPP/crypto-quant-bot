from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dt_index_1min():
    return pd.date_range("2025-01-01", periods=20, freq="1min", tz="UTC")


@pytest.fixture
def toy_prices_flat(dt_index_1min) -> pd.DataFrame:
    # Two coins: ETH and BTC
    return pd.DataFrame(
        {
            "ETH": np.full(len(dt_index_1min), 100.0, dtype=float),
            "BTC": np.full(len(dt_index_1min), 50.0, dtype=float),
        },
        index=dt_index_1min,
    )


@pytest.fixture
def toy_pair():
    return "ETH-BTC"


def make_entries_exits(index: pd.Index, pairs: list[str], entry_t: int, exit_t: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = pd.DataFrame(False, index=index, columns=pairs)
    exits = pd.DataFrame(False, index=index, columns=pairs)
    entries.iloc[entry_t, :] = True
    exits.iloc[exit_t, :] = True
    return entries, exits
