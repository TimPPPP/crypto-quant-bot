from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.data_segmenter import validate_data_continuity


def test_validate_data_continuity_raises_on_gap():
    # Build 1-min index then introduce a 6-minute gap
    t0 = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx_part1 = pd.date_range(t0, periods=5, freq="1min")
    idx_part2 = pd.date_range(t0 + pd.Timedelta(minutes=11), periods=5, freq="1min")  # gap is 6 minutes (from 00:04 to 00:11)
    idx = idx_part1.append(idx_part2)

    df = pd.DataFrame({"ETH": np.arange(len(idx), dtype=float)}, index=idx)

    with pytest.raises(ValueError):
        validate_data_continuity(df)
