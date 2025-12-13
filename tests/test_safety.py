import sys
from pathlib import Path

# Add project root to path so imports work when running tests directly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest
from src.backtest.data_segmenter import validate_data_continuity

def test_gap_detection():
    # Create valid data (1 min intervals)
    dates = pd.date_range("2024-01-01", periods=10, freq="1min")
    df = pd.DataFrame({"close": 100}, index=dates)
    
    # Inject a 1-hour gap
    dates_gap = dates.append(pd.Index([pd.Timestamp("2024-01-01 02:00:00")]))
    df_gap = pd.DataFrame({"close": 100}, index=dates_gap)
    
    # Should PASS
    validate_data_continuity(df)
    
    # Should FAIL
    with pytest.raises(ValueError):
        validate_data_continuity(df_gap)
        
    print("✅ Safety Check Test Passed")

if __name__ == "__main__":
    test_gap_detection()