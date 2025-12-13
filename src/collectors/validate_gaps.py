# src/collectors/validate_gaps.py

import os
import sys
import pandas as pd
import requests

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.universe import get_liquid_universe
from src.backtest.data_segmenter import validate_data_continuity  # canonical rule
from src.backtest import config_backtest as cfg                  # MAX_DATA_GAP_MINS, etc.

# --- CONFIGURATION ---
QUESTDB_HOST = os.getenv("QUESTDB_HOST", "localhost")
HTTP_PORT = int(os.getenv("QUESTDB_HTTP_PORT", "9000"))
URL = f"http://{QUESTDB_HOST}:{HTTP_PORT}/exec"


def check_table(symbol: str, table_name: str) -> str:
    """
    Checks a specific QuestDB table for:
    - emptiness
    - duplicate timestamps
    - fatal continuity gaps using the backtest's canonical rule (MAX_DATA_GAP_MINS)

    Returns a short status string for console reporting.
    """
    query = f"SELECT timestamp FROM {table_name} WHERE symbol = '{symbol}' ORDER BY timestamp ASC"

    try:
        r = requests.get(URL, params={"query": query}, timeout=30)

        if r.status_code != 200:
            if "table does not exist" in r.text:
                return "⚪ Missing Table"
            return "❌ DB Error"

        data = r.json()
        if "dataset" not in data or not data["dataset"]:
            return "⚪ Empty"

        df = pd.DataFrame(data["dataset"], columns=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            return "⚪ Empty"

        # Duplicate check
        start_len = len(df)
        df = df.drop_duplicates(subset=["timestamp"])
        if len(df) != start_len:
            return "⚠️ Duplicates"

        # Canonical continuity check (same rule as backtest)
        ts_df = df.set_index("timestamp").sort_index()
        try:
            validate_data_continuity(ts_df)
        except ValueError:
            return f"❌ Gap > {cfg.MAX_DATA_GAP_MINS}m"

        return "✅ OK"

    except Exception as e:
        return f"❌ Err: {str(e)[:20]}"


def run_full_scan(top_n: int = 50) -> None:
    print("🏥 Starting Data Lake Check (1M Candles + Funding)...")

    symbols = get_liquid_universe(top_n=top_n)

    print(f"\n{'SYMBOL':<10} | {'1-MIN (Price)':<18} | {'FUNDING (Rate)':<18}")
    print("-" * 55)

    for sym in symbols:
        status_1m = check_table(sym, "candles_1m")
        status_funding = check_table(sym, "funding_history")
        print(f"{sym:<10} | {status_1m:<18} | {status_funding:<18}")

    print("-" * 55)
    print("⚪ Empty = You haven't ingested this symbol/table yet.")
    print("⚠️ Duplicates = Consider 'VACUUM PARTITIONS <table>;' in QuestDB.")
    print(f"❌ Gap > {cfg.MAX_DATA_GAP_MINS}m = Fatal gap by backtest rule (fix before backtesting).")


if __name__ == "__main__":
    run_full_scan()
