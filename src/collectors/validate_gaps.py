import sys
import os
import pandas as pd
import requests

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.universe import get_liquid_universe

# --- CONFIGURATION ---
QUESTDB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
HTTP_PORT = 9000
URL = f"http://{QUESTDB_HOST}:{HTTP_PORT}/exec"

def check_table(symbol, table_name, interval_sec):
    """
    Checks a specific table for gaps and duplicates.
    """
    # 1. Fetch Timestamp Column
    query = f"SELECT timestamp FROM {table_name} WHERE symbol = '{symbol}' ORDER BY timestamp ASC"
    
    try:
        r = requests.get(URL, params={'query': query})
        
        # Handle non-existent table or DB errors
        if r.status_code != 200:
            if "table does not exist" in r.text:
                return "⚪ Missing Table"
            return f"❌ DB Error"

        data = r.json()
        if 'dataset' not in data or not data['dataset']:
            return "⚪ Empty"

        # 2. Parse Data
        df = pd.DataFrame(data['dataset'], columns=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 3. Check Duplicates
        start_len = len(df)
        df = df.drop_duplicates(subset=['timestamp'])
        if start_len != len(df):
            return "⚠️ Duplicates"
        
        # 4. Check Gaps
        # We define a "Gap" as missing a whole interval (interval * 1.5 allows for small jitter)
        df['delta'] = df['timestamp'].diff().dt.total_seconds()
        gaps = df[df['delta'] > (interval_sec * 1.5)]
        
        if not gaps.empty:
            return f"❌ {len(gaps)} Gaps"
        
        return "✅ OK"

    except Exception as e:
        return f"❌ Err: {str(e)[:10]}"

def run_full_scan():
    print("🏥 Starting Complete Data Lake Check (1H, 1M, Funding)...")
    
    # 1. Get Universe
    symbols = get_liquid_universe(top_n=50)
    
    # 2. Print Header
    # Formats the table nicely
    print(f"\n{'SYMBOL':<8} | {'1-HOUR (Price)':<18} | {'FUNDING (Rate)':<18} | {'1-MIN (Price)':<18}")
    print("-" * 70)
    
    # 3. Scan Loop
    for sym in symbols:
        # Check 1H Candles (Interval: 3600s)
        status_1h = check_table(sym, 'candles_1h', 3600)
        
        # Check Funding History (Interval: 3600s)
        # Note: Funding sometimes has slight jitter, but should never be > 1.5 hours apart
        status_funding = check_table(sym, 'funding_history', 3600)
        
        # Check 1M Candles (Interval: 60s)
        status_1m = check_table(sym, 'candles_1m', 60)
        
        print(f"{sym:<8} | {status_1h:<18} | {status_funding:<18} | {status_1m:<18}")

    print("-" * 70)
    print("⚪ Empty = You haven't run the ingestion for this table yet.")
    print("⚠️ Duplicates = Run 'VACUUM PARTITIONS [table_name];' in SQL.")
    print("❌ Gaps = Data missing. Re-run ingestion or check logs.")

if __name__ == "__main__":
    run_full_scan()
