import os
import requests
import pandas as pd
import logging
import shutil

# --- CONFIGURATION ---
QUESTDB_HOST = os.getenv('QUESTDB_HOST', 'localhost')
QUESTDB_EXPORT_URL = f"http://{QUESTDB_HOST}:9000/exp"
TABLE_NAME = "candles_1m" 
LOOKBACK_DAYS = 180
OUTPUT_FILE = "data/raw_downloads/crypto_prices_1m.parquet"
MIN_DATA_THRESHOLD = 0.90  # Drop coins missing >10% of data

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataExport")

def export_csv_stream_to_parquet():
    """
    Exports 1-minute candle data from QuestDB to a Parquet Matrix.
    optimized for memory usage and atomic writing.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    logger.info(f"📤 Exporting last {LOOKBACK_DAYS} days of '{TABLE_NAME}'...")
    
    # SQL Query: Selects data using database-side filtering
    query = f"""
    SELECT timestamp, symbol, close 
    FROM {TABLE_NAME} 
    WHERE timestamp >= dateadd('d', -{LOOKBACK_DAYS}, now())
    ORDER BY timestamp ASC;
    """
    
    try:
        # Stream the CSV response to avoid loading raw text into RAM
        with requests.get(QUESTDB_EXPORT_URL, params={'query': query}, stream=True) as r:
            r.raise_for_status()
            logger.info("   ↳ Connection established. Streaming & Parsing CSV...")
            
            # Read directly into Pandas with optimized types
            df = pd.read_csv(
                r.raw, 
                parse_dates=['timestamp'],
                dtype={'close': 'float32', 'symbol': 'category'} 
            )
            
        if df.empty:
            logger.warning(f"⚠️ No data found in table '{TABLE_NAME}'. Run ingest_1m.py first.")
            return

        logger.info(f"   ↳ Loaded {len(df):,} rows. Cleaning data...")

        # 1. Deduplicate
        # critical for backfilled data which often has overlaps
        df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)

        # 2. Pivot to Matrix (Index=Time, Columns=Coins)
        price_matrix = df.pivot(index='timestamp', columns='symbol', values='close')
        
        # 3. Quality Control (Ghost Coin Removal)
        # Calculate valid data percentage per coin
        valid_data_pct = 1.0 - price_matrix.isna().mean()
        drop_coins = valid_data_pct[valid_data_pct < MIN_DATA_THRESHOLD].index.tolist()
        
        if drop_coins:
            logger.info(f"   🗑️ Dropping {len(drop_coins)} coins with insufficient history (<{MIN_DATA_THRESHOLD*100}%).")
            price_matrix.drop(columns=drop_coins, inplace=True)
            
        # 4. Fill & Trim
        price_matrix.ffill(inplace=True)
        price_matrix.dropna(axis=0, how='all', inplace=True)
        
        logger.info(f"   ↳ Final Matrix Shape: {price_matrix.shape} (Time x Coins)")

        # 5. Atomic Write
        # Write to a .tmp file in the SAME directory to ensure atomic rename
        temp_path = OUTPUT_FILE + ".tmp"
        price_matrix.to_parquet(temp_path)
        shutil.move(temp_path, OUTPUT_FILE)
        
        logger.info(f"✅ Success! Data saved to: {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"❌ Export Failed: {e}")
        # Cleanup temp file if it was created
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    export_csv_stream_to_parquet()