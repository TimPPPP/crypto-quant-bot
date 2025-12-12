import sys
import os
# [omitted imports for brevity]

# 1. Force /app to be in the Python Path
sys.path.append('/app')

# --- CONFIGURATION ---

# ⚠️ FIX HERE: List all the coins you need to patch from your validation report.
# Example: If SEI and SOL were empty, you would use: ['SEI', 'SOL']
PATCH_LIST = ['SEI']  # <-- Start with SEI, add others if necessary.

try:
    # [omitted imports for brevity]
    from src.collectors.ingest_1h import run_ingest as run_1h_ingest
    from src.collectors.ingest_funding import run_funding_ingest as run_funding_ingest
    from src.collectors.ingest_1m import run_ingest as run_1m_ingest
    import src.utils.universe as universe_module
except ImportError as e:
    # [omitted error handling]
    sys.exit(1)

def patch_batch_coins_all_tables():
    print(f"==================================================")
    print(f"⚙️ Starting BATCH Patch for {len(PATCH_LIST)} coins.")
    print(f"==================================================")
    
    # 3. Monkey-Patch the Universe Function Globally
    # The universe function now returns our PATCH_LIST, so all ingest scripts loop through only these coins.
    universe_module.get_liquid_universe = lambda top_n=50: PATCH_LIST

    # 4. Sequential Execution (Hole-Patching)
    
    # --- PATCH 1: 1-HOUR CANDLES ---
    print("\n[1/3] Patching 1-Hour Candles (OHLCV)...")
    # This loop runs the 1H ingestion logic for every coin in PATCH_LIST
    run_1h_ingest()
    
    # --- PATCH 2: FUNDING HISTORY ---
    print("\n[2/3] Patching Funding History (Rate & Premium)...")
    run_funding_ingest()

    # --- PATCH 3: 1-MINUTE CANDLES ---
    print("\n[3/3] Patching 1-Minute Candles (High-Frequency)...")
    run_1m_ingest()

    print(f"\n✅ Full Batch Patch complete.")

if __name__ == "__main__":
    if not PATCH_LIST:
        print("⚠️ PATCH_LIST is empty. Nothing to do.")
    else:
        patch_batch_coins_all_tables()
