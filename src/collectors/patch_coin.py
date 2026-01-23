import sys
import os
from pathlib import Path
# [omitted imports for brevity]

# 1. Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- CONFIGURATION ---

# ⚠️ FIX HERE: List all the coins you need to patch from your validation report.
# Additional coins needed to reach 60+ target (beyond the initial 40 at $1M threshold)
# These 39 coins are available at $250k volume threshold on both Hyperliquid and Coinbase
PATCH_LIST = [
    'ZEC', 'NEAR', 'SEI', 'kFLOKI', 'STRK', 'PENDLE', 'DOT', 'MOODENG', 'HYPER', 'MET',
    'ONDO', 'BIO', 'TRUMP', 'ATOM', 'IO', 'BERA', 'ZEN', 'FIL', 'IP', 'FET',
    'ICP', 'XLM', 'SNX', 'EIGEN', 'TON', 'LINEA', 'LDO', 'MINA', 'AVNT', 'ME',
    'AERO', 'POL', 'OP', 'KAITO', 'ETC', 'ZK', 'SYRUP', 'ALGO', 'PYTH'
]

# Optional override via env: PATCH_LIST="BTC,ETH,..." 
env_list = os.getenv("PATCH_LIST")
if env_list:
    PATCH_LIST = [s.strip().upper() for s in env_list.split(",") if s.strip()]

try:
    from src.collectors.ingest_coinbase import run_ingest as run_coinbase_ingest
    import src.utils.universe as universe_module
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root.")
    sys.exit(1)

def patch_batch_coins_all_tables():
    print(f"==================================================")
    print(f"⚙️ Ingesting {len(PATCH_LIST)} additional coins")
    print(f"==================================================")
    print(f"Coins: {', '.join(PATCH_LIST)}")
    print(f"Target: Reach 60+ total coins for backtest")
    print(f"==================================================\n")

    # Monkey-patch the universe function to return our PATCH_LIST
    # This ensures the ingester only processes these specific coins
    original_universe = universe_module.get_liquid_universe
    universe_module.get_liquid_universe = lambda top_n=50, use_buffer=False, force_refresh=False: PATCH_LIST

    try:
        # Ingest 1-minute Coinbase candles for additional coins
        print("Starting Coinbase 1-minute candle ingestion...")
        print("This will take ~6-8 hours for 39 coins × 180 days\n")
        lookback_days = int(os.getenv("BACKTEST_LOOKBACK_DAYS", "365"))
        run_coinbase_ingest(resume=True, lookback_days=lookback_days)
        print(f"\n✅ Ingestion complete for {len(PATCH_LIST)} coins.")
    finally:
        # Restore original universe function
        universe_module.get_liquid_universe = original_universe

if __name__ == "__main__":
    if not PATCH_LIST:
        print("⚠️ PATCH_LIST is empty. Nothing to do.")
    else:
        patch_batch_coins_all_tables()
