import os
import time
import logging

# Hyperliquid SDK is optional. If it's not installed the code will run in
# a "mock" mode that returns a small fallback universe. This prevents
# ModuleNotFoundError when running the container without the SDK.
try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    HYPERLIQUID_AVAILABLE = True
except Exception:
    Info = None
    constants = None
    HYPERLIQUID_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Universe")

# Configuration
MIN_VOLUME_USD = float(os.getenv('MIN_VOLUME_USD', 1_000_000))
BUFFER_SIZE = int(os.getenv('UNIVERSE_BUFFER', 30))
CACHE_TTL_SECONDS = int(os.getenv('UNIVERSE_CACHE_TTL', 3600))  # 1 hour default

# Cache for universe data
_universe_cache = {
    'data': None,
    'timestamp': 0
}


def get_liquid_universe(top_n: int = 50, use_buffer: bool = False, force_refresh: bool = False):
    """
    Fetches the top liquid assets from Hyperliquid On-Chain Data.

    Args:
        top_n: Number of primary assets to select (default 50).
        use_buffer: If True, fetches an extra BUFFER_SIZE coins for rank shifting.
        force_refresh: If True, bypass cache and fetch fresh data.

    Returns:
        List of symbol strings sorted by volume.
    """
    global _universe_cache

    # Check cache
    cache_age = time.time() - _universe_cache['timestamp']
    if not force_refresh and _universe_cache['data'] is not None and cache_age < CACHE_TTL_SECONDS:
        logger.debug(f"Using cached universe (age: {cache_age:.0f}s)")
        all_symbols = _universe_cache['data']
    else:
            # If the Hyperliquid SDK is unavailable, provide a safe fallback so
            # the bot can run in mock mode. Prefer an explicit env var override.
            if not HYPERLIQUID_AVAILABLE:
                logger.warning("Hyperliquid SDK not installed. Using fallback universe.")

                # 1) Allow explicit override via environment variable
                env_list = os.getenv('UNIVERSE_FALLBACK')
                if env_list:
                    all_symbols = [s.strip().upper() for s in env_list.split(',') if s.strip()]
                else:
                    # 2) Try to read a small sample from data/state or hardcoded safe list
                    try:
                        import json
                        state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'state', 'positions.json')
                        state_path = os.path.abspath(state_path)
                        if os.path.exists(state_path):
                            with open(state_path, 'r') as f:
                                j = json.load(f)
                            # Expect a dict of positions; extract keys that look like symbols
                            all_symbols = [k.upper() for k in j.keys()] if isinstance(j, dict) else []
                        else:
                            # Minimal safe default
                            all_symbols = ['BTC', 'ETH', 'SOL', 'MATIC', 'LINK']
                    except Exception:
                        all_symbols = ['BTC', 'ETH', 'SOL', 'MATIC', 'LINK']

                # Update cache with fallback
                _universe_cache = {
                    'data': all_symbols,
                    'timestamp': time.time()
                }

            else:
                # Fetch fresh data from Hyperliquid
                logger.info("Fetching universe from Hyperliquid...")
                try:
                    info = Info(constants.MAINNET_API_URL, skip_ws=True)

                    meta = info.meta()
                    universe_list = meta.get('universe', [])

                    asset_ctxs = info.meta_and_asset_ctxs()

                    valid_pairs = []

                    # asset_ctxs may be a tuple/list with multiple sections
                    ctx_list = asset_ctxs[1] if isinstance(asset_ctxs, (list, tuple)) and len(asset_ctxs) > 1 else asset_ctxs

                    for i, asset_info in enumerate(ctx_list):
                        symbol = universe_list[i]['name'] if i < len(universe_list) else None
                        try:
                            volume = float(asset_info.get('dayNtlVlm', 0))
                        except Exception:
                            volume = 0.0

                        if not symbol:
                            continue

                        if volume < MIN_VOLUME_USD:
                            continue

                        valid_pairs.append({
                            'symbol': symbol,
                            'volume': volume
                        })

                    sorted_pairs = sorted(valid_pairs, key=lambda x: x['volume'], reverse=True)
                    all_symbols = [x['symbol'] for x in sorted_pairs]

                    # Update cache
                    _universe_cache = {
                        'data': all_symbols,
                        'timestamp': time.time()
                    }

                    logger.info(f"Fetched {len(all_symbols)} liquid coins (min vol: ${MIN_VOLUME_USD:,.0f})")

                except Exception as e:
                    logger.error(f"Failed to fetch universe: {e}")
                    # Return cached data if available
                    if _universe_cache['data'] is not None:
                        logger.warning("Using stale cache due to fetch error")
                        all_symbols = _universe_cache['data']
                    else:
                        return []

    # Apply limit
    limit = top_n + BUFFER_SIZE if use_buffer else top_n
    return all_symbols[:limit]


def clear_universe_cache():
    """Clear the universe cache."""
    global _universe_cache
    _universe_cache = {'data': None, 'timestamp': 0}
    logger.info("Universe cache cleared")


if __name__ == "__main__":
    print("Standard Universe (Top 10):")
    print(get_liquid_universe(10, use_buffer=False))

    print("\nBuffered Universe (Top 10 + buffer):")
    print(get_liquid_universe(10, use_buffer=True))

    print("\nCached call (should be instant):")
    print(get_liquid_universe(10, use_buffer=False))