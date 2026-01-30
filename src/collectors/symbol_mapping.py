"""
Symbol Mapping Module for Multi-Exchange Data Integration.

Maps between Hyperliquid symbols (canonical) and symbols on other exchanges.
Hyperliquid uses the canonical symbol format for the system.
"""

from typing import Dict, Optional, Set

# =============================================================================
# BINANCE MAPPINGS
# =============================================================================

# Hyperliquid symbol -> Binance Futures symbol
BINANCE_SYMBOL_MAP: Dict[str, str] = {
    # Major coins (direct mapping with USDT suffix)
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "DOT": "DOTUSDT",
    "UNI": "UNIUSDT",
    "LTC": "LTCUSDT",
    "BCH": "BCHUSDT",
    "ATOM": "ATOMUSDT",
    "XLM": "XLMUSDT",
    "NEAR": "NEARUSDT",
    "FIL": "FILUSDT",
    "ARB": "ARBUSDT",
    "OP": "OPUSDT",
    "SUI": "SUIUSDT",
    "SEI": "SEIUSDT",
    "INJ": "INJUSDT",
    "TIA": "TIAUSDT",
    "APT": "APTUSDT",
    "HBAR": "HBARUSDT",
    "ALGO": "ALGOUSDT",
    "AAVE": "AAVEUSDT",
    "CRV": "CRVUSDT",
    "FET": "FETUSDT",
    "RENDER": "RENDERUSDT",
    "TAO": "TAOUSDT",
    "WIF": "WIFUSDT",
    "ONDO": "ONDOUSDT",
    "ICP": "ICPUSDT",
    "STX": "STXUSDT",
    "ENA": "ENAUSDT",
    "PENDLE": "PENDLEUSDT",
    "ZRO": "ZROUSDT",
    "EIGEN": "EIGENUSDT",
    "WLD": "WLDUSDT",
    "STRK": "STRKUSDT",
    "ZK": "ZKUSDT",
    "BLAST": "BLASTUSDT",
    "IO": "IOUSDT",
    "ZEN": "ZENUSDT",
    "COMP": "COMPUSDT",
    "SNX": "SNXUSDT",
    "LDO": "LDOUSDT",
    "GMT": "GMTUSDT",
    "MINA": "MINAUSDT",
    "TNSR": "TNSRUSDT",
    "ETC": "ETCUSDT",
    "XMR": "XMRUSDT",
    "ZEC": "ZECUSDT",
    "POPCAT": "POPCATUSDT",
    "TURBO": "TURBOUSDT",
    "MOODENG": "MOODENGUSDT",
    "PNUT": "PNUTUSDT",
    "POL": "POLUSDT",
    "PENGU": "PENGUUSDT",

    # 1000x tokens (Hyperliquid uses 'k' prefix)
    "kPEPE": "1000PEPEUSDT",
    "kBONK": "1000BONKUSDT",
    "kFLOKI": "1000FLOKIUSDT",
    "kSHIB": "1000SHIBUSDT",
    "kLUNC": "1000LUNCUSDT",
}

# Coins only available on Hyperliquid (not on Binance)
HYPERLIQUID_ONLY: Set[str] = {
    "HYPE",       # Hyperliquid native token
    "PURR",       # Hyperliquid native
    "FARTCOIN",   # Hyperliquid-first meme
    "TRUMP",      # Political meme - not on Binance
    "KAITO",      # Not on Binance
    "BERA",       # Berachain - not on Binance yet
    "IP",         # Story Protocol
    "PROMPT",     # Not on Binance
    "PAXG",       # Gold token
    "ME",         # Magic Eden
    "AERO",       # Aerodrome
    "XPL",        # Not on Binance
    "ASTER",      # Not on Binance
    "FOGO",       # Not on Binance
    "WLFI",       # World Liberty
    "MON",        # Not on Binance
    "LIT",        # Limited
    "PUMP",       # Not on Binance
}


# =============================================================================
# COINBASE MAPPINGS
# =============================================================================

# Hyperliquid symbol -> Coinbase product ID
COINBASE_SYMBOL_MAP: Dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
    "DOT": "DOT-USD",
    "UNI": "UNI-USD",
    "LTC": "LTC-USD",
    "BCH": "BCH-USD",
    "ATOM": "ATOM-USD",
    "XLM": "XLM-USD",
    "NEAR": "NEAR-USD",
    "FIL": "FIL-USD",
    "APT": "APT-USD",
    "HBAR": "HBAR-USD",
    "ALGO": "ALGO-USD",
    "AAVE": "AAVE-USD",
    "CRV": "CRV-USD",
    "FET": "FET-USD",
    "RENDER": "RENDER-USD",
    "ICP": "ICP-USD",
    "COMP": "COMP-USD",
    "SNX": "SNX-USD",
    "LDO": "LDO-USD",
    "ETC": "ETC-USD",
    "ZEC": "ZEC-USD",
    # Special mappings
    "kPEPE": "PEPE-USD",
    "kBONK": "BONK-USD",
    "kFLOKI": "FLOKI-USD",
    "kSHIB": "SHIB-USD",
}

# Coins not available on Coinbase
COINBASE_UNAVAILABLE: Set[str] = {
    "HYPE", "PURR", "FARTCOIN", "TRUMP", "KAITO", "BERA", "IP", "PROMPT",
    "PAXG", "ME", "AERO", "XPL", "ASTER", "FOGO", "WLFI", "MON", "LIT",
    "PUMP", "ARB", "OP", "SUI", "SEI", "INJ", "TIA", "TAO", "WIF", "ONDO",
    "STX", "ENA", "PENDLE", "ZRO", "EIGEN", "WLD", "STRK", "ZK", "BLAST",
    "IO", "ZEN", "GMT", "MINA", "TNSR", "POPCAT", "TURBO", "MOODENG", "PNUT",
    "POL", "PENGU",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_binance_symbol(hl_symbol: str) -> Optional[str]:
    """
    Convert Hyperliquid symbol to Binance Futures symbol.

    Returns None if the symbol is not available on Binance.
    """
    if hl_symbol in HYPERLIQUID_ONLY:
        return None
    if hl_symbol in BINANCE_SYMBOL_MAP:
        return BINANCE_SYMBOL_MAP[hl_symbol]
    # Default: append USDT
    return f"{hl_symbol}USDT"


def get_coinbase_symbol(hl_symbol: str) -> Optional[str]:
    """
    Convert Hyperliquid symbol to Coinbase product ID.

    Returns None if the symbol is not available on Coinbase.
    """
    if hl_symbol in COINBASE_UNAVAILABLE:
        return None
    if hl_symbol in COINBASE_SYMBOL_MAP:
        return COINBASE_SYMBOL_MAP[hl_symbol]
    # Default: append -USD
    return f"{hl_symbol}-USD"


def is_1000x_token(hl_symbol: str) -> bool:
    """Check if a symbol is a 1000x token (kPEPE, kBONK, etc.)."""
    return hl_symbol.startswith('k') and hl_symbol[1:].isupper()


def get_price_multiplier(hl_symbol: str, source: str) -> float:
    """
    Get price multiplier for converting between exchange formats.

    Binance uses 1000PEPEUSDT, Hyperliquid uses kPEPE (1/1000th).
    Returns multiplier to convert source price to Hyperliquid format.
    """
    if is_1000x_token(hl_symbol):
        if source == 'binance':
            return 1.0 / 1000.0  # Binance price / 1000 = HL price
        elif source == 'coinbase':
            return 1.0  # Coinbase uses actual price
    return 1.0


def get_canonical_symbol(exchange: str, exchange_symbol: str) -> Optional[str]:
    """
    Convert an exchange-specific symbol back to Hyperliquid canonical format.

    Example: 'BTCUSDT' (Binance) -> 'BTC'
    """
    if exchange == 'binance':
        # Remove USDT suffix
        if exchange_symbol.endswith('USDT'):
            base = exchange_symbol[:-4]
            # Handle 1000x tokens
            if base.startswith('1000'):
                return 'k' + base[4:]
            return base
    elif exchange == 'coinbase':
        # Remove -USD suffix
        if exchange_symbol.endswith('-USD'):
            return exchange_symbol[:-4]
    elif exchange == 'hyperliquid':
        return exchange_symbol

    return None


def get_available_sources(hl_symbol: str) -> list:
    """
    Get list of exchanges where a symbol is available.

    Returns list of source names in priority order.
    """
    sources = ['hyperliquid']  # Always available

    if hl_symbol not in HYPERLIQUID_ONLY and get_binance_symbol(hl_symbol):
        sources.append('binance')

    if hl_symbol not in COINBASE_UNAVAILABLE and get_coinbase_symbol(hl_symbol):
        sources.append('coinbase')

    return sources


# =============================================================================
# OKX MAPPINGS
# =============================================================================

# Hyperliquid symbol -> OKX instrument ID
OKX_SYMBOL_MAP: Dict[str, str] = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT",
    "DOGE": "DOGE-USDT",
    "ADA": "ADA-USDT",
    "AVAX": "AVAX-USDT",
    "LINK": "LINK-USDT",
    "DOT": "DOT-USDT",
    "UNI": "UNI-USDT",
    "LTC": "LTC-USDT",
    "BCH": "BCH-USDT",
    "ATOM": "ATOM-USDT",
    "XLM": "XLM-USDT",
    "NEAR": "NEAR-USDT",
    "FIL": "FIL-USDT",
    "ARB": "ARB-USDT",
    "OP": "OP-USDT",
    "SUI": "SUI-USDT",
    "SEI": "SEI-USDT",
    "INJ": "INJ-USDT",
    "TIA": "TIA-USDT",
    "APT": "APT-USDT",
    "HBAR": "HBAR-USDT",
    "ALGO": "ALGO-USDT",
    "AAVE": "AAVE-USDT",
    "CRV": "CRV-USDT",
    "kPEPE": "PEPE-USDT",
    "kBONK": "BONK-USDT",
    "kFLOKI": "FLOKI-USDT",
    "kSHIB": "SHIB-USDT",
}

# Coins not available on OKX
OKX_UNAVAILABLE: Set[str] = {
    "HYPE", "PURR", "FARTCOIN", "TRUMP", "KAITO", "BERA", "IP", "PROMPT",
    "ME", "AERO", "XPL", "ASTER", "FOGO", "WLFI", "MON", "PUMP",
}


def get_okx_symbol(hl_symbol: str) -> Optional[str]:
    """
    Convert Hyperliquid symbol to OKX instrument ID.

    Returns None if the symbol is not available on OKX.
    """
    if hl_symbol in OKX_UNAVAILABLE:
        return None
    if hl_symbol in OKX_SYMBOL_MAP:
        return OKX_SYMBOL_MAP[hl_symbol]
    # Default: append -USDT
    return f"{hl_symbol}-USDT"


# Priority order for data sources (higher priority first)
DATA_SOURCE_PRIORITY = {
    'hyperliquid': 1,  # Native, highest priority
    'okx': 2,          # Good liquidity, available in most regions
    'binance': 3,      # Best liquidity but geo-restricted
    'coinbase': 4,     # Good quality, limited coverage
}


def get_source_priority(source: str) -> int:
    """Get priority rank for a data source (lower = higher priority)."""
    return DATA_SOURCE_PRIORITY.get(source, 99)
