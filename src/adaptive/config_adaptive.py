"""
Adaptive configuration for live and backtest parameter tuning.
"""

# Enable/disable adaptive behavior globally
ADAPTIVE_ENABLED: bool = False

# Update cadence and minimum data requirements
ADAPTIVE_WINDOW_DAYS: int = 30
ADAPTIVE_MIN_TRADES: int = 20

# Where to store live trade history and overrides
ADAPTIVE_TRADES_PATH: str = "data/performance/trades.jsonl"
ADAPTIVE_OVERRIDES_PATH: str = "data/performance/adaptive_overrides.json"

# Risk regime thresholds
ADAPTIVE_RISK_OFF_DD: float = 0.03
ADAPTIVE_RISK_ON_DD: float = 0.02
ADAPTIVE_RISK_OFF_WIN_RATE: float = 0.45
ADAPTIVE_RISK_ON_WIN_RATE: float = 0.55
ADAPTIVE_RISK_ON_TOTAL_RETURN: float = 0.01
ADAPTIVE_RISK_ON_SHARPE: float = 0.5
ADAPTIVE_RISK_OFF_STOP_LOSS_RATE: float = 0.30
ADAPTIVE_RISK_OFF_PROFIT_FACTOR: float = 0.90
ADAPTIVE_RISK_ON_PROFIT_FACTOR: float = 1.30
ADAPTIVE_RISK_OFF_MAX_HOLD_HOURS: float = 72.0
ADAPTIVE_RISK_OFF_MIN_HOLD_HOURS: float = 2.0
ADAPTIVE_RISK_ON_MAX_HOLD_HOURS: float = 72.0

# Parameter bounds
ENTRY_Z_BOUNDS = (2.0, 2.8)
EXIT_Z_BOUNDS = (0.3, 0.8)
MIN_PROFIT_HURDLE_BOUNDS = (0.008, 0.02)
MAX_PORTFOLIO_POSITIONS_BOUNDS = (4, 10)
MAX_POSITIONS_PER_COIN_BOUNDS = (1, 3)
STOP_LOSS_PCT_BOUNDS = (0.015, 0.04)

# Parameter step sizes per update
ENTRY_Z_STEP: float = 0.1
EXIT_Z_STEP: float = 0.05
MIN_PROFIT_STEP: float = 0.002
MAX_POSITIONS_STEP: int = 1
MAX_POSITIONS_PER_COIN_STEP: int = 1
STOP_LOSS_PCT_STEP: float = 0.0025
