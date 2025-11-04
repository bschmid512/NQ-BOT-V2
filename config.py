"""
Configuration file for NQ Futures Trading Bot - CORRECTED VERSION
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
LOG_DIR = BASE_DIR / 'logs'

# Data Files
LIVE_DATA_FILE = DATA_DIR / 'nq_live_data.csv'
HISTORICAL_DATA_FILE = DATA_DIR / 'nq_historical.csv'
TRADES_FILE = DATA_DIR / 'trades.csv'
SIGNALS_FILE = DATA_DIR / 'signals.csv'

# TradingView Webhook Settings
WEBHOOK_PORT = 8050
WEBHOOK_PASSPHRASE = os.getenv('WEBHOOK_PASSPHRASE', 'change_this_secure_passphrase')

# Trading Parameters - CORRECTED FOR NQ
TICKER = 'NQ'
POINT_VALUE = 20  # $20 per point for NQ futures
TICK_SIZE = 0.25  # NQ tick size in points (0.25 points = 1 tick)
TICK_VALUE = 5.00  # $20/point × 0.25 = $5 per tick

# Realistic costs
COMMISSION = 6.00  # Realistic retail commission
SLIPPAGE_TICKS = 3  # Average slippage in NQ
TOTAL_COST_PER_RT = 21.00  # Real cost
REALISTIC_COSTS = True
SLIPPAGE_BY_SESSION = {
    'PREMARKET': 4,      # 4 ticks = $20
    'OPENING': 3,        # 3 ticks = $15
    'MIDDAY': 2,         # 2 ticks = $10
    'AFTERNOON': 2,
    'CLOSE': 3
}
# Risk Management - REALISTIC VALUES
MAX_POSITION_SIZE = 1  # Start with 1 contract for testing
MAX_DAILY_LOSS = -1000  # Halt trading if daily loss exceeds $1000
MAX_DRAWDOWN = 0.15  # 15% max drawdown threshold
RISK_PER_TRADE = 0.02  # 2% of capital per trade
KELLY_FRACTION = 0.25  # Quarter-Kelly (safer than half-Kelly)

# Account size (for paper trading)
STARTING_CAPITAL = 25000  # Minimum for NQ day trading

# ML Model Settings
ML_LOOKBACK_BARS = 100  # Number of bars for feature calculation
ML_RETRAIN_DAYS = 7  # Retrain models every N days
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'vwap', 'rsi', 'macd', 'signal', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr', 'adx', 'ema_9', 'ema_21', 'ema_50'
]

# Strategy Settings - FIXED PARAMETERS
STRATEGIES = {
    'orb': {
        'enabled': True,
        'or_period': 15,  # 15-minute opening range
        'target_pct': 0.5,  # 50% of range size
        'max_sl_points': 50,  # Max 50 points ($1000) stop loss
        'min_range_pct': 0.0015,  # Min 0.15% range (40 points @ 25,600)
        'max_range_pct': 0.004,   # Max 0.4% range (100 points @ 25,600)
        'weight': 0.30,
        'optimal_days': [0, 2, 4]  # Mon, Wed, Fri
    },
    'mean_reversion': {
        'enabled': True,
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_stop_multiplier': 1.5,  # Use 1.5x ATR for stops
        'min_reward_risk': 1.5,  # Minimum 1.5:1 R:R ratio
        'weight': 0.25
    },
    'fvg': {
        'enabled': False,  # Disabled until implemented
        'min_gap_points': 3,
        'lookback': 20,
        'weight': 0.25
    },
    'pivot': {
        'enabled': False,  # Disabled until implemented
        'pivot_type': 'camarilla',
        'weight': 0.10
    },
    'pullback': {
        'enabled': False,  # Disabled until implemented
        'ema_fast': 20,
        'ema_slow': 50,
        'atr_period': 14,
        'weight': 0.10
    }
}

# Dashboard Settings
DASHBOARD_UPDATE_INTERVAL = 10000  # milliseconds (10 seconds)
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = False  # Set to False for production

# Trading Hours (ET) - NQ trades 23 hours/day, 5 days/week
# Futures open: Sunday 6:00pm ET
# Futures close: Friday 5:00pm ET
MARKET_OPEN = '18:00'  # 6:00pm ET Sunday
MARKET_CLOSE = '17:00'  # 5:00pm ET Friday
RTH_OPEN = '09:30'  # Regular Trading Hours open
RTH_CLOSE = '16:00'  # Regular Trading Hours close

# Session filters
AVOID_ASIAN_SESSION = True  # 6pm-2am ET (low volume)
AVOID_EARLY_EUROPEAN = True  # 2am-6am ET (very thin)
TRADE_PREMARKET = True  # 6am-9:30am ET (ORB setup)
TRADE_RTH = True  # 9:30am-4pm ET (best liquidity)
TRADE_AFTERHOURS = False  # 4pm-6pm ET (earnings reactions)

# Logging Settings
LOG_LEVEL = 'INFO'  # DEBUG for development, INFO for production
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB
LOG_BACKUP_COUNT = 10

# Performance Metrics
METRICS_WINDOW = 252  # Trading days for performance calculation
SHARPE_RISK_FREE_RATE = 0.04  # 4% annual risk-free rate (2024 rates)

# Alert Settings
ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
ENABLE_ALERTS = False  # Disable until configured

# Database Settings (Optional - for TimescaleDB)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'nq_trading')
DB_USER = os.getenv('DB_USER', 'trader')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
USE_DATABASE = False  # Use CSV for now

# Paper Trading Settings
PAPER_TRADING = True  # ALWAYS True until you've proven profitability
SIMULATED_FILLS = True  # Assume fills at market price
REALISTIC_SLIPPAGE = True  # Apply slippage to fills

# Safety Limits
MAX_TRADES_PER_DAY = 10  # Circuit breaker
MAX_CONSECUTIVE_LOSSES = 5  # Stop after 5 losses in a row
MIN_TIME_BETWEEN_TRADES = 300  # 5 minutes between entries (seconds)

# High-Impact Economic Events (Update monthly)
# Bot will NOT trade on these dates
HIGH_IMPACT_DATES = [
    # November 2024
    '2024-11-07',  # FOMC Decision
    '2024-11-13',  # CPI Report
    '2024-11-15',  # Retail Sales
    # December 2024
    '2024-12-06',  # NFP (Jobs Report)
    '2024-12-11',  # CPI Report
    '2024-12-18',  # FOMC Decision
    # January 2025
    '2025-01-10',  # NFP
    '2025-01-15',  # CPI Report
    '2025-01-29',  # FOMC Decision
]

# Convert to datetime objects at runtime
from datetime import datetime
HIGH_IMPACT_DATES = [datetime.strptime(d, '%Y-%m-%d') for d in HIGH_IMPACT_DATES]