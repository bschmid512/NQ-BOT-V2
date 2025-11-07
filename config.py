"""
Configuration file for NQ Futures Trading Bot
--- COMPLETE FUSION VERSION ---
Contains all keys for all strategies and core components.
"""
import os
from pathlib import Path
import logging

# --- TECHNICAL ANALYSIS ---
ATR_PERIOD = 14

# --- PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
LOG_DIR = BASE_DIR / 'logs'

# -----------------------------------------------------------------
# --- LOGGING CONFIGURATION (Fixes 'LOG_LEVEL' error) ---
# -----------------------------------------------------------------
LOG_LEVEL = logging.DEBUG  # Level for the log files (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
LOG_BACKUP_COUNT = 3  # Keep 3 old log files

# --- DATA FILES ---
LIVE_DATA_FILE = DATA_DIR / 'nq_live_data.csv'
HISTORICAL_DATA_FILE = DATA_DIR / 'nq_historical.csv'
TRADES_FILE = DATA_DIR / 'trades.csv'
SIGNALS_FILE = DATA_DIR / 'signals.csv'

# --- WEBHOOK & SERVER ---
WEBHOOK_PORT = 8050
WEBHOOK_PASSPHRASE = os.getenv('WEBHOOK_PASSPHRASE', 'change_this_secure_passphrase')

# --- TRADING PARAMETERS (NQ) ---
TICKER = 'NQ'
POINT_VALUE = 20  # $20 per point for NQ futures
TICK_SIZE = 0.25  # NQ tick size in points (0.25 points = 1 tick)
TICK_VALUE = 5.00  # $20/point × 0.25 = $5 per tick

# -----------------------------------------------------------------
# --- COSTS (Fixes 'COMMISSION_PER_SIDE' error) ---
# -----------------------------------------------------------------
# This must be PER_SIDE because position_manager multiplies by 2
COMMISSION_PER_SIDE = 3.00  # $3.00 per side ($6.00 round-trip)
SLIPPAGE_TICKS = 3  # Average slippage in NQ
TOTAL_COST_PER_RT = (COMMISSION_PER_SIDE * 2) + (SLIPPAGE_TICKS * TICK_VALUE)
REALISTIC_COSTS = True
SLIPPAGE_BY_SESSION = {
    'PREMARKET': 4,
    'REGULAR': 2,
    'AFTERHOURS': 4,
    'OVERNIGHT': 6
}

# --- RISK & POSITION SIZING ---
CONTRACT_SIZE = 1  # Default number of contracts to trade
MAX_POSITION_SIZE = 3  # Max contracts at any time
MAX_OPEN_POSITIONS = 1  # Only allow one position at a time
MAX_DAILY_LOSS = -500.0  # Daily stop: -$500
MAX_DAILY_TRADES = 10     # Max trades per day
MIN_TIME_BETWEEN_TRADES = 300 # 5 minutes

# --- TRADING SESSIONS (US/Eastern) ---
SESSION_START = '09:30'
SESSION_END = '16:00'
ALLOW_PREMARKET = True
ALLOW_AFTERHOURS = False

# --- ECONOMIC EVENTS (Bot will NOT trade on these dates) ---
HIGH_IMPACT_DATES = [
    # November 2024
    '2024-11-07',  # FOMC Decision
    '2024-11-13',  # CPI Report
    '2024-11-15',  # Retail Sales
    # December 2024
    '2024-12-06',  # NFP (Non-Farm Payroll)
    '2024-12-11',  # CPI Report
    '2024-12-18',  # FOMC Decision
]

# -----------------------------------------------------------------
# --- FUSION ENGINE CONFIGURATION ---
# -----------------------------------------------------------------
FUSION_CONFIG = {
    'min_signals_required': 1,      # How many strategies must agree (1 = at least one)
    'min_total_weight': 60,         # The combined "score" needed to approve a trade
    'max_weight': 100,              # The maximum possible weight
    'vision_weight_multiplier': 1.2, # 20% boost if vision confirms
    'convergence_bonus': 10         # 10 point bonus if 2+ strategies agree
}

# -----------------------------------------------------------------
# --- STRATEGY CONFIGURATIONS (FINAL, ALL KEYS) ---
# -----------------------------------------------------------------
STRATEGIES = {
    'orb': {
        'enabled': True,
        'weight': 60,
        'or_period': 15,
        'min_range': 20.0,
        'max_range': 100.0,
        'atr_multiplier': 1.0,
        'target_pct': 1.5,
        'max_sl_points': 40,
        'use_atr_sl': True,
        'min_time': '09:46:00',
        'max_time': '12:00:00',
        'sl_to_be_pct': 0.5
    },
    
    'mean_reversion': {
        'enabled': True,
        'weight': 50,
        'rsi_overbought': 70,   # Correct key
        'rsi_oversold': 30,   # Correct key
        'adx_max': 25,
        'bb_period': 20,
        'bb_std': 2.0,            # Correct key
        'rsi_period': 14,
        'adx_period': 14,
        'atr_multiplier': 1.5,
        'use_adx_filter': True,
        'use_context_filter': True
    },
    
    'trend_following': {
        'enabled': True,
        'weight': 60,
        'fast_ema': 21,
        'slow_ema': 50,
        'adx_min': 25,
        'adx_period': 14,
        'atr_multiplier': 2.0
    },
    
    'breakout': {
        'enabled': True,
        'weight': 65,
        'breakout_period': 20,
        'atr_multiplier': 2.0,
        'min_volume_spike': 1.5,
        'use_context_filter': True
    },
    
    'pullback': {
        'enabled': True,
        'weight': 70,
        'trend_ema': 50,
        'fast_ema': 8,
        'pullback_pct': 0.005,
        'atr_multiplier': 1.5,
        'adx_min': 20
    },
    
    'momentum': {
        'enabled': True,
        'weight': 65,
        'fast_ema': 8,
        'mid_ema': 21,
        'long_ema': 50,
        'adx_min': 25
    }
}

# --- DASHBOARD ---
DASHBOARD_UPDATE_INTERVAL = 5 * 1000  # 5 seconds