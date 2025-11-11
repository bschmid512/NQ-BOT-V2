"""
Enhanced Configuration for NQ Trading Bot Optimization
Phase 1-4 Implementation Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# PHASE 1: DATA PIPELINE OPTIMIZATION
# =============================================================================

# Redis Configuration for High-Performance Caching
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'password': os.getenv('REDIS_PASSWORD', None),
    'decode_responses': True,
    'socket_keepalive': True,
    'socket_keepalive_options': {}
}

# Data Pipeline Optimization Settings
DATA_PIPELINE_CONFIG = {
    'use_redis_cache': True,
    'cache_ttl_seconds': 300,  # 5 minutes TTL for market data
    'batch_write_interval': 1.0,  # seconds
    'batch_size': 100,  # number of bars to batch write
    'async_processing': True,
    'max_memory_usage_gb': 2.0,
    'compression_enabled': True
}

# File Paths
LIVE_DATA_FILE = DATA_DIR / "nq_live_data.csv"
TRADES_FILE = DATA_DIR / "trades.csv"
SIGNALS_FILE = DATA_DIR / "signals.csv"
PERFORMANCE_FILE = DATA_DIR / "performance_metrics.csv"

# =============================================================================
# PHASE 2: STRATEGY ENGINE ENHANCEMENT
# =============================================================================

# Strategy Optimization Settings
STRATEGY_CONFIG = {
    'cache_indicators': True,
    'cache_ttl_indicators': 60,  # 1 minute for indicators
    'use_numba_optimization': True,
    'precompute_common_indicators': True,
    'signal_memoization': True,
    'max_cache_size_mb': 500
}

# Individual Strategy Configurations
ORB_STRATEGY_CONFIG = {
    'enabled': True,
    'or_period': 15,  # minutes
    'target_pct': 1.0,
    'max_sl_points': 50,
    'min_range_pct': 0.0015,
    'max_range_pct': 0.0040,
    'weight': 0.8,  # Increased weight for high-performance strategy
    'optimal_days': [0, 2, 4],  # Mon, Wed, Fri
    'position_size_multiplier': 2.5,  # 2.5x normal size for ORB
    'cooldown_minutes': 2  # Reduced cooldown
}

TREND_FOLLOWING_CONFIG = {
    'enabled': True,
    'fast_ema': 8,
    'medium_ema': 21,
    'slow_ema': 50,
    'cooldown_minutes': 5,
    'weight': 0.6,
    'min_adx': 25
}

PULLBACK_STRATEGY_CONFIG = {
    'enabled': True,
    'ema_short': 8,
    'ema_long': 50,
    'adx_threshold': 25,
    'cooldown_minutes': 3,
    'weight': 0.7,
    'pullback_threshold': 0.002  # 0.2% pullback
}

MEAN_REVERSION_CONFIG = {
    'enabled': True,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'bb_period': 20,
    'bb_std': 2,
    'cooldown_minutes': 10,
    'weight': 0.5
}

MOMENTUM_CONTINUATION_CONFIG = {
    'enabled': True,
    'momentum_period': 12,
    'volume_confirmation': True,
    'cooldown_minutes': 5,
    'weight': 0.65
}

# =============================================================================
# PHASE 3: RISK MANAGEMENT ENHANCEMENT
# =============================================================================

# Dynamic Position Sizing Configuration
RISK_MANAGEMENT_CONFIG = {
    'use_dynamic_position_sizing': True,
    'base_position_size': 1,
    'max_position_size': 5,
    'min_position_size': 0.5,
    'kelly_fraction': 0.25,  # Use 25% of Kelly Criterion
    'confidence_threshold_increase': 0.7,  # Increase size above 70% confidence
    'confidence_threshold_decrease': 0.3,  # Decrease size below 30% confidence
}

# ATR-Based Risk Management
ATR_CONFIG = {
    'period': 14,
    'stop_loss_multiplier': 2.0,
    'take_profit_multiplier': 3.0,
    'position_size_multiplier': 1.0 / 2.0,  # Risk 1% per trade
    'max_risk_per_trade_pct': 1.0
}

# Risk Limits
MAX_OPEN_POSITIONS = 5  # Increased from 3
MAX_DAILY_LOSS = 1500   # Increased from 1000
MAX_DAILY_TRADES = 50   # Increased from previous limit
MIN_TIME_BETWEEN_TRADES = 1  # Reduced from 5 seconds

# =============================================================================
# PHASE 4: PERFORMANCE MONITORING
# =============================================================================

# Performance Monitoring Configuration
PERFORMANCE_CONFIG = {
    'enable_real_time_metrics': True,
    'metrics_update_interval': 60,  # seconds
    'log_performance_data': True,
    'alert_on_performance_degradation': True,
    'performance_threshold_win_rate': 0.55,
    'performance_threshold_latency_ms': 20
}

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Trading Parameters
CONTRACT_SIZE = 20  # NQ contract size
COMMISSION_PER_SIDE = 2.50  # Commission per contract per side
TICK_SIZE = 0.25
TICK_VALUE = 5.00

# Session Configuration
TRADING_SESSIONS = {
    'us_session': {
        'start': '09:30',
        'end': '16:00',
        'timezone': 'US/Eastern',
        'enabled': True
    },
    'extended_session': {
        'start': '18:00',
        'end': '09:30',
        'timezone': 'US/Eastern',
        'enabled': False  # Disable extended hours for scalping
    }
}

# Webhook Configuration
WEBHOOK_PORT = 5000
WEBHOOK_PASSPHRASE = "your_secure_passphrase_here"
TICKER = "NQ"

# Dashboard Configuration
DASHBOARD_UPDATE_INTERVAL = 5000  # 5 seconds
DASHBOARD_PORT = 8050

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': True,
    'max_file_size_mb': 100,
    'backup_count': 5
}

# Technical Indicator Configuration
INDICATOR_CONFIG = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'vwap_anchor': 'session'
}

# Market Context Configuration
CONTEXT_THRESHOLDS = {
    'mom_1m_bull': 0.001,    # 0.1% for 1-minute momentum
    'mom_1m_bear': -0.001,   # -0.1% for 1-minute momentum
    'mom_5m_bull': 0.002,    # 0.2% for 5-minute momentum
    'mom_5m_bear': -0.002,   # -0.2% for 5-minute momentum
    'mom_10m_trending': 0.005,  # 0.5% for 10-minute trend detection
    'ema_slope_min': 0.0001,    # Minimum EMA slope for trend
    'adx_trend_min': 20         # Minimum ADX for trending market
}

# Signal Fusion Configuration
SIGNAL_FUSION_CONFIG = {
    'min_total_weight': 0.35,
    'min_signals_required': 1,
    'trade_cooldown_seconds': 1,  # Reduced from 5 seconds
    'require_vision': False,
    'min_atr': 0.0,
    'force_once': False,
    'allow_globex': False  # Disable globex for scalping
}

# =============================================================================
# OPTIMIZATION FLAGS
# =============================================================================

# Enable/Disable optimization features
OPTIMIZATION_FLAGS = {
    'phase1_data_pipeline': True,
    'phase2_strategy_engine': True,
    'phase3_risk_management': True,
    'phase4_performance_monitoring': True,
    'use_redis_cache': True,
    'enable_async_processing': True,
    'dynamic_position_sizing': True,
    'enhanced_market_context': True,
    'real_time_performance_tracking': True
}