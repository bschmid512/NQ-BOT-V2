"""
Enhanced Strategy Engine with Caching and Optimizations
Phase 2 Implementation: High-Performance Signal Generation
"""
from __future__ import annotations

import time
import threading
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import pandas as pd

# Try to import numba for performance optimization
try:
    from numba import njit, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func
    def jit(func):
        return func

# Import configuration
from config import (
    STRATEGY_CONFIG, ORB_STRATEGY_CONFIG, TREND_FOLLOWING_CONFIG,
    PULLBACK_STRATEGY_CONFIG, MEAN_REVERSION_CONFIG, MOMENTUM_CONTINUATION_CONFIG,
    OPTIMIZATION_FLAGS, INDICATOR_CONFIG
)
from enhanced_data_handler import enhanced_data_handler

# Configure logging
import logging
logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """Track strategy performance metrics"""
    name: str
    total_signals: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    last_signal_time: Optional[datetime] = None
    confidence_sum: float = 0.0
    
    def add_signal(self, confidence: float, pnl: float = 0.0):
        """Add signal performance data"""
        self.total_signals += 1
        self.confidence_sum += confidence
        self.total_pnl += pnl
        self.last_signal_time = datetime.now()
        
        if self.total_signals > 0:
            self.avg_pnl = self.total_pnl / self.total_signals
    
    def get_avg_confidence(self) -> float:
        """Get average signal confidence"""
        return self.confidence_sum / max(self.total_signals, 1)

class IndicatorCache:
    """Cache for technical indicators to avoid recalculation"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache = {}
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def _get_cache_key(self, indicator: str, params: tuple, data_hash: str) -> str:
        """Generate cache key for indicator"""
        return f"{indicator}:{params}:{data_hash}"
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached data is still valid"""
        return time.time() - timestamp < self.ttl
    
    def get_indicator(self, indicator: str, params: tuple, data_hash: str) -> Optional[Any]:
        """Get cached indicator if available and valid"""
        cache_key = self._get_cache_key(indicator, params, data_hash)
        
        with self._lock:
            if cache_key in self.cache:
                value, timestamp = self.cache[cache_key]
                if self._is_cache_valid(timestamp):
                    return value
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
        
        return None
    
    def set_indicator(self, indicator: str, params: tuple, data_hash: str, value: Any):
        """Cache indicator calculation result"""
        cache_key = self._get_cache_key(indicator, params, data_hash)
        
        with self._lock:
            self.cache[cache_key] = (value, time.time())
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]

# Global indicator cache
indicator_cache = IndicatorCache(STRATEGY_CONFIG.get('cache_ttl_indicators', 60))

# =========================================================================
# OPTIMIZED TECHNICAL INDICATORS (NUMBA-ACCELERATED)
# =========================================================================

if NUMBA_AVAILABLE:
    @njit
    def _ema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """Numba-accelerated EMA calculation"""
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @njit
    def _rsi_numba(close_prices: np.ndarray, period: int) -> np.ndarray:
        """Numba-accelerated RSI calculation"""
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.empty_like(close_prices)
        avg_loss = np.empty_like(close_prices)
        
        # Initial averages
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Rolling averages
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @njit
    def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Numba-accelerated ATR calculation"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        true_range[0] = high[0] - low[0]  # First element
        
        atr = np.empty_like(true_range)
        atr[period-1] = np.mean(true_range[:period])
        
        for i in range(period, len(true_range)):
            atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
        
        return atr

class OptimizedIndicators:
    """Optimized technical indicators with caching"""
    
    @staticmethod
    def _get_data_hash(data: pd.Series) -> str:
        """Generate hash for data to use as cache key"""
        # Use last 10 values for hash to detect changes
        last_values = data.tail(10).values.tobytes()
        return str(hash(last_values))
    
    @classmethod
    def ema(cls, data: pd.Series, period: int) -> pd.Series:
        """Optimized EMA calculation with caching"""
        if not OPTIMIZATION_FLAGS.get('cache_indicators', False):
            return data.ewm(span=period, adjust=False).mean()
        
        data_hash = cls._get_data_hash(data)
        cache_key = f"ema_{period}_{data_hash}"
        
        # Try to get from cache
        cached_result = indicator_cache.get_indicator('ema', (period,), data_hash)
        if cached_result is not None:
            return pd.Series(cached_result, index=data.index)
        
        # Calculate EMA
        if NUMBA_AVAILABLE and len(data) > period:
            result = _ema_numba(data.values, period)
        else:
            result = data.ewm(span=period, adjust=False).mean().values
        
        # Cache the result
        indicator_cache.set_indicator('ema', (period,), data_hash, result)
        
        return pd.Series(result, index=data.index)
    
    @classmethod
    def rsi(cls, data: pd.Series, period: int = 14) -> pd.Series:
        """Optimized RSI calculation with caching"""
        if not OPTIMIZATION_FLAGS.get('cache_indicators', False):
            return cls._rsi_pandas(data, period)
        
        data_hash = cls._get_data_hash(data)
        
        # Try to get from cache
        cached_result = indicator_cache.get_indicator('rsi', (period,), data_hash)
        if cached_result is not None:
            return pd.Series(cached_result, index=data.index)
        
        # Calculate RSI
        if NUMBA_AVAILABLE and len(data) > period * 2:
            result = _rsi_numba(data.values, period)
        else:
            result = cls._rsi_pandas(data, period).values
        
        # Cache the result
        indicator_cache.set_indicator('rsi', (period,), data_hash, result)
        
        return pd.Series(result, index=data.index)
    
    @staticmethod
    def _rsi_pandas(data: pd.Series, period: int) -> pd.Series:
        """Standard pandas RSI calculation"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @classmethod
    def atr(cls, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Optimized ATR calculation with caching"""
        if not OPTIMIZATION_FLAGS.get('cache_indicators', False):
            return cls._atr_pandas(high, low, close, period)
        
        # Create combined hash from all series
        combined_hash = hash((
            high.tail(10).values.tobytes(),
            low.tail(10).values.tobytes(),
            close.tail(10).values.tobytes()
        ))
        
        # Try to get from cache
        cached_result = indicator_cache.get_indicator('atr', (period,), str(combined_hash))
        if cached_result is not None:
            return pd.Series(cached_result, index=close.index)
        
        # Calculate ATR
        if NUMBA_AVAILABLE and len(close) > period * 2:
            result = _atr_numba(high.values, low.values, close.values, period)
        else:
            result = cls._atr_pandas(high, low, close, period).values
        
        # Cache the result
        indicator_cache.set_indicator('atr', (period,), str(combined_hash), result)
        
        return pd.Series(result, index=close.index)
    
    @staticmethod
    def _atr_pandas(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Standard pandas ATR calculation"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @classmethod
    def bollinger_bands(cls, data: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands calculation with caching"""
        if not OPTIMIZATION_FLAGS.get('cache_indicators', False):
            return cls._bb_pandas(data, period, std)
        
        data_hash = cls._get_data_hash(data)
        cache_key = f"bb_{period}_{std}_{data_hash}"
        
        # Try to get from cache
        cached_result = indicator_cache.get_indicator('bb', (period, std), data_hash)
        if cached_result is not None:
            upper, middle, lower = cached_result
            return (
                pd.Series(upper, index=data.index),
                pd.Series(middle, index=data.index),
                pd.Series(lower, index=data.index)
            )
        
        # Calculate Bollinger Bands
        upper, middle, lower = cls._bb_pandas(data, period, std)
        
        # Cache the result
        result_tuple = (upper.values, middle.values, lower.values)
        indicator_cache.set_indicator('bb', (period, std), data_hash, result_tuple)
        
        return upper, middle, lower
    
    @staticmethod
    def _bb_pandas(data: pd.Series, period: int, std: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Standard pandas Bollinger Bands calculation"""
        middle = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

# =========================================================================
# ENHANCED STRATEGY ENGINE
# =========================================================================

class EnhancedStrategyEngine:
    """
    Enhanced strategy engine with caching, optimization, and improved performance
    Phase 2: High-performance signal generation for scalping operations
    """
    
    def __init__(self):
        self.strategies = {}
        self.performance_tracker = {}
        self.last_cleanup = datetime.now()
        self._lock = threading.Lock()
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Start background tasks
        if OPTIMIZATION_FLAGS.get('cache_indicators', False):
            self._start_cache_cleanup_task()
        
        logger.info("✅ Enhanced Strategy Engine initialized with optimizations")
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        from enhanced_strategies.opening_range_breakout import OptimizedOpeningRangeBreakout
        from enhanced_strategies.trend_following import OptimizedTrendFollowing
        from enhanced_strategies.pullback import OptimizedPullbackStrategy
        from enhanced_strategies.mean_reversion import OptimizedMeanReversion
        from enhanced_strategies.momentum_continuation import OptimizedMomentumContinuation
        
        self.strategies = {
            'orb': OptimizedOpeningRangeBreakout(),
            'trend_following': OptimizedTrendFollowing(),
            'pullback': OptimizedPullbackStrategy(),
            'mean_reversion': OptimizedMeanReversion(),
            'momentum_continuation': OptimizedMomentumContinuation()
        }
        
        # Initialize performance trackers
        for name in self.strategies.keys():
            self.performance_tracker[name] = StrategyPerformance(name)
    
    def _start_cache_cleanup_task(self):
        """Start background task for cache cleanup"""
        def cleanup_task():
            while True:
                time.sleep(300)  # Clean up every 5 minutes
                indicator_cache.clear_expired()
                logger.debug("Cache cleanup completed")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def process_new_bar(self, bar_data: Dict, context: Dict = None) -> List[Dict]:
        """
        Process new bar through all enabled strategies with optimization
        Returns list of generated signals
        """
        start_time = time.time()
        signals = []
        
        try:
            # Get recent bars for analysis
            df = enhanced_data_handler.get_latest_bars(500)
            if df.empty:
                return signals
            
            current_price = float(bar_data['close'])
            timestamp = pd.to_datetime(bar_data['timestamp'])
            
            # Pre-compute common indicators if caching is enabled
            if OPTIMIZATION_FLAGS.get('precompute_common_indicators', False):
                df = self._precompute_indicators(df)
            
            # Process each strategy
            for strategy_name, strategy in self.strategies.items():
                if not strategy.is_enabled():
                    continue
                
                try:
                    # Check cooldown period
                    if not strategy.can_generate_signal(timestamp):
                        continue
                    
                    # Generate signal
                    signal = strategy.generate_signal(df, current_price, context)
                    
                    if signal and self._validate_signal(signal, strategy_name):
                        # Add metadata
                        signal['strategy'] = strategy_name
                        signal['timestamp'] = timestamp.isoformat()
                        signal['processing_time_ms'] = (time.time() - start_time) * 1000
                        
                        signals.append(signal)
                        
                        # Update performance tracking
                        self.performance_tracker[strategy_name].add_signal(
                            signal.get('confidence', 0.5)
                        )
                        
                        logger.info(f"✅ {strategy_name.upper()} signal: {signal['signal']} @ {signal['price']:.2f} "
                                  f"(confidence: {signal.get('confidence', 0):.2f})")
                        
                        # Store signal
                        enhanced_data_handler.append_signal(signal)
                        
                except Exception as e:
                    logger.error(f"Error in {strategy_name} strategy: {e}")
            
            total_processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Processed bar in {total_processing_time:.2f}ms, generated {len(signals)} signals")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in strategy engine: {e}")
            return signals
    
    def _precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute common technical indicators"""
        df = df.copy()
        
        # Common EMA periods
        ema_periods = [8, 20, 21, 50]
        for period in ema_periods:
            col_name = f'ema_{period}'
            if col_name not in df.columns:
                df[col_name] = OptimizedIndicators.ema(df['close'], period)
        
        # RSI
        if 'rsi' not in df.columns:
            df['rsi'] = OptimizedIndicators.rsi(df['close'], INDICATOR_CONFIG['rsi_period'])
        
        # Bollinger Bands
        bb_cols = ['bb_upper', 'bb_middle', 'bb_lower']
        if not all(col in df.columns for col in bb_cols):
            upper, middle, lower = OptimizedIndicators.bollinger_bands(
                df['close'], 
                INDICATOR_CONFIG['bb_period'], 
                INDICATOR_CONFIG['bb_std']
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        
        # ATR
        if 'atr' not in df.columns:
            df['atr'] = OptimizedIndicators.atr(
                df['high'], df['low'], df['close'], 14
            )
        
        return df
    
    def _validate_signal(self, signal: Dict, strategy_name: str) -> bool:
        """Validate signal before processing"""
        required_fields = ['signal', 'price', 'confidence']
        
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Missing required field '{field}' in {strategy_name} signal")
                return False
        
        # Validate signal direction
        if signal['signal'] not in ['LONG', 'SHORT']:
            logger.warning(f"Invalid signal direction '{signal['signal']}' in {strategy_name}")
            return False
        
        # Validate price
        if not isinstance(signal['price'], (int, float)) or signal['price'] <= 0:
            logger.warning(f"Invalid price '{signal['price']}' in {strategy_name} signal")
            return False
        
        # Validate confidence
        if not isinstance(signal['confidence'], (int, float)) or not (0 <= signal['confidence'] <= 1):
            logger.warning(f"Invalid confidence '{signal['confidence']}' in {strategy_name} signal")
            return False
        
        return True
    
    def get_strategy_performance(self, strategy_name: str) -> Dict:
        """Get performance metrics for a specific strategy"""
        if strategy_name not in self.performance_tracker:
            return {'error': 'Strategy not found'}
        
        performance = self.performance_tracker[strategy_name]
        return {
            'total_signals': performance.total_signals,
            'win_rate': performance.win_rate,
            'avg_pnl': performance.avg_pnl,
            'total_pnl': performance.total_pnl,
            'avg_confidence': performance.get_avg_confidence(),
            'last_signal_time': performance.last_signal_time
        }
    
    def get_all_strategy_performance(self) -> Dict:
        """Get performance metrics for all strategies"""
        return {
            name: self.get_strategy_performance(name)
            for name in self.strategies.keys()
        }
    
    def enable_strategy(self, strategy_name: str, enabled: bool = True):
        """Enable or disable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].set_enabled(enabled)
            logger.info(f"Strategy {strategy_name} {'enabled' if enabled else 'disabled'}")
    
    def update_strategy_config(self, strategy_name: str, config: Dict):
        """Update configuration for a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_config(config)
            logger.info(f"Updated configuration for {strategy_name}")


# Global singleton instance
enhanced_strategy_engine = EnhancedStrategyEngine()

# Convenience functions for backward compatibility
def process_new_bar(bar_data: Dict, context: Dict = None) -> List[Dict]:
    """Process new bar (backward compatibility)"""
    return enhanced_strategy_engine.process_new_bar(bar_data, context)

def get_strategy_performance(strategy_name: str) -> Dict:
    """Get strategy performance (backward compatibility)"""
    return enhanced_strategy_engine.get_strategy_performance(strategy_name)