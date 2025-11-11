"""
Optimized Opening Range Breakout Strategy
Enhanced for high-performance scalping operations
"""
from __future__ import annotations

import time
import threading
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import pytz

from config import ORB_STRATEGY_CONFIG
import logging
logger = logging.getLogger(__name__)

class OptimizedOpeningRangeBreakout:
    """
    High-performance Opening Range Breakout strategy
    Optimized for low-latency signal generation and enhanced profitability
    """
    
    def __init__(self):
        self.config = ORB_STRATEGY_CONFIG
        self.name = "orb"
        self.enabled = self.config.get('enabled', True)
        
        # Strategy state
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_size: Optional[float] = None
        self.current_date: Optional[datetime.date] = None
        self.trade_taken_today = False
        self.last_signal_time: Optional[datetime] = None
        
        # Timezone
        self._et = pytz.timezone("US/Eastern")
        
        # Cooldown tracking
        self.signal_cooldown = {}  # Track signal cooldowns
        self._lock = threading.Lock()
        
        logger.info(f"✅ Optimized ORB Strategy initialized: period={self.config['or_period']}m, "
                   f"weight={self.config['weight']}, position_multiplier={self.config['position_size_multiplier']}")
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable strategy"""
        self.enabled = enabled
    
    def update_config(self, config: Dict):
        """Update strategy configuration"""
        self.config.update(config)
    
    def can_generate_signal(self, timestamp: datetime) -> bool:
        """Check if strategy can generate signal based on cooldown"""
        if self.last_signal_time is None:
            return True
        
        time_since_last = (timestamp - self.last_signal_time).total_seconds() / 60
        cooldown_minutes = self.config.get('cooldown_minutes', 2)
        
        return time_since_last >= cooldown_minutes
    
    def reset_daily_state(self, current_date: datetime.date):
        """Reset daily strategy state"""
        with self._lock:
            if self.current_date != current_date:
                self.or_high = self.or_low = self.or_size = None
                self.trade_taken_today = False
                self.current_date = current_date
                logger.info(f"ORB daily state reset for {current_date}")
    
    def should_trade_today(self, day_of_week: int) -> bool:
        """Check if we should trade today based on optimal days"""
        return day_of_week in self.config.get('optimal_days', [0, 2, 4])
    
    def _is_within_trading_hours(self, timestamp: datetime) -> bool:
        """Check if we're within optimal trading hours"""
        try:
            if isinstance(timestamp, pd.Timestamp):
                ts_et = timestamp.tz_convert(self._et)
            else:
                if timestamp.tzinfo is None:
                    timestamp = pytz.UTC.localize(timestamp)
                ts_et = timestamp.astimezone(self._et)
            
            current_time = ts_et.time()
            
            # ORB trading window: after OR period until market close
            or_end_time = time(9, 30 + self.config['or_period'])
            market_close = time(16, 0)
            
            return or_end_time <= current_time <= market_close
            
        except Exception as e:
            logger.error(f"Error checking trading hours: {e}")
            return False
    
    def _calculate_opening_range(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate opening range with enhanced validation"""
        try:
            if df is None or df.empty:
                return None
            
            # Ensure we have enough data
            if len(df) < 10:
                return None
            
            # Get the last timestamp to determine the trading day
            last_ts = df.index.max()
            if pd.isna(last_ts):
                return None
            
            # Convert to Eastern time for OR calculation
            if isinstance(last_ts, pd.Timestamp):
                last_ts_et = last_ts.tz_convert(self._et) if last_ts.tzinfo else self._et.localize(last_ts)
            else:
                last_ts_et = self._et.localize(last_ts)
            
            # Define OR period
            market_open = last_ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
            or_end = market_open + pd.Timedelta(minutes=self.config['or_period'])
            
            # Filter bars within OR period
            start_utc = market_open.tz_convert('UTC')
            end_utc = or_end.tz_convert('UTC')
            
            # Ensure DataFrame index is timezone-aware
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC')
            
            or_bars = df[(df.index >= start_utc) & (df.index < end_utc)]
            
            if or_bars.empty or len(or_bars) < 2:
                logger.debug(f"Insufficient OR data: {len(or_bars)} bars")
                return None
            
            # Calculate OR levels
            or_high = float(or_bars["high"].max())
            or_low = float(or_bars["low"].min())
            or_size = max(or_high - or_low, 0.0)
            
            # Enhanced validation
            if or_size <= 0:
                logger.debug("Invalid OR size (<= 0)")
                return None
            
            # Validate range size relative to recent price
            recent_close = float(df["close"].iloc[-1])
            range_pct = or_size / max(recent_close, 1e-6)
            
            min_range_pct = self.config.get('min_range_pct', 0.0015)
            max_range_pct = self.config.get('max_range_pct', 0.0040)
            
            if not (min_range_pct <= range_pct <= max_range_pct):
                logger.debug(f"OR range rejected: {range_pct:.4f} not in [{min_range_pct:.4f}, {max_range_pct:.4f}]")
                return None
            
            # Additional validation: minimum volume or volatility
            or_volume = or_bars["volume"].sum() if "volume" in or_bars.columns else None
            
            logger.info(f"Opening Range calculated: High={or_high:.2f}, Low={or_low:.2f}, Size={or_size:.2f}, Range%={range_pct:.3f}")
            
            return {
                "high": or_high,
                "low": or_low,
                "size": or_size,
                "volume": or_volume,
                "range_pct": range_pct,
                "start": start_utc,
                "end": end_utc
            }
            
        except Exception as e:
            logger.error(f"Error calculating opening range: {e}")
            return None
    
    def _is_after_or_window(self, timestamp: pd.Timestamp) -> bool:
        """Check if we're after the OR window"""
        try:
            if isinstance(timestamp, pd.Timestamp):
                ts_et = timestamp.tz_convert(self._et) if timestamp.tzinfo else self._et.localize(timestamp)
            else:
                ts_et = self._et.localize(timestamp)
            
            or_end = ts_et.replace(hour=9, minute=30, second=0, microsecond=0) + \
                    pd.Timedelta(minutes=self.config['or_period'])
            
            return ts_et >= or_end
            
        except Exception as e:
            logger.error(f"Error checking OR window: {e}")
            return True
    
    def generate_signal(self, df: pd.DataFrame, current_price: float, context: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Generate ORB trading signal with enhanced logic
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            context: Market context (optional)
            
        Returns:
            Signal dictionary or None
        """
        start_time = time.time()
        
        try:
            # Get current timestamp
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                timestamp = df.index[-1]
            else:
                timestamp = pd.Timestamp.now(tz='UTC')
            
            # Reset daily state if needed
            self.reset_daily_state(timestamp.date())
            
            # Check if we should trade today
            if not self.should_trade_today(timestamp.weekday()):
                return None
            
            # Check if we're within trading hours
            if not self._is_after_or_window(timestamp):
                return None
            
            # Check cooldown
            if not self.can_generate_signal(timestamp):
                return None
            
            # Calculate or retrieve opening range
            if self.or_high is None:
                or_data = self._calculate_opening_range(df)
                if or_data is None:
                    return None
                
                self.or_high = or_data["high"]
                self.or_low = or_data["low"]
                self.or_size = or_data["size"]
            
            # Check for breakout
            broke_up = current_price > self.or_high
            broke_dn = current_price < self.or_low
            
            if not (broke_up or broke_dn):
                return None
            
            # Enhanced signal generation with volume confirmation
            signal = self._create_signal(
                current_price, broke_up, df, context, or_data={
                    'high': self.or_high,
                    'low': self.or_low,
                    'size': self.or_size
                }
            )
            
            if signal:
                self.last_signal_time = timestamp
                processing_time = (time.time() - start_time) * 1000
                signal['processing_time_ms'] = processing_time
                
                logger.debug(f"ORB signal generated in {processing_time:.2f}ms")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ORB signal: {e}")
            return None
    
    def _create_signal(self, price: float, is_long: bool, df: pd.DataFrame, 
                      context: Dict = None, or_data: Dict = None) -> Optional[Dict[str, Any]]:
        """Create trading signal with enhanced risk management"""
        try:
            direction = "LONG" if is_long else "SHORT"
            
            # Base stop loss size
            base_sl_size = or_data['size'] if or_data else 20.0
            max_sl_points = self.config.get('max_sl_points', 50)
            
            # Calculate stop loss
            if is_long:
                stop_price = price - min(base_sl_size, max_sl_points)
                target_dist = max(
                    base_sl_size * self.config['target_pct'],
                    1.5 * (price - stop_price)
                )
                target_price = price + target_dist
            else:
                stop_price = price + min(base_sl_size, max_sl_points)
                target_dist = max(
                    base_sl_size * self.config['target_pct'],
                    1.5 * (stop_price - price)
                )
                target_price = price - target_dist
            
            # Enhanced confidence calculation
            confidence = self._calculate_confidence(price, direction, df, context)
            
            # Position size multiplier for high-confidence signals
            position_multiplier = self.config.get('position_size_multiplier', 2.5)
            
            signal = {
                'strategy': self.name,
                'signal': direction,
                'direction': direction,  # For compatibility
                'price': float(price),
                'stop_loss': float(stop_price),
                'stop': float(stop_price),  # For compatibility
                'take_profit': float(target_price),
                'target': float(target_price),  # For compatibility
                'confidence': float(confidence),
                'weight': float(self.config['weight']),
                'position_size': position_multiplier if confidence > 0.7 else 1.0,
                'reason': f'ORB breakout {direction} (range size: {base_sl_size:.1f})',
                'or_high': or_data['high'] if or_data else None,
                'or_low': or_data['low'] if or_data else None,
                'or_size': or_data['size'] if or_data else None,
                'metadata': {
                    'range_size': base_sl_size,
                    'risk_reward_ratio': target_dist / max(price - stop_price, stop_price - price, 1),
                    'confidence_factors': self._get_confidence_factors(price, direction, df, context)
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating signal: {e}")
            return None
    
    def _calculate_confidence(self, price: float, direction: str, df: pd.DataFrame, 
                            context: Dict = None) -> float:
        """Calculate enhanced signal confidence"""
        try:
            base_confidence = self.config.get('weight', 0.6)
            confidence_factors = []
            
            # Volume confirmation (if available)
            if 'volume' in df.columns and len(df) > 5:
                recent_volume = df['volume'].tail(5).mean()
                avg_volume = df['volume'].tail(20).mean()
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    if volume_ratio > 1.5:
                        confidence_factors.append(0.1)  # 10% boost for high volume
            
            # Momentum confirmation
            if len(df) > 5:
                recent_change = (price - df['close'].iloc[-5]) / df['close'].iloc[-5]
                if (direction == 'LONG' and recent_change > 0.002) or \
                   (direction == 'SHORT' and recent_change < -0.002):
                    confidence_factors.append(0.05)  # 5% boost for momentum
            
            # Time of day factor (higher confidence during US session)
            current_time = datetime.now(self._et).time()
            us_session_start = time(9, 30)
            us_session_end = time(11, 0)  # First hour is best for ORB
            
            if us_session_start <= current_time <= us_session_end:
                confidence_factors.append(0.15)  # 15% boost for optimal timing
            
            # Market context factors
            if context:
                if direction == 'LONG' and context.get('market_regime') == 'trending':
                    confidence_factors.append(0.05)
                elif direction == 'SHORT' and context.get('market_regime') == 'trending':
                    confidence_factors.append(0.05)
            
            # Calculate final confidence (capped at 0.95)
            final_confidence = min(base_confidence + sum(confidence_factors), 0.95)
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return self.config.get('weight', 0.6)
    
    def _get_confidence_factors(self, price: float, direction: str, df: pd.DataFrame, 
                               context: Dict = None) -> List[str]:
        """Get list of confidence factors for signal metadata"""
        factors = []
        
        # Volume factor
        if 'volume' in df.columns and len(df) > 5:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].tail(20).mean()
            if avg_volume > 0 and recent_volume / avg_volume > 1.5:
                factors.append("High volume confirmation")
        
        # Momentum factor
        if len(df) > 5:
            recent_change = (price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            if abs(recent_change) > 0.002:
                factors.append("Momentum confirmation")
        
        # Time factor
        current_time = datetime.now(self._et).time()
        us_session_start = time(9, 30)
        us_session_end = time(11, 0)
        if us_session_start <= current_time <= us_session_end:
            factors.append("Optimal trading window")
        
        # Market regime
        if context and context.get('market_regime') == 'trending':
            factors.append("Trending market")
        
        return factors
    
    # =========================================================================
    # PERFORMANCE MONITORING
    # =========================================================================
    
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'config': self.config,
            'or_levels': {
                'high': self.or_high,
                'low': self.or_low,
                'size': self.or_size
            } if self.or_high else None,
            'last_signal_time': self.last_signal_time,
            'signals_today': self.trade_taken_today
        }