"""
Opening Range Breakout Strategy - FIXED VERSION
Trades breakouts of the first 15-minute range after market open
"""
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from typing import Optional, Dict
from config import STRATEGIES
from utils.logger import trading_logger


class OpeningRangeBreakout:
    """
    Opening Range Breakout Strategy - REALISTIC VERSION
    
    Expected Win Rate: ~55% (realistic, not 74%)
    Uses 15-minute opening range (9:30-9:45 ET)
    Target: ATR-based (typically 0.5-1.0x range)
    Stop: Opposite side of range or max 50 points
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or STRATEGIES['orb']
        self.or_period = self.config['or_period']
        self.target_pct = self.config['target_pct']
        self.max_sl_points = self.config['max_sl_points']
        self.min_range_pct = self.config.get('min_range_pct', 0.0015)
        self.max_range_pct = self.config.get('max_range_pct', 0.004)
        self.weight = self.config['weight']
        self.optimal_days = self.config.get('optimal_days', [0, 2, 4])
        
        self.logger = trading_logger.strategy_logger
        
        # State tracking
        self.or_high = None
        self.or_low = None
        self.or_size = None
        self.trade_taken_today = False
        self.current_date = None
        
        self.logger.info(f"ORB Strategy initialized: {self.or_period}min period, {self.target_pct*100}% target")
    
    def reset_daily_state(self, current_date: datetime.date):
        """Reset state for new trading day"""
        if self.current_date != current_date:
            self.or_high = None
            self.or_low = None
            self.or_size = None
            self.trade_taken_today = False
            self.current_date = current_date
            self.logger.info(f"ORB state reset for {current_date}")
    
    def calculate_opening_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate opening range from morning data
        
        Args:
            df: DataFrame with OHLCV data for opening period
            
        Returns:
            Dict with OR high, low, size, or None if insufficient data
        """
        try:
            if df.empty or len(df) < 1:
                return None
            
            or_high = df['high'].max()
            or_low = df['low'].min()
            or_size = or_high - or_low
            
            # FIXED: More realistic range filter
            # At 25,600: 0.15% = 38 points, 0.4% = 102 points
            or_pct = or_size / df['close'].iloc[-1]
            
            if or_pct < self.min_range_pct:
                self.logger.info(f"OR too small: {or_pct:.3%} ({or_size:.2f} pts) < {self.min_range_pct:.3%}")
                return None
            
            if or_pct > self.max_range_pct:
                self.logger.info(f"OR too large: {or_pct:.3%} ({or_size:.2f} pts) > {self.max_range_pct:.3%}")
                return None
            
            self.logger.info(f"Valid OR: High={or_high:.2f}, Low={or_low:.2f}, Size={or_size:.2f} ({or_pct:.3%})")
            
            return {
                'high': or_high,
                'low': or_low,
                'size': or_size,
                'pct': or_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating opening range: {e}")
            return None
    
    # <-- UPDATED SIGNATURE (removed levels, added context)
    def generate_signal(self, current_bar, or_data=None, context: Dict = None) -> Optional[Dict]:
        """
        Generate ORB signal
        
        Args:
            current_bar: The latest price bar
            or_data (Optional): Provided by engine if calculated just-in-time
            context (Dict): Market context (ES trend, VIX status)
        """
        try:
            # Check if we should trade today
            today = pd.to_datetime(current_bar.name if hasattr(current_bar, 'name') else datetime.now()).weekday()
            if today not in self.optimal_days:
                self.logger.debug(f"ORB: Skipping trade, not an optimal day ({today})")
                self.trade_taken_today = True # Prevent trades
                return None

            # Check if a trade was already taken
            if self.trade_taken_today:
                return None
            
            # Check if we have OR data
            if or_data is None:
                if self.or_high is None:
                    return None # Still waiting for OR to be set by engine
                # Use state set by engine
                or_data = {
                    'high': self.or_high,
                    'low': self.or_low,
                    'size': self.or_size
                }
            
            current_price = current_bar['close']
            
            # Get context
            es_trend = context.get('es_trend', 'NEUTRAL') if context else 'NEUTRAL'
            
            # Long breakout above OR high
            if current_price > or_data['high']:
                
                # ⭐⭐⭐ NEW CONTEXT FILTER ("WHY") ⭐⭐⭐
                if es_trend == 'STRONG_DOWN':
                    self.logger.info("ORB LONG rejected: ES trend is STRONG_DOWN.")
                    return None
                
                # Stop at OR low, but cap at max points
                stop_distance = or_data['size']
                if stop_distance > self.max_sl_points:
                    stop_distance = self.max_sl_points
                
                stop_loss = current_price - stop_distance
                
                # Target: 50% of range or minimum 1.5:1 R:R
                target_distance = or_data['size'] * self.target_pct
                min_target = stop_distance * 1.5  # Minimum 1.5:1 R:R
                
                if target_distance < min_target:
                    target_distance = min_target
                
                take_profit = current_price + target_distance
                
                signal = {
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else datetime.now(),
                    'strategy': 'orb',
                    'signal': 'LONG',
                    'confidence': self.weight,
                    'price': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'or_high': or_data['high'],
                    'or_low': or_data['low'],
                    'or_size': or_data['size']
                }
                
                self.trade_taken_today = True
                self.logger.info(
                    f"ORB LONG: Entry={current_price:.2f}, Target={take_profit:.2f} (+{target_distance:.2f}), "
                    f"Stop={stop_loss:.2f} (-{stop_distance:.2f}), R:R={(target_distance/stop_distance):.2f}:1"
                )
                return signal
            
            # Short breakout below OR low
            elif current_price < or_data['low']:
                
                # ⭐⭐⭐ NEW CONTEXT FILTER ("WHY") ⭐⭐⭐
                if es_trend == 'STRONG_UP':
                    self.logger.info("ORB SHORT rejected: ES trend is STRONG_UP.")
                    return None
                
                # Stop at OR high, but cap at max points
                stop_distance = or_data['size']
                if stop_distance > self.max_sl_points:
                    stop_distance = self.max_sl_points
                
                stop_loss = current_price + stop_distance
                
                # Target: 50% of range or minimum 1.5:1 R:R
                target_distance = or_data['size'] * self.target_pct
                min_target = stop_distance * 1.5
                
                if target_distance < min_target:
                    target_distance = min_target
                
                take_profit = current_price - target_distance
                
                signal = {
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else datetime.now(),
                    'strategy': 'orb',
                    'signal': 'SHORT',
                    'confidence': self.weight,
                    'price': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'or_high': or_data['high'],
                    'or_low': or_data['low'],
                    'or_size': or_data['size']
                }
                
                self.trade_taken_today = True
                self.logger.info(
                    f"ORB SHORT: Entry={current_price:.2f}, Target={take_profit:.2f} (-{target_distance:.2f}), "
                    f"Stop={stop_loss:.2f} (+{stop_distance:.2f}), R:R={(target_distance/stop_distance):.2f}:1"
                )
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating ORB signal: {e}")
            return None
    
    def is_within_trading_hours(self, timestamp: datetime) -> bool:
        """Check if current time is after opening range period"""
        et_tz = pytz.timezone('US/Eastern')
        current_time = timestamp.astimezone(et_tz).time() if timestamp.tzinfo else et_tz.localize(timestamp).time()
        
        # Market opens at 9:30 ET, OR ends at 9:30 + or_period
        or_end_hour = 9
        or_end_minute = 30 + self.or_period
        if or_end_minute >= 60:
            or_end_hour += 1
            or_end_minute -= 60
        
        or_end_time = time(or_end_hour, or_end_minute)
        market_close = time(16, 0)
        
        return or_end_time < current_time < market_close
    
    def should_trade_today(self, day_of_week: int) -> bool:
        """
        Determine if we should trade based on day of week
        Best performance: Monday (0), Wednesday (2), Friday (4)
        
        Args:
            day_of_week: 0=Monday, 6=Sunday
        """
        return day_of_week in self.optimal_days


# Create global strategy instance
orb_strategy = OpeningRangeBreakout()