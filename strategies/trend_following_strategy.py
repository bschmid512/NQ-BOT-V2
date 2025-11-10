"""
Trend Following Strategy
Trades in the direction of the prevailing trend using EMA crossovers
"""
from datetime import datetime
from typing import Dict, Optional
import pandas as pd


class TrendFollowingStrategy:
    """
    EMA-based trend following strategy
    
    Signals:
    - LONG: Fast EMA crosses above Medium EMA in uptrend (Fast > Medium > Slow)
    - SHORT: Fast EMA crosses below Medium EMA in downtrend (Slow > Medium > Fast)
    """
    
    def __init__(self, fast_ema: int = 8, medium_ema: int = 21, slow_ema: int = 50):
        """
        Initialize trend following strategy
        
        Args:
            fast_ema: Fast EMA period (default: 8)
            medium_ema: Medium EMA period (default: 21)
            slow_ema: Slow EMA period (default: 50)
        """
        self.name = "trend_following"
        self.fast_ema = fast_ema
        self.medium_ema = medium_ema
        self.slow_ema = slow_ema
        self.last_signal_time = None
        self.cooldown_minutes = 5
        
        print(f"✓ Trend Following Strategy initialized (EMA {fast_ema}/{medium_ema}/{slow_ema})")
    
    # FIXED: Added current_price parameter
    def generate_signal(self, df, current_price, context=None):
        """
        Generate trading signal based on EMA crossovers
        
        Args:
            df: pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            current_price: Current market price
            context: Market context (optional)
            
        Returns:
            dict with signal info or None
        """
        # Need enough data for slow EMA
        if len(df) < self.slow_ema:
            return None
        
        # Check cooldown period
        if self.last_signal_time:
            time_since_signal = (datetime.now() - self.last_signal_time).total_seconds() / 60
            if time_since_signal < self.cooldown_minutes:
                return None
        
        # Calculate EMAs
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=self.medium_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # Get current and previous values
        ema_fast = df['ema_fast'].iloc[-1]
        ema_medium = df['ema_medium'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        
        ema_fast_prev = df['ema_fast'].iloc[-2]
        ema_medium_prev = df['ema_medium'].iloc[-2]
        
        # LONG Signal: Fast crosses above Medium in uptrend
        if ema_fast > ema_medium > ema_slow and ema_fast_prev <= ema_medium_prev:
            self.last_signal_time = datetime.now()
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'strategy': self.name,
                'signal': 'LONG',  # Standardized to 'signal' instead of 'direction'
                'price': current_price,
                'confidence': 35,
                'reason': f'EMA {self.fast_ema} crossed above EMA {self.medium_ema} in uptrend',
                'metadata': {
                    'ema_fast': float(ema_fast),
                    'ema_medium': float(ema_medium),
                    'ema_slow': float(ema_slow)
                }
            }
        
        # SHORT Signal: Fast crosses below Medium in downtrend
        if ema_slow > ema_medium > ema_fast and ema_fast_prev >= ema_medium_prev:
            self.last_signal_time = datetime.now()
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'strategy': self.name,
                'signal': 'SHORT',  # Standardized to 'signal' instead of 'direction'
                'price': current_price,
                'confidence': 35,
                'reason': f'EMA {self.fast_ema} crossed below EMA {self.medium_ema} in downtrend',
                'metadata': {
                    'ema_fast': float(ema_fast),
                    'ema_medium': float(ema_medium),
                    'ema_slow': float(ema_slow)
                }
            }
        
        return None