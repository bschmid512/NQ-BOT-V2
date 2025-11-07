"""
Momentum Continuation Strategy
Enters strong trends that are already in motion.
Designed to get into a trade *during* a strong rally or selloff.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
from config import STRATEGIES 
from utils.indicators import TechnicalIndicators 

class MomentumContinuationStrategy:
    """
    Enters when price shows strong, sustained momentum.
    """
    
    def __init__(self):
        # Add 'momentum' to your STRATEGIES dict in config.py
        self.config = STRATEGIES.get('momentum', {'enabled': True, 'weight': 70})
        self.name = 'momentum'
        self.weight = self.config['weight']
        
        self.fast_ema_span = 8
        self.mid_ema_span = 21
        self.long_ema_span = 50
        self.adx_min = 25
        
        self.last_signal_time = None
        self.signal_cooldown_minutes = 15
        
        print(f"✅ Momentum Strategy initialized")
    
    def generate_signal(self, df: pd.DataFrame, current_price: float,
                       context: Dict = None) -> Optional[Dict]:
        
        if not self.config['enabled']:
            return None
        
        if len(df) < self.long_ema_span + 10:
            return None
        
        if self._in_cooldown():
            return None
        
        # Use your existing TechnicalIndicators class from utils/indicators.py
        df_with_indicators = TechnicalIndicators.add_all_indicators(df.copy())
        
        # Make sure your add_all_indicators includes 'ema_8', 'ema_50', 'adx', 'atr'
        # For this example, I'll assume they exist.
        if 'ema_21' not in df_with_indicators.columns or 'adx' not in df_with_indicators.columns:
            # Try to calculate them manually if missing
            df_with_indicators['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
            df_with_indicators['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df_with_indicators['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df_with_indicators['adx'] = self._calculate_adx(df) # Simple ADX
            df_with_indicators['atr'] = self._calculate_atr(df) # Simple ATR

        ema_fast = df_with_indicators['ema_8'].iloc[-1]
        ema_mid = df_with_indicators['ema_21'].iloc[-1]
        ema_long = df_with_indicators['ema_50'].iloc[-1]
        adx = df_with_indicators['adx'].iloc[-1]
        
        if adx < self.adx_min:
            return None # Not a strong enough trend

        trend_direction = None
        
        # --- Trend Definition: All EMAs aligned ---
        if current_price > ema_fast > ema_mid > ema_long:
            trend_direction = 'LONG'
        elif current_price < ema_fast < ema_mid < ema_long:
            trend_direction = 'SHORT'
        else:
            return None # Not in a clean, stacked trend

        # --- Entry Condition: Is price breaking out? ---
        is_breakout = False
        if trend_direction == 'LONG':
            if current_price >= df['high'].tail(5).max():
                is_breakout = True
        else: # SHORT
            if current_price <= df['low'].tail(5).min():
                is_breakout = True
        
        if not is_breakout:
            return None # Not actively breaking out, might be a pullback

        atr = df_with_indicators['atr'].iloc[-1]
        
        if trend_direction == 'LONG':
            stop_loss = current_price - (atr * 2.0)
            target = current_price + (atr * 3.0)
        else:
            stop_loss = current_price + (atr * 2.0)
            target = current_price - (atr * 3.0)
        
        # Confidence is scaled by ADX strength
        confidence = min((adx - self.adx_min) / (40 - self.adx_min), 1.0)
        
        self.last_signal_time = datetime.now()
        
        signal = {
            'strategy': self.name,
            'direction': trend_direction,
            'price': current_price,
            'stop': stop_loss,
            'target': target,
            'confidence': confidence, # 0-1 confidence
            'weight': self.weight,   # 0-100 weight
            'reason': f"Momentum {trend_direction} (ADX: {adx:.1f})",
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🚀 MOMENTUM Signal: {trend_direction} @ {current_price:.2f}")
        return signal

    def _in_cooldown(self) -> bool:
        if self.last_signal_time is None:
            return False
        minutes_since_last = (datetime.now() - self.last_signal_time).total_seconds() / 60
        return minutes_since_last < self.signal_cooldown_minutes

    # --- Minimal indicator calculations (in case they aren't in utils) ---
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        tr = np.max(ranges, axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        df['tr'] = self._calculate_atr(df, period)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / df['tr'])
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / df['tr'])
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx.fillna(0)