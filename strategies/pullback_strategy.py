"""
Pullback Strategy
Enters on pullbacks in trending markets
"""
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np


class PullbackStrategy:
    """
    Pullback strategy that enters on retracements in trending markets
    
    Signals:
    - LONG: Price pulls back to EMA8 in uptrend (EMA8 > EMA50, ADX > 25)
    - SHORT: Price bounces to EMA8 in downtrend (EMA8 < EMA50, ADX > 25)
    """
    
    def __init__(self, ema_short: int = 8, ema_long: int = 50, adx_threshold: float = 25):
        """
        Initialize pullback strategy
        
        Args:
            ema_short: Short EMA period (default: 8)
            ema_long: Long EMA period (default: 50)
            adx_threshold: Minimum ADX for trend strength (default: 25)
        """
        self.name = "pullback"
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.adx_threshold = adx_threshold
        self.last_signal_time = None
        self.cooldown_minutes = 3
        
        print(f"✓ Pullback Strategy initialized (EMA {ema_short}/{ema_long}, ADX > {adx_threshold})")
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed +DI and -DI
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    # FIXED: Added current_price parameter
    def generate_signal(self, df, current_price, context=None):
        """
        Generate pullback trading signals
        
        Args:
            df: pandas DataFrame with OHLCV data
            current_price: Current market price
            context: Market context (optional)
            
        Returns:
            dict with signal info or None
        """
        if len(df) < max(self.ema_short, self.ema_long):
            return None
        
        # Check cooldown
        if self.last_signal_time:
            time_since_signal = (datetime.now() - self.last_signal_time).total_seconds() / 60
            if time_since_signal < self.cooldown_minutes:
                return None
        
        # Calculate indicators
        df = df.copy()
        df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
        df['adx'] = self._calculate_adx(df)
        
        # Get current values
        ema_short = df['ema_short'].iloc[-1]
        ema_long = df['ema_long'].iloc[-1]
        adx = df['adx'].iloc[-1]
        
        # Check for valid ADX
        if pd.isna(adx) or adx < self.adx_threshold:
            return None
        
        # Calculate pullback distance
        pullback_distance = abs(current_price - ema_short) / ema_short
        
        # LONG Signal: Uptrend pullback
        if ema_short > ema_long:
            # Price near EMA8 (within 0.2%)
            if pullback_distance < 0.002:
                confidence = min(0.8, adx / 40)  # Scale confidence with ADX strength
                self.last_signal_time = datetime.now()
                
                print(f"🔄 PULLBACK Signal: LONG @ {current_price}")
                print(f"   EMA8: {ema_short:.2f} | EMA50: {ema_long:.2f}")
                print(f"   ADX: {adx:.1f} | Confidence: {confidence:.2f}")
                
                return {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'strategy': self.name,
                    'signal': 'LONG',  # Standardized key
                    'confidence': confidence * 100,
                    'price': current_price,
                    'reason': f'Pullback to EMA{self.ema_short} in uptrend (ADX: {adx:.1f})',
                    'metadata': {
                        'ema_short': float(ema_short),
                        'ema_long': float(ema_long),
                        'adx': float(adx),
                        'pullback_distance': float(pullback_distance)
                    }
                }
        
        # SHORT Signal: Downtrend pullback
        elif ema_short < ema_long:
            # Price near EMA8 (within 0.2%)
            if pullback_distance < 0.002:
                confidence = min(0.8, adx / 40)
                self.last_signal_time = datetime.now()
                
                print(f"🔄 PULLBACK Signal: SHORT @ {current_price}")
                print(f"   EMA8: {ema_short:.2f} | EMA50: {ema_long:.2f}")
                print(f"   ADX: {adx:.1f} | Confidence: {confidence:.2f}")
                
                return {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'strategy': self.name,
                    'signal': 'SHORT',  # Standardized key
                    'confidence': confidence * 100,
                    'price': current_price,
                    'reason': f'Pullback to EMA{self.ema_short} in downtrend (ADX: {adx:.1f})',
                    'metadata': {
                        'ema_short': float(ema_short),
                        'ema_long': float(ema_long),
                        'adx': float(adx),
                        'pullback_distance': float(pullback_distance)
                    }
                }
        
        return None