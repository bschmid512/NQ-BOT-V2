"""
Mean Reversion Strategy
Trades when price deviates significantly from its mean
"""
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np


class MeanReversionStrategy:
    """
    Mean reversion strategy using Bollinger Bands and RSI
    
    Signals:
    - LONG: Price below lower BB and RSI < 30 (oversold)
    - SHORT: Price above upper BB and RSI > 70 (overbought)
    """
    
    def __init__(self, lookback_period: int = 20, num_std: float = 2.0, rsi_period: int = 14):
        """
        Initialize mean reversion strategy
        
        Args:
            lookback_period: Period for moving average (default: 20)
            num_std: Number of standard deviations for bands (default: 2.0)
            rsi_period: RSI calculation period (default: 14)
        """
        self.name = "mean_reversion"
        self.lookback_period = lookback_period
        self.num_std = num_std
        self.rsi_period = rsi_period
        self.last_signal_time = None
        self.cooldown_minutes = 5
        
        print(f"✓ Mean Reversion Strategy initialized (BB {lookback_period}/{num_std}σ, RSI {rsi_period})")
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # FIXED: Added current_price parameter
    def generate_signal(self, df, current_price, context=None):
        """
        Generate mean reversion signals
        
        Args:
            df: pandas DataFrame with OHLCV data
            current_price: Current market price
            context: Market context (optional)
            
        Returns:
            dict with signal info or None
        """
        if len(df) < self.lookback_period:
            return None
        
        # Check cooldown
        if self.last_signal_time:
            time_since_signal = (datetime.now() - self.last_signal_time).total_seconds() / 60
            if time_since_signal < self.cooldown_minutes:
                return None
        
        # Calculate indicators
        df = df.copy()
        df['sma'] = df['close'].rolling(window=self.lookback_period).mean()
        df['std'] = df['close'].rolling(window=self.lookback_period).std()
        df['upper_bb'] = df['sma'] + (self.num_std * df['std'])
        df['lower_bb'] = df['sma'] - (self.num_std * df['std'])
        df['rsi'] = self._calculate_rsi(df, self.rsi_period)
        
        # Get current values
        sma = df['sma'].iloc[-1]
        upper_bb = df['upper_bb'].iloc[-1]
        lower_bb = df['lower_bb'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        # Check for valid values
        if pd.isna(rsi) or pd.isna(upper_bb) or pd.isna(lower_bb):
            return None
        
        # LONG Signal: Oversold conditions
        if current_price < lower_bb and rsi < 30:
            distance_from_bb = (lower_bb - current_price) / lower_bb
            confidence = min(0.8, distance_from_bb * 100 + (30 - rsi) / 30 * 0.5)
            self.last_signal_time = datetime.now()
            
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'strategy': self.name,
                'signal': 'LONG',  # Standardized key
                'confidence': confidence * 100,
                'price': current_price,
                'reason': f'Oversold: Price below BB ({distance_from_bb:.2%}), RSI {rsi:.1f}',
                'metadata': {
                    'sma': float(sma),
                    'upper_bb': float(upper_bb),
                    'lower_bb': float(lower_bb),
                    'rsi': float(rsi),
                    'distance_from_bb': float(distance_from_bb)
                }
            }
        
        # SHORT Signal: Overbought conditions
        elif current_price > upper_bb and rsi > 70:
            distance_from_bb = (current_price - upper_bb) / upper_bb
            confidence = min(0.8, distance_from_bb * 100 + (rsi - 70) / 30 * 0.5)
            self.last_signal_time = datetime.now()
            
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'strategy': self.name,
                'signal': 'SHORT',  # Standardized key
                'confidence': confidence * 100,
                'price': current_price,
                'reason': f'Overbought: Price above BB ({distance_from_bb:.2%}), RSI {rsi:.1f}',
                'metadata': {
                    'sma': float(sma),
                    'upper_bb': float(upper_bb),
                    'lower_bb': float(lower_bb),
                    'rsi': float(rsi),
                    'distance_from_bb': float(distance_from_bb)
                }
            }
        
        return None