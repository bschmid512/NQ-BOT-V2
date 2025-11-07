"""
Pullback Entry Strategy
Enters established trends on minor pullbacks
Catches continuation moves after healthy retracements
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
from config import STRATEGIES


class PullbackEntryStrategy:
    """
    Enters trending markets on pullbacks to moving averages
    
    Key Features:
    - Identifies established trend (price above/below 50 EMA)
    - Waits for pullback to 8 EMA
    - Enters when price bounces off fast EMA
    - Uses ADX to confirm trend strength
    
    Perfect for catching continuation after brief pauses in trends
    """
    
    def __init__(self):
        self.config = STRATEGIES['pullback']
        self.name = 'pullback'
        self.weight = self.config['weight']
        
        self.trend_ema = self.config['trend_ema']
        self.fast_ema = self.config['fast_ema']
        self.pullback_pct = self.config['pullback_pct']
        self.atr_multiplier = self.config['atr_multiplier']
        self.adx_min = self.config['adx_min']
        
        # Track signals
        self.last_signal_time = None
        self.signal_cooldown_minutes = 8
        
        print(f"✅ Pullback Entry Strategy initialized")
    
    def generate_signal(self, df: pd.DataFrame, current_price: float,
                       context: Dict = None) -> Optional[Dict]:
        """
        Generate pullback entry signal
        
        Logic:
        1. Confirm trend with 50 EMA
        2. Detect pullback to 8 EMA
        3. Enter when price bounces off fast EMA
        4. Require ADX > minimum for trend strength
        """
        if not self.config['enabled']:
            return None
        
        if len(df) < self.trend_ema + 10:
            return None
        
        # Check cooldown
        if self._in_cooldown():
            return None
        
        # Calculate EMAs
        df = self._add_indicators(df)
        
        ema_fast = df['ema_fast'].iloc[-1]
        ema_trend = df['ema_trend'].iloc[-1]
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else self.adx_min
        
        # Need minimum trend strength
        if adx < self.adx_min:
            return None
        
        # Determine trend direction
        if current_price > ema_trend and ema_fast > ema_trend:
            trend_direction = 'LONG'
        elif current_price < ema_trend and ema_fast < ema_trend:
            trend_direction = 'SHORT'
        else:
            return None  # No clear trend
        
        # Check for pullback to fast EMA
        pullback_detected = False
        
        if trend_direction == 'LONG':
            # Price should be near or just above fast EMA
            distance_pct = (current_price - ema_fast) / current_price
            
            # Recent bars should show pullback (price went below fast EMA recently)
            recent_lows = df['low'].tail(3)
            touched_ema = any(low <= ema_fast * 1.002 for low in recent_lows)
            
            if touched_ema and 0 <= distance_pct < self.pullback_pct:
                pullback_detected = True
        
        else:  # SHORT
            distance_pct = (ema_fast - current_price) / current_price
            
            recent_highs = df['high'].tail(3)
            touched_ema = any(high >= ema_fast * 0.998 for high in recent_highs)
            
            if touched_ema and 0 <= distance_pct < self.pullback_pct:
                pullback_detected = True
        
        if not pullback_detected:
            return None
        
        # Calculate ATR for stops and targets
        atr = self._calculate_atr(df)
        
        if trend_direction == 'LONG':
            stop_loss = ema_fast - (atr * self.atr_multiplier)
            target = current_price + (atr * self.atr_multiplier * 2.0)
        else:
            stop_loss = ema_fast + (atr * self.atr_multiplier)
            target = current_price - (atr * self.atr_multiplier * 2.0)
        
        # Calculate confidence
        # Higher ADX = higher confidence
        confidence = min(adx / 40, 1.0)  # Scale ADX to 0-1 (assuming ADX of 40 is very strong)
        
        # Boost if context confirms
        if context:
            regime = context.get('market_regime', 'unknown')
            if regime in ['strong_trend', 'trending']:
                confidence = min(confidence * 1.15, 1.0)
            
            # Check vision confirmation
            if context.get('vision_available'):
                vision_sentiment = context.get('vision_sentiment', 'neutral')
                if (trend_direction == 'LONG' and vision_sentiment == 'bullish') or \
                   (trend_direction == 'SHORT' and vision_sentiment == 'bearish'):
                    confidence = min(confidence * 1.1, 1.0)
        
        # Record signal
        self.last_signal_time = datetime.now()
        
        signal = {
            'strategy': self.name,
            'direction': trend_direction,
            'price': current_price,
            'stop': stop_loss,
            'target': target,
            'targets': [
                target,
                target + (atr * 3.0 * (1 if trend_direction == 'LONG' else -1)),
                target + (atr * 4.0 * (1 if trend_direction == 'LONG' else -1))
            ],
            'confidence': confidence,
            'weight': self.weight,
            'reason': f"Pullback to EMA{self.fast_ema} in {trend_direction} trend (ADX: {adx:.1f})",
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🔄 PULLBACK Signal: {trend_direction} @ {current_price:.2f}")
        print(f"   EMA{self.fast_ema}: {ema_fast:.2f} | EMA{self.trend_ema}: {ema_trend:.2f}")
        print(f"   ADX: {adx:.1f} | Confidence: {confidence:.2f}")
        
        return signal
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required indicators"""
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()
        
        # ADX
        if 'adx' not in df.columns:
            df['adx'] = self._calculate_adx(df)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        if len(df) < period + 1:
            return pd.Series([self.adx_min] * len(df), index=df.index)
        
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
        
        # Smooth with EMA
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 20.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.tail(period).mean()
        
        return atr
    
    def _in_cooldown(self) -> bool:
        """Check if in cooldown period"""
        if self.last_signal_time is None:
            return False
        
        minutes_since_last = (datetime.now() - self.last_signal_time).total_seconds() / 60
        return minutes_since_last < self.signal_cooldown_minutes
    
    def get_current_state(self, df: pd.DataFrame) -> Dict:
        """Get current strategy state for debugging"""
        if len(df) < self.trend_ema:
            return {'status': 'insufficient_data'}
        
        df = self._add_indicators(df)
        current_price = df['close'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_trend = df['ema_trend'].iloc[-1]
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0
        
        trend = 'bullish' if current_price > ema_trend else 'bearish'
        
        return {
            'status': 'active',
            'trend': trend,
            'current_price': current_price,
            'ema_fast': ema_fast,
            'ema_trend': ema_trend,
            'adx': adx,
            'trend_strong': adx >= self.adx_min,
            'in_cooldown': self._in_cooldown()
        }
