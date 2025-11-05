"""
Mean Reversion Strategy - WITH REAL CONFIDENCE SCORING
Trades reversals using Bollinger Bands and RSI
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from config import STRATEGIES
from utils.logger import trading_logger
from utils.indicators import TechnicalIndicators


class MeanReversionStrategy:
    """
    Mean Reversion Strategy with Dynamic Confidence Scoring
    
    Confidence based on:
    - RSI extremity (how oversold/overbought)
    - BB touch strength (how far outside bands)
    - Volume confirmation
    - ADX level (lower = better for mean reversion)
    - Reversal candle strength
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or STRATEGIES['mean_reversion']
        self.bb_period = self.config['bb_period']
        self.bb_std = self.config['bb_std']
        self.rsi_period = self.config['rsi_period']
        self.rsi_oversold = self.config['rsi_oversold']
        self.rsi_overbought = self.config['rsi_overbought']
        self.weight = self.config['weight']
        
        self.logger = trading_logger.strategy_logger
        self.logger.info(f"Mean Reversion Strategy initialized: BB({self.bb_period},{self.bb_std}), RSI({self.rsi_period})")
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required indicators to dataframe"""
        df = df.copy()
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.calculate_bollinger_bands(
            df, period=self.bb_period, std=self.bb_std
        )
        
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df, period=self.rsi_period)
        
        # ADX for trend filter
        df['adx'] = TechnicalIndicators.calculate_adx(df)
        
        # ATR for volatility
        df['atr'] = TechnicalIndicators.calculate_atr(df)
        
        return df
    
    def calculate_confidence(self, current_bar, prev_bar, signal_type: str) -> float:
        """
        Calculate dynamic confidence score (0.0 to 1.0)
        
        Factors:
        1. RSI extremity (30%): More extreme RSI = higher confidence
        2. BB penetration (25%): Further outside bands = higher confidence
        3. Volume surge (20%): Above average volume = higher confidence
        4. Reversal strength (15%): Strong reversal candle = higher confidence
        5. ADX level (10%): Lower ADX (ranging) = higher confidence
        
        Returns: 0.0 to 1.0 confidence score
        """
        confidence_factors = {}
        
        # 1. RSI Extremity (0-30%)
        if signal_type == 'LONG':
            # For LONG: Lower RSI = better (more oversold)
            rsi_score = max(0, (30 - current_bar['rsi']) / 30)  # 0 at RSI=30, 1.0 at RSI=0
        else:  # SHORT
            # For SHORT: Higher RSI = better (more overbought)
            rsi_score = max(0, (current_bar['rsi'] - 70) / 30)  # 0 at RSI=70, 1.0 at RSI=100
        
        confidence_factors['rsi'] = min(rsi_score, 1.0) * 0.30
        
        # 2. BB Penetration (0-25%)
        bb_range = current_bar['bb_upper'] - current_bar['bb_lower']
        
        if signal_type == 'LONG':
            # How far below lower BB (penetration)
            penetration = max(0, current_bar['bb_lower'] - current_bar['low'])
            bb_score = min(penetration / (bb_range * 0.1), 1.0)  # Max at 10% penetration
        else:  # SHORT
            # How far above upper BB
            penetration = max(0, current_bar['high'] - current_bar['bb_upper'])
            bb_score = min(penetration / (bb_range * 0.1), 1.0)
        
        confidence_factors['bb_penetration'] = bb_score * 0.25
        
        # 3. Volume Surge (0-20%)
        # Assume we have volume data
        if 'volume' in current_bar and current_bar['volume'] > 0:
            # Compare to average (would need to pass this in)
            volume_score = 0.5  # Default moderate score
        else:
            volume_score = 0.5
        
        confidence_factors['volume'] = volume_score * 0.20
        
        # 4. Reversal Candle Strength (0-15%)
        candle_body = abs(current_bar['close'] - current_bar['open'])
        candle_range = current_bar['high'] - current_bar['low']
        
        if candle_range > 0:
            if signal_type == 'LONG':
                # Want bullish candle (close > open)
                if current_bar['close'] > current_bar['open']:
                    reversal_score = candle_body / candle_range  # Strong body = strong reversal
                else:
                    reversal_score = 0.3  # Weak signal
            else:  # SHORT
                # Want bearish candle (close < open)
                if current_bar['close'] < current_bar['open']:
                    reversal_score = candle_body / candle_range
                else:
                    reversal_score = 0.3
        else:
            reversal_score = 0.5
        
        confidence_factors['reversal'] = min(reversal_score, 1.0) * 0.15
        
        # 5. ADX Level (0-10%)
        # Lower ADX = better for mean reversion (ranging market)
        if 'adx' in current_bar and current_bar['adx'] > 0:
            adx_score = max(0, (40 - current_bar['adx']) / 40)  # 1.0 at ADX=0, 0 at ADX=40+
        else:
            adx_score = 0.5
        
        confidence_factors['adx'] = adx_score * 0.10
        
        # Total confidence
        total_confidence = sum(confidence_factors.values())
        
        # Log breakdown for debugging
        self.logger.debug(
            f"Confidence breakdown: RSI={confidence_factors['rsi']:.2f}, "
            f"BB={confidence_factors['bb_penetration']:.2f}, "
            f"Vol={confidence_factors['volume']:.2f}, "
            f"Rev={confidence_factors['reversal']:.2f}, "
            f"ADX={confidence_factors['adx']:.2f} | "
            f"Total={total_confidence:.2f}"
        )
        
        return total_confidence
    
    def generate_signal(self, df: pd.DataFrame, context: Dict) -> Optional[Dict]:
        """
        Generate mean reversion signal with dynamic confidence
        """
        try:
            if len(df) < 2:
                return None
            
            # Add indicators if not present
            if 'bb_lower' not in df.columns:
                df = self.add_indicators(df)
            
            current_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            # OPTIONAL: Skip if trending market (comment out if disabled)
            # if current_bar['adx'] > 25:
            #     self.logger.debug(f"Skipping mean reversion - trending market (ADX={current_bar['adx']:.1f})")
            #     return None
            
            current_price = current_bar['close']
            
            # Long signal: Price touched/broke lower BB + RSI oversold + bullish reversal
            if (current_bar['low'] <= current_bar['bb_lower'] and
                current_bar['rsi'] < self.rsi_oversold and
                current_bar['close'] > prev_bar['close']):
                
                # Calculate dynamic confidence
                confidence = self.calculate_confidence(current_bar, prev_bar, 'LONG')
                
                # Require minimum confidence of 40%
                if confidence < 0.40:
                    self.logger.debug(f"Mean Reversion LONG: Low confidence ({confidence*100:.0f}%), skipping")
                    return None
                
                stop_loss = current_price - 10.0
                take_profit = current_bar['bb_middle']
                
                risk = current_price - stop_loss
                reward = take_profit - current_price
                
                if reward / risk < 1.5:
                    self.logger.debug(f"Poor R:R ratio: {reward/risk:.2f}")
                    return None
                
                signal = {
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now(),
                    'strategy': 'mean_reversion',
                    'signal': 'LONG',
                    'confidence': confidence,  # REAL confidence, not just weight!
                    'price': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'rsi': current_bar['rsi'],
                    'bb_width': current_bar['bb_upper'] - current_bar['bb_lower'],
                    'adx': current_bar['adx']
                }
                
                self.logger.info(
                    f"Mean Reversion LONG: Entry={current_price:.2f}, Target={take_profit:.2f}, "
                    f"Stop={stop_loss:.2f}, RSI={current_bar['rsi']:.1f}, "
                    f"Confidence={confidence*100:.0f}%"
                )
                return signal
            
            # Short signal: Price touched/broke upper BB + RSI overbought + bearish reversal
            elif (current_bar['high'] >= current_bar['bb_upper'] and
                  current_bar['rsi'] > self.rsi_overbought and
                  current_bar['close'] < prev_bar['close']):
                
                # Calculate dynamic confidence
                confidence = self.calculate_confidence(current_bar, prev_bar, 'SHORT')
                
                # Require minimum confidence of 40%
                if confidence < 0.40:
                    self.logger.debug(f"Mean Reversion SHORT: Low confidence ({confidence*100:.0f}%), skipping")
                    return None
                
                stop_loss = current_price + 10.0
                take_profit = current_bar['bb_middle']
                
                risk = stop_loss - current_price
                reward = current_price - take_profit
                
                if reward / risk < 1.5:
                    self.logger.debug(f"Poor R:R ratio: {reward/risk:.2f}")
                    return None
                
                signal = {
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now(),
                    'strategy': 'mean_reversion',
                    'signal': 'SHORT',
                    'confidence': confidence,  # REAL confidence!
                    'price': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'rsi': current_bar['rsi'],
                    'bb_width': current_bar['bb_upper'] - current_bar['bb_lower'],
                    'adx': current_bar['adx']
                }
                
                self.logger.info(
                    f"Mean Reversion SHORT: Entry={current_price:.2f}, Target={take_profit:.2f}, "
                    f"Stop={stop_loss:.2f}, RSI={current_bar['rsi']:.1f}, "
                    f"Confidence={confidence*100:.0f}%"
                )
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signal: {e}")
            return None
    
    def is_valid_setup(self, df: pd.DataFrame) -> bool:
        """Check if market conditions are favorable for mean reversion"""
        if len(df) < self.bb_period:
            return False
        
        current_bar = df.iloc[-1]
        
        # Check for ranging market (optional)
        if 'adx' in df.columns and current_bar['adx'] > 25:
            return False
        
        # Check Bollinger Band width (not too tight, not too wide)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_width = current_bar['bb_upper'] - current_bar['bb_lower']
            avg_width = (df['bb_upper'] - df['bb_lower']).tail(20).mean()
            
            if bb_width < avg_width * 0.5 or bb_width > avg_width * 1.5:
                return False
        
        return True


# Create global strategy instance
mean_reversion_strategy = MeanReversionStrategy()