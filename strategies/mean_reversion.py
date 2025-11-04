"""
Mean Reversion Strategy
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
    Mean Reversion Strategy using Bollinger Bands + RSI
    
    Win Rate: ~62%
    Works best in ranging markets (ADX < 25)
    Entry: Price touches BB band + RSI confirmation
    Target: Middle BB band (20 SMA)
    Stop: 10 points from entry
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
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate mean reversion signal
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Signal dict or None
        """
        try:
            if len(df) < 2:
                return None
            
            # Add indicators if not present
            if 'bb_lower' not in df.columns:
                df = self.add_indicators(df)
            
            current_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            # Skip if trending market (ADX > 25)
            if current_bar['adx'] > 25:
                self.logger.debug(f"Skipping mean reversion - trending market (ADX={current_bar['adx']:.1f})")
                return None
            
            current_price = current_bar['close']
            
            # Long signal: Price touched/broke lower BB + RSI oversold + bullish reversal
            if (current_bar['low'] <= current_bar['bb_lower'] and
                current_bar['rsi'] < self.rsi_oversold and
                current_bar['close'] > prev_bar['close']):  # Bullish reversal candle
                
                stop_loss = current_price - 10.0  # 10 points stop
                take_profit = current_bar['bb_middle']  # Target middle band
                
                # Calculate risk/reward
                risk = current_price - stop_loss
                reward = take_profit - current_price
                
                if reward / risk < 1.5:  # Minimum 1.5:1 R:R
                    self.logger.debug(f"Poor R:R ratio: {reward/risk:.2f}")
                    return None
                
                signal = {
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now(),
                    'strategy': 'mean_reversion',
                    'signal': 'LONG',
                    'confidence': self.weight,
                    'price': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'rsi': current_bar['rsi'],
                    'bb_width': current_bar['bb_upper'] - current_bar['bb_lower'],
                    'adx': current_bar['adx']
                }
                
                self.logger.info(
                    f"Mean Reversion LONG: Entry={current_price:.2f}, Target={take_profit:.2f}, "
                    f"Stop={stop_loss:.2f}, RSI={current_bar['rsi']:.1f}"
                )
                return signal
            
            # Short signal: Price touched/broke upper BB + RSI overbought + bearish reversal
            elif (current_bar['high'] >= current_bar['bb_upper'] and
                  current_bar['rsi'] > self.rsi_overbought and
                  current_bar['close'] < prev_bar['close']):  # Bearish reversal candle
                
                stop_loss = current_price + 10.0  # 10 points stop
                take_profit = current_bar['bb_middle']  # Target middle band
                
                # Calculate risk/reward
                risk = stop_loss - current_price
                reward = current_price - take_profit
                
                if reward / risk < 1.5:  # Minimum 1.5:1 R:R
                    self.logger.debug(f"Poor R:R ratio: {reward/risk:.2f}")
                    return None
                
                signal = {
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now(),
                    'strategy': 'mean_reversion',
                    'signal': 'SHORT',
                    'confidence': self.weight,
                    'price': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'rsi': current_bar['rsi'],
                    'bb_width': current_bar['bb_upper'] - current_bar['bb_lower'],
                    'adx': current_bar['adx']
                }
                
                self.logger.info(
                    f"Mean Reversion SHORT: Entry={current_price:.2f}, Target={take_profit:.2f}, "
                    f"Stop={stop_loss:.2f}, RSI={current_bar['rsi']:.1f}"
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
        
        # Check for ranging market
        if 'adx' in df.columns and current_bar['adx'] > 25:
            return False
        
        # Check Bollinger Band width (not too tight, not too wide)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_width = current_bar['bb_upper'] - current_bar['bb_lower']
            avg_width = (df['bb_upper'] - df['bb_lower']).tail(20).mean()
            
            # Width should be within 50-150% of average
            if bb_width < avg_width * 0.5 or bb_width > avg_width * 1.5:
                return False
        
        return True


# Create global strategy instance
mean_reversion_strategy = MeanReversionStrategy()
