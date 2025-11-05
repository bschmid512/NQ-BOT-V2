"""
Breakout Strategy Instance for NQ Trading Bot
Wrapper to integrate with existing strategy engine
"""
from datetime import datetime
import pandas as pd
from config import STRATEGIES

# Import the actual strategy class
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from breakout_strategy import BreakoutStrategy
except ImportError:
    # Fallback if not in same directory
    from strategies.breakout_strategy import BreakoutStrategy


class BreakoutStrategyWrapper:
    """
    Wrapper to make BreakoutStrategy compatible with existing bot structure
    """
    
    def __init__(self):
        # Load config or use defaults
        self.config = STRATEGIES.get('breakout', {
            'enabled': True,
            'weight': 40,
            'bb_period': 20,
            'bb_std_dev': 2,
            'volume_multiplier': 1.5,
            'cooldown_minutes': 10
        })
        
        # Create the actual strategy instance
        self.strategy = BreakoutStrategy(
            bb_period=self.config.get('bb_period', 20),
            bb_std_dev=self.config.get('bb_std_dev', 2),
            volume_multiplier=self.config.get('volume_multiplier', 1.5),
            cooldown_minutes=self.config.get('cooldown_minutes', 10)
        )
    
    def generate_signal(self, df: pd.DataFrame, context: dict = None) -> dict:
        """
        Generate signal using the breakout strategy
        Adapted to match your bot's expected format
        
        Args:
            df: DataFrame with OHLCV data
            context: Market context (optional)
            
        Returns:
            Signal dict matching your bot's format or None
        """
        if df.empty or len(df) < self.config.get('bb_period', 20):
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Call the actual strategy
        signal = self.strategy.generate_signal(df, current_price)
        
        if signal:
            # Convert to your bot's expected format
            return {
                'timestamp': signal['timestamp'],
                'strategy': 'breakout',
                'signal': signal['direction'],  # 'LONG' or 'SHORT'
                'price': signal['price'],
                'confidence': signal['confidence'] / 100.0,  # Convert to 0.0-1.0
                'weight': self.config.get('weight', 40),
                'reason': signal['reason'],
                'target': None,  # Can add target calculation if needed
                'stop': None     # Can add stop calculation if needed
            }
        
        return None


# Create singleton instance
breakout_strategy = BreakoutStrategyWrapper()

# Ensure config exists
if 'breakout' not in STRATEGIES:
    STRATEGIES['breakout'] = {
        'enabled': True,
        'weight': 40,
        'bb_period': 20,
        'bb_std_dev': 2,
        'volume_multiplier': 1.5,
        'cooldown_minutes': 10
    }
