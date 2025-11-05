"""
Breakout Strategy
Uses Bollinger Bands and volume confirmation to catch strong moves
"""

from datetime import datetime
import pandas as pd


class BreakoutStrategy:
    """
    Breakout Strategy using Bollinger Bands and Volume
    
    Signals:
    - LONG: Price breaks above upper Bollinger Band with high volume
    - SHORT: Price breaks below lower Bollinger Band with high volume
    """
    
    def __init__(self, bb_period=20, bb_std_dev=2, volume_multiplier=1.5, cooldown_minutes=10):
        """
        Initialize Breakout Strategy
        
        Args:
            bb_period: Period for Bollinger Bands calculation (default 20)
            bb_std_dev: Number of standard deviations for bands (default 2)
            volume_multiplier: Volume must be this many times average (default 1.5)
            cooldown_minutes: Minutes to wait between signals (default 10)
        """
        self.name = "BREAKOUT"
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.volume_multiplier = volume_multiplier
        self.cooldown_minutes = cooldown_minutes
        self.last_signal_time = None
        
    def generate_signal(self, df, current_price):
        """
        Generate trading signal based on Bollinger Band breakouts
        
        Args:
            df: pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            current_price: Current market price
            
        Returns:
            dict with signal info or None
            Example: {
                'timestamp': '12:30:45',
                'signal_type': 'BREAKOUT',
                'direction': 'LONG',
                'price': 25650.50,
                'confidence': 40,
                'reason': 'High volume breakout above BB upper (25645.00)'
            }
        """
        # Need enough data for Bollinger Bands
        if len(df) < self.bb_period:
            return None
        
        # Check cooldown period
        if self.last_signal_time:
            time_since_signal = (datetime.now() - self.last_signal_time).total_seconds() / 60
            if time_since_signal < self.cooldown_minutes:
                return None
        
        # Calculate Bollinger Bands
        df = df.copy()
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std_dev)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std_dev)
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=self.bb_period).mean()
        
        # Get current values
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['avg_volume'].iloc[-1]
        
        # Check for high volume
        is_high_volume = current_volume > (avg_volume * self.volume_multiplier)
        
        # LONG Signal: Breakout above upper band with high volume
        if current_price > bb_upper and is_high_volume:
            self.last_signal_time = datetime.now()
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'signal_type': self.name,
                'direction': 'LONG',
                'price': current_price,
                'confidence': 40,
                'reason': f'High volume breakout above BB upper ({bb_upper:.2f})'
            }
        
        # SHORT Signal: Breakdown below lower band with high volume
        if current_price < bb_lower and is_high_volume:
            self.last_signal_time = datetime.now()
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'signal_type': self.name,
                'direction': 'SHORT',
                'price': current_price,
                'confidence': 40,
                'reason': f'High volume breakdown below BB lower ({bb_lower:.2f})'
            }
        
        return None
    
    def get_current_levels(self, df):
        """
        Get current Bollinger Band levels (useful for visualization)
        
        Args:
            df: pandas DataFrame with price data
            
        Returns:
            dict with current BB levels or None
        """
        if len(df) < self.bb_period:
            return None
        
        df = df.copy()
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std_dev)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std_dev)
        
        return {
            'upper': df['bb_upper'].iloc[-1],
            'middle': df['sma'].iloc[-1],
            'lower': df['bb_lower'].iloc[-1]
        }
    
    def reset(self):
        """Reset strategy state (useful for backtesting)"""
        self.last_signal_time = None


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Breakout Strategy...")
    print("=" * 60)
    
    # Create sample data with breakout
    dates = pd.date_range(start='2025-01-01 09:30:00', periods=100, freq='1min')
    
    # Create ranging market that breaks out
    prices = np.concatenate([
        np.random.randn(50) * 5 + 25500,  # Range-bound
        25500 + np.cumsum(np.random.randn(50) * 3 + 1)  # Breakout
    ])
    
    # Create volume spike during breakout
    volumes = np.concatenate([
        np.random.randint(5000, 8000, 50),  # Normal volume
        np.random.randint(12000, 18000, 50)  # High volume
    ])
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.uniform(0, 1, 100),
        'high': prices + np.random.uniform(0, 3, 100),
        'low': prices - np.random.uniform(0, 3, 100),
        'close': prices,
        'volume': volumes
    })
    
    # Initialize strategy
    strategy = BreakoutStrategy()
    
    # Test signal generation
    signals = []
    for i in range(20, len(df)):
        signal = strategy.generate_signal(df.iloc[:i+1], df['close'].iloc[i])
        
        # Also get current BB levels for display
        levels = strategy.get_current_levels(df.iloc[:i+1])
        
        if signal:
            signals.append(signal)
            print(f"\n✓ Signal Generated:")
            print(f"  Time: {signal['timestamp']}")
            print(f"  Direction: {signal['direction']}")
            print(f"  Price: ${signal['price']:.2f}")
            print(f"  Confidence: {signal['confidence']}%")
            print(f"  Reason: {signal['reason']}")
            if levels:
                print(f"  BB Upper: ${levels['upper']:.2f}")
                print(f"  BB Middle: ${levels['middle']:.2f}")
                print(f"  BB Lower: ${levels['lower']:.2f}")
    
    print(f"\n{'=' * 60}")
    print(f"Total Signals Generated: {len(signals)}")
    print(f"Strategy Name: {strategy.name}")
    print(f"BB Settings: {strategy.bb_period} period, {strategy.bb_std_dev} std dev")
    print(f"Volume Multiplier: {strategy.volume_multiplier}x")
    print(f"Cooldown: {strategy.cooldown_minutes} minutes")
