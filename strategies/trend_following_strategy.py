"""
Trend Following Strategy
Uses EMA crossovers to catch sustained directional moves
"""

from datetime import datetime
import pandas as pd


class TrendFollowingStrategy:
    """
    Trend Following Strategy using EMA crossovers
    
    Signals:
    - LONG: When EMA 9 crosses above EMA 21, with both above EMA 50 (uptrend)
    - SHORT: When EMA 9 crosses below EMA 21, with both below EMA 50 (downtrend)
    """
    
    def __init__(self, fast_ema=9, medium_ema=21, slow_ema=50, cooldown_minutes=5):
        """
        Initialize Trend Following Strategy
        
        Args:
            fast_ema: Period for fast EMA (default 9)
            medium_ema: Period for medium EMA (default 21)
            slow_ema: Period for slow EMA (default 50)
            cooldown_minutes: Minutes to wait between signals (default 5)
        """
        self.name = "TREND_FOLLOWING"
        self.fast_ema = fast_ema
        self.medium_ema = medium_ema
        self.slow_ema = slow_ema
        self.cooldown_minutes = cooldown_minutes
        self.last_signal_time = None
        
    def generate_signal(self, df, current_price):
        """
        Generate trading signal based on EMA crossovers
        
        Args:
            df: pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            current_price: Current market price
            
        Returns:
            dict with signal info or None
            Example: {
                'timestamp': '12:30:45',
                'signal_type': 'TREND_FOLLOWING',
                'direction': 'LONG',
                'price': 25650.50,
                'confidence': 35,
                'reason': 'EMA 9 crossed above EMA 21 in uptrend'
            }
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
                'signal_type': self.name,
                'direction': 'LONG',
                'price': current_price,
                'confidence': 35,
                'reason': f'EMA {self.fast_ema} crossed above EMA {self.medium_ema} in uptrend'
            }
        
        # SHORT Signal: Fast crosses below Medium in downtrend
        if ema_slow > ema_medium > ema_fast and ema_fast_prev >= ema_medium_prev:
            self.last_signal_time = datetime.now()
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'signal_type': self.name,
                'direction': 'SHORT',
                'price': current_price,
                'confidence': 35,
                'reason': f'EMA {self.fast_ema} crossed below EMA {self.medium_ema} in downtrend'
            }
        
        return None
    
    def reset(self):
        """Reset strategy state (useful for backtesting)"""
        self.last_signal_time = None


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Trend Following Strategy...")
    print("=" * 60)
    
    # Create sample data with uptrend
    dates = pd.date_range(start='2025-01-01 09:30:00', periods=100, freq='1min')
    prices = 25500 + np.cumsum(np.random.randn(100) * 2 + 0.5)  # Uptrending
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.uniform(0, 1, 100),
        'high': prices + np.random.uniform(0, 3, 100),
        'low': prices - np.random.uniform(0, 3, 100),
        'close': prices,
        'volume': np.random.randint(5000, 15000, 100)
    })
    
    # Initialize strategy
    strategy = TrendFollowingStrategy()
    
    # Test signal generation
    signals = []
    for i in range(50, len(df)):
        signal = strategy.generate_signal(df.iloc[:i+1], df['close'].iloc[i])
        if signal:
            signals.append(signal)
            print(f"\n✓ Signal Generated:")
            print(f"  Time: {signal['timestamp']}")
            print(f"  Direction: {signal['direction']}")
            print(f"  Price: ${signal['price']:.2f}")
            print(f"  Confidence: {signal['confidence']}%")
            print(f"  Reason: {signal['reason']}")
    
    print(f"\n{'=' * 60}")
    print(f"Total Signals Generated: {len(signals)}")
    print(f"Strategy Name: {strategy.name}")
    print(f"EMA Settings: {strategy.fast_ema}/{strategy.medium_ema}/{strategy.slow_ema}")
    print(f"Cooldown: {strategy.cooldown_minutes} minutes")
