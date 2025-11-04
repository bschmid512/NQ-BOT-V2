"""
Technical Indicators for NQ Trading Bot
"""
import pandas as pd
import numpy as np
import pandas_ta as ta


class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period=20, std=2.0):
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period=14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period=14) -> pd.Series:
        """Calculate Average Directional Index"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = TechnicalIndicators.calculate_atr(df, period)
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_standard_error_bands(df: pd.DataFrame, period=20, std=2.0):
        """Calculate Standard Error Bands"""
        middle = df['close'].rolling(window=period).mean()
        se = df['close'].rolling(window=period).std() / np.sqrt(period)
        upper = middle + (se * std)
        lower = middle - (se * std)
        return upper, middle, lower
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window=20, num_levels=3):
        """
        Calculate support and resistance levels using swing highs/lows
        
        Returns: dict with support and resistance levels
        """
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Find swing highs and lows
        swing_highs = df['high'][(df['high'] == highs) & (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])]
        swing_lows = df['low'][(df['low'] == lows) & (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])]
        
        # Get most recent levels
        resistance = sorted(swing_highs.tail(num_levels * 2).unique(), reverse=True)[:num_levels]
        support = sorted(swing_lows.tail(num_levels * 2).unique())[:num_levels]
        
        return {
            'resistance': resistance,
            'support': support
        }
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame, pivot_type='standard'):
        """
        Calculate pivot points
        
        Args:
            pivot_type: 'standard' or 'camarilla'
        """
        # Use previous day's data
        prev_high = df['high'].iloc[-1]
        prev_low = df['low'].iloc[-1]
        prev_close = df['close'].iloc[-1]
        
        if pivot_type == 'standard':
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
        else:  # camarilla
            range_hl = prev_high - prev_low
            r4 = prev_close + range_hl * 1.1 / 2
            r3 = prev_close + range_hl * 1.1 / 4
            r2 = prev_close + range_hl * 1.1 / 6
            r1 = prev_close + range_hl * 1.1 / 12
            s1 = prev_close - range_hl * 1.1 / 12
            s2 = prev_close - range_hl * 1.1 / 6
            s3 = prev_close - range_hl * 1.1 / 4
            s4 = prev_close - range_hl * 1.1 / 2
            pivot = prev_close
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # VWAP
        df['vwap'] = TechnicalIndicators.calculate_vwap(df)
        
        # EMAs
        df['ema_9'] = TechnicalIndicators.calculate_ema(df, 9)
        df['ema_21'] = TechnicalIndicators.calculate_ema(df, 21)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df, 50)
        
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df, 14)
        
        # MACD
        df['macd'], df['signal'], df['macd_hist'] = TechnicalIndicators.calculate_macd(df)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.calculate_bollinger_bands(df)
        
        # ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df)
        
        # ADX
        df['adx'] = TechnicalIndicators.calculate_adx(df)
        
        # Standard Error Bands
        df['se_upper'], df['se_middle'], df['se_lower'] = TechnicalIndicators.calculate_standard_error_bands(df)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['close'].pct_change(periods=10)
        
        return df


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_points=3) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG) in price action
    
    Returns DataFrame with FVG information
    """
    fvgs = []
    
    for i in range(2, len(df)):
        c1_high = df['high'].iloc[i-2]
        c1_low = df['low'].iloc[i-2]
        c3_high = df['high'].iloc[i]
        c3_low = df['low'].iloc[i]
        
        # Bullish FVG (gap up)
        if c3_low > c1_high:
            gap_size = c3_low - c1_high
            if gap_size >= min_gap_points * 0.25:  # 0.25 points = 1 tick for NQ
                fvgs.append({
                    'timestamp': df.index[i],
                    'type': 'bullish',
                    'upper': c3_low,
                    'lower': c1_high,
                    'size': gap_size,
                    'filled': False
                })
        
        # Bearish FVG (gap down)
        elif c3_high < c1_low:
            gap_size = c1_low - c3_high
            if gap_size >= min_gap_points * 0.25:
                fvgs.append({
                    'timestamp': df.index[i],
                    'type': 'bearish',
                    'upper': c1_low,
                    'lower': c3_high,
                    'size': gap_size,
                    'filled': False
                })
    
    return pd.DataFrame(fvgs)
