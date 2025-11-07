# indicators/divergence.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill().fillna(50.0)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _pivots(series: pd.Series, left: int = 2, right: int = 2) -> Tuple[List[int], List[int]]:
    s = series.values
    highs, lows = [], []
    n = len(s)
    for i in range(left, n - right):
        window = s[i-left:i+right+1]
        c = s[i]
        if c == window.max() and (window > c).sum() == 0:
            highs.append(i)
        if c == window.min() and (window < c).sum() == 0:
            lows.append(i)
    return highs, lows

@dataclass
class DivSignal:
    bullish: bool = False
    bearish: bool = False
    age_bars: Optional[int] = None

def _latest_divergence(price: pd.Series,
                       osc: pd.Series,
                       highs: List[int],
                       lows: List[int],
                       lookback: int = 80) -> Tuple[DivSignal, DivSignal]:
    n = len(price)
    highs = [i for i in highs if i >= n - lookback]
    lows  = [i for i in lows  if i >= n - lookback]

    bear = DivSignal(False, False, None)
    bull = DivSignal(False, False, None)

    # bearish: price HH, osc LH
    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if price.iloc[h2] > price.iloc[h1] and osc.iloc[h2] < osc.iloc[h1]:
            bear.bearish = True
            bear.age_bars = (n - 1) - h2

    # bullish: price LL, osc HL
    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if price.iloc[l2] < price.iloc[l1] and osc.iloc[l2] > osc.iloc[l1]:
            bull.bullish = True
            bull.age_bars = (n - 1) - l2

    return bull, bear

def analyze_divergence(df: pd.DataFrame,
                       rsi_period: int = 14,
                       pivot_left: int = 2,
                       pivot_right: int = 2,
                       lookback: int = 80) -> Dict:
    """
    Returns:
      {
        'bullish_rsi': bool, 'bearish_rsi': bool,
        'bullish_macd': bool, 'bearish_macd': bool,
        'score': float, 'details': {...}
      }
    """
    if df is None or df.empty or 'close' not in df.columns:
        return {'bullish_rsi': False, 'bearish_rsi': False,
                'bullish_macd': False, 'bearish_macd': False,
                'score': 0.0, 'details': {}}

    close = df['close'].astype(float)
    rsi_series = rsi(close, rsi_period)
    macd_line, sig, hist = macd(close)

    highs, lows = _pivots(close, pivot_left, pivot_right)
    rsi_bull, rsi_bear = _latest_divergence(close, rsi_series, highs, lows, lookback)
    mac_bull, mac_bear = _latest_divergence(close, macd_line, highs, lows, lookback)

    def weight(age):
        if age is None: return 0.0
        return max(0.0, 1.0 - (age / float(lookback)))  # newer = stronger

    score = 0.0
    if rsi_bull.bullish: score += 1.0 * weight(rsi_bull.age_bars)
    if mac_bull.bullish: score += 1.0 * weight(mac_bull.age_bars)
    if rsi_bear.bearish: score -= 1.0 * weight(rsi_bear.age_bars)
    if mac_bear.bearish: score -= 1.0 * weight(mac_bear.age_bars)

    return {
        'bullish_rsi': bool(rsi_bull.bullish),
        'bearish_rsi': bool(rsi_bear.bearish),
        'bullish_macd': bool(mac_bull.bullish),
        'bearish_macd': bool(mac_bear.bearish),
        'score': float(round(score, 2)),
        'details': {
            'rsi_age': rsi_bull.age_bars if rsi_bull.bullish else rsi_bear.age_bars,
            'macd_age': mac_bull.age_bars if mac_bull.bullish else mac_bear.age_bars,
            'pivots_highs': highs[-6:],
            'pivots_lows': lows[-6:],
        }
    }
