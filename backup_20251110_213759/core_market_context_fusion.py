"""
Market Context Fusion Layer
Combines vision analysis with real-time webhook data into a single context
"""
from __future__ import annotations

from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from config import ATR_PERIOD, CONTEXT_THRESHOLDS
from indicators.divergence import analyze_divergence

# Lightweight EMA without external deps
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return np.nan
    a = float(series.iloc[-periods-1])
    b = float(series.iloc[-1])
    if a == 0:
        return np.nan
    return (b - a) / a

class MarketContextFusion:
    """
    Fuses vision system analysis with webhook price data
    Creates unified market context for strategies and the fusion engine
    """

    def __init__(self):
        self.vision_data: Optional[Dict] = None
        self.price_data: Optional[Dict] = None
        self.last_update: Optional[datetime] = None
        print("✅ Market Context Fusion initialized")

    # ------------------- updaters -------------------

    def update_vision_data(self, vision_analysis: Dict):
        self.vision_data = vision_analysis or {}
        self.last_update = datetime.now()

    def update_price_data(self, df: pd.DataFrame, current_price: float):
        self.price_data = {
            'current_price': float(current_price),
            'df': df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(),
            'bars_available': 0 if df is None else int(len(df))
        }

    # ------------------- core context -------------------

    def get_unified_context(self) -> Dict:
        """
        Returns a comprehensive, robust market context dict.
        Keys consumed by the rest of your system are preserved.
        New key 'diagnostics' explains the label decision.
        """
        th = CONTEXT_THRESHOLDS
        context = {
            'timestamp': datetime.now().isoformat(),
            'current_price': None,
            'market_regime': 'unknown',
            'regime_confidence': 0.0,
            'visual_patterns': [],
            'key_levels': {'support': [], 'resistance': []},
            'momentum': 'neutral',
            'volatility': None,
            'volume_profile': 'normal',
            'trend_direction': 'neutral',
            'trend_strength': 0.0,
            'vision_sentiment': 'neutral',
            'vision_available': False,
            'price_data_available': False,
            'diagnostics': ''
        }

        reasons: List[str] = []

        # ----- PRICE DATA -----
        df = None
        if self.price_data and isinstance(self.price_data.get('df'), pd.DataFrame) and not self.price_data['df'].empty:
            context['price_data_available'] = True
            context['current_price'] = float(self.price_data['current_price'])
            df = self.price_data['df'].sort_index()
            # ensure numeric & clean
            close = pd.to_numeric(df['close'], errors='coerce').ffill().bfill()
            high = pd.to_numeric(df['high'], errors='coerce').ffill().bfill()
            low  = pd.to_numeric(df['low'],  errors='coerce').ffill().bfill()

            bars = len(close)
            reasons.append(f"bars={bars}")
            if bars >= 3:
                hl = (high - low).tail(ATR_PERIOD)
                context['volatility'] = float(hl.mean())

            # Momentum: 1m & 5m pct change
            mom1 = _pct_change(close, 1)
            mom5 = _pct_change(close, 5)
            mom10 = _pct_change(close, 10)

            mom_label = "neutral"
            if not np.isnan(mom5) and mom5 >= th["mom_5m_bull"]:
                mom_label = "bullish"; reasons.append(f"mom5={mom5:.3%}≥{th['mom_5m_bull']:.2%}")
            elif not np.isnan(mom5) and mom5 <= th["mom_5m_bear"]:
                mom_label = "bearish"; reasons.append(f"mom5={mom5:.3%}≤{th['mom_5m_bear']:.2%}")
            elif not np.isnan(mom1) and mom1 >= th["mom_1m_bull"]:
                mom_label = "bullish"; reasons.append(f"mom1={mom1:.3%}≥{th['mom_1m_bull']:.2%}")
            elif not np.isnan(mom1) and mom1 <= th["mom_1m_bear"]:
                mom_label = "bearish"; reasons.append(f"mom1={mom1:.3%}≤{th['mom_1m_bear']:.2%}")
            else:
                reasons.append(f"mom1={mom1 if not np.isnan(mom1) else 'NaN'}, mom5={mom5 if not np.isnan(mom5) else 'NaN'}")
            context['momentum'] = mom_label

            # Trend: EMA20 vs EMA50 + slope; ADX proxy via return std
            ema20 = _ema(close, 20)
            ema50 = _ema(close, 50)
            ema_ok = bars >= 50 and not (np.isnan(ema20.iloc[-1]) or np.isnan(ema50.iloc[-1]))
            ema20_slope = float(ema20.diff().iloc[-1]) if ema_ok else np.nan

            ret = close.pct_change()
            vol14 = ret.rolling(14, min_periods=14).std()
            adx_proxy = float((vol14.iloc[-1] or 0.0) * 100) if len(vol14.dropna()) else 0.0

            trend_label = "neutral"
            trend_strength = 0.0
            if ema_ok:
                if (ema20.iloc[-1] > ema50.iloc[-1]) and (ema20_slope > th["ema_slope_min"]) and (adx_proxy >= th["adx_trend_min"]):
                    trend_label = "uptrend"
                    trend_strength = float((ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1])
                    reasons.append(f"ema20>ema50 slope={ema20_slope:.4f} adx≈{adx_proxy:.1f}")
                elif (ema20.iloc[-1] < ema50.iloc[-1]) and (ema20_slope < -th["ema_slope_min"]) and (adx_proxy >= th["adx_trend_min"]):
                    trend_label = "downtrend"
                    trend_strength = float((ema50.iloc[-1] - ema20.iloc[-1]) / ema50.iloc[-1])
                    reasons.append(f"ema20<ema50 slope={ema20_slope:.4f} adx≈{adx_proxy:.1f}")
                else:
                    reasons.append(f"trend-neutral (ema20={ema20.iloc[-1]:.2f}, ema50={ema50.iloc[-1]:.2f}, slope={ema20_slope:.4f}, adx≈{adx_proxy:.1f})")
            else:
                reasons.append("not-enough-bars-for-EMA")

            context['trend_direction'] = trend_label
            context['trend_strength'] = float(trend_strength)

            # Regime: default by adx proxy
            regime = "ranging" if adx_proxy < th["adx_trend_min"] else "trending"
            conf = 0.40 if regime == "ranging" else min(0.90, 0.60 + abs(trend_strength) * 4)

            # Regime override: big 10-bar move
            if not np.isnan(mom10) and abs(mom10) >= th["mom_10m_trending"]:
                regime = "trending"
                conf = max(conf, 0.85)
                reasons.append(f"mom10 override: {mom10:.2%}")

            context['market_regime'] = regime
            context['regime_confidence'] = float(conf)

        # ----- VISION DATA -----
        if self.vision_data:
            context['vision_available'] = True
            stats = self.vision_data.get('statistics', {})
            context['vision_sentiment'] = stats.get('sentiment', 'neutral')
            if 'patterns' in self.vision_data:
                context['visual_patterns'] = self.vision_data['patterns']
            if 'support_resistance' in self.vision_data:
                context['key_levels'] = self.vision_data['support_resistance']

        # ----- DIVERGENCE (safe) -----
        try:
            _div = analyze_divergence(df) if isinstance(df, pd.DataFrame) else {}
        except Exception:
            _div = {}
        context['divergence'] = _div or {'score': 0.0}
        context['divergence_score'] = float((context['divergence'] or {}).get('score', 0.0))

        # ----- DIAGNOSTICS -----
        price = context['current_price']
        mom5 = 'n/a' if not context['price_data_available'] else f"{_pct_change(self.price_data['df']['close'].sort_index().ffill().bfill(), 5):.3%}"
        context['diagnostics'] = f"p={price} | regime={context['market_regime']}({context['regime_confidence']:.2f}) | mom={context['momentum']} (5m={mom5}) | trend={context['trend_direction']} | {'vision✓' if context['vision_available'] else 'vision✗'} | " + "; ".join(reasons)

        return context

    # ------------------- helpers used elsewhere -------------------

    def check_vision_confirmation(self, signal_direction: str) -> tuple[bool, float]:
        if not self.vision_data:
            return False, 0.0
        sentiment = self.vision_data.get('statistics', {}).get('sentiment', 'neutral')
        if signal_direction == 'LONG' and sentiment == 'bullish':
            return True, 0.15
        if signal_direction == 'SHORT' and sentiment == 'bearish':
            return True, 0.15
        return False, 0.0

    def get_nearest_support_resistance(self, current_price: float) -> Dict:
        if not self.vision_data or 'support_resistance' not in self.vision_data:
            return {'nearest_support': None, 'nearest_resistance': None}
        levels = self.vision_data['support_resistance']
        supports = [l for l in levels.get('support', []) if l < current_price]
        resistances = [l for l in levels.get('resistance', []) if l > current_price]
        return {
            'nearest_support': max(supports) if supports else None,
            'nearest_resistance': min(resistances) if resistances else None
        }

# Global singleton
market_context_fusion = MarketContextFusion()
