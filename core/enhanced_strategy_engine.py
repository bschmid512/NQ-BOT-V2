"""
Enhanced Strategy Engine with Signal Fusion
Coordinates multiple strategies and uses fusion engine for final decisions
"""
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import numpy as np

from core.market_context_fusion import market_context_fusion
from strategies.pullback_strategy import PullbackStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy


def call_generate_signal(strategy, df, current_price, context, current_bar):
    """
    Safely call generate_signal with proper parameter handling
    """
    try:
        # Try with all parameters first
        return strategy.generate_signal(
            df=df,
            current_price=current_price,
            context=context
        )
    except TypeError:
        # Fallback to minimal parameters if strategy doesn't accept all
        try:
            return strategy.generate_signal(df=df, current_price=current_price)
        except TypeError:
            # Last resort - just df
            return strategy.generate_signal(df=df)


class EnhancedStrategyEngine:
    """
    Coordinates multiple trading strategies and uses fusion engine for decisions
    """
    
    def __init__(self, signal_fusion_engine):
        """
        Initialize with fusion engine
        
        Args:
            signal_fusion_engine: SignalFusionEngine instance for final decisions
        """
        self.fusion_engine = signal_fusion_engine
        
        # Initialize all strategies
        self.strategies = {
            'pullback': PullbackStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy()
        }
        
        print("✓ Enhanced Strategy Engine initialized with Signal Fusion")
    
    def _ensure_common_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure common indicators are calculated"""
        df = df.copy()
        
        # EMAs
        for period in [8, 20, 50, 200]:
            col_name = f'ema_{period}'
            if col_name not in df.columns:
                df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def process_new_bar(self, bar_data: Dict, df: pd.DataFrame,
                        vision_data: Optional[Dict] = None) -> Optional[Dict]:
        """Process new bar and generate trading signals using fusion."""
        current_price = float(bar_data['close'])
        df = self._ensure_common_indicators(df)

        # Update market context fusion with latest data
        market_context_fusion.update_price_data(df, current_price)
        if vision_data:
            market_context_fusion.update_vision_data(vision_data)

        # Get unified market context
        context = market_context_fusion.get_unified_context()

        # FIXED: Removed verbose context logging that runs on every bar
        # self._log_context(context)  # Commented out - too verbose

        # Collect signals from all strategies
        strategy_signals = []
        for name, strategy in self.strategies.items():
            try:
                signal = call_generate_signal(
                    strategy,
                    df=df,
                    current_price=current_price,
                    context=context,
                    current_bar=bar_data
                )
                if signal:
                    signal = self._apply_divergence_confluence(signal, context)
                if signal:
                    strategy_signals.append(signal)
                    print(f"   ✓ {name}: {signal.get('direction', signal.get('signal'))} "
                          f"(confidence: {float(signal.get('confidence', 0)):.2f})")
            except Exception as e:
                print(f"   ✗ Error in {name}: {e}")

        # FIXED: Only log context when we have signals
        if strategy_signals:
            self._log_context(context)

        if not strategy_signals:
            return None

        # Use fusion engine to make final decision
        fused_signal = self.fusion_engine.evaluate_trade_setup(context, strategy_signals)
        
        return fused_signal
    
    def _log_context(self, context: Dict):
        """Log current market context"""
        print("─" * 60)
        print(" Market Context Update")
        print(f"   Price: ${context.get('current_price', 0):.2f}")
        print(f"   Regime: {context.get('market_regime', 'unknown')} "
              f"({context.get('regime_confidence', 0):.1%})")
        print(f"   Momentum: {context.get('momentum', 'neutral')}")
        print(f"   Trend: {context.get('trend_direction', 'neutral')}")
        print(f"   Vision: {'✓' if context.get('vision_available') else '✗'} | "
              f"Sentiment: {context.get('vision_sentiment', 'neutral')}")
        
        # Detailed context line
        p = context.get('current_price', 0)
        regime = context.get('market_regime', 'unknown')
        regime_conf = context.get('regime_confidence', 0)
        mom = context.get('momentum', 'neutral')
        mom1 = context.get('momentum_1m', 0)
        mom5 = context.get('momentum_5m', 0)
        trend = context.get('trend_direction', 'neutral')
        vision = '✓' if context.get('vision_available') else '✗'
        bars = context.get('bars_count', 0)
        
        ema20 = context.get('ema_20', 0)
        ema50 = context.get('ema_50', 0)
        slope = context.get('trend_slope', 0)
        adx = context.get('adx', 0)
        
        print(f"   ➜ p={p:.2f} | regime={regime}({regime_conf:.2f}) | "
              f"mom={mom} (5m={mom5:.3%}) | trend={trend} | vision{vision} | bars={bars}; "
              f"mom1={mom1}, mom5={mom5}; "
              f"trend-{trend} (ema20={ema20:.2f}, ema50={ema50:.2f}, slope={slope:.4f}, adx≈{adx:.1f})")
        print("─" * 60)
    
    def _apply_divergence_confluence(self, signal: Dict, context: Dict) -> Optional[Dict]:
        """Apply divergence and confluence filters"""
        # Check for divergence (price vs indicator disagreement)
        if self._has_divergence(signal, context):
            return None
        
        # Check for confluence (multiple indicators agreeing)
        if not self._has_confluence(signal, context):
            signal['confidence'] = signal.get('confidence', 50) * 0.8
        
        return signal
    
    def _has_divergence(self, signal: Dict, context: Dict) -> bool:
        """Check for bearish/bullish divergence"""
        direction = signal.get('direction', signal.get('signal'))
        
        # If RSI and price disagree significantly
        rsi = context.get('rsi', 50)
        if direction == 'LONG' and rsi < 30:
            return True
        if direction == 'SHORT' and rsi > 70:
            return True
        
        return False
    
    def _has_confluence(self, signal: Dict, context: Dict) -> bool:
        """Check if multiple indicators confirm the signal"""
        direction = signal.get('direction', signal.get('signal'))
        
        confirmations = 0
        
        # Trend confirmation
        if context.get('trend_direction') == direction.lower():
            confirmations += 1
        
        # Momentum confirmation
        if direction == 'LONG' and context.get('momentum') == 'bullish':
            confirmations += 1
        elif direction == 'SHORT' and context.get('momentum') == 'bearish':
            confirmations += 1
        
        # Volume confirmation (if available)
        if context.get('volume_trend') == 'increasing':
            confirmations += 1
        
        return confirmations >= 2