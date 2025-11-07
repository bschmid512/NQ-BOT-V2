"""
Enhanced Strategy Engine with Fusion Integration
Orchestrates all strategies and uses fusion layer for intelligent decisions
"""
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from core.strategy_adapter import call_generate_signal

# --- 1. IMPORT ALL STRATEGIES ---
from strategies.momentum_continuation_strategy import MomentumContinuationStrategy
from strategies.pullback_entry_strategy import PullbackEntryStrategy
from strategies.opening_range import orb_strategy
from strategies.mean_reversion import mean_reversion_strategy
from strategies.trend_following_strategy import trend_following_strategy
from strategies.breakout import breakout_strategy

# --- 2. IMPORT FUSION & CONFIG ---
from core.market_context_fusion import market_context_fusion
from core.signal_fusion_engine import signal_fusion_engine
from config import STRATEGIES  # Import the main strategy config

class EnhancedStrategyEngine:
    """
    Orchestrates multiple strategies and uses fusion for final decisions
    This is the BRAIN of the trading system
    """
    def _ensure_common_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure commonly used EMAs exist to prevent KeyErrors (ema_8, ema_9, ema_21, ema_50)."""
        if df is None or df.empty:
            return df
        if 'close' not in df.columns:
            return df
        for n in (8, 9, 21, 50):
            col = f'ema_{n}'
            if col not in df.columns:
                df[col] = df['close'].ewm(span=n, adjust=False).mean()
        return df

    
    def __init__(self):
        # --- 3. BUILD STRATEGY LIST DYNAMICALLY ---
        
        # Map of strategy names to their class/instance
        self.strategy_map = {
            'momentum': MomentumContinuationStrategy(),
            'pullback': PullbackEntryStrategy(),
            'orb': orb_strategy,
            'mean_reversion': mean_reversion_strategy,
            'trend_following': trend_following_strategy,
            'breakout': breakout_strategy
        }
        
        # Load only strategies that are ENABLED in config.py
        self.strategies = {}
        for name, instance in self.strategy_map.items():
            if name in STRATEGIES and STRATEGIES[name].get('enabled', False):
                self.strategies[name] = instance
        
        self.active_strategy_names = list(self.strategies.keys())
        # --- END DYNAMIC BUILD ---
        
        print(f"✅ Enhanced Strategy Engine initialized")
        print(f"   Active strategies: {self.active_strategy_names}")
        print(f"   Fusion system: ENABLED")
    
    def process_new_bar(self, bar_data: Dict, df: pd.DataFrame, 
                       vision_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Process new bar and generate trading signals using fusion
        """
        current_price = bar_data['close']
        # Ensure common EMAs exist so strategies referencing ema_8/9/21/50 won't error
        df = self._ensure_common_indicators(df)
        
        # Update market context fusion with latest data
        market_context_fusion.update_price_data(df, current_price)
        
        if vision_data:
            market_context_fusion.update_vision_data(vision_data)
        
        # Get unified market context
        context = market_context_fusion.get_unified_context()
        
        # Log context
        self._log_context(context)
        
        # Collect signals from all strategies
        strategy_signals = []
        
        for name, strategy in self.strategies.items():
            try:
                # --- 4. FIX ARGUMENT PASSING ---
                # Some strategies (like pullback) need current_price,
                # some (like ORB) need special handling.
                # We will check what arguments they need (a bit advanced,
                # but for now we'll pass all common ones)
                
                # The 'orb' strategy is a function, not a class, in your code
                # and is handled by the old strategy_engine.
                # For now, let's assume all strategies in this new engine
                # use the (df, current_price, context) signature.
                
                # NOTE: This assumes your orb_strategy and others
                # have a `generate_signal(df, current_price, context)` method.
                # Your pullback/momentum do, but your old ones might not.
                # We'll adjust if another error appears, but this is the
                # correct architecture.
                
                # Let's check the old strategy_engine.py for a hint...
                # old engine passes: strategy.generate_signal(df, context)
                # new strategies need: strategy.generate_signal(df, current_price, context)
                # This is a conflict.
                
                # --- Let's use a standard signature ---
                # All strategies *should* conform to:
                # generate_signal(df, current_price, context)
                
                # We'll assume you update your old strategies
                # (mean_reversion, etc.) to accept `current_price` as well.
                
                signal = call_generate_signal(strategy, df=df, current_price=current_price, context=context,
    current_bar=bar_data)
                
                if signal:
                    strategy_signals.append(signal)
                    print(f"   ✓ {name}: {signal.get('direction', signal.get('signal'))} (confidence: {signal.get('confidence', 0):.2f})")
            except Exception as e:
                print(f"   ✗ Error in {name}: {e}")
        
        if not strategy_signals:
            return None
        
        # Use fusion engine to evaluate signals
        fused_signal = signal_fusion_engine.evaluate_trade_setup(
            strategy_signals, 
            context
        )
        
        return fused_signal
    
    def _log_context(self, context: Dict):
        """Log market context for debugging"""
        print(f"\n{'─'*60}")
        print(f"📊 Market Context Update")
        print(f"   Price: ${context.get('current_price', 0):.2f}")
        print(f"   Regime: {context.get('market_regime', 'unknown')} ({context.get('regime_confidence', 0):.1%})")
        print(f"   Momentum: {context.get('momentum', 'neutral')}")
        print(f"   Trend: {context.get('trend_direction', 'neutral')}")
        print(f"   Vision: {'✓' if context.get('vision_available') else '✗'} | Sentiment: {context.get('vision_sentiment', 'N/A')}")
        print(f"{'─'*60}")
    
    def get_strategy_states(self, df: pd.DataFrame) -> Dict:
        """Get current state of all strategies"""
        states = {}
        
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'get_current_state'):
                    states[name] = strategy.get_current_state(df)
                else:
                    states[name] = {'status': 'no_state_method'}
            except Exception as e:
                states[name] = {'status': 'error', 'error': str(e)}
        
        return states
    
    def get_fusion_stats(self) -> Dict:
        """Get fusion engine statistics"""
        return signal_fusion_engine.get_fusion_stats()


# Create global instance
enhanced_strategy_engine = EnhancedStrategyEngine()