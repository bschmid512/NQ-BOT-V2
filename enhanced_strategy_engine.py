"""
Enhanced Strategy Engine with Fusion Integration
Orchestrates all strategies and uses fusion layer for intelligent decisions
"""
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

# Import all strategies
from strategies.momentum_continuation_strategy import MomentumContinuationStrategy
from strategies.pullback_entry_strategy import PullbackEntryStrategy

# Import fusion systems
from market_context_fusion import market_context_fusion
from signal_fusion_engine import signal_fusion_engine


class EnhancedStrategyEngine:
    """
    Orchestrates multiple strategies and uses fusion for final decisions
    This is the BRAIN of the trading system
    """
    
    def __init__(self):
        # Initialize all strategies
        self.strategies = {
            'momentum': MomentumContinuationStrategy(),
            'pullback': PullbackEntryStrategy(),
        }
        
        # Track active strategies
        self.active_strategy_names = [name for name, strat in self.strategies.items()]
        
        print(f"✅ Enhanced Strategy Engine initialized")
        print(f"   Active strategies: {self.active_strategy_names}")
        print(f"   Fusion system: ENABLED")
    
    def process_new_bar(self, bar_data: Dict, df: pd.DataFrame, 
                       vision_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Process new bar and generate trading signals using fusion
        
        Args:
            bar_data: Latest bar data from webhook
            df: DataFrame with recent historical bars
            vision_data: Optional vision system analysis
            
        Returns:
            Fused signal if approved by fusion engine, None otherwise
        """
        current_price = bar_data['close']
        
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
                signal = strategy.generate_signal(df, current_price, context)
                if signal:
                    strategy_signals.append(signal)
                    print(f"   ✓ {name}: {signal['direction']} (confidence: {signal['confidence']:.2f})")
            except Exception as e:
                print(f"   ✗ Error in {name}: {e}")
        
        # If no signals, return None
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
