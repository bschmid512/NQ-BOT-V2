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
    """Orchestrates multiple strategies and uses fusion for final decisions."""

    def _ensure_common_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or 'close' not in df.columns:
            return df
        for n in (8, 9, 21, 50):
            col = f'ema_{n}'
            if col not in df.columns:
                df[col] = df['close'].ewm(span=n, adjust=False, min_periods=n).mean()
        return df

    def __init__(self):
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
        self.strategies = {name: inst for name, inst in self.strategy_map.items()
                           if name in STRATEGIES and STRATEGIES[name].get('enabled', False)}
        self.active_strategy_names = list(self.strategies.keys())

        print(f"✅ Enhanced Strategy Engine initialized")
        print(f"   Active strategies: {self.active_strategy_names}")
        print(f"   Fusion system: ENABLED")

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

        # Log context (+ diagnostics explains why labels were chosen)
        self._log_context(context)

        # Collect signals
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

        if not strategy_signals:
            return None

        # Use fusion engine to evaluate signals
        fused_signal = signal_fusion_engine.evaluate_trade_setup(strategy_signals, context)
        return fused_signal

    def _log_context(self, context: Dict):
        print(f"\n{'─'*60}")
        print(f" Market Context Update")
        print(f"   Price: ${context.get('current_price', 0):.2f}")
        print(f"   Regime: {context.get('market_regime', 'unknown')} ({context.get('regime_confidence', 0):.1%})")
        print(f"   Momentum: {context.get('momentum', 'neutral')}")
        print(f"   Trend: {context.get('trend_direction', 'neutral')}")
        print(f"   Vision: {'✓' if context.get('vision_available') else '✗'} | Sentiment: {context.get('vision_sentiment', 'N/A')}")
        diag = context.get('diagnostics')
        if diag:
            print(f"   ➜ {diag}")
        print(f"{'─'*60}")

    def get_strategy_states(self, df: pd.DataFrame) -> Dict:
        states = {}
        for name, strategy in self.strategies.items():
            try:
                states[name] = strategy.get_current_state(df) if hasattr(strategy, 'get_current_state') \
                               else {'status': 'no_state_method'}
            except Exception as e:
                states[name] = {'status': 'error', 'error': str(e)}
        return states

    def get_fusion_stats(self) -> Dict:
        return signal_fusion_engine.get_fusion_stats()

    def _apply_divergence_confluence(self, signal: dict, context: dict) -> dict:
        try:
            if not signal or not context:
                return signal
            score = float(context.get('divergence_score', 0.0))
            direction = signal.get('direction')
            adj = 0.0
            if direction == 'LONG':
                adj = 0.10 if score > 0 else (-0.10 if score < 0 else 0.0)
            elif direction == 'SHORT':
                adj = 0.10 if score < 0 else (-0.10 if score > 0 else 0.0)
            conf = float(signal.get('confidence', 0.5))
            conf = max(0.0, min(1.0, conf + adj))
            signal['confidence'] = round(conf, 3)
            signal['divergence_score'] = score
            return signal
        except Exception:
            return signal

# Create global instance
enhanced_strategy_engine = EnhancedStrategyEngine()
