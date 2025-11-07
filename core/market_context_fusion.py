"""
Market Context Fusion Layer
Combines visual analysis from screen capture with real-time webhook data
This is the KEY integration that makes the hybrid system work
"""
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
from config import ATR_PERIOD
from core.comprehensive_logger import comprehensive_logger # <-- ADD THIS

class MarketContextFusion:
    """
    Fuses vision system analysis with webhook price data
    Creates unified market context for intelligent decision-making
    """
    
    def __init__(self):
        self.vision_data = None
        self.price_data = None
        self.last_update = None
        
        # Track market regime
        self.current_regime = 'unknown'
        self.regime_confidence = 0.0
        
        print("✅ Market Context Fusion initialized")
    
    def update_vision_data(self, vision_analysis: Dict):
        """
        Update with latest vision system analysis
        
        Args:
            vision_analysis: Dict from vision system with patterns, levels, sentiment
        """
        self.vision_data = vision_analysis
        self.last_update = datetime.now()
        
        # Extract key visual elements
        if vision_analysis:
            print(f"👁️ Vision Update: {vision_analysis.get('statistics', {}).get('sentiment', 'unknown')}")
        try:
            if self.price_data and not self.price_data['df'].empty:
                df = self.price_data['df']
                price_action = {
                    'close': self.price_data['current_price'],
                    'direction': 'bullish' if df.iloc[-1]['close'] > df.iloc[-1]['open'] else 'bearish',
                    'change_pct': (df.iloc[-1]['close'] - df.iloc[-10]['close']) / df.iloc[-10]['close'] if len(df) > 10 else 0
                }
                self.logger.log_vision_analysis(vision_analysis, price_action)
        except Exception as e:
            print(f"Error logging vision analysis: {e}")
    def update_price_data(self, df: pd.DataFrame, current_price: float):
        """
        Update with latest price data from webhook
        
        Args:
            df: DataFrame with recent bars
            current_price: Current market price
        """
        self.price_data = {
            'current_price': current_price,
            'df': df,
            'bars_available': len(df)
        }
    
    def get_unified_context(self) -> Dict:
        """
        Create unified market context combining ALL available data
        This is what strategies will use for decision-making
        
        Returns:
            Comprehensive market context dictionary
        """
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
            'price_data_available': False
        }
        
        # Add price data if available
        if self.price_data and not self.price_data['df'].empty:
            context['price_data_available'] = True
            context['current_price'] = self.price_data['current_price']
            
            df = self.price_data['df']
            
            # Calculate momentum
            if len(df) >= 20:
                momentum_pct = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
                if momentum_pct > 0.003:
                    context['momentum'] = 'strong_bullish'
                elif momentum_pct > 0.001:
                    context['momentum'] = 'bullish'
                elif momentum_pct < -0.003:
                    context['momentum'] = 'strong_bearish'
                elif momentum_pct < -0.001:
                    context['momentum'] = 'bearish'
            
            # Calculate volatility (ATR-based)
            if len(df) >= ATR_PERIOD:
                high_low = df['high'] - df['low']
                context['volatility'] = high_low.tail(ATR_PERIOD).mean()
            
            # Determine trend
            if len(df) >= 50:
                sma_50 = df['close'].tail(50).mean()
                current = df['close'].iloc[-1]
                
                if current > sma_50:
                    context['trend_direction'] = 'bullish'
                    context['trend_strength'] = (current - sma_50) / sma_50
                else:
                    context['trend_direction'] = 'bearish'
                    context['trend_strength'] = (sma_50 - current) / sma_50
        
        # Add vision data if available
        if self.vision_data:
            context['vision_available'] = True
            
            # Get visual sentiment
            stats = self.vision_data.get('statistics', {})
            context['vision_sentiment'] = stats.get('sentiment', 'neutral')
            
            # Extract detected patterns
            if 'patterns' in self.vision_data:
                context['visual_patterns'] = self.vision_data['patterns']
            
            # Extract key levels from vision
            if 'support_resistance' in self.vision_data:
                sr_levels = self.vision_data['support_resistance']
                context['key_levels'] = sr_levels
        
        # FUSION: Determine market regime using BOTH sources
        context['market_regime'], context['regime_confidence'] = self._determine_market_regime(context)
        
        return context
    
    def _determine_market_regime(self, context: Dict) -> tuple[str, float]:
        """
        Use AI-like logic to determine market regime from multiple signals
        
        Returns:
            (regime, confidence) where:
            - regime: 'strong_trend', 'weak_trend', 'ranging', 'breakout_pending', 'choppy'
            - confidence: 0.0 to 1.0
        """
        signals = []
        
        # Signal 1: Price momentum
        if context['momentum'] in ['strong_bullish', 'strong_bearish']:
            signals.append(('trending', 0.8))
        elif context['momentum'] in ['bullish', 'bearish']:
            signals.append(('weak_trend', 0.6))
        else:
            signals.append(('ranging', 0.5))
        
        # Signal 2: Vision sentiment
        if context['vision_sentiment'] in ['bullish', 'bearish']:
            # Vision confirms directional bias
            if context['vision_sentiment'] == context['momentum'].replace('strong_', ''):
                signals.append(('trending', 0.9))  # High confidence when aligned
            else:
                signals.append(('ranging', 0.4))  # Conflicting signals = ranging
        
        # Signal 3: Trend strength
        if context['trend_strength'] > 0.015:  # 1.5% from mean
            signals.append(('strong_trend', 0.85))
        elif context['trend_strength'] > 0.007:  # 0.7% from mean
            signals.append(('weak_trend', 0.65))
        
        # Signal 4: Visual patterns
        if len(context['visual_patterns']) > 0:
            # If we see consolidation patterns = breakout pending
            consolidation_patterns = ['triangle', 'wedge', 'rectangle']
            has_consolidation = any(p in str(context['visual_patterns']).lower() 
                                   for p in consolidation_patterns)
            if has_consolidation:
                signals.append(('breakout_pending', 0.75))
        
        # FUSION: Weight and combine signals
        regime_scores = {}
        for regime, confidence in signals:
            if regime not in regime_scores:
                regime_scores[regime] = []
            regime_scores[regime].append(confidence)
        
        # Average confidences for each regime
        regime_avg = {r: np.mean(scores) for r, scores in regime_scores.items()}
        
        if not regime_avg:
            return 'unknown', 0.0
        
        # Pick regime with highest confidence
        best_regime = max(regime_avg, key=regime_avg.get)
        best_confidence = regime_avg[best_regime]
        
        return best_regime, best_confidence
    
    def check_vision_confirmation(self, signal_direction: str) -> tuple[bool, float]:
        """
        Check if vision system confirms a trading signal
        
        Args:
            signal_direction: 'LONG' or 'SHORT'
            
        Returns:
            (confirmed, confidence_boost)
        """
        if not self.vision_data:
            return False, 0.0
        
        sentiment = self.vision_data.get('statistics', {}).get('sentiment', 'neutral')
        
        # Check alignment
        if signal_direction == 'LONG' and sentiment == 'bullish':
            return True, 0.15  # 15% confidence boost
        elif signal_direction == 'SHORT' and sentiment == 'bearish':
            return True, 0.15
        
        return False, 0.0
    
    def get_nearest_support_resistance(self, current_price: float) -> Dict:
        """
        Find nearest S/R levels from vision system
        
        Returns:
            {'nearest_support': price, 'nearest_resistance': price, 'distance': points}
        """
        if not self.vision_data or 'support_resistance' not in self.vision_data:
            return {'nearest_support': None, 'nearest_resistance': None}
        
        levels = self.vision_data['support_resistance']
        
        # Find nearest support (below current price)
        supports = [l for l in levels.get('support', []) if l < current_price]
        nearest_support = max(supports) if supports else None
        
        # Find nearest resistance (above current price)
        resistances = [l for l in levels.get('resistance', []) if l > current_price]
        nearest_resistance = min(resistances) if resistances else None
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': (current_price - nearest_support) if nearest_support else None,
            'resistance_distance': (nearest_resistance - current_price) if nearest_resistance else None
        }


# Create global instance
market_context_fusion = MarketContextFusion()
