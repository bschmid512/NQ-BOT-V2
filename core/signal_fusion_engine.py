"""
Signal Fusion Engine - The AI Decision Layer
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import FUSION_CONFIG
from core.comprehensive_logger import comprehensive_logger # <-- 1. IMPORT
from core.market_context_fusion import market_context_fusion # <-- 2. IMPORT FOR STOPS

class SignalFusionEngine:
    """
    Advanced signal fusion system that requires CONFLUENCE
    """
    
    def __init__(self):
        # ... (all your existing init properties are fine) ...
        self.min_signals_required = FUSION_CONFIG['min_signals_required']
        self.min_total_weight = FUSION_CONFIG['min_total_weight']
        self.max_weight = FUSION_CONFIG['max_weight']
        self.vision_weight_multiplier = FUSION_CONFIG['vision_weight_multiplier']
        self.convergence_bonus = FUSION_CONFIG['convergence_bonus']
        self.recent_signals = []
        self.signal_cooldown = timedelta(minutes=5)

        self.logger = comprehensive_logger # <-- 3. SET UP LOGGER
        
        print(f"✅ Signal Fusion Engine initialized")
        print(f"   Requires {self.min_signals_required}+ signals with {self.min_total_weight}+ weight")
    
    def evaluate_trade_setup(self, strategy_signals: List[Dict], 
                                market_context: Dict) -> Optional[Dict]:
            """
            Evaluate if we should trade based on multiple signals
            
            Args:
                strategy_signals: List of signals from different strategies
                market_context: Unified context from MarketContextFusion
            """
            
            # --- 1. Filter by Signal Count ---
            if len(strategy_signals) < self.min_signals_required:
                self.logger.log_trade_rejected(
                    strategy_signals, 
                    f"Not enough signals. Need {self.min_signals_required}, got {len(strategy_signals)}",
                    market_context
                )
                return None
            
            # --- 2. Fuse Signals ---
            # (This calls your existing _fuse_signals method)
            fused_result = self._fuse_signals(strategy_signals, market_context)
            
            if not fused_result:
                self.logger.log_trade_rejected(strategy_signals, "Signal fusion failed (e.g., conflicting)", market_context)
                return None
            
            # --- 3. Check Cooldown ---
            if self._in_cooldown_period(fused_result['direction']):
                self.logger.log_trade_rejected([fused_result], "Cooldown active", market_context)
                return None

            # --- 4. Final Weight/Confidence Check ---
            if fused_result['weight'] < self.min_total_weight:
                self.logger.log_trade_rejected(
                    [fused_result], 
                    f"Fused weight {fused_result['weight']} < {self.min_total_weight}",
                    market_context
                )
                return None
            
            # --- 5. ROADMAP: Implement Dynamic Stops ---
            # "Use vision-detected levels for stops, not fixed points."
            current_price = market_context.get('current_price', fused_result['price'])
            levels = market_context_fusion.get_nearest_support_resistance(current_price)
            
            if fused_result['direction'] == 'LONG' and levels.get('nearest_support'):
                fused_result['stop'] = levels['nearest_support'] - 2 # 2 pts below vision support
            elif fused_result['direction'] == 'SHORT' and levels.get('nearest_resistance'):
                fused_result['stop'] = levels['nearest_resistance'] + 2 # 2 pts above vision resistance

            # --- 6. ROADMAP: Implement Scale In/Out ---
            # "Scale In/Out... When conviction is high... scale to 2-3 contracts"
            if fused_result['weight'] > (self.min_total_weight * 1.5) and fused_result.get('vision_confirmed', False):
                fused_result['size'] = 3 # High conviction
            elif fused_result['weight'] > (self.min_total_weight * 1.2) or fused_result.get('vision_confirmed', False):
                fused_result['size'] = 2 # Medium conviction
            else:
                fused_result['size'] = 1 # Standard conviction
            
            # --- 7. LOG APPROVED TRADE & RETURN ---
            self.logger.log_trade_taken(fused_result, position={}, context=market_context)
            
            # --- 8. (THE FIX) Standardize key and return the correct variable ---
            if 'direction' in fused_result:
                fused_result['signal'] = fused_result.pop('direction')

            self.recent_signals.append({
                'timestamp': datetime.now(),
                'direction': fused_result['signal'],
                'weight': fused_result['weight']
            })
            
            return fused_result


# ... (This function goes INSIDE the SignalFusionEngine class)

    def _fuse_signals(self, signals: List[Dict], context: Dict) -> Optional[Dict]:
        """
        Advanced fusion logic.
        This is the "brain" that combines signals and checks for confluence.
        """
        # --- 1. Separate signals by direction ---
        long_signals = [s for s in signals if s.get('direction', s.get('signal')) == 'LONG']
        short_signals = [s for s in signals if s.get('direction', s.get('signal')) == 'SHORT']

        # --- 2. Reject if signals are conflicting ---
        if long_signals and short_signals:
            self.logger.log_trade_rejected(signals, "Conflicting signals (LONG and SHORT)", context)
            return None
        
        if not long_signals and not short_signals:
            return None # No signals to fuse
        
        target_signals = long_signals if long_signals else short_signals
        direction = 'LONG' if long_signals else 'SHORT'
        
        # --- 3. Calculate fused weight and confidence ---
        fused_weight = sum(s['weight'] for s in target_signals)
        fused_confidence = sum(s.get('confidence', 0.5) for s in target_signals) / len(target_signals) # Use 0.5 as default
        
        # --- 4. Check for Vision Confirmation ---
        vision_sentiment = context.get('vision_sentiment', 'neutral')
        vision_confirmed = False
        
        if (direction == 'LONG' and vision_sentiment == 'bullish') or \
           (direction == 'SHORT' and vision_sentiment == 'bearish'):
            vision_confirmed = True
            fused_weight *= self.vision_weight_multiplier # Apply 20% boost
            fused_weight = min(fused_weight, self.max_weight) # Cap at max
        
        # --- 5. Check for Convergence Bonus ---
        if len(target_signals) > 1:
            fused_weight += self.convergence_bonus
            fused_weight = min(fused_weight, self.max_weight) # Cap at max
        
        # --- 6. Build the final, approved signal ---
        # We take the price/stop/target from the *highest confidence* signal
        best_signal = max(target_signals, key=lambda s: s.get('confidence', 0))
        
        return {
            'strategy': 'fusion',
            'direction': direction,
            'price': best_signal.get('price'),
            'stop': best_signal.get('stop'),   # Will be overwritten by dynamic stops
            'target': best_signal.get('target'),
            'weight': fused_weight,
            'confidence': fused_confidence,
            'vision_confirmed': vision_confirmed,
            'source_signals': [s['strategy'] for s in target_signals],
            'reason': f"Fused {len(target_signals)} signals. Vision: {vision_confirmed}"
        }
    # ... (Your _fuse_signals, _in_cooldown_period, get_fusion_stats methods are here) ...
    # (No changes needed to them)
    
    def _calculate_fusion_score(self, signals: List[Dict], context: Dict, 
                                direction: str) -> Dict:
        """
        Calculate fusion score from multiple signals
        
        Returns:
            Dict with total_weight, num_signals, vision_confirmed, etc.
        """
        total_weight = 0
        max_confidence = 0
        
        for signal in signals:
            weight = signal.get('weight', 25)
            confidence = signal.get('confidence', 0.5)
            
            # Boost weight based on confidence
            adjusted_weight = weight * (0.5 + confidence)
            
            total_weight += adjusted_weight
            max_confidence = max(max_confidence, confidence)
        
        # Check vision confirmation
        vision_confirmed = False
        if context.get('vision_available'):
            vision_sentiment = context.get('vision_sentiment', 'neutral')
            
            if (direction == 'LONG' and vision_sentiment == 'bullish') or \
               (direction == 'SHORT' and vision_sentiment == 'bearish'):
                vision_confirmed = True
                total_weight *= self.vision_weight_multiplier
        
        # Convergence bonus (3+ signals agreeing)
        if len(signals) >= 3:
            total_weight += self.convergence_bonus
        
        # Cap at max weight
        total_weight = min(total_weight, self.max_weight)
        
        return {
            'total_weight': total_weight,
            'num_signals': len(signals),
            'max_confidence': max_confidence,
            'vision_confirmed': vision_confirmed
        }
    
    def _create_fused_signal(self, signals: List[Dict], fusion_result: Dict,
                            context: Dict) -> Dict:
        """
        Create final fused signal combining best elements from all signals
        """
        direction = signals[0]['direction']
        
        # Use weighted average for entry, stop, target
        weights = [s.get('weight', 25) for s in signals]
        total_weight = sum(weights)
        
        # Entry: Use current price from context
        entry = context.get('current_price', signals[0].get('price', 0))
        
        # Stop: Use tightest stop from high-weight signals
        stops = [s.get('stop') for s in signals if s.get('stop') is not None]
        if stops:
            if direction == 'LONG':
                stop = max(stops)  # Tightest stop for LONG
            else:
                stop = min(stops)  # Tightest stop for SHORT
        else:
            # Fallback: 50 points
            stop = entry - 50 if direction == 'LONG' else entry + 50
        
        # Targets: Use weighted average of all targets
        targets_lists = [s.get('targets', [s.get('target')]) for s in signals 
                        if s.get('targets') or s.get('target')]
        
        if targets_lists:
            # Flatten and average
            all_targets = [t for sublist in targets_lists for t in (sublist if isinstance(sublist, list) else [sublist]) if t is not None]
            if all_targets:
                avg_target = sum(all_targets) / len(all_targets)
                
                # Create multiple targets (T1, T2, T3)
                target_range = abs(avg_target - entry)
                targets = [
                    entry + (target_range * 0.5 * (1 if direction == 'LONG' else -1)),  # T1: 50%
                    entry + (target_range * 1.0 * (1 if direction == 'LONG' else -1)),  # T2: 100%
                    entry + (target_range * 1.5 * (1 if direction == 'LONG' else -1)),  # T3: 150%
                ]
            else:
                # Fallback targets
                target_distance = 75
                targets = [
                    entry + (target_distance * (1 if direction == 'LONG' else -1)),
                ]
        else:
            target_distance = 75
            targets = [
                entry + (target_distance * (1 if direction == 'LONG' else -1)),
            ]
        
        # Calculate conviction percentage (0-100)
        conviction = min(int(fusion_result['total_weight']), 100)
        
        # Create fusion reason
        strategy_names = [s.get('strategy', 'unknown') for s in signals]
        reason = f"Fusion of {len(signals)} strategies: {', '.join(strategy_names)}"
        
        if fusion_result['vision_confirmed']:
            reason += " | Vision CONFIRMED"
        
        # Add market regime to reason
        regime = context.get('market_regime', 'unknown')
        reason += f" | Regime: {regime}"
        
        return {
            'strategy': 'fusion',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': targets[0],  # Primary target for compatibility
            'targets': targets,
            'conviction': conviction,
            'confidence': fusion_result['max_confidence'],
            'weight': fusion_result['total_weight'],
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'contributing_signals': len(signals),
            'vision_confirmed': fusion_result['vision_confirmed'],
            'market_regime': context.get('market_regime')
        }
    
    def _in_cooldown_period(self, direction: str) -> bool:
        """Check if we're in cooldown period for this direction"""
        now = datetime.now()
        
        # Clean old signals
        self.recent_signals = [s for s in self.recent_signals 
                               if now - s['timestamp'] < timedelta(hours=1)]
        
        # Check recent signals for this direction
        recent_same_direction = [s for s in self.recent_signals 
                                 if s['direction'] == direction and 
                                 now - s['timestamp'] < self.signal_cooldown]
        
        return len(recent_same_direction) > 0
    
    def get_fusion_stats(self) -> Dict:
        """Get statistics on fusion decisions"""
        now = datetime.now()
        
        # Count signals in last hour
        recent = [s for s in self.recent_signals 
                 if now - s['timestamp'] < timedelta(hours=1)]
        
        return {
            'signals_last_hour': len(recent),
            'avg_weight': sum(s['weight'] for s in recent) / len(recent) if recent else 0,
            'long_signals': len([s for s in recent if s['direction'] == 'LONG']),
            'short_signals': len([s for s in recent if s['direction'] == 'SHORT'])
        }


# Create global instance
signal_fusion_engine = SignalFusionEngine()
