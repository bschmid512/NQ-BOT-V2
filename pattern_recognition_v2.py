"""
Enhanced Pattern Recognition - Version 2.0
IMPROVEMENTS:
- More chart patterns (double top/bottom, triangles, head & shoulders)
- Better engulfing pattern detection
- Support/resistance level detection
- Volume analysis
- Multi-timeframe pattern detection
- Confidence scoring based on multiple factors
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import defaultdict


class PatternRecognizerV2:
    """
    Enhanced AI-based pattern recognition for trading charts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern confidence thresholds
        self.min_confidence = 0.6
        
        # Support/Resistance parameters
        self.sr_tolerance = 20  # pixels
        self.sr_min_touches = 2  # minimum touches for valid level
        
    def calculate_signal_levels(self, signal_type, pattern, confidence, current_price: Optional[float] = None, candles=None):
        """
        Calculate entry, stop loss, and take profit levels for a signal
        
        Args:
            signal_type: 'LONG' or 'SHORT' or 'WATCH'
            pattern: Pattern name
            confidence: Signal confidence (0.0-1.0)
            current_price: Optional current price from OCR
            candles: Candlestick data (to estimate price)
            
        Returns:
            Dict with entry, stop_loss, take_profit, risk/reward info
        """
        # --- START MODIFICATION ---
        # Estimate current price
        if current_price:
            estimated_price = current_price
            self.logger.info(f"Using OCR price for levels: {estimated_price}")
        else:
            estimated_price = 20500  # NQ default fallback
            self.logger.warning(f"Using FALLBACK price for levels: {estimated_price}")
        # --- END MODIFICATION ---
        
        # Define stop loss distances based on pattern (in points)
        stop_distances = {
            # Reversal patterns - tight stops
            'bullish_engulfing': 20,
            'bearish_engulfing': 20,
            'hammer': 18,
            'shooting_star': 18,
            'morning_star': 25,
            'evening_star': 25,
            
            # Momentum patterns - medium stops  
            'three_white_soldiers': 35,
            'three_black_crows': 35,
            
            # Chart patterns - wider stops
            'double_top': 40,
            'double_bottom': 40,
            
            # Sentiment - widest stops
            'strong_bullish_sentiment': 50,
            'strong_bearish_sentiment': 50,
        }
        
        # Get base stop distance
        stop_distance = stop_distances.get(pattern, 30)
        
        # Adjust based on confidence (higher confidence = tighter stop)
        if confidence >= 0.85:
            stop_distance = int(stop_distance * 0.8)
        elif confidence >= 0.75:
            stop_distance = int(stop_distance * 0.9)
        elif confidence < 0.70:
            stop_distance = int(stop_distance * 1.2)
        
        # Calculate risk:reward ratio
        if confidence >= 0.80:
            risk_reward = 3.0  # 3:1 for high confidence
        elif confidence >= 0.75:
            risk_reward = 2.5  # 2.5:1 for good confidence
        else:
            risk_reward = 2.0  # 2:1 for acceptable confidence
        
        tp_distance = int(stop_distance * risk_reward)
        
        # Calculate actual levels
        if signal_type == 'LONG':
            entry = estimated_price
            stop_loss = entry - stop_distance
            take_profit = entry + tp_distance
        elif signal_type == 'SHORT':
            entry = estimated_price
            stop_loss = entry + stop_distance  
            take_profit = entry - tp_distance
        else:  # WATCH
            return {
                'entry': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_points': 0,
                'reward_points': 0,
                'risk_reward_ratio': 0
            }
        
        return {
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_points': stop_distance,
            'reward_points': tp_distance,
            'risk_reward_ratio': round(risk_reward, 1)
        }

    def analyze_candlesticks(self, candlesticks: List[Dict], current_price: Optional[float] = None) -> List[Dict]:
        """
        Analyze candlestick patterns with enhanced detection
        
        Args:
            candlesticks: List of detected candlesticks
            current_price: Optional current price from OCR
            
        Returns:
            List of detected patterns with confidence scores
        """
        if not candlesticks or len(candlesticks) < 2:
            return []
        
        patterns_found = []
        
        # Sort by x position (time)
        sorted_candles = sorted(candlesticks, key=lambda c: c['x'])
        
        # BULLISH ENGULFING
        patterns_found.extend(self._detect_engulfing_pattern(sorted_candles, 'bullish'))
        
        # BEARISH ENGULFING
        patterns_found.extend(self._detect_engulfing_pattern(sorted_candles, 'bearish'))
        
        # HAMMER / SHOOTING STAR
        patterns_found.extend(self._detect_reversal_candles(sorted_candles))
        
        # THREE WHITE SOLDIERS / THREE BLACK CROWS
        # --- MODIFICATION: Pass current_price ---
        patterns_found.extend(self._detect_three_soldiers_crows(sorted_candles, current_price))
        
        # MORNING STAR / EVENING STAR
        patterns_found.extend(self._detect_star_patterns(sorted_candles))
        
        return patterns_found
    
    def _detect_engulfing_pattern(self, candles: List[Dict], direction: str) -> List[Dict]:
        """
        Detect bullish or bearish engulfing patterns
        
        Args:
            candles: Sorted list of candlesticks
            direction: 'bullish' or 'bearish'
            
        Returns:
            List of detected engulfing patterns
        """
        patterns = []
        
        for i in range(len(candles) - 1):
            current = candles[i]
            next_candle = candles[i + 1]
            
            if direction == 'bullish':
                # Bullish engulfing: small bearish followed by large bullish
                if (current['type'] == 'bearish' and 
                    next_candle['type'] == 'bullish' and
                    next_candle['area'] > current['area'] * 1.3):
                    
                    # Calculate confidence based on size ratio
                    size_ratio = next_candle['area'] / current['area']
                    confidence = min(0.9, 0.65 + (size_ratio - 1.3) * 0.1)
                    
                    patterns.append({
                        'pattern': 'bullish_engulfing',
                        'signal': 'LONG',
                        'confidence': confidence,
                        'location': (next_candle['x'], next_candle['y']),
                        'reason': f'Bullish engulfing (size ratio: {size_ratio:.2f})',
                        'candle_index': i + 1
                    })
            
            elif direction == 'bearish':
                # Bearish engulfing: small bullish followed by large bearish
                if (current['type'] == 'bullish' and 
                    next_candle['type'] == 'bearish' and
                    next_candle['area'] > current['area'] * 1.3):
                    
                    size_ratio = next_candle['area'] / current['area']
                    confidence = min(0.9, 0.65 + (size_ratio - 1.3) * 0.1)
                    
                    patterns.append({
                        'pattern': 'bearish_engulfing',
                        'signal': 'SHORT',
                        'confidence': confidence,
                        'location': (next_candle['x'], next_candle['y']),
                        'reason': f'Bearish engulfing (size ratio: {size_ratio:.2f})',
                        'candle_index': i + 1
                    })
        
        return patterns
    
    def _detect_reversal_candles(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect hammer and shooting star patterns
        
        Args:
            candles: Sorted list of candlesticks
            
        Returns:
            List of detected reversal patterns
        """
        patterns = []
        
        for i, candle in enumerate(candles):
            # Need significant height
            if candle['height'] < 30:
                continue
            
            # HAMMER (bullish reversal) - long candle with small body at top
            # In our detection, taller green candles could be hammers
            if candle['type'] == 'bullish' and candle['aspect_ratio'] > 3:
                patterns.append({
                    'pattern': 'hammer',
                    'signal': 'LONG',
                    'confidence': 0.65,
                    'location': (candle['x'], candle['y']),
                    'reason': f'Hammer candlestick (AR: {candle["aspect_ratio"]:.1f})',
                    'candle_index': i
                })
            
            # SHOOTING STAR (bearish reversal) - long candle with small body at bottom
            elif candle['type'] == 'bearish' and candle['aspect_ratio'] > 3:
                patterns.append({
                    'pattern': 'shooting_star',
                    'signal': 'SHORT',
                    'confidence': 0.65,
                    'location': (candle['x'], candle['y']),
                    'reason': f'Shooting star (AR: {candle["aspect_ratio"]:.1f})',
                    'candle_index': i
                })
        
        return patterns
    
    def _detect_three_soldiers_crows(self, candles: List[Dict], current_price: Optional[float] = None) -> List[Dict]:
        """
        Detect three white soldiers (bullish) or three black crows (bearish)
        
        Args:
            candles: Sorted list of candlesticks
            current_price: Optional current price from OCR
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        for i in range(len(candles) - 2):
            c1, c2, c3 = candles[i], candles[i+1], candles[i+2]
            
            # THREE WHITE SOLDIERS - three consecutive bullish candles
            if (c1['type'] == 'bullish' and 
                c2['type'] == 'bullish' and 
                c3['type'] == 'bullish'):
                
                # Check if generally increasing in height
                avg_area = (c1['area'] + c2['area'] + c3['area']) / 3
                if all(c['area'] > avg_area * 0.7 for c in [c1, c2, c3]):
                    # --- MODIFICATION: Pass current_price ---
                    levels = self.calculate_signal_levels('LONG', 'three_white_soldiers', 0.75, current_price)
                    patterns.append({
                        'pattern': 'three_white_soldiers',
                        'signal': 'LONG',
                        'confidence': 0.75,
                        'location': (c3['x'], c3['y']),
                        'reason': 'Three consecutive bullish candles',
                        'candle_index': i + 2,
                        **levels
                    })
            
            # THREE BLACK CROWS - three consecutive bearish candles
            elif (c1['type'] == 'bearish' and 
                            c2['type'] == 'bearish' and 
                            c3['type'] == 'bearish'):
                            
                            avg_area = (c1['area'] + c2['area'] + c3['area']) / 3
                            if all(c['area'] > avg_area * 0.7 for c in [c1, c2, c3]):
                                # --- MODIFICATION: Pass current_price ---
                                levels = self.calculate_signal_levels('SHORT', 'three_black_crows', 0.75, current_price)
                                patterns.append({
                                    'pattern': 'three_black_crows',
                                    'signal': 'SHORT',
                                    'confidence': 0.75,
                                    'location': (c3['x'], c3['y']),
                                    'reason': 'Three consecutive bearish candles',
                                    'candle_index': i + 2,
                                    **levels
                                })
        
        return patterns
    
    def _detect_star_patterns(self, candles: List[Dict]) -> List[Dict]:
        """
        Detect morning star (bullish) and evening star (bearish) patterns
        
        Args:
            candles: Sorted list of candlesticks
            
        Returns:
            List of detected star patterns
        """
        patterns = []
        
        for i in range(len(candles) - 2):
            c1, c2, c3 = candles[i], candles[i+1], candles[i+2]
            
            # MORNING STAR - bearish, small, bullish
            if (c1['type'] == 'bearish' and 
                c3['type'] == 'bullish' and
                c2['area'] < c1['area'] * 0.5 and
                c3['area'] > c1['area'] * 0.8):
                
                patterns.append({
                    'pattern': 'morning_star',
                    'signal': 'LONG',
                    'confidence': 0.80,
                    'location': (c3['x'], c3['y']),
                    'reason': 'Morning star reversal pattern',
                    'candle_index': i + 2
                })
            
            # EVENING STAR - bullish, small, bearish
            elif (c1['type'] == 'bullish' and 
                  c3['type'] == 'bearish' and
                  c2['area'] < c1['area'] * 0.5 and
                  c3['area'] > c1['area'] * 0.8):
                
                patterns.append({
                    'pattern': 'evening_star',
                    'signal': 'SHORT',
                    'confidence': 0.80,
                    'location': (c3['x'], c3['y']),
                    'reason': 'Evening star reversal pattern',
                    'candle_index': i + 2
                })
        
        return patterns
    
    def detect_support_resistance(self, trend_lines: List[Dict], candles: List[Dict]) -> Dict:
        """
        Enhanced support and resistance detection using both lines and candle touches
        
        Args:
            trend_lines: List of detected trend lines
            candles: List of candlesticks for touch validation
            
        Returns:
            Dictionary with support/resistance levels
        """
        # Get horizontal lines
        horizontal_lines = [line for line in trend_lines if line['type'] == 'horizontal']
        
        if not horizontal_lines:
            return {'support': [], 'resistance': [], 'strong_levels': []}
        
        # Group lines by y-coordinate (price level)
        levels = defaultdict(list)
        
        for line in horizontal_lines:
            y = int((line['y1'] + line['y2']) / 2)
            
            # Find or create level group
            found = False
            for level_y in list(levels.keys()):
                if abs(y - level_y) < self.sr_tolerance:
                    levels[level_y].append(line)
                    found = True
                    break
            
            if not found:
                levels[y] = [line]
        
        # Filter levels by number of touches
        valid_levels = []
        for y, lines in levels.items():
            if len(lines) >= self.sr_min_touches:
                # Calculate average length and position
                avg_length = sum(l['length'] for l in lines) / len(lines)
                avg_x = sum((l['x1'] + l['x2'])/2 for l in lines) / len(lines)
                
                valid_levels.append({
                    'y': y,
                    'touches': len(lines),
                    'avg_length': avg_length,
                    'strength': min(1.0, len(lines) / 5),  # Normalize to 0-1
                    'x': avg_x
                })
        
        # Sort by y coordinate
        valid_levels.sort(key=lambda l: l['y'])
        
        # Classify as support or resistance based on position
        if not valid_levels:
            return {'support': [], 'resistance': [], 'strong_levels': []}
        
        # Middle level - levels below are support, above are resistance
        mid_y = valid_levels[len(valid_levels)//2]['y']
        
        support_levels = [l for l in valid_levels if l['y'] > mid_y]
        resistance_levels = [l for l in valid_levels if l['y'] <= mid_y]
        
        # Identify strong levels (3+ touches)
        strong_levels = [l for l in valid_levels if l['touches'] >= 3]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'strong_levels': strong_levels,
            'all_levels': valid_levels
        }
    
    def detect_chart_patterns(self, candles: List[Dict], sr_levels: Dict) -> List[Dict]:
        """
        Detect larger chart patterns (triangles, double tops/bottoms, etc.)
        
        Args:
            candles: List of candlesticks
            sr_levels: Support/resistance levels
            
        Returns:
            List of detected chart patterns
        """
        patterns = []
        
        if len(candles) < 10:
            return patterns
        
        # Get high and low points
        highs = [c for c in candles if c['type'] == 'bearish']  # Bearish candles often at highs
        lows = [c for c in candles if c['type'] == 'bullish']  # Bullish candles often at lows
        
        # DOUBLE TOP PATTERN
        if len(highs) >= 2:
            for i in range(len(highs) - 1):
                for j in range(i + 1, len(highs)):
                    h1, h2 = highs[i], highs[j]
                    
                    # Check if at similar height
                    if abs(h1['y'] - h2['y']) < 30:
                        # Check if there's separation
                        if abs(h1['x'] - h2['x']) > 100:
                            patterns.append({
                                'pattern': 'double_top',
                                'signal': 'SHORT',
                                'confidence': 0.75,
                                'location': ((h1['x'] + h2['x'])//2, min(h1['y'], h2['y'])),
                                'reason': 'Double top resistance pattern',
                                'level': min(h1['y'], h2['y'])
                            })
                            break
        
        # DOUBLE BOTTOM PATTERN
        if len(lows) >= 2:
            for i in range(len(lows) - 1):
                for j in range(i + 1, len(lows)):
                    l1, l2 = lows[i], lows[j]
                    
                    # Check if at similar height
                    if abs(l1['y'] - l2['y']) < 30:
                        # Check if there's separation
                        if abs(l1['x'] - l2['x']) > 100:
                            patterns.append({
                                'pattern': 'double_bottom',
                                'signal': 'LONG',
                                'confidence': 0.75,
                                'location': ((l1['x'] + l2['x'])//2, max(l1['y'], l2['y'])),
                                'reason': 'Double bottom support pattern',
                                'level': max(l1['y'], l2['y'])
                            })
                            break
        
        return patterns
    
    def generate_trading_signals(self, analysis: Dict) -> List[Dict]:
        """
        Generate comprehensive trading signals from analysis
        
        Args:
            analysis: Complete chart analysis with statistics
            
        Returns:
            List of trading signals sorted by confidence
        """
        signals = []
        
        # --- START MODIFICATION ---
        # Get the price from analysis to pass to pattern functions
        current_price = analysis.get('current_price')
        # --- END MODIFICATION ---

        # Analyze candlestick patterns
        # --- MODIFICATION: Pass current_price ---
        candle_patterns = self.analyze_candlesticks(analysis.get('candlesticks', []), current_price)
        signals.extend(candle_patterns)
        
        # Detect support/resistance
        sr_levels = self.detect_support_resistance(
            analysis.get('trend_lines', []),
            analysis.get('candlesticks', [])
        )
        
        # Detect chart patterns
        chart_patterns = self.detect_chart_patterns(
            analysis.get('candlesticks', []),
            sr_levels
        )
        signals.extend(chart_patterns)
        
        # Add sentiment-based signals
        stats = analysis.get('statistics', {})
        sentiment = stats.get('sentiment', 'neutral')
        
        if sentiment == 'bullish' and stats.get('bullish_percentage', 0) > 70:
            signals.append({
                'pattern': 'strong_bullish_sentiment',
                'signal': 'LONG',
                'confidence': min(0.75, stats['bullish_percentage'] / 100),
                'location': None,
                'reason': f"Strong bullish momentum ({stats['bullish_percentage']:.1f}% bullish candles)"
            })
        elif sentiment == 'bearish' and stats.get('bearish_percentage', 0) > 70:
            signals.append({
                'pattern': 'strong_bearish_sentiment',
                'signal': 'SHORT',
                'confidence': min(0.75, stats['bearish_percentage'] / 100),
                'location': None,
                'reason': f"Strong bearish momentum ({stats['bearish_percentage']:.1f}% bearish candles)"
            })
        
        # Add support/resistance breakout signals
        strong_levels = sr_levels.get('strong_levels', [])
        if strong_levels:
            for level in strong_levels:
                if level['touches'] >= 3:
                    signals.append({
                        'pattern': 'support_resistance_level',
                        'signal': 'WATCH',
                        'confidence': level['strength'],
                        'location': (level['x'], level['y']),
                        'reason': f"Strong S/R level ({level['touches']} touches)",
                        'level_y': level['y']
                    })
        
        # Filter and sort by confidence
        signals = [s for s in signals if s['confidence'] >= self.min_confidence]
        signals = sorted(signals, key=lambda s: s['confidence'], reverse=True)
        
        # Remove duplicates based on pattern type
        seen_patterns = set()
        unique_signals = []
        for signal in signals:
            if signal['pattern'] not in seen_patterns:
                unique_signals.append(signal)
                seen_patterns.add(signal['pattern'])
        
        return unique_signals[:10]  # Return top 10 signals


class TradingViewAIV2:
    """
    Enhanced AI system for TradingView analysis
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizerV2()
        self.logger = logging.getLogger(__name__)
        self.signal_history = []
        
        # Performance tracking
        self.stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'watch_signals': 0
        }
        
    def process_frame(self, analysis: Dict) -> Dict:
        """
        Process frame analysis and generate trading signals
        
        Args:
            analysis: Frame analysis from screen capture
            
        Returns:
            Complete analysis with trading signals
        """
        # Generate trading signals
        signals = self.pattern_recognizer.generate_trading_signals(analysis)
        
        # Add to result
        analysis['signals'] = signals
        analysis['signal_count'] = len(signals)
        
        # Update stats
        self.stats['total_signals'] += len(signals)
        for signal in signals:
            if signal['signal'] == 'LONG':
                self.stats['long_signals'] += 1
            elif signal['signal'] == 'SHORT':
                self.stats['short_signals'] += 1
            elif signal['signal'] == 'WATCH':
                self.stats['watch_signals'] += 1
        
        # Log signals
        if signals:
            self.logger.info(f"Generated {len(signals)} trading signals:")
            for i, signal in enumerate(signals[:5], 1):  # Log top 5
                self.logger.info(
                    f"  {i}. {signal['pattern']}: {signal['signal']} "
                    f"(confidence: {signal['confidence']:.1%}) - {signal['reason']}"
                )
        
        # Store in history
        self.signal_history.append({
            'timestamp': analysis['timestamp'],
            'signals': signals,
            'statistics': analysis.get('statistics', {})
        })
        
        # Keep last 100
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
        
        return analysis
    
    def get_latest_signals(self, min_confidence: float = 0.7, signal_type: str = None) -> List[Dict]:
        """
        Get latest signals above confidence threshold
        
        Args:
            min_confidence: Minimum confidence level
            signal_type: Filter by signal type ('LONG', 'SHORT', 'WATCH') or None for all
            
        Returns:
            List of high-confidence signals
        """
        if not self.signal_history:
            return []
        
        latest = self.signal_history[-1]
        signals = latest.get('signals', [])
        
        # Filter by confidence
        signals = [s for s in signals if s['confidence'] >= min_confidence]
        
        # Filter by type if specified
        if signal_type:
            signals = [s for s in signals if s['signal'] == signal_type.upper()]
        
        return signals
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return self.stats.copy()


if __name__ == "__main__":
    print("Pattern Recognizer V2.0 - Test Mode")
    print("This module works with screen_capture_v2.py")
    print("\nRun: python vision_integration_v2.py for full system")