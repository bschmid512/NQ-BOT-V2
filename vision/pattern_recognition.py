"""
AI Pattern Recognition for TradingView
Detects chart patterns, setups, and generates trading signals
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging


class PatternRecognizer:
    """
    AI-based pattern recognition for trading charts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern templates (will be improved with ML)
        self.patterns = {
            'double_bottom': self._detect_double_bottom,
            'double_top': self._detect_double_top,
            'head_and_shoulders': self._detect_head_shoulders,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'engulfing': self._detect_engulfing
        }
    
    def analyze_candlesticks(self, candlesticks: List[Dict]) -> List[Dict]:
        """
        Analyze candlestick patterns
        
        Args:
            candlesticks: List of detected candlesticks
            
        Returns:
            List of detected patterns
        """
        if not candlesticks or len(candlesticks) < 2:
            return []
        
        patterns_found = []
        
        # Sort by x position (time)
        sorted_candles = sorted(candlesticks, key=lambda c: c['x'])
        
        # Check for bullish engulfing
        for i in range(len(sorted_candles) - 1):
            current = sorted_candles[i]
            next_candle = sorted_candles[i + 1]
            
            # Bullish engulfing: small red followed by large green
            if (current['type'] == 'bearish' and 
                next_candle['type'] == 'bullish' and
                next_candle['height'] > current['height'] * 1.5):
                
                patterns_found.append({
                    'pattern': 'bullish_engulfing',
                    'signal': 'LONG',
                    'confidence': 0.7,
                    'location': (next_candle['x'], next_candle['y']),
                    'reason': 'Bullish engulfing pattern detected'
                })
            
            # Bearish engulfing: small green followed by large red
            elif (current['type'] == 'bullish' and 
                  next_candle['type'] == 'bearish' and
                  next_candle['height'] > current['height'] * 1.5):
                
                patterns_found.append({
                    'pattern': 'bearish_engulfing',
                    'signal': 'SHORT',
                    'confidence': 0.7,
                    'location': (next_candle['x'], next_candle['y']),
                    'reason': 'Bearish engulfing pattern detected'
                })
        
        # Check for hammer/shooting star
        for candle in sorted_candles:
            # Hammer (bullish): long lower wick
            if candle['type'] == 'bullish' and candle['height'] > 20:
                # This is simplified - would need more detailed analysis
                patterns_found.append({
                    'pattern': 'potential_hammer',
                    'signal': 'LONG',
                    'confidence': 0.5,
                    'location': (candle['x'], candle['y']),
                    'reason': 'Potential hammer candlestick'
                })
        
        return patterns_found
    
    def detect_support_resistance(self, trend_lines: List[Dict], img_height: int) -> Dict:
        """
        Identify support and resistance levels
        
        Args:
            trend_lines: List of detected lines
            img_height: Image height for coordinate mapping
            
        Returns:
            Dictionary with support/resistance levels
        """
        horizontal_lines = [line for line in trend_lines if line['type'] == 'horizontal']
        
        if not horizontal_lines:
            return {'support': [], 'resistance': []}
        
        # Group lines by y-coordinate (price level)
        levels = {}
        tolerance = 20  # pixels
        
        for line in horizontal_lines:
            y = (line['y1'] + line['y2']) // 2
            
            # Find or create level group
            found_group = False
            for level_y in list(levels.keys()):
                if abs(y - level_y) < tolerance:
                    levels[level_y].append(line)
                    found_group = True
                    break
            
            if not found_group:
                levels[y] = [line]
        
        # Sort by strength (number of touches)
        sorted_levels = sorted(levels.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Determine support (lower) vs resistance (upper)
        mid_point = img_height // 2
        
        support = [{'y': y, 'strength': len(lines)} 
                  for y, lines in sorted_levels if y > mid_point]
        resistance = [{'y': y, 'strength': len(lines)} 
                     for y, lines in sorted_levels if y <= mid_point]
        
        return {
            'support': support[:3],  # Top 3 support levels
            'resistance': resistance[:3]  # Top 3 resistance levels
        }
    
    def detect_breakout(self, current_price_y: int, support_resistance: Dict) -> Optional[Dict]:
        """
        Detect potential breakouts
        
        Args:
            current_price_y: Current price y-coordinate
            support_resistance: Support/resistance levels
            
        Returns:
            Breakout signal or None
        """
        # Check resistance breakout
        for res in support_resistance.get('resistance', []):
            if current_price_y < res['y'] - 10:  # Price moved above resistance
                return {
                    'pattern': 'resistance_breakout',
                    'signal': 'LONG',
                    'confidence': 0.8,
                    'reason': f'Price broke above resistance at y={res["y"]}'
                }
        
        # Check support breakdown
        for sup in support_resistance.get('support', []):
            if current_price_y > sup['y'] + 10:  # Price moved below support
                return {
                    'pattern': 'support_breakdown',
                    'signal': 'SHORT',
                    'confidence': 0.8,
                    'reason': f'Price broke below support at y={sup["y"]}'
                }
        
        return None
    
    def _detect_double_bottom(self, candlesticks: List[Dict]) -> Optional[Dict]:
        """Detect double bottom pattern"""
        # Simplified implementation
        # Would need more sophisticated analysis in production
        return None
    
    def _detect_double_top(self, candlesticks: List[Dict]) -> Optional[Dict]:
        """Detect double top pattern"""
        return None
    
    def _detect_head_shoulders(self, candlesticks: List[Dict]) -> Optional[Dict]:
        """Detect head and shoulders pattern"""
        return None
    
    def _detect_triangle(self, trend_lines: List[Dict]) -> Optional[Dict]:
        """Detect triangle pattern"""
        return None
    
    def _detect_flag(self, candlesticks: List[Dict]) -> Optional[Dict]:
        """Detect flag/pennant pattern"""
        return None
    
    def _detect_engulfing(self, candlesticks: List[Dict]) -> Optional[Dict]:
        """Detect engulfing patterns"""
        return None
    
    def generate_trading_signals(self, analysis: Dict) -> List[Dict]:
        """
        Generate trading signals from analysis
        
        Args:
            analysis: Complete chart analysis
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Analyze candlestick patterns
        candle_patterns = self.analyze_candlesticks(analysis.get('candlesticks', []))
        signals.extend(candle_patterns)
        
        # Detect support/resistance
        sr_levels = self.detect_support_resistance(
            analysis.get('trend_lines', []),
            1080  # Assume 1080p, adjust as needed
        )
        
        # Check for breakouts (would need current price)
        # breakout_signal = self.detect_breakout(current_price_y, sr_levels)
        # if breakout_signal:
        #     signals.append(breakout_signal)
        
        # Combine with sentiment analysis
        sentiment = analysis.get('sentiment', 'neutral')
        
        if sentiment == 'bullish' and not any(s['signal'] == 'SHORT' for s in signals):
            signals.append({
                'pattern': 'bullish_sentiment',
                'signal': 'LONG',
                'confidence': 0.6,
                'reason': f'Strong bullish sentiment: {analysis.get("bullish_count", 0)} bullish candles'
            })
        elif sentiment == 'bearish' and not any(s['signal'] == 'LONG' for s in signals):
            signals.append({
                'pattern': 'bearish_sentiment',
                'signal': 'SHORT',
                'confidence': 0.6,
                'reason': f'Strong bearish sentiment: {analysis.get("bearish_count", 0)} bearish candles'
            })
        
        # Filter and rank by confidence
        signals = sorted(signals, key=lambda s: s['confidence'], reverse=True)
        
        return signals[:5]  # Return top 5 signals


class TradingViewAI:
    """
    Main AI system for TradingView analysis
    Combines screen capture, pattern recognition, and signal generation
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.logger = logging.getLogger(__name__)
        self.signal_history = []
        
    def process_frame(self, analysis: Dict) -> Dict:
        """
        Process a frame analysis and generate trading signals
        
        Args:
            analysis: Frame analysis from screen capture
            
        Returns:
            Complete analysis with trading signals
        """
        # Generate trading signals
        signals = self.pattern_recognizer.generate_trading_signals(analysis)
        
        # Add to result
        analysis['signals'] = signals
        
        # Log signals
        if signals:
            self.logger.info(f"Generated {len(signals)} trading signals")
            for signal in signals:
                self.logger.info(
                    f"  {signal['pattern']}: {signal['signal']} "
                    f"(confidence: {signal['confidence']:.1%}) - {signal['reason']}"
                )
        
        # Store in history
        self.signal_history.append({
            'timestamp': analysis['timestamp'],
            'signals': signals
        })
        
        # Keep last 100
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
        
        return analysis
    
    def get_latest_signals(self, min_confidence: float = 0.6) -> List[Dict]:
        """
        Get latest signals above confidence threshold
        
        Args:
            min_confidence: Minimum confidence level
            
        Returns:
            List of high-confidence signals
        """
        if not self.signal_history:
            return []
        
        latest = self.signal_history[-1]
        return [s for s in latest['signals'] if s['confidence'] >= min_confidence]
    
    def should_take_trade(self, min_confidence: float = 0.7) -> Tuple[bool, Optional[Dict]]:
        """
        Determine if a trade should be taken
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            (should_trade, best_signal)
        """
        signals = self.get_latest_signals(min_confidence)
        
        if not signals:
            return False, None
        
        # Take the highest confidence signal
        best_signal = max(signals, key=lambda s: s['confidence'])
        
        return True, best_signal


# Test function
if __name__ == "__main__":
    print("AI Pattern Recognition Test")
    print("=" * 60)
    
    # Create test data
    test_candlesticks = [
        {'type': 'bearish', 'x': 100, 'y': 200, 'width': 10, 'height': 30, 'area': 300},
        {'type': 'bullish', 'x': 120, 'y': 180, 'width': 10, 'height': 50, 'area': 500},
        {'type': 'bullish', 'x': 140, 'y': 170, 'width': 10, 'height': 40, 'area': 400},
    ]
    
    test_analysis = {
        'timestamp': datetime.now().isoformat(),
        'candlesticks': test_candlesticks,
        'trend_lines': [],
        'bullish_count': 2,
        'bearish_count': 1,
        'sentiment': 'bullish'
    }
    
    # Test pattern recognition
    ai = TradingViewAI()
    result = ai.process_frame(test_analysis)
    
    print("\nDetected Patterns:")
    for signal in result['signals']:
        print(f"  Pattern: {signal['pattern']}")
        print(f"  Signal: {signal['signal']}")
        print(f"  Confidence: {signal['confidence']:.1%}")
        print(f"  Reason: {signal['reason']}")
        print()
    
    # Test trading decision
    should_trade, best_signal = ai.should_take_trade(min_confidence=0.6)
    
    if should_trade:
        print(f"TRADE RECOMMENDED: {best_signal['signal']}")
        print(f"Pattern: {best_signal['pattern']}")
        print(f"Confidence: {best_signal['confidence']:.1%}")
    else:
        print("No trade recommended at this time")
