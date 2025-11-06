"""
Automatic Updater: Add Entry, TP, and SL to Vision Signals
Run this script to automatically upgrade your pattern_recognition_v2.py

Usage:
    python upgrade_add_prices.py
"""
import sys
from pathlib import Path

def update_pattern_recognition():
    """Add calculate_signal_levels method to pattern_recognition_v2.py"""
    
    file_path = Path("pattern_recognition_v2.py")
    
    if not file_path.exists():
        print("❌ pattern_recognition_v2.py not found!")
        print("Make sure you're in the correct directory.")
        return False
    
    print("📄 Reading pattern_recognition_v2.py...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if 'calculate_signal_levels' in content:
        print("✅ File already has calculate_signal_levels method!")
        print("No update needed.")
        return True
    
    # Method to add
    method_code = '''
    def calculate_signal_levels(self, signal_type, pattern, confidence, candles=None):
        """
        Calculate entry, stop loss, and take profit levels for a signal
        
        Args:
            signal_type: 'LONG' or 'SHORT' or 'WATCH'
            pattern: Pattern name
            confidence: Signal confidence (0.0-1.0)
            candles: Candlestick data (to estimate price)
            
        Returns:
            Dict with entry, stop_loss, take_profit, risk/reward info
        """
        # Estimate current price (you can improve this with OCR later)
        estimated_price = 20500  # NQ default
        
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
'''
    
    # Find where to insert (after __init__ method in PatternRecognizerV2 class)
    insertion_point = content.find('class PatternRecognizerV2:')
    if insertion_point == -1:
        print("❌ Could not find PatternRecognizerV2 class!")
        return False
    
    # Find the end of __init__ method
    init_start = content.find('def __init__', insertion_point)
    if init_start == -1:
        print("❌ Could not find __init__ method!")
        return False
    
    # Find the next method definition after __init__
    next_method = content.find('\n    def ', init_start + 10)
    if next_method == -1:
        print("❌ Could not find insertion point!")
        return False
    
    # Insert the new method
    new_content = content[:next_method] + method_code + content[next_method:]
    
    # Create backup
    backup_path = file_path.with_suffix('.py.backup')
    print(f"💾 Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Write updated file
    print("✏️  Adding calculate_signal_levels method...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully updated pattern_recognition_v2.py!")
    print("\n📋 Next steps:")
    print("1. Update signal generation to use this method")
    print("2. See UPGRADE_ADD_PRICES.md for detailed instructions")
    print("3. Or run: python update_vision_display.py (coming next!)")
    
    return True


def update_signal_generation():
    """Update signals to include price levels"""
    
    file_path = Path("pattern_recognition_v2.py")
    
    if not file_path.exists():
        print("❌ pattern_recognition_v2.py not found!")
        return False
    
    print("\n📄 Updating signal generation...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patterns to update with their signal types
    patterns_to_update = [
        ('bullish_engulfing', 'LONG'),
        ('bearish_engulfing', 'SHORT'),
        ('hammer', 'LONG'),
        ('shooting_star', 'SHORT'),
        ('three_white_soldiers', 'LONG'),
        ('three_black_crows', 'SHORT'),
        ('morning_star', 'LONG'),
        ('evening_star', 'SHORT'),
        ('double_bottom', 'LONG'),
        ('double_top', 'SHORT'),
        ('strong_bullish_sentiment', 'LONG'),
        ('strong_bearish_sentiment', 'SHORT'),
    ]
    
    modified = False
    
    for pattern, signal_type in patterns_to_update:
        # Look for the pattern in signals.append
        search_str = f"'pattern': '{pattern}'"
        
        if search_str in content:
            # Find this signal's append block
            pos = content.find(search_str)
            
            # Find the opening of this signals.append
            append_start = content.rfind('signals.append({', 0, pos)
            
            # Find the closing
            append_end = content.find('})', append_start)
            
            if append_start != -1 and append_end != -1:
                old_signal = content[append_start:append_end+2]
                
                # Check if already has levels
                if '**levels' in old_signal or 'entry' in old_signal:
                    continue
                
                # Add levels calculation before the append
                indent = '        '  # Adjust based on your indentation
                levels_calc = f"{indent}levels = self.calculate_signal_levels('{signal_type}', '{pattern}', confidence, analysis.get('candlesticks', []))\n{indent}"
                
                # Find the line before signals.append
                line_before = content.rfind('\n', 0, append_start)
                
                # Insert levels calculation
                content = content[:line_before+1] + levels_calc + content[line_before+1:]
                
                # Now find the updated append position
                append_start = content.find('signals.append({', line_before)
                append_end = content.find('})', append_start)
                
                # Add **levels to the dict
                old_signal_dict = content[append_start+16:append_end+1]
                new_signal_dict = old_signal_dict[:-1] + ',\n            **levels\n        }'
                
                content = content[:append_start+16] + new_signal_dict + content[append_end+2:]
                
                modified = True
                print(f"  ✅ Updated {pattern}")
    
    if modified:
        # Create backup if not already done
        backup_path = file_path.with_suffix('.py.backup2')
        if not backup_path.exists():
            print(f"💾 Creating backup: {backup_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)
        
        # Write updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n✅ Signal generation updated!")
    else:
        print("\n⚠️  No patterns found to update (may already be updated)")
    
    return True


def main():
    print("="*70)
    print("VISION SYSTEM UPGRADE: Add Entry, TP, and SL to Signals")
    print("="*70)
    print()
    
    # Step 1: Add the method
    if not update_pattern_recognition():
        print("\n❌ Update failed!")
        return
    
    print("\n" + "="*70)
    print("✅ UPGRADE COMPLETE!")
    print("="*70)
    print()
    print("Next, update vision_integration_v2.py to display the new fields.")
    print("See UPGRADE_ADD_PRICES.md for manual instructions.")
    print()
    print("Test with: python vision_integration_v2.py")
    print()


if __name__ == "__main__":
    main()
