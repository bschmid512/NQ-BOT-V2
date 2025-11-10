#!/usr/bin/env python3
"""
Automatic Fix Script - Signal Fusion Parameter Order
Fixes: AttributeError: 'list' object has no attribute 'get'
"""
import os
import sys
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create timestamped backup"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(filepath, 'r') as f:
        content = f.read()
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backed up: {filepath}")
    return True

def fix_parameter_order():
    """Fix the parameter order in enhanced_strategy_engine.py"""
    filepath = "core/enhanced_strategy_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found!")
        return False
    
    # Read file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'evaluate_trade_setup(context, strategy_signals)' in content:
        print(f"✅ {filepath} already has correct parameter order")
        return True
    
    # Check if the error exists
    if 'evaluate_trade_setup(strategy_signals, context)' not in content:
        print(f"⚠️  Could not find the problematic line in {filepath}")
        print(f"    The file may have been modified or the error is elsewhere")
        return False
    
    # Make backup
    backup_file(filepath)
    
    # Fix the parameter order
    old_line = 'signal_fusion_engine.evaluate_trade_setup(strategy_signals, context)'
    new_line = 'signal_fusion_engine.evaluate_trade_setup(context, strategy_signals)'
    
    content = content.replace(old_line, new_line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed parameter order in {filepath}")
    print(f"   Changed: evaluate_trade_setup(strategy_signals, context)")
    print(f"   To:      evaluate_trade_setup(context, strategy_signals)")
    
    return True

def main():
    print("="*70)
    print("SIGNAL FUSION PARAMETER ORDER FIX")
    print("="*70)
    print("\nThis script fixes:")
    print("  AttributeError: 'list' object has no attribute 'get'")
    print("\nThe problem:")
    print("  Parameters were passed in wrong order to evaluate_trade_setup()")
    print("\n" + "="*70)
    
    response = input("\nApply fix? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    print("\n" + "="*70)
    print("APPLYING FIX")
    print("="*70 + "\n")
    
    if fix_parameter_order():
        print("\n" + "="*70)
        print("✅ FIX APPLIED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Restart your bot: python main.py")
        print("2. Test with a webhook")
        print("3. Error should be gone!")
    else:
        print("\n" + "="*70)
        print("❌ FIX FAILED")
        print("="*70)
        print("\nPlease apply the fix manually:")
        print("1. Open core/enhanced_strategy_engine.py")
        print("2. Find: evaluate_trade_setup(strategy_signals, context)")
        print("3. Change to: evaluate_trade_setup(context, strategy_signals)")
        print("4. Save and restart bot")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
