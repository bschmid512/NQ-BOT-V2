#!/usr/bin/env python3
"""
Comprehensive Fix Script - All Signal Fusion Errors
Fixes:
1. Parameter order (list/dict swap)
2. Price key name (price -> current_price)
3. None safety checks
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

def fix_enhanced_strategy_engine():
    """Fix parameter order in enhanced_strategy_engine.py"""
    filepath = "core/enhanced_strategy_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found!")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'evaluate_trade_setup(context, strategy_signals)' in content:
        print(f"✅ {filepath} - parameter order already correct")
        return True
    
    # Make backup
    backup_file(filepath)
    
    # Fix parameter order
    old = 'signal_fusion_engine.evaluate_trade_setup(strategy_signals, context)'
    new = 'signal_fusion_engine.evaluate_trade_setup(context, strategy_signals)'
    
    if old in content:
        content = content.replace(old, new)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Fixed parameter order in {filepath}")
        return True
    else:
        print(f"⚠️  Could not find problematic line in {filepath}")
        return False

def fix_signal_fusion_engine():
    """Fix price key and add None checks in signal_fusion_engine.py"""
    filepath = "core/signal_fusion_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found!")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    modified = False
    
    # Fix 1: Change price = market_context.get("price") to current_price
    old_price_line = 'price = market_context.get("price")'
    new_price_line = 'price = market_context.get("current_price")'
    
    if old_price_line in content:
        if new_price_line not in content:
            backup_file(filepath)
            content = content.replace(old_price_line, new_price_line)
            print(f"✅ Fixed price key: 'price' -> 'current_price'")
            modified = True
        else:
            print(f"✅ Price key already correct")
    
    # Fix 2: Add None check before using price
    # Look for the line that creates fused_result
    if 'if price is None:' not in content:
        # Find where to insert the safety check
        search_for = '# --- 2) preliminary trade object'
        
        if search_for in content:
            safety_check = '''
    # --- 2) Safety check - price must exist
    if price is None:
        self._reject("price", "current_price not in market_context")
        return None

    # --- 3) preliminary trade object'''
            
            content = content.replace(search_for, safety_check)
            print(f"✅ Added None safety check for price")
            modified = True
        else:
            print(f"⚠️  Could not find insertion point for safety check")
    else:
        print(f"✅ None safety check already present")
    
    if modified:
        # Also need to update the comment numbers
        content = content.replace('# --- 3) normalize keys', '# --- 4) normalize keys')
        content = content.replace('# --- 4) optional one-off', '# --- 5) optional one-off')
        content = content.replace('# --- 5) log & cooldown', '# --- 6) log & cooldown')
        
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ All fixes applied to {filepath}")
        return True
    
    return True

def verify_fixes():
    """Verify that fixes were applied correctly"""
    print("\n" + "="*70)
    print("VERIFYING FIXES")
    print("="*70 + "\n")
    
    success = True
    
    # Check enhanced_strategy_engine
    if os.path.exists("core/enhanced_strategy_engine.py"):
        with open("core/enhanced_strategy_engine.py", 'r') as f:
            content = f.read()
        
        if 'evaluate_trade_setup(context, strategy_signals)' in content:
            print("✅ Parameter order correct in enhanced_strategy_engine.py")
        else:
            print("❌ Parameter order still wrong in enhanced_strategy_engine.py")
            success = False
    
    # Check signal_fusion_engine
    if os.path.exists("core/signal_fusion_engine.py"):
        with open("core/signal_fusion_engine.py", 'r') as f:
            content = f.read()
        
        if 'price = market_context.get("current_price")' in content:
            print("✅ Price key correct in signal_fusion_engine.py")
        else:
            print("❌ Price key still wrong in signal_fusion_engine.py")
            success = False
        
        if 'if price is None:' in content:
            print("✅ None check present in signal_fusion_engine.py")
        else:
            print("⚠️  None check not found in signal_fusion_engine.py")
    
    return success

def main():
    print("="*70)
    print("COMPREHENSIVE SIGNAL FUSION FIX")
    print("="*70)
    print("\nThis script will fix:")
    print("1. ❌ AttributeError: 'list' object has no attribute 'get'")
    print("   → Fix: Swap parameter order in evaluate_trade_setup()")
    print()
    print("2. ❌ TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'")
    print("   → Fix: Change 'price' to 'current_price' key")
    print("   → Fix: Add None safety check")
    print("\n" + "="*70)
    
    response = input("\nApply all fixes? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    print("\n" + "="*70)
    print("APPLYING FIXES")
    print("="*70 + "\n")
    
    success = True
    
    # Fix 1: Enhanced strategy engine
    print("Fix 1: Parameter order in enhanced_strategy_engine.py")
    if not fix_enhanced_strategy_engine():
        success = False
    print()
    
    # Fix 2: Signal fusion engine
    print("Fix 2: Price key and None checks in signal_fusion_engine.py")
    if not fix_signal_fusion_engine():
        success = False
    print()
    
    # Verify
    verify_fixes()
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Restart your bot:")
        print("   python main.py")
        print()
        print("2. Test with webhook:")
        print('   curl -X POST http://localhost:8050/webhook \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"passphrase":"change_this_secure_passphrase","close":25709,"open":25705,"high":25715,"low":25700,"volume":1000}\'')
        print()
        print("3. Should work without errors! ✅")
    else:
        print("⚠️  SOME FIXES MAY HAVE FAILED")
        print("="*70)
        print("\nPlease check the messages above for details")
        print("You may need to apply some fixes manually")
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
