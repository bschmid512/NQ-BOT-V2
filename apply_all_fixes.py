# NQ TRADING BOT - COMPREHENSIVE FIX BUNDLE
# Run this script to apply all critical fixes

import os
import shutil
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modifying"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up: {filepath} → {backup_path}")
        return True
    return False

def fix_position_manager():
    """Fix hardcoded commission in position_manager.py"""
    filepath = "core/position_manager.py"
    
    if not os.path.exists(filepath):
        print(f"⚠️  {filepath} not found - skipping")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'from config import COMMISSION_PER_SIDE' in content:
        print(f"✅ {filepath} already fixed")
        return
    
    # Add import at top
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('from config import'):
            # Add to existing config import
            if 'COMMISSION_PER_SIDE' not in line:
                lines[i] = line.rstrip(')') + ', COMMISSION_PER_SIDE, CONTRACT_SIZE)'
            break
        elif line.startswith('from datetime import'):
            # Insert new import after datetime
            lines.insert(i+1, 'from config import COMMISSION_PER_SIDE, CONTRACT_SIZE, MAX_POSITION_SIZE')
            break
    
    content = '\n'.join(lines)
    
    # Fix hardcoded commission
    content = content.replace(
        'self.pnl -= 5.0  # Hardcoded commission',
        'total_commission = COMMISSION_PER_SIDE * 2 * self.size  # Both sides\n        self.pnl -= total_commission'
    )
    
    content = content.replace(
        'self.pnl -= 5',
        'total_commission = COMMISSION_PER_SIDE * 2 * self.size\n        self.pnl -= total_commission'
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed: {filepath}")

def check_files_exist():
    """Check if critical files exist"""
    print("\n" + "="*60)
    print("CHECKING FILE STRUCTURE")
    print("="*60 + "\n")
    
    critical_files = [
        "main.py",
        "config.py",
        "core/enhanced_strategy_engine.py",
        "core/signal_fusion_engine.py",
        "core/position_manager.py",
        "core/market_context_fusion.py",
        "utils/data_handler.py",
        "strategies/opening_range.py",
        "strategies/momentum_continuation_strategy.py",
        "strategies/pullback_entry_strategy.py"
    ]
    
    all_exist = True
    for filepath in critical_files:
        if os.path.exists(filepath):
            print(f"✅ Found: {filepath}")
        else:
            print(f"❌ Missing: {filepath}")
            all_exist = False
    
    return all_exist

def verify_config():
    """Verify config.py has all required settings"""
    print("\n" + "="*60)
    print("VERIFYING CONFIGURATION")
    print("="*60 + "\n")
    
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        
        required_settings = [
            'FUSION_CONFIG',
            'STRATEGIES',
            'MAX_POSITION_SIZE',
            'MAX_DAILY_LOSS',
            'COMMISSION_PER_SIDE',
            'CONTRACT_SIZE'
        ]
        
        for setting in required_settings:
            if setting in content:
                print(f"✅ Found: {setting}")
            else:
                print(f"❌ Missing: {setting}")
        
    except FileNotFoundError:
        print("❌ config.py not found!")

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60 + "\n")
    
    test_modules = [
        ('core.position_manager', 'position_manager'),
        ('core.signal_fusion_engine', 'signal_fusion_engine'),
        ('core.enhanced_strategy_engine', 'enhanced_strategy_engine'),
        ('core.market_context_fusion', 'market_context_fusion'),
        ('utils.data_handler', 'data_handler'),
    ]
    
    import sys
    success = True
    
    for module_path, object_name in test_modules:
        try:
            module = __import__(module_path, fromlist=[object_name])
            obj = getattr(module, object_name, None)
            if obj:
                print(f"✅ Imported: {module_path}.{object_name}")
            else:
                print(f"⚠️  Imported {module_path} but {object_name} not found")
        except Exception as e:
            print(f"❌ Failed to import {module_path}: {e}")
            success = False
    
    return success

def main():
    print("\n" + "="*70)
    print("NQ TRADING BOT - COMPREHENSIVE FIX BUNDLE")
    print("="*70)
    print("\nThis script will:")
    print("1. Check file structure")
    print("2. Verify configuration")
    print("3. Fix position_manager.py (hardcoded commission)")
    print("4. Test imports")
    print("\n" + "="*70 + "\n")
    
    input("Press Enter to continue (or Ctrl+C to cancel)...")
    
    # Step 1: Check files
    if not check_files_exist():
        print("\n❌ Some critical files are missing!")
        print("Make sure you're running this from the bot's root directory.")
        return
    
    # Step 2: Verify config
    verify_config()
    
    # Step 3: Fix position manager
    print("\n" + "="*60)
    print("APPLYING FIXES")
    print("="*60 + "\n")
    fix_position_manager()
    
    # Step 4: Test imports
    if test_imports():
        print("\n" + "="*70)
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the changes in the backup files")
        print("2. Run: python main.py")
        print("3. Test with: python test_webhook.py")
        print("\n")
    else:
        print("\n" + "="*70)
        print("⚠️  FIXES APPLIED BUT IMPORT ERRORS DETECTED")
        print("="*70)
        print("\nPlease check the error messages above.")
        print("You may need to install missing dependencies:")
        print("  pip install flask pandas numpy pytz --break-system-packages")
        print("\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Fix process cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error during fix process: {e}")
        import traceback
        traceback.print_exc()