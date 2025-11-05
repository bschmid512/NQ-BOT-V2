#!/usr/bin/env python3
"""
Temporary Patch: Disable ADX Filter
Allows Mean Reversion to trade in trending markets (for testing only)

⚠️ WARNING: This is ONLY for testing! In production, you want the ADX filter
   because mean reversion strategies lose money in strong trends.
"""
from pathlib import Path

def disable_adx_filter():
    """Comment out ADX trending filter in mean reversion strategy"""
    
    filepath = Path("strategies/mean_reversion.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find and comment out ADX check
    old_code = """            # Skip if trending market (ADX > 25)
            if current_bar['adx'] > 25:
                self.logger.debug(f"Skipping mean reversion - trending market (ADX={current_bar['adx']:.1f})")
                return None"""
    
    new_code = """            # ⚠️ ADX FILTER TEMPORARILY DISABLED FOR TESTING
            # # Skip if trending market (ADX > 25)
            # if current_bar['adx'] > 25:
            #     self.logger.debug(f"Skipping mean reversion - trending market (ADX={current_bar['adx']:.1f})")
            #     return None
            # ⚠️ NOTE: Re-enable this in production! Mean reversion loses in trends."""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ ADX filter disabled in mean_reversion.py")
        print("⚠️  CAUTION: Mean reversion will now trade in trending markets")
        print("   This is for TESTING only. Re-enable for live trading!")
        return True
    else:
        print("⚠️  Pattern not found - may already be patched")
        return False


def main():
    print("\n" + "=" * 70)
    print("  ⚠️  DISABLE ADX FILTER (TESTING ONLY)")
    print("=" * 70)
    print("\nThis will allow Mean Reversion to trade even when ADX > 25")
    print("(i.e., in trending markets where it normally shouldn't trade)")
    print("\n⚠️  USE FOR TESTING ONLY!")
    print("   Mean reversion strategies lose money in strong trends.")
    print("   This is ONLY to verify signal generation is working.")
    
    response = input("\nAre you sure? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print("\n🔧 Patching...")
    
    if disable_adx_filter():
        print("\n✅ DONE!")
        print("\nNext steps:")
        print("   1. Restart bot: python main.py")
        print("   2. Watch for Mean Reversion signals")
        print("   3. IMPORTANT: Re-enable ADX filter after testing!")
        print("\nTo re-enable, restore from backup or re-run automated fix.")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
