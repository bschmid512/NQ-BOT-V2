#!/usr/bin/env python3
"""
Quick Patch: Disable Context Manager Blocking
Use this if Context Manager isn't running but you want to test strategies
"""
import sys
from pathlib import Path

def patch_context_blocking():
    """Remove context-based trade blocking"""
    
    # Patch strategy_engine.py
    filepath = Path("strategy_engine.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Comment out VIX blocking
    old_code = """        # <-- NEW CONTEXT-BASED CHECK
        # Do not trade if VIX is spiking (market is in panic)
        if context.get('vix_status') == 'SPIKING':
            return False, f"VIX is spiking - market too volatile, staying flat\""""
    
    new_code = """        # <-- CONTEXT CHECK TEMPORARILY DISABLED FOR TESTING
        # # Do not trade if VIX is spiking (market is in panic)
        # if context.get('vix_status') == 'SPIKING':
        #     return False, f"VIX is spiking - market too volatile, staying flat\""""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Patched strategy_engine.py - Context blocking disabled")
        return True
    else:
        print("⚠️  Pattern not found - may already be patched")
        return False


def patch_mean_reversion_context():
    """Remove ES trend context blocking from mean reversion"""
    
    filepath = Path("strategies/mean_reversion.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find and comment out ES trend checks
    changes = 0
    
    # LONG context filter
    old_long = """                # ⭐⭐⭐ NEW CONTEXT FILTER ("WHY") ⭐⭐⭐
                # Do not take a long if the broad market (ES) is in a STRONG_DOWN trend
                es_trend = context.get('es_trend', 'NEUTRAL')
                if es_trend == 'STRONG_DOWN':
                    self.logger.info(f"Mean Reversion LONG rejected: ES trend is STRONG_DOWN.")
                    return None
                # ⭐⭐⭐ END OF FILTER ⭐⭐⭐"""
    
    new_long = """                # ⭐⭐⭐ CONTEXT FILTER TEMPORARILY DISABLED ⭐⭐⭐
                # # Do not take a long if the broad market (ES) is in a STRONG_DOWN trend
                # es_trend = context.get('es_trend', 'NEUTRAL')
                # if es_trend == 'STRONG_DOWN':
                #     self.logger.info(f"Mean Reversion LONG rejected: ES trend is STRONG_DOWN.")
                #     return None
                # ⭐⭐⭐ END OF FILTER ⭐⭐⭐"""
    
    if old_long in content:
        content = content.replace(old_long, new_long)
        changes += 1
    
    # SHORT context filter
    old_short = """                # ⭐⭐⭐ NEW CONTEXT FILTER ("WHY") ⭐⭐⭐
                # Do not take a short if the broad market (ES) is in a STRONG_UP trend
                es_trend = context.get('es_trend', 'NEUTRAL')
                if es_trend == 'STRONG_UP':
                    self.logger.info(f"Mean Reversion SHORT rejected: ES trend is STRONG_UP.")
                    return None
                # ⭐⭐⭐ END OF FILTER ⭐⭐⭐"""
    
    new_short = """                # ⭐⭐⭐ CONTEXT FILTER TEMPORARILY DISABLED ⭐⭐⭐
                # # Do not take a short if the broad market (ES) is in a STRONG_UP trend
                # es_trend = context.get('es_trend', 'NEUTRAL')
                # if es_trend == 'STRONG_UP':
                #     self.logger.info(f"Mean Reversion SHORT rejected: ES trend is STRONG_UP.")
                #     return None
                # ⭐⭐⭐ END OF FILTER ⭐⭐⭐"""
    
    if old_short in content:
        content = content.replace(old_short, new_short)
        changes += 1
    
    if changes > 0:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Patched mean_reversion.py - Context blocking disabled ({changes} filters)")
        return True
    else:
        print("⚠️  Patterns not found - may already be patched")
        return False


def patch_orb_context():
    """Remove ES trend context blocking from ORB"""
    
    filepath = Path("strategies/opening_range.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    changes = 0
    
    # LONG context filter
    old_long = """                # ⭐ CONTEXT FILTER
                if es_trend == 'STRONG_DOWN':
                    self.logger.info("ORB LONG rejected: ES trend is STRONG_DOWN.")
                    return None"""
    
    new_long = """                # ⭐ CONTEXT FILTER TEMPORARILY DISABLED
                # if es_trend == 'STRONG_DOWN':
                #     self.logger.info("ORB LONG rejected: ES trend is STRONG_DOWN.")
                #     return None"""
    
    if old_long in content:
        content = content.replace(old_long, new_long)
        changes += 1
    
    # SHORT context filter
    old_short = """                # ⭐ CONTEXT FILTER
                if es_trend == 'STRONG_UP':
                    self.logger.info("ORB SHORT rejected: ES trend is STRONG_UP.")
                    return None"""
    
    new_short = """                # ⭐ CONTEXT FILTER TEMPORARILY DISABLED
                # if es_trend == 'STRONG_UP':
                #     self.logger.info("ORB SHORT rejected: ES trend is STRONG_UP.")
                #     return None"""
    
    if old_short in content:
        content = content.replace(old_short, new_short)
        changes += 1
    
    if changes > 0:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Patched opening_range.py - Context blocking disabled ({changes} filters)")
        return True
    else:
        print("⚠️  Patterns not found - may already be patched")
        return False


def main():
    print("\n" + "=" * 70)
    print("  DISABLE CONTEXT MANAGER BLOCKING")
    print("=" * 70)
    print("\n⚠️  This will allow strategies to trade without Context Manager")
    print("    Use this temporarily if Context Manager isn't working")
    
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print("\n🔧 Applying patches...")
    
    patch_context_blocking()
    patch_mean_reversion_context()
    patch_orb_context()
    
    print("\n✅ DONE! Restart the bot:")
    print("   python main.py")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
