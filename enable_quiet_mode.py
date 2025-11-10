#!/usr/bin/env python3
"""
Automatic Fix Script - Enable Quiet Mode
Suppresses verbose market context updates, shows only signals/trades
"""
import os
import sys
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create timestamped backup"""
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return False
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(filepath, 'r') as f:
        content = f.read()
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backed up: {filepath} → {backup_path}")
    return True

def fix_config():
    """Add verbosity controls to config.py"""
    filepath = "config.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found!")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already added
    if 'SHOW_CONTEXT_UPDATES' in content:
        print(f"✅ {filepath} already has verbosity controls")
        return True
    
    # Find where to insert (after LOG_BACKUP_COUNT)
    verbosity_config = '''
# -----------------------------------------------------------------
# --- CONSOLE OUTPUT VERBOSITY ---
# -----------------------------------------------------------------
SHOW_CONTEXT_UPDATES = False  # Set to True to see market context on every bar
SHOW_ONLY_SIGNALS = True      # Only print when signals are generated
VERBOSE_STARTUP = False       # Show detailed startup information
'''
    
    # Insert after LOG_BACKUP_COUNT line
    if 'LOG_BACKUP_COUNT' in content:
        lines = content.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if 'LOG_BACKUP_COUNT' in line:
                # Add after this line
                new_lines.append('')
                new_lines.extend(verbosity_config.split('\n'))
        
        content = '\n'.join(new_lines)
    else:
        # Just add at the end
        content += '\n' + verbosity_config
    
    backup_file(filepath)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {filepath} with verbosity controls")
    return True

def fix_enhanced_strategy_engine():
    """Update enhanced strategy engine to use quiet mode"""
    filepath = "core/enhanced_strategy_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found!")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'SHOW_CONTEXT_UPDATES' in content:
        print(f"✅ {filepath} already using quiet mode")
        return True
    
    backup_file(filepath)
    
    # Find and replace the _log_context method
    old_method = '''    def _log_context(self, context: Dict):
        print(f"\\n{'─'*60}")
        print(f" Market Context Update")
        print(f"   Price: ${context.get('current_price', 0):.2f}")
        print(f"   Regime: {context.get('market_regime', 'unknown')} ({context.get('regime_confidence', 0):.1%})")
        print(f"   Momentum: {context.get('momentum', 'neutral')}")
        print(f"   Trend: {context.get('trend_direction', 'neutral')}")
        print(f"   Vision: {'✓' if context.get('vision_available') else '✗'} | Sentiment: {context.get('vision_sentiment', 'N/A')}")
        diag = context.get('diagnostics')
        if diag:
            print(f"   ➜ {diag}")
        print(f"{'─'*60}")'''
    
    new_method = '''    def _log_context(self, context: Dict):
        """Log market context - controlled by SHOW_CONTEXT_UPDATES config"""
        try:
            from config import SHOW_CONTEXT_UPDATES
            if not SHOW_CONTEXT_UPDATES:
                return  # Quiet mode - skip context updates
        except ImportError:
            pass  # Config not updated yet, show anyway
        
        print(f"\\n{'─'*60}")
        print(f"📊 Market Context Update")
        print(f"   Price: ${context.get('current_price', 0):.2f}")
        print(f"   Regime: {context.get('market_regime', 'unknown')} ({context.get('regime_confidence', 0):.1%})")
        print(f"   Momentum: {context.get('momentum', 'neutral')}")
        print(f"   Trend: {context.get('trend_direction', 'neutral')}")
        print(f"   Vision: {'✓' if context.get('vision_available') else '✗'} | Sentiment: {context.get('vision_sentiment', 'N/A')}")
        diag = context.get('diagnostics')
        if diag:
            print(f"   ➜ {diag}")
        print(f"{'─'*60}")'''
    
    content = content.replace(old_method, new_method)
    
    # Also improve signal output
    old_signal_print = '''                if signal:
                    strategy_signals.append(signal)
                    print(f"   ✓ {name}: {signal.get('direction', signal.get('signal'))} "
                          f"(confidence: {float(signal.get('confidence', 0)):.2f})")'''
    
    new_signal_print = '''                if signal:
                    strategy_signals.append(signal)
                    # Print signal with prominence
                    print(f"\\n{'='*70}")
                    print(f"🎯 SIGNAL: {name.upper()}")
                    print(f"   Direction: {signal.get('direction', signal.get('signal'))}")
                    print(f"   Confidence: {float(signal.get('confidence', 0)):.1%}")
                    print(f"   Price: ${signal.get('price', 0):.2f}")
                    if signal.get('stop'):
                        print(f"   Stop: ${signal.get('stop', 0):.2f}")
                    if signal.get('target'):
                        print(f"   Target: ${signal.get('target', 0):.2f}")
                    print(f"{'='*70}\\n")'''
    
    if old_signal_print in content:
        content = content.replace(old_signal_print, new_signal_print)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {filepath} with quiet mode")
    return True

def main():
    print("="*70)
    print("QUIET MODE FIX - Suppress Verbose Console Output")
    print("="*70)
    print("\nThis script will:")
    print("1. Add verbosity controls to config.py")
    print("2. Update strategy engine to use quiet mode")
    print("3. Show only signals and trades (not every context update)")
    print("\n" + "="*70)
    
    response = input("\nProceed with fixes? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        return
    
    print("\n" + "="*70)
    print("APPLYING FIXES")
    print("="*70 + "\n")
    
    success = True
    
    # Fix 1: Update config
    if not fix_config():
        success = False
    
    # Fix 2: Update strategy engine
    if not fix_enhanced_strategy_engine():
        success = False
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Restart your bot: python main.py")
        print("2. Console will now be much quieter")
        print("3. Only signals and trades will be shown")
        print("\nTo re-enable verbose output:")
        print("   Edit config.py and set SHOW_CONTEXT_UPDATES = True")
    else:
        print("⚠️  SOME FIXES FAILED")
        print("="*70)
        print("\nPlease check the error messages above")
        print("You may need to apply fixes manually")
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
