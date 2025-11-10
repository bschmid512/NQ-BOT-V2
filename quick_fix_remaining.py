#!/usr/bin/env python3
"""Quick fix for remaining issues"""

# Fix #1: Update min_signals_required to 2
with open('config.py', 'r') as f:
    content = f.read()

# Backup first
with open('config.py.backup2', 'w') as f:
    f.write(content)

# Fix the config
content = content.replace(
    "'min_signals_required': 1,",
    "'min_signals_required': 2,  # Fixed: require 2+ signals"
)

with open('config.py', 'w') as f:
    f.write(content)

print("✅ Fixed: min_signals_required set to 2")

# Fix #2: Check fusion engine has global instance
try:
    with open('core/signal_fusion_engine.py', 'r') as f:
        content = f.read()
    
    if 'signal_fusion_engine = SignalFusionEngine()' in content:
        print("✅ Fusion engine instance exists")
    else:
        print("⚠️  Need to add fusion engine instance")
        print("   Add this line at the end of core/signal_fusion_engine.py:")
        print("   signal_fusion_engine = SignalFusionEngine()")
except FileNotFoundError:
    print("❌ core/signal_fusion_engine.py not found")

print("\n✅ Fixes applied! Run test again:")
print("   python test_all_fixes.py")