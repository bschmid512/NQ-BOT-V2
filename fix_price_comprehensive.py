#!/usr/bin/env python3
"""
COMPREHENSIVE FIX: Price key mismatch in signal_fusion_engine.py
Fixes both the key name (price vs current_price) and None checking
"""

import os
import shutil
from datetime import datetime

# Path to your file
file_path = r"C:\Users\bschm\OneDrive\Documents\nq_trading_bot\NQ-BOT-V2\core\signal_fusion_engine.py"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

print("🔧 COMPREHENSIVE FIX: Price handling in signal_fusion_engine.py")
print("="*70)

# Backup first
backup_path = file_path + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(file_path, backup_path)
print(f"✅ Backup created: {backup_path}")

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"📄 Original file: {len(content.splitlines())} lines")

# Fix 1: Change how we get price to handle both 'price' and 'current_price'
old_price_line = '        price = market_context.get("price")'
new_price_lines = '''        # Get price - handle both 'price' and 'current_price' keys
        price = market_context.get("price") or market_context.get("current_price")'''

if old_price_line in content:
    content = content.replace(old_price_line, new_price_lines)
    print("✅ Fix 1: Updated price retrieval to handle both keys")
else:
    print("⚠️  Could not find exact price line")

# Fix 2: Add None check right after price assignment
# Find where to insert the None check
search_str = '''        # Get price - handle both 'price' and 'current_price' keys
        price = market_context.get("price") or market_context.get("current_price")
        bar_age = market_context.get("bar_age_sec")'''

replacement_str = '''        # Get price - handle both 'price' and 'current_price' keys
        price = market_context.get("price") or market_context.get("current_price")
        bar_age = market_context.get("bar_age_sec")

        # Reject if price is None
        if price is None:
            self._reject("price", "No price in market_context")
            return None'''

if search_str in content:
    content = content.replace(search_str, replacement_str)
    print("✅ Fix 2: Added None check for price")
else:
    # Try alternative approach
    alt_search = '        bar_age = market_context.get("bar_age_sec")'
    alt_insert = '''        bar_age = market_context.get("bar_age_sec")

        # Reject if price is None
        if price is None:
            self._reject("price", "No price in market_context")
            return None'''
    
    if alt_search in content and "if price is None:" not in content:
        content = content.replace(alt_search, alt_insert, 1)
        print("✅ Fix 2: Added None check for price (alternative method)")

# Write fixed file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ File fixed: {len(content.splitlines())} lines")
print(f"✅ Backup saved to: {backup_path}")

print("\n" + "="*70)
print("FIXES APPLIED:")
print("="*70)
print("1. Price retrieval now checks BOTH 'price' AND 'current_price' keys")
print("2. Added None check to reject trades when price is missing")
print("\n" + "="*70)
print("You can now run your bot:")
print("  python main.py")
print("="*70)
