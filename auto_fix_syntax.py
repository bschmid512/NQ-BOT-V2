#!/usr/bin/env python3
"""
AUTOMATIC FIX for signal_fusion_engine.py syntax error
Removes the orphaned 'return None' at line 142
"""

import os
import shutil
from datetime import datetime

# Path to your file
file_path = r"C:\Users\bschm\OneDrive\Documents\nq_trading_bot\NQ-BOT-V2\core\signal_fusion_engine.py"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    print("Please update the file_path in this script")
    exit(1)

print("🔧 AUTOMATIC FIX for signal_fusion_engine.py")
print("="*70)

# Backup first
backup_path = file_path + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(file_path, backup_path)
print(f"✅ Backup created: {backup_path}")

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"📄 Original file: {len(lines)} lines")

# Check line 142
if len(lines) >= 142:
    line_142 = lines[141]  # 0-indexed
    print(f"Line 142 before: '{line_142.rstrip()}'")
    
    # Remove line 142 if it's the problematic return
    if 'return None' in line_142 or 'return' in line_142:
        # Delete the line
        del lines[141]
        print(f"❌ Line 142 removed")
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ File fixed: {len(lines)} lines")
        print(f"✅ Backup saved to: {backup_path}")
        print("\nYou can now run your bot:")
        print("  python main.py")
    else:
        print("⚠️  Line 142 doesn't contain 'return' - may not be the issue")
        print(f"   Content: '{line_142.strip()}'")
else:
    print("❌ File has fewer than 142 lines - unexpected")

print("="*70)
