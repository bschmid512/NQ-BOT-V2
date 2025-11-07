import os
import shutil

"""
This script reorganizes the NQ-BOT-V2 project into a clean,
modular folder structure.

It moves root .py files into new subdirectories:
- 'core/' for main logic
- 'vision_v2/' for screen analysis
- 'tests/' for all test scripts
- 'docs/' for markdown guides
- 'scripts/' for helper/utility scripts

It also archives the old 'vision/' V1 folder into 'archive_vision_v1/'.
"""

# Define the target structure
# Mapping: 'target_directory': ['list_of_files_to_move_from_root']
file_map = {
    'core': [
        'comprehensive_logger.py',
        'market_context_fusion.py',
        'position_manager.py',
        'signal_fusion_engine.py',
        'strategy_engine.py',
        'webhook_server.py',
        'enhanced_strategy_engine.py'
    ],
    'vision_v2': [
        'vision_integration_v2.py',
        'screen_capture_v2.py',
        'pattern_recognition_v2.py',
        'setup_vision_v2.py',
        'pattern_recognition_v2.py.backup'
    ],
    'tests': [
        'test_ocr.py',
        'test_vision_v2.py',
        'test_webhook.py',
        'diagnose_trading.py'
    ],
    'docs': [
        'DEPLOYMENT_CHECKLIST.md',
        'QUICKSTART.md',
        'VISION_V2_GUIDE.md'
        # README.md is correctly left in the root
    ],
    'scripts': [
        'csv_data_feeder.py',
        'disable_adx_filter.py',
        'disable_context_blocking.py',
        'setup.py',
        'upgrade_add_prices.py'
    ]
}

# --- Main Script ---
print("Starting project organization...")
total_moved = 0
total_errors = 0

# 1. Create all target directories
all_folders = list(file_map.keys())
all_folders.append('archive_vision_v1') # For the old V1 vision code
print(f"Ensuring {len(all_folders)} directories exist...")

for folder in all_folders:
    try:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✅ Directory: {folder}/")
    except OSError as e:
        print(f"  ❌ ERROR creating directory {folder}: {e}")
        total_errors += 1

# 2. Move files from root into new directories
print("\nMoving files into new directories...")
for folder, files in file_map.items():
    print(f"--- Processing folder: {folder}/ ---")
    for file_name in files:
        source_path = file_name
        dest_path = os.path.join(folder, file_name)
        
        if os.path.isfile(source_path):
            try:
                shutil.move(source_path, dest_path)
                print(f"  Moved: {source_path}  ->  {dest_path}")
                total_moved += 1
            except Exception as e:
                print(f"  ❌ ERROR moving {source_path}: {e}")
                total_errors += 1
        else:
            print(f"  Skipped: {source_path} (not found in root, may already be moved)")

# 3. Archive the old 'vision/' V1 folder to declutter
print("\n--- Archiving old 'vision' V1 folder ---")
v1_folder = 'vision'
archive_dest = 'archive_vision_v1/vision'

if os.path.isdir(v1_folder):
    try:
        shutil.move(v1_folder, archive_dest)
        print(f"  Moved: {v1_folder}/  ->  {archive_dest}/")
    except Exception as e:
        print(f"  ❌ ERROR archiving {v1_folder}: {e}")
        total_errors += 1
else:
    print(f"  Skipped: {v1_folder}/ (not found)")

# --- Summary ---
print("\n" + "="*50)
print("Organization Complete! 🎉")
print(f"Files Moved: {total_moved}")
print(f"Errors:      {total_errors}")
print("="*50)
print("\n⚠️ IMPORTANT: NEXT STEPS ⚠️")
print("Your project structure is clean, but your imports are now broken.")
print("You must now search your project and update all import statements.")
print("\nExamples:")
print("  `from strategy_engine import ...`  ->  `from core.strategy_engine import ...`")
print("  `from vision_integration_v2 import ...`  ->  `from vision_v2.vision_integration_v2 import ...`")
print("  `from utils.data_handler import ...` (This should still work if it was correct before)")
print("  `import config` (This should still work as config.py is in the root)")
print("="*50)