#!/usr/bin/env python3
"""
Setup Script for NQ Trading Bot
Initializes the trading system and checks dependencies
"""
import sys
import subprocess
import os
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python_version():
    """Ensure Python 3.8+"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print("✅ Python version OK")


def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    try:
        print("Installing Python packages from requirements.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        print("Try manually: pip install -r requirements.txt")
        sys.exit(1)


def check_talib():
    """Check if TA-Lib is installed"""
    print_header("Checking TA-Lib")
    
    try:
        import talib
        print(f"✅ TA-Lib version {talib.__version__} installed")
    except ImportError:
        print("⚠️  TA-Lib not installed (optional but recommended)")
        print("\nTo install TA-Lib:")
        print("  Ubuntu/Debian: sudo apt-get install ta-lib")
        print("  macOS: brew install ta-lib")
        print("  Windows: https://github.com/mrjbq7/ta-lib")
        print("\nContinuing without TA-Lib...")


def create_directories():
    """Create required directories"""
    print_header("Creating Directories")
    
    directories = [
        'data', 'models', 'models/trained_models',
        'logs', 'backtest', 'strategies', 'dashboard', 'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")


def initialize_data_files():
    """Initialize empty CSV files"""
    print_header("Initializing Data Files")
    
    import pandas as pd
    from config import LIVE_DATA_FILE, TRADES_FILE, SIGNALS_FILE
    
    # Live data file
    if not LIVE_DATA_FILE.exists():
        df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(LIVE_DATA_FILE, index=False)
        print(f"✅ Created: {LIVE_DATA_FILE}")
    
    # Trades file
    if not TRADES_FILE.exists():
        df = pd.DataFrame(columns=[
            'timestamp', 'ticker', 'action', 'price', 'size',
            'signal', 'stop_loss', 'take_profit', 'pnl', 'status',
            'entry_price','exit_price','entry_time','exit_time','r_multiple'
        ])
        df.to_csv(TRADES_FILE, index=False)
        print(f"✅ Created: {TRADES_FILE}")
    
    # Signals file
    if not SIGNALS_FILE.exists():
        df = pd.DataFrame(columns=[
            'timestamp', 'strategy', 'signal', 'confidence',
            'price', 'target', 'stop'
        ])
        df.to_csv(SIGNALS_FILE, index=False)
        print(f"✅ Created: {SIGNALS_FILE}")


def create_env_file():
    """Create .env file template"""
    print_header("Creating Environment File")
    
    env_template = """# NQ Trading Bot Environment Variables

# Webhook Security
WEBHOOK_PASSPHRASE=change_this_secure_passphrase

# Database Settings (Optional - for TimescaleDB)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nq_trading
DB_USER=trader
DB_PASSWORD=your_secure_password

# Alert Settings (Optional)
ALERT_EMAIL=your_email@example.com
SLACK_WEBHOOK_URL=your_slack_webhook_url
"""
    
    env_file = Path('.env')
    if not env_file.exists():
        env_file.write_text(env_template)
        print(f"✅ Created: .env")
        print("⚠️  Please edit .env and set your secure passphrase!")
    else:
        print("ℹ️  .env already exists, skipping...")


def print_next_steps():
    """Print next steps for user"""
    print_header("Setup Complete! 🎉")
    
    print("Next steps:")
    print("\n1. Configure your settings:")
    print("   - Edit config.py to customize trading parameters")
    print("   - Edit .env to set your webhook passphrase")
    
    print("\n2. Start the bot:")
    print("   python main.py")
    
    print("\n3. Access the dashboard:")
    print("   http://localhost:8050/dashboard/")
    
    print("\n4. Configure TradingView:")
    print("   - Create alert on NQ 1-minute chart")
    print("   - Set webhook URL: http://your-server:8050/webhook")
    print("   - See README.md for detailed webhook configuration")
    
    print("\n5. Test the webhook:")
    print("   curl -X POST http://localhost:8050/webhook \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"passphrase":"your_passphrase","close":16500,"volume":1000}\'')
    
    print("\nFor more information, see README.md")
    print("\n🚀 Happy Trading!\n")


def main():
    """Main setup function"""
    print("\n🚀 NQ Futures Trading Bot Setup")
    print("================================\n")
    
    try:
        check_python_version()
        create_directories()
        install_dependencies()
        check_talib()
        initialize_data_files()
        create_env_file()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
