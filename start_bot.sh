#!/bin/bash
# NQ-BOT-V2 Enhanced Startup Script

echo "[+] Starting Enhanced NQ-BOT-V2..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "[+] Activating virtual environment..."
    source venv/bin/activate
fi

# Check Redis status
if command -v redis-cli &> /dev/null; then
    echo "[+] Checking Redis status..."
    if redis-cli ping &> /dev/null; then
        echo "[OK] Redis is running"
    else
        echo "[WARN] Redis not running, attempting to start..."
        # Try starting (this may require sudo)
        redis-server --daemonize yes
        sleep 2
        if redis-cli ping &> /dev/null; then
            echo "[OK] Redis started successfully"
        else
            echo "[FAIL] Failed to start Redis. Bot will continue without caching."
        fi
    fi
else
    echo "[WARN] redis-cli not found. Cannot check Redis status."
fi

# Start the trading bot
echo "[+] Starting main application (main.py)..."
python main.py

echo "[INFO] Trading bot stopped"
