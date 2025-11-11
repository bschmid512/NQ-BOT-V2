@echo off
echo [+] Starting Enhanced NQ-BOT-V2...

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo [+] Activating virtual environment...
    call venv\Scripts\activate
)

REM Check Redis status
echo [+] Checking Redis status...
redis-cli ping
IF ERRORLEVEL 1 (
    echo [WARN] Redis is not running or redis-cli not in PATH.
    echo [INFO] Please start Redis manually for caching to work.
) ELSE (
    echo [OK] Redis is running.
)

REM Start the trading bot
echo [+] Starting main application (main.py)...
python main.py

echo [INFO] Trading bot stopped
pause
