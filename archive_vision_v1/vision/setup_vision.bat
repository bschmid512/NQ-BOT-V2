@echo off
REM Setup Script for Computer Vision Trading System

echo ======================================================================
echo Computer Vision Trading System - Setup
echo ======================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

echo [OK] Python found: 
python --version
echo.

REM Check if in correct directory
if not exist "vision" (
    echo ERROR: vision folder not found!
    echo Please run this script from your bot directory
    pause
    exit /b 1
)

echo [OK] Vision folder found
echo.

REM Check Tesseract
if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ======================================================================
    echo WARNING: Tesseract OCR not found!
    echo ======================================================================
    echo.
    echo Tesseract is required for reading prices from the screen.
    echo.
    echo Download from:
    echo https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo Get: tesseract-ocr-w64-setup-5.3.3.20231005.exe
    echo Install to: C:\Program Files\Tesseract-OCR
    echo.
    echo Press any key to continue without Tesseract (limited functionality)
    echo Or press Ctrl+C to exit and install Tesseract first
    pause >nul
    echo.
) else (
    echo [OK] Tesseract OCR found
    echo.
)

REM Install Python packages
echo ======================================================================
echo Installing Python packages...
echo ======================================================================
echo.

pip install opencv-python==4.8.1.78 --break-system-packages
pip install numpy==1.26.2 --break-system-packages
pip install Pillow==10.1.0 --break-system-packages
pip install mss==9.0.1 --break-system-packages
pip install pytesseract==0.3.10 --break-system-packages

echo.
echo ======================================================================
echo Package installation complete
echo ======================================================================
echo.

REM Create logs directory if needed
if not exist "logs" (
    mkdir logs
    echo [OK] Created logs directory
)

echo.
echo ======================================================================
echo Setup Complete!
echo ======================================================================
echo.
echo Next steps:
echo.
echo 1. Open TradingView and make your chart visible
echo.
echo 2. Test screen capture:
echo    cd vision
echo    python screen_capture.py
echo.
echo 3. Run the full system:
echo    python vision_trading.py
echo.
echo See COMPUTER_VISION_GUIDE.md for detailed instructions
echo.
pause
