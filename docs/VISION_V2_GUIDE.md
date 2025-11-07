# Vision System V2.0 - Complete Guide

## 🎯 What's New in V2.0

### Fixed Issues
- ✅ **Threading errors resolved** - No more `'_thread._local' object has no attribute 'srcdc'` errors
- ✅ **Better candlestick detection** - Enhanced color detection and filtering
- ✅ **Improved price extraction** - Multiple OCR methods for better accuracy

### New Features
- 🎨 **Enhanced Pattern Recognition**
  - Bullish/Bearish Engulfing
  - Hammer & Shooting Star
  - Three White Soldiers / Three Black Crows
  - Morning Star / Evening Star
  - Double Top / Double Bottom
  - Support & Resistance Levels

- 📊 **Better Visual Analysis**
  - Real-time candlestick detection with size filtering
  - Trend line detection with classification
  - Support/resistance level identification
  - Pattern confidence scoring

- ⚡ **Improved Performance**
  - Region selection for faster processing
  - Better memory management
  - Enhanced thread handling

## 📦 Installation

### 1. Install Required Packages

```bash
pip install opencv-python==4.8.1.78 --break-system-packages
pip install numpy --break-system-packages
pip install Pillow --break-system-packages
pip install mss --break-system-packages
pip install pytesseract --break-system-packages
```

### 2. Install Tesseract OCR (for price reading)

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
- Get: `tesseract-ocr-w64-setup-5.3.3.20231005.exe`
- Install to: `C:\Program Files\Tesseract-OCR`

The system will work without Tesseract, but won't be able to read prices from the screen.

## 🚀 Quick Start

### Test the Enhanced Screen Capture

```bash
python screen_capture_v2.py
```

This will:
1. Ask if you want to select a specific region (recommended)
2. Start capturing your TradingView screen
3. Display real-time analysis with overlays
4. Show candlestick detection, trend lines, and statistics

**Controls:**
- `q` - Quit
- `s` - Save current frame
- `r` - Select new region

### Run the Complete Vision Trading System

```bash
python vision_integration_v2.py
```

This will:
1. Guide you through setup
2. Let you choose region capture
3. Select trading mode (Analysis or Live Trading)
4. Configure capture interval and confidence threshold
5. Start analyzing TradingView charts in real-time

## 📋 Features Explained

### Candlestick Detection
- Detects both **bullish** (green) and **bearish** (red) candles
- Filters by size to remove noise
- Calculates aspect ratios
- Shows counts and percentages

### Pattern Recognition

**Reversal Patterns:**
- **Bullish Engulfing**: Small red candle followed by large green
- **Bearish Engulfing**: Small green candle followed by large red
- **Hammer**: Long bullish candle (potential reversal)
- **Shooting Star**: Long bearish candle (potential reversal)

**Continuation Patterns:**
- **Three White Soldiers**: Three consecutive green candles (strong bullish)
- **Three Black Crows**: Three consecutive red candles (strong bearish)

**Star Patterns:**
- **Morning Star**: Bearish → Small → Bullish (reversal to upside)
- **Evening Star**: Bullish → Small → Bearish (reversal to downside)

**Chart Patterns:**
- **Double Top**: Two peaks at similar levels (bearish reversal)
- **Double Bottom**: Two troughs at similar levels (bullish reversal)

### Support & Resistance Detection
- Identifies horizontal lines from chart analysis
- Groups similar levels together
- Calculates strength based on number of touches
- Classifies as support or resistance based on position

### Signal Confidence
Each signal includes a confidence score (0.0 - 1.0) based on:
- Pattern clarity
- Size ratios
- Number of confirming factors
- Market sentiment

## ⚙️ Configuration

### Capture Settings

```python
system = VisionTradingV2(
    monitor_number=1,              # Which monitor (1, 2, etc.)
    capture_region=None,           # None for full screen, or dict with top/left/width/height
    capture_interval=2.0,          # Seconds between captures (2.0 recommended)
    min_confidence=0.75,           # Minimum confidence to act on signals
    trading_enabled=False          # True to execute trades automatically
)
```

### Region Selection
For best performance, select just your TradingView chart area:

```python
# Interactive region selection
capture_region = TradingViewCaptureV2.select_capture_region()
```

Or manually specify:
```python
capture_region = {
    'left': 100,
    'top': 100,
    'width': 1600,
    'height': 900
}
```

### Customizing Pattern Detection

Edit `pattern_recognition_v2.py`:

```python
# Minimum confidence for signals
self.min_confidence = 0.6  # Lower = more signals, higher = fewer but better

# Support/Resistance settings
self.sr_tolerance = 20      # Pixel tolerance for grouping levels
self.sr_min_touches = 2     # Minimum touches for valid level
```

### Customizing Candlestick Detection

Edit `screen_capture_v2.py`:

```python
# Detection parameters
self.min_candle_area = 100    # Minimum area for valid candle
self.max_candle_area = 50000  # Maximum area to filter out noise
```

## 🎮 Usage Modes

### 1. Analysis Mode (Safe)
- Captures and analyzes charts
- Generates signals
- Logs everything
- **No trades executed**

Perfect for:
- Testing the system
- Learning how patterns work
- Backtesting pattern recognition
- Running alongside your main bot

### 2. Trading Mode (Live)
- Everything from Analysis Mode
- **Automatically executes trades** via `position_manager`
- Respects signal cooldown (default 60s)
- Only trades signals above confidence threshold

⚠️ **Warning**: Only use this mode when you're confident in the system!

## 📊 Understanding the Display

### Real-Time Window Shows:

**Overlays:**
- 🟩 Green boxes: Bullish candlesticks
- 🟥 Red boxes: Bearish candlesticks
- 🟨 Yellow lines: Horizontal support/resistance
- 🟪 Purple lines: Diagonal trend lines

**Statistics Panel:**
```
Candles: 47
Bullish: 28 (59.6%)
Bearish: 19 (40.4%)
Sentiment: BULLISH
H-Lines: 3
D-Lines: 12
```

### Console Log Shows:

**Analysis:**
```
Captured: 28 bullish (59.6%), 19 bearish (40.4%) | Sentiment: BULLISH
Generated 3 trading signals:
  1. bullish_engulfing: LONG (confidence: 75%) - Bullish engulfing (size ratio: 1.87)
  2. three_white_soldiers: LONG (confidence: 75%) - Three consecutive bullish candles
  3. strong_bullish_sentiment: LONG (confidence: 60%) - Strong bullish momentum
```

## 🔧 Troubleshooting

### "Tesseract not found" Error
- Install Tesseract OCR (see Installation section)
- Or edit path in `screen_capture_v2.py`:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'YOUR_PATH_HERE'
  ```

### Threading Errors
- V2.0 fixes these! Each thread gets its own MSS instance
- If you still see errors, try reducing `capture_interval`

### No Candles Detected
- Check TradingView chart colors (should be red/green)
- Try adjusting color ranges in `detect_candlesticks_enhanced()`
- Use region selection to focus on chart area only

### Poor Performance
- Select a specific region instead of full screen
- Increase `capture_interval` (try 3.0 or 4.0 seconds)
- Close other applications
- Reduce your screen resolution

### Signals Not Executing
- Check that `position_manager` is available
- Verify `trading_enabled=True`
- Check signal confidence vs `min_confidence` threshold
- Look for signal cooldown messages in logs

## 📁 File Overview

### Core Files
- `screen_capture_v2.py` - Enhanced screen capture with fixed threading
- `pattern_recognition_v2.py` - Improved pattern detection algorithms
- `vision_integration_v2.py` - Complete integration with trading bot

### Your Original Files (Keep These!)
- `vision/screen_capture.py` - Original version (backup)
- `vision/pattern_recognition.py` - Original patterns (backup)
- `vision/vision_trading.py` - Original integration (backup)

## 🎯 Best Practices

### For Testing
1. Start with Analysis Mode
2. Use region selection for better performance
3. Set `min_confidence=0.70` to see various signals
4. Watch for patterns you recognize
5. Compare with your own chart analysis

### For Live Trading
1. Test thoroughly in Analysis Mode first
2. Start with `min_confidence=0.80` for safer signals
3. Use longer `capture_interval` (3.0s+) to avoid overtrading
4. Monitor the first hour closely
5. Keep signal cooldown at 60s or higher

### TradingView Setup
- Use clean chart with minimal indicators
- Ensure good contrast (dark theme works well)
- Standard candlestick display (not Heikin-Ashi)
- 1-minute or 5-minute timeframe recommended
- Chart should be clearly visible and not obscured

## 📈 Performance Tips

### Optimize Detection
```python
# For cleaner detection, adjust in screen_capture_v2.py:

# Tighter candle area filtering
self.min_candle_area = 200    # Ignore tiny candles
self.max_candle_area = 30000  # Ignore huge anomalies

# More selective colors
green_lower = np.array([40, 50, 50])  # Tighter green range
green_upper = np.array([80, 255, 255])
```

### Optimize Performance
```python
# Faster capture
capture_interval = 3.0  # Capture every 3 seconds

# Process fewer candles
# In pattern_recognition_v2.py, limit analysis to most recent N candles
candles = sorted_candles[-50:]  # Only analyze last 50 candles
```

## 🆘 Support

If you encounter issues:

1. **Check logs**: `logs/vision_trading_v2.log`
2. **Test components individually**:
   - Run `screen_capture_v2.py` alone
   - Test region selection
3. **Verify TradingView is visible** on your screen
4. **Check that candlesticks are displaying** (red/green)
5. **Try different confidence thresholds**

## 🚀 Next Steps

1. **Test the system**: Run `python screen_capture_v2.py`
2. **Watch patterns**: See what it detects on your charts
3. **Compare signals**: Check against your own analysis
4. **Tune parameters**: Adjust confidence and intervals
5. **Go live**: When confident, enable trading mode

Happy trading! 📊🎯
