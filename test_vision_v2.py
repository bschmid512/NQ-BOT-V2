"""
Quick Test Script for Vision System V2.0
Verifies all components are working correctly
"""
import sys
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def check_imports():
    """Check if all required packages are installed"""
    print("\n" + "="*70)
    print("CHECKING IMPORTS")
    print("="*70 + "\n")
    
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'mss': 'mss',
        'pytesseract': 'pytesseract'
    }
    
    all_good = True
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - Installed")
        except ImportError:
            print(f"✗ {package:20s} - MISSING")
            all_good = False
    
    if not all_good:
        print("\n❌ Some packages are missing!")
        print("Install with: pip install <package> --break-system-packages")
        return False
    
    print("\n✓ All packages installed")
    return True


def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\n" + "="*70)
    print("CHECKING TESSERACT OCR")
    print("="*70 + "\n")
    
    import pytesseract
    
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract found: v{version}")
        return True
    except Exception as e:
        print("⚠ Tesseract not found or not configured")
        print("  Price extraction will not work")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def test_screen_capture():
    """Test basic screen capture"""
    print("\n" + "="*70)
    print("TESTING SCREEN CAPTURE")
    print("="*70 + "\n")
    
    try:
        import mss
        
        with mss.mss() as sct:
            monitors = sct.monitors
            print(f"✓ Found {len(monitors)-1} monitor(s)")
            
            for i, monitor in enumerate(monitors[1:], 1):
                print(f"  Monitor {i}: {monitor['width']}x{monitor['height']}")
            
            # Capture primary monitor
            screenshot = sct.grab(monitors[1])
            img = np.array(screenshot)
            
            print(f"\n✓ Captured frame: {img.shape}")
            print(f"  Resolution: {img.shape[1]}x{img.shape[0]}")
            
            return True
            
    except Exception as e:
        print(f"✗ Screen capture failed: {e}")
        return False


def test_opencv():
    """Test OpenCV functionality"""
    print("\n" + "="*70)
    print("TESTING OPENCV")
    print("="*70 + "\n")
    
    try:
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw test rectangle (simulating candlestick)
        cv2.rectangle(img, (100, 100), (150, 200), (0, 255, 0), -1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Test color detection
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✓ OpenCV working correctly")
        print(f"  Test image: {img.shape}")
        print(f"  Detected {len(contours)} contour(s)")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False


def test_vision_modules():
    """Test V2.0 vision modules"""
    print("\n" + "="*70)
    print("TESTING VISION V2.0 MODULES")
    print("="*70 + "\n")
    
    try:
        # Test screen_capture_v2
        from screen_capture_v2 import TradingViewCaptureV2
        print("✓ screen_capture_v2.py - Loaded")
        
        # Test pattern_recognition_v2
        from pattern_recognition_v2 import PatternRecognizerV2, TradingViewAIV2
        print("✓ pattern_recognition_v2.py - Loaded")
        
        # Test vision_integration_v2
        from vision_integration_v2 import VisionTradingV2
        print("✓ vision_integration_v2.py - Loaded")
        
        print("\n✓ All V2.0 modules loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Module loading failed: {e}")
        print("\nMake sure these files are in the same directory:")
        print("  - screen_capture_v2.py")
        print("  - pattern_recognition_v2.py")
        print("  - vision_integration_v2.py")
        return False


def test_capture_instance():
    """Test creating capture instance"""
    print("\n" + "="*70)
    print("TESTING CAPTURE INSTANCE")
    print("="*70 + "\n")
    
    try:
        from screen_capture_v2 import TradingViewCaptureV2
        
        # Create instance
        capture = TradingViewCaptureV2(monitor_number=1, display_mode=False)
        print("✓ Created TradingViewCaptureV2 instance")
        
        # Test capture
        print("  Attempting single capture...")
        img = capture.capture_screen()
        print(f"✓ Captured frame: {img.shape}")
        
        # Test analysis
        print("  Running analysis...")
        analysis = capture.analyze_frame_enhanced(img)
        
        print(f"✓ Analysis complete:")
        print(f"  Candles detected: {len(analysis['candlesticks'])}")
        print(f"  Lines detected: {len(analysis['trend_lines'])}")
        print(f"  Sentiment: {analysis['statistics'].get('sentiment', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Capture instance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_recognition():
    """Test pattern recognition"""
    print("\n" + "="*70)
    print("TESTING PATTERN RECOGNITION")
    print("="*70 + "\n")
    
    try:
        from pattern_recognition_v2 import PatternRecognizerV2, TradingViewAIV2
        
        # Create AI instance
        ai = TradingViewAIV2()
        print("✓ Created TradingViewAIV2 instance")
        
        # Create test candles
        test_candles = [
            {'type': 'bearish', 'x': 100, 'y': 100, 'width': 20, 'height': 80, 'area': 1600, 'aspect_ratio': 4},
            {'type': 'bullish', 'x': 130, 'y': 80, 'width': 30, 'height': 120, 'area': 3600, 'aspect_ratio': 4},
        ]
        
        # Create test analysis
        test_analysis = {
            'timestamp': '2025-11-06',
            'candlesticks': test_candles,
            'trend_lines': [],
            'statistics': {
                'bullish_count': 1,
                'bearish_count': 1,
                'sentiment': 'bullish'
            }
        }
        
        # Process
        result = ai.process_frame(test_analysis)
        signals = result.get('signals', [])
        
        print(f"✓ Pattern recognition working")
        print(f"  Test signals generated: {len(signals)}")
        
        if signals:
            print("\n  Example signals:")
            for i, signal in enumerate(signals[:3], 1):
                print(f"    {i}. {signal['pattern']} ({signal['signal']}) - {signal['confidence']:.0%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pattern recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("VISION SYSTEM V2.0 - DIAGNOSTIC TEST")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", check_imports()))
    results.append(("Tesseract OCR", check_tesseract()))
    results.append(("Screen Capture", test_screen_capture()))
    results.append(("OpenCV", test_opencv()))
    results.append(("V2.0 Modules", test_vision_modules()))
    
    # Only run these if modules loaded
    if results[-1][1]:
        results.append(("Capture Instance", test_capture_instance()))
        results.append(("Pattern Recognition", test_pattern_recognition()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou're ready to run:")
        print("  python screen_capture_v2.py        (Test capture)")
        print("  python vision_integration_v2.py    (Full system)")
    else:
        print("\n⚠ Some tests failed - check the output above")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install <package> --break-system-packages")
        print("  - Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  - Make sure all V2.0 files are in the same directory")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
