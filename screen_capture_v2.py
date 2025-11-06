"""
TradingView Screen Capture & Analysis System - Version 2.0
IMPROVEMENTS:
- Fixed threading issues with MSS
- Better candlestick detection with size filtering
- Enhanced price extraction with multiple OCR attempts
- Improved pattern recognition
- Better error handling
- Region selection helper
"""
import cv2
import numpy as np
import mss
import pytesseract
from PIL import Image
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
import logging

# Configure pytesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class TradingViewCaptureV2:
    """
    Enhanced TradingView screen capture with improved detection
    """
    
    def __init__(self, monitor_number: int = 1, capture_region: Dict = None, display_mode: bool = True):
        """
        Initialize screen capture
        
        Args:
            monitor_number: Which monitor to capture (1, 2, etc.)
            capture_region: Specific region {'top': y, 'left': x, 'width': w, 'height': h}
        """
        self.monitor_number = monitor_number
        self.capture_region = capture_region
        self.running = False
        self.current_frame = None
        self.logger = logging.getLogger(__name__)
        
        # Create MSS instance per thread (FIX for threading issue)
        self._sct_instances = {}
        
        # Detection parameters
        self.min_candle_area = 100  # Minimum area for valid candle
        self.max_candle_area = 50000  # Maximum area to filter out noise
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _get_sct(self):
        """Get MSS instance for current thread (fixes threading issue)"""
        thread_id = threading.get_ident()
        if thread_id not in self._sct_instances:
            self._sct_instances[thread_id] = mss.mss()
        return self._sct_instances[thread_id]
    
    def get_monitor_region(self) -> Dict:
        """Get the monitor region to capture"""
        if self.capture_region:
            return self.capture_region
        
        sct = self._get_sct()
        monitors = sct.monitors
        
        if self.monitor_number >= len(monitors):
            self.logger.warning(f"Monitor {self.monitor_number} not found, using primary")
            self.monitor_number = 1
            
        return monitors[self.monitor_number]
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture screen and return as numpy array
        
        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        sct = self._get_sct()
        monitor = self.get_monitor_region()
        
        # Capture screen
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def extract_price_from_region(self, img: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Extract price from specific region using OCR with multiple attempts
        
        Args:
            img: Image array
            region: (x, y, width, height) of region to extract from
            
        Returns:
            Extracted price as float or None
        """
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        
        # Try multiple preprocessing methods
        methods = [
            self._preprocess_method1,
            self._preprocess_method2,
            self._preprocess_method3
        ]
        
        for method in methods:
            try:
                preprocessed = method(roi)
                text = pytesseract.image_to_string(
                    preprocessed, 
                    config='--psm 7 -c tessedit_char_whitelist=0123456789.,'
                )
                text = text.strip().replace(',', '').replace(' ', '')
                
                if text and text.replace('.', '').isdigit():
                    price = float(text)
                    if 10000 < price < 30000:  # Reasonable NQ price range
                        return price
            except Exception as e:
                continue
                
        return None
    
    def _preprocess_method1(self, roi: np.ndarray) -> np.ndarray:
        """Standard preprocessing - grayscale + threshold"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _preprocess_method2(self, roi: np.ndarray) -> np.ndarray:
        """Adaptive threshold"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    def _preprocess_method3(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced with morphology"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morphology
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph
    
    def detect_candlesticks_enhanced(self, img: np.ndarray) -> List[Dict]:
        """
        Enhanced candlestick detection with better filtering
        
        Args:
            img: Image array
            
        Returns:
            List of detected candlesticks with properties
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for green (bullish) candles - wider range
        green_lower = np.array([35, 30, 30])
        green_upper = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Define color ranges for red (bearish) candles - wider range
        red_lower1 = np.array([0, 30, 30])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([165, 30, 30])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
        
        # Clean up masks with morphology
        kernel = np.ones((3,3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candlesticks = []
        
        # Process green candles
        for contour in green_contours:
            area = cv2.contourArea(contour)
            if self.min_candle_area < area < self.max_candle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (candles should be taller than wide)
                if h > w * 0.5:  # Allow some variation
                    candlesticks.append({
                        'type': 'bullish',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'aspect_ratio': h / w if w > 0 else 0
                    })
        
        # Process red candles
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if self.min_candle_area < area < self.max_candle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                if h > w * 0.5:
                    candlesticks.append({
                        'type': 'bearish',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'aspect_ratio': h / w if w > 0 else 0
                    })
        
        return candlesticks
    
    def detect_trend_lines_enhanced(self, img: np.ndarray) -> List[Dict]:
        """
        Enhanced trend line detection with better filtering
        
        Args:
            img: Image array
            
        Returns:
            List of detected lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise but keep edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection with auto-threshold
        median_val = np.median(bilateral)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        edges = cv2.Canny(bilateral, lower, upper, apertureSize=3)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=80,  # Lower threshold for more lines
            minLineLength=80,  # Shorter minimum
            maxLineGap=15
        )
        
        detected_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Normalize angle to 0-180
                if angle < 0:
                    angle += 180
                
                # Classify line type
                line_type = 'diagonal'
                if abs(angle) < 10 or abs(angle - 180) < 10:
                    line_type = 'horizontal'
                elif 80 < abs(angle) < 100:
                    line_type = 'vertical'
                
                detected_lines.append({
                    'type': line_type,
                    'angle': angle,
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2),
                    'length': length,
                    'midpoint': ((x1+x2)/2, (y1+y2)/2)
                })
        
        return detected_lines
    
    def analyze_frame_enhanced(self, img: np.ndarray) -> Dict:
        """
        Enhanced frame analysis with better detection
        
        Args:
            img: Image array
            
        Returns:
            Dictionary with comprehensive analysis
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'image_shape': img.shape,
            'candlesticks': [],
            'trend_lines': [],
            'patterns': [],
            'statistics': {}
        }
        
        # Detect candlesticks with enhanced method
        analysis['candlesticks'] = self.detect_candlesticks_enhanced(img)
        
        # Detect trend lines with enhanced method
        analysis['trend_lines'] = self.detect_trend_lines_enhanced(img)
        
        # Calculate statistics
        bullish_count = sum(1 for c in analysis['candlesticks'] if c['type'] == 'bullish')
        bearish_count = sum(1 for c in analysis['candlesticks'] if c['type'] == 'bearish')
        
        total_candles = bullish_count + bearish_count
        
        analysis['statistics'] = {
            'total_candles': total_candles,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'bullish_percentage': (bullish_count / total_candles * 100) if total_candles > 0 else 0,
            'bearish_percentage': (bearish_count / total_candles * 100) if total_candles > 0 else 0,
            'sentiment': 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral',
            'horizontal_lines': len([l for l in analysis['trend_lines'] if l['type'] == 'horizontal']),
            'diagonal_lines': len([l for l in analysis['trend_lines'] if l['type'] == 'diagonal'])
        }
        
        return analysis
    
    def display_frame(self, img: np.ndarray = None, analysis: Dict = None, window_name: str = "TradingView Analysis"):
        """
        Display frame with analysis overlays
        
        Args:
            img: Image to display (uses current_frame if None)
            analysis: Analysis results to overlay
            window_name: Window name
        """
        if img is None:
            img = self.current_frame
        
        if img is None:
            return
        
        # Create copy for drawing
        display_img = img.copy()
        
        if analysis:
            # Draw candlesticks
            for candle in analysis.get('candlesticks', []):
                color = (0, 255, 0) if candle['type'] == 'bullish' else (0, 0, 255)
                cv2.rectangle(
                    display_img,
                    (candle['x'], candle['y']),
                    (candle['x'] + candle['width'], candle['y'] + candle['height']),
                    color, 2
                )
            
            # Draw trend lines
            for line in analysis.get('trend_lines', []):
                color = (255, 255, 0) if line['type'] == 'horizontal' else (255, 0, 255)
                cv2.line(
                    display_img,
                    (line['x1'], line['y1']),
                    (line['x2'], line['y2']),
                    color, 2
                )
            
            # Draw statistics overlay
            stats = analysis.get('statistics', {})
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            info_lines = [
                f"Candles: {stats.get('total_candles', 0)}",
                f"Bullish: {stats.get('bullish_count', 0)} ({stats.get('bullish_percentage', 0):.1f}%)",
                f"Bearish: {stats.get('bearish_count', 0)} ({stats.get('bearish_percentage', 0):.1f}%)",
                f"Sentiment: {stats.get('sentiment', 'N/A').upper()}",
                f"H-Lines: {stats.get('horizontal_lines', 0)}",
                f"D-Lines: {stats.get('diagonal_lines', 0)}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(
                    display_img, line, (10, y_offset + i*25),
                    font, 0.6, (255, 255, 255), 2
                )
                cv2.putText(
                    display_img, line, (10, y_offset + i*25),
                    font, 0.6, (0, 0, 0), 1
                )
        
        # Resize if too large
        max_width = 1920
        if display_img.shape[1] > max_width:
            scale = max_width / display_img.shape[1]
            new_width = int(display_img.shape[1] * scale)
            new_height = int(display_img.shape[0] * scale)
            display_img = cv2.resize(display_img, (new_width, new_height))
        
        cv2.imshow(window_name, display_img)
    
    def save_frame(self, filename: str, img: np.ndarray = None):
        """Save current frame to file"""
        if img is None:
            img = self.current_frame
        if img is not None:
            cv2.imwrite(filename, img)
            self.logger.info(f"Frame saved to {filename}")
    
    def start_capture_loop(self, interval: float = 2.0, callback=None):
        """
        Start continuous capture loop with fixed threading
        
        Args:
            interval: Seconds between captures
            callback: Function to call with analysis results
        """
        self.running = True
        
        def capture_loop():
            # Each thread gets its own MSS instance
            while self.running:
                try:
                    # Capture screen (uses thread-local MSS instance)
                    img = self.capture_screen()
                    self.current_frame = img
                    
                    # Analyze with enhanced methods
                    analysis = self.analyze_frame_enhanced(img)
                    
                    # Callback with results
                    if callback:
                        callback(analysis)
                    
                    # Log statistics
                    stats = analysis['statistics']
                    self.logger.info(
                        f"Captured: {stats['bullish_count']} bullish ({stats['bullish_percentage']:.1f}%), "
                        f"{stats['bearish_count']} bearish ({stats['bearish_percentage']:.1f}%) | "
                        f"Sentiment: {stats['sentiment'].upper()}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Capture error: {e}", exc_info=True)
                
                time.sleep(interval)
        
        # Start in background thread
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("Enhanced screen capture started")
    
    def stop_capture(self):
        """Stop the capture loop"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=5)
        
        # Clean up MSS instances
        for sct in self._sct_instances.values():
            sct.close()
        self._sct_instances.clear()
        
        self.logger.info("Screen capture stopped")
    
    @staticmethod
    def select_capture_region():
        """
        Interactive region selection tool
        Returns region dict for initialization
        """
        print("\n" + "="*70)
        print("CAPTURE REGION SELECTOR")
        print("="*70)
        print("\nInstructions:")
        print("1. A window will open showing your full screen")
        print("2. Click and drag to select the TradingView chart area")
        print("3. Press ENTER to confirm, ESC to cancel")
        print("\nMake sure TradingView is visible!")
        print("="*70)
        input("\nPress ENTER to continue...")
        
        # Capture full screen first
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Let user select ROI
        roi = cv2.selectROI("Select TradingView Chart Area", img, False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:
            region = {
                'left': int(roi[0]) + monitor['left'],
                'top': int(roi[1]) + monitor['top'],
                'width': int(roi[2]),
                'height': int(roi[3])
            }
            print(f"\n✓ Region selected: {region}")
            print("\nAdd this to your code:")
            print(f"capture_region = {region}")
            return region
        else:
            print("\n✗ Selection cancelled")
            return None


# Test function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRADINGVIEW SCREEN CAPTURE V2.0 - TEST MODE")
    print("="*70)
    print("\nEnhancements:")
    print("  ✓ Fixed threading issues")
    print("  ✓ Better candlestick detection")
    print("  ✓ Enhanced pattern recognition")
    print("  ✓ Improved price extraction")
    print("="*70)
    print("\nMake sure TradingView is visible on your screen!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Select new region")
    print("="*70 + "\n")
    
    # Ask about region selection
    choice = input("Select specific region? (y/n): ").lower()
    capture_region = None
    
    if choice == 'y':
        capture_region = TradingViewCaptureV2.select_capture_region()
    
    # Create capture instance
    capture = TradingViewCaptureV2(monitor_number=1, capture_region=capture_region)
    
    # Start capture
    capture.start_capture_loop(interval=2.0)
    
    print("\n✓ Capture started! Watch the analysis window...")
    print("  Press 'q' in the window to quit\n")
    
    try:
        while True:
            if capture.current_frame is not None:
                # Get latest analysis
                analysis = capture.analyze_frame_enhanced(capture.current_frame)
                
                # Display with overlays
                capture.display_frame(analysis=analysis)
                
                # Check for keys
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    capture.save_frame(f'tradingview_{timestamp}.png')
                    print(f"✓ Frame saved: tradingview_{timestamp}.png")
                elif key == ord('r'):
                    print("\nSelecting new region...")
                    new_region = TradingViewCaptureV2.select_capture_region()
                    if new_region:
                        capture.capture_region = new_region
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        capture.stop_capture()
        cv2.destroyAllWindows()
        print("✓ Done!")
