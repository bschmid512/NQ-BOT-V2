"""
TradingView Screen Capture & Analysis System
Real-time computer vision for trading signals
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
# Update this path if tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class TradingViewCapture:
    """
    Captures TradingView screen and extracts trading information
    """
    
    def __init__(self, monitor_number: int = 1, capture_region: Dict = None):
        """
        Initialize screen capture
        
        Args:
            monitor_number: Which monitor to capture (1, 2, etc.)
            capture_region: Specific region to capture {'top': y, 'left': x, 'width': w, 'height': h}
        """
        self.monitor_number = monitor_number
        self.capture_region = capture_region
        self.sct = mss.mss()
        self.running = False
        self.current_frame = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def get_monitor_region(self) -> Dict:
        """Get the monitor region to capture"""
        if self.capture_region:
            return self.capture_region
        
        # Get monitor info
        monitors = self.sct.monitors
        
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
        monitor = self.get_monitor_region()
        
        # Capture screen
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def extract_price_from_region(self, img: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Extract price from specific region using OCR
        
        Args:
            img: Image array
            region: (x, y, width, height) of region to extract from
            
        Returns:
            Extracted price as float or None
        """
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        try:
            text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789.,')
            text = text.strip().replace(',', '')
            
            if text:
                return float(text)
        except Exception as e:
            self.logger.debug(f"OCR extraction failed: {e}")
            
        return None
    
    def detect_candlesticks(self, img: np.ndarray) -> List[Dict]:
        """
        Detect candlesticks in the chart
        
        Args:
            img: Image array
            
        Returns:
            List of detected candlesticks with their properties
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for green (bullish) candles
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Define color ranges for red (bearish) candles
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
        
        # Find contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candlesticks = []
        
        # Process green candles
        for contour in green_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                candlesticks.append({
                    'type': 'bullish',
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area
                })
        
        # Process red candles
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                candlesticks.append({
                    'type': 'bearish',
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area
                })
        
        return candlesticks
    
    def detect_trend_lines(self, img: np.ndarray) -> List[Dict]:
        """
        Detect trend lines and support/resistance levels
        
        Args:
            img: Image array
            
        Returns:
            List of detected lines with their properties
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)
        
        detected_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter for mostly horizontal lines (support/resistance)
                if abs(angle) < 15 or abs(angle - 180) < 15:
                    detected_lines.append({
                        'type': 'horizontal',
                        'angle': angle,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    })
                # Filter for diagonal trend lines
                elif 15 < abs(angle) < 75:
                    detected_lines.append({
                        'type': 'diagonal',
                        'angle': angle,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    })
        
        return detected_lines
    
    def analyze_frame(self, img: np.ndarray) -> Dict:
        """
        Analyze a single frame and extract all trading information
        
        Args:
            img: Image array
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'candlesticks': [],
            'trend_lines': [],
            'price': None,
            'patterns': []
        }
        
        # Detect candlesticks
        analysis['candlesticks'] = self.detect_candlesticks(img)
        
        # Detect trend lines
        analysis['trend_lines'] = self.detect_trend_lines(img)
        
        # Count bullish vs bearish candles
        bullish_count = sum(1 for c in analysis['candlesticks'] if c['type'] == 'bullish')
        bearish_count = sum(1 for c in analysis['candlesticks'] if c['type'] == 'bearish')
        
        analysis['bullish_count'] = bullish_count
        analysis['bearish_count'] = bearish_count
        analysis['sentiment'] = 'bullish' if bullish_count > bearish_count else 'bearish'
        
        return analysis
    
    def start_capture_loop(self, interval: float = 1.0, callback=None):
        """
        Start continuous capture loop
        
        Args:
            interval: Seconds between captures
            callback: Function to call with analysis results
        """
        self.running = True
        
        def capture_loop():
            while self.running:
                try:
                    # Capture screen
                    img = self.capture_screen()
                    self.current_frame = img
                    
                    # Analyze
                    analysis = self.analyze_frame(img)
                    
                    # Callback with results
                    if callback:
                        callback(analysis)
                    
                    # Log basic stats
                    self.logger.info(
                        f"Captured: {analysis['bullish_count']} bullish, "
                        f"{analysis['bearish_count']} bearish | "
                        f"Sentiment: {analysis['sentiment']}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Capture error: {e}")
                
                time.sleep(interval)
        
        # Start in background thread
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("Screen capture started")
    
    def stop_capture(self):
        """Stop the capture loop"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2)
        self.logger.info("Screen capture stopped")
    
    def save_frame(self, filename: str):
        """Save current frame to file"""
        if self.current_frame is not None:
            cv2.imwrite(filename, self.current_frame)
            self.logger.info(f"Frame saved to {filename}")
    
    def display_frame(self, img: np.ndarray = None, analysis: Dict = None):
        """
        Display frame with overlays
        
        Args:
            img: Image to display (uses current_frame if None)
            analysis: Analysis results to overlay
        """
        if img is None:
            img = self.current_frame
            
        if img is None:
            return
        
        display_img = img.copy()
        
        if analysis:
            # Draw candlesticks
            for candle in analysis.get('candlesticks', []):
                color = (0, 255, 0) if candle['type'] == 'bullish' else (0, 0, 255)
                cv2.rectangle(display_img, 
                            (candle['x'], candle['y']),
                            (candle['x'] + candle['width'], candle['y'] + candle['height']),
                            color, 2)
            
            # Draw trend lines
            for line in analysis.get('trend_lines', []):
                color = (255, 0, 0) if line['type'] == 'horizontal' else (255, 255, 0)
                cv2.line(display_img,
                        (line['x1'], line['y1']),
                        (line['x2'], line['y2']),
                        color, 2)
            
            # Add text info
            sentiment = analysis.get('sentiment', 'unknown')
            text = f"Sentiment: {sentiment} | Bulls: {analysis.get('bullish_count', 0)} Bears: {analysis.get('bearish_count', 0)}"
            cv2.putText(display_img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show
        cv2.imshow('TradingView Analysis', display_img)
        cv2.waitKey(1)


# Test function
if __name__ == "__main__":
    print("TradingView Screen Capture Test")
    print("=" * 60)
    print()
    print("This will capture your screen and analyze TradingView charts.")
    print("Make sure TradingView is visible on your screen!")
    print()
    print("Press 'q' to quit, 's' to save frame")
    print()
    
    # Create capture instance
    capture = TradingViewCapture(monitor_number=1)
    
    # Define callback for analysis results
    def on_analysis(analysis):
        """Called with each analysis result"""
        # You can process results here
        # For now, we'll just display them
        pass
    
    # Start capture
    capture.start_capture_loop(interval=2.0, callback=on_analysis)
    
    print("Capturing... Press 'q' to quit")
    
    try:
        while True:
            if capture.current_frame is not None:
                # Get latest analysis
                analysis = capture.analyze_frame(capture.current_frame)
                
                # Display with overlays
                capture.display_frame(analysis=analysis)
                
                # Check for quit
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    capture.save_frame(f'tradingview_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                    print("Frame saved!")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        capture.stop_capture()
        cv2.destroyAllWindows()
        print("Done!")
