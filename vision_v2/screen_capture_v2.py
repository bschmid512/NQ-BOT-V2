"""
TradingView Screen Capture & Analysis System - Version 2.0
IMPROVEMENTS:
- Fixed threading issues with MSS
- Better candlestick detection with size filtering
- Enhanced price extraction with multiple OCR attempts
- Improved pattern recognition
- Better error handling
- Region selection helper
- ADDED: Display mode control to prevent unwanted overlays
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
    
    # --- START DEBUGGING ---
    # This will ensure we only save the debug images once per run
    debug_ocr_saved = False
    debug_candle_saved = False # NEW
    # --- END DEBUGGING ---
    
    def __init__(self, 
                 monitor_number: int = 1, 
                 capture_region: Dict = None,  # This will be treated as the CHART region
                 price_region: Dict = None, 
                 display_mode: bool = True):
        """
        Initialize screen capture
        
        Args:
            monitor_number: Which monitor to capture (1, 2, etc.)
            capture_region: Specific region for the CHART
            price_region: Specific region for price OCR
            display_mode: Enable/disable visual overlays (False for testing)
        """
        self.monitor_number = monitor_number
        self.chart_region = capture_region  # User's chart selection
        self.price_region = price_region # User's price selection
        self.display_mode = display_mode
        self.running = False
        self.current_frame = None
        self.logger = logging.getLogger(__name__)
        self._sct_instances = {}
        
        # This is the new region that MSS will capture, combining both chart and price
        self.combined_capture_region = self._calculate_combined_region()

        # Detection parameters
        self.min_candle_area = 100  # Minimum area for valid candle
        self.max_candle_area = 50000  # Maximum area to filter out noise
        
    def _get_sct(self):
        """Get MSS instance for current thread (fixes threading issue)"""
        thread_id = threading.get_ident()
        if thread_id not in self._sct_instances:
            self._sct_instances[thread_id] = mss.mss()
        return self._sct_instances[thread_id]

    def _calculate_combined_region(self) -> Dict:
        """Calculates a single bounding box to capture all selected regions."""
        
        regions = []
        if self.chart_region:
            regions.append(self.chart_region)
        if self.price_region:
            regions.append(self.price_region)

        # If no regions, capture primary monitor
        if not regions:
            self.logger.info("No regions selected, capturing full monitor.")
            with mss.mss() as sct:
                return sct.monitors[self.monitor_number]
            
        # If regions exist, calculate combined bounding box
        
        left = min(r['left'] for r in regions)
        top = min(r['top'] for r in regions)
        
        right = max(r['left'] + r['width'] for r in regions)
        bottom = max(r['top'] + r['height'] for r in regions)
        
        combined = {
            'left': left,
            'top': top,
            'width': right - left,
            'height': bottom - top
        }
        self.logger.info(f"Chart region: {self.chart_region}")
        self.logger.info(f"Price region: {self.price_region}")
        self.logger.info(f"Combined capture region calculated: {combined}")
        return combined

    def get_monitor_region(self) -> Dict:
        """Get the monitor region to capture"""
        # This function now returns the new combined region
        return self.combined_capture_region
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture screen and return as numpy array
        
        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        sct = self._get_sct()
        # Grabs the new combined_capture_region
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
        
        # Ensure region is valid
        if w <= 0 or h <= 0:
            self.logger.warning(f"Invalid OCR region: {region}")
            return None
        
        # Ensure region is within the image dimensions
        img_h, img_w = img.shape[:2]
        if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
            self.logger.warning(f"OCR region {region} is outside image bounds ({img_w}x{img_h})")
            # Try to clip it
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            if w <= 0 or h <= 0:
                self.logger.error("Clipped OCR region is invalid.")
                return None
            self.logger.warning(f"Using clipped OCR region: {(x,y,w,h)}")
        
        roi = img[y:y+h, x:x+w]
        
        methods = [
            ("method1_otsu", self._preprocess_method1),
            ("method2_adaptive", self._preprocess_method2),
            ("method3_morph", self._preprocess_method3),
            ("method4_grayscale", self._preprocess_method4),
            ("method5_inverted", self._preprocess_method5),
        ]
        
        # Save debug images on the first run
        if not TradingViewCaptureV2.debug_ocr_saved:
            try:
                cv2.imwrite("debug_ocr_raw_region.png", roi)
                self.logger.info("Saved 'debug_ocr_raw_region.png'")
                for name, func in methods:
                    preprocessed_img = func(roi)
                    cv2.imwrite(f"debug_ocr_{name}.png", preprocessed_img)
                self.logger.info("Saved all preprocessed debug images.")
                TradingViewCaptureV2.debug_ocr_saved = True
            except Exception as e:
                self.logger.error(f"Failed to save debug OCR image: {e}")
        
        for name, method in methods:
            try:
                preprocessed = method(roi)
                text = pytesseract.image_to_string(
                    preprocessed, 
                    config='--psm 7 -c tessedit_char_whitelist=0123456789.,'
                )
                text = text.strip().replace(',', '').replace(' ', '')
            except ValueError:
                    continue
            if text:
                try:
                    price = float(text)
                    if 10000 < price < 30000:
                        self.logger.info(f"OCR Success with {name}: {price}")
                        return price
                except ValueError:
                    continue
                
        return None
    
    def _resize_for_ocr(self, img: np.ndarray, target_height: int = 50) -> np.ndarray:
        """Resize image to a target height for better OCR, maintaining aspect ratio."""
        try:
            h, w = img.shape[:2]
            if h == 0 or h == target_height:
                return img
            scale = target_height / h
            target_width = int(w * scale)
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            return resized
        except Exception as e:
            self.logger.warning(f"Could not resize OCR image: {e}")
            return img

    def _preprocess_method1(self, roi: np.ndarray) -> np.ndarray:
        """Standard preprocessing - grayscale + threshold"""
        resized = self._resize_for_ocr(roi)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _preprocess_method2(self, roi: np.ndarray) -> np.ndarray:
        """Adaptive threshold"""
        resized = self._resize_for_ocr(roi)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    def _preprocess_method3(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced with morphology"""
        resized = self._resize_for_ocr(roi)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morphology
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph
    
    def _preprocess_method4(self, roi: np.ndarray) -> np.ndarray:
        """Simple grayscale"""
        resized = self._resize_for_ocr(roi)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray
    
    def _preprocess_method5(self, roi: np.ndarray) -> np.ndarray:
        """Inverted binary threshold (good for light text on dark bg)"""
        resized = self._resize_for_ocr(roi)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    
    def detect_candlesticks_enhanced(self, img: np.ndarray) -> List[Dict]:
        """
        Enhanced candlestick detection with better filtering
        
        Args:
            img: Image array (THIS SHOULD BE THE CROPPED CHART IMAGE)
            
        Returns:
            List of detected candlesticks with properties
        """
        # Convert to HSV for better color detection

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Green (bullish) – classic + teal band (dark theme)
        green_lower1 = np.array([35,  30, 30])
        green_upper1 = np.array([100, 255, 255])
        green_lower2 = np.array([85,  20, 30])
        green_upper2 = np.array([125, 255, 255])
        green_mask  = cv2.inRange(hsv, green_lower1, green_upper1)
        green_mask |= cv2.inRange(hsv, green_lower2, green_upper2)

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
        
        # --- START NEW DEBUGGING CODE ---
        # Save debug images on the first run
        if not TradingViewCaptureV2.debug_candle_saved:
            try:
                cv2.imwrite("debug_chart_image_raw.png", img) # Save the cropped chart
                cv2.imwrite("debug_candle_green_mask.png", green_mask)
                cv2.imwrite("debug_candle_red_mask.png", red_mask)
                self.logger.info("Saved candle detection debug images.")
                TradingViewCaptureV2.debug_candle_saved = True
            except Exception as e:
                self.logger.error(f"Failed to save candle debug images: {e}")
        # --- END NEW DEBUGGING CODE ---
        
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
        
        # Sort by x-coordinate (left to right)
        candlesticks.sort(key=lambda c: c['x'])
        
        return candlesticks
    
    def detect_trend_lines(self, img: np.ndarray) -> List[Dict]:
        """
        Detect horizontal and diagonal lines (support/resistance, trendlines)
        
        Args:
            img: Image array (THIS SHOULD BE THE CROPPED CHART IMAGE)
            
        Returns:
            List of detected lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        detected_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                else:
                    angle = 90
                
                # Classify line type
                if abs(angle) < 5:  # Nearly horizontal
                    line_type = 'horizontal'
                elif abs(angle) > 85:  # Nearly vertical (ignore)
                    continue
                else:
                    line_type = 'diagonal'
                
                detected_lines.append({
                    'type': line_type,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'angle': float(angle),
                    'length': float(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                })
        
        return detected_lines
    
    def analyze_frame_enhanced(self, img: np.ndarray) -> Dict:
        """
        Comprehensive frame analysis with enhanced detection
        
        Args:
            img: Image array to analyze (THIS IS THE *COMBINED* IMAGE)
            
        Returns:
            Dictionary with analysis results
        """
        
        # 1. Extract Price from the combined image
        current_price = None
        if self.price_region:
            # Calculate price region's coordinates *relative* to the combined image
            relative_price_x = self.price_region['left'] - self.combined_capture_region['left']
            relative_price_y = self.price_region['top'] - self.combined_capture_region['top']
            
            price_region_coords = (
                relative_price_x,
                relative_price_y,
                self.price_region['width'],
                self.price_region['height']
            )
            current_price = self.extract_price_from_region(img, price_region_coords)
        
        if current_price:
            self.logger.info(f"Extracted price: {current_price}")
        else:
            self.logger.warning("Could not extract price. OCR failed or price not in region.")

        # 2. Extract Chart from the combined image
        if self.chart_region:
            relative_chart_x = self.chart_region['left'] - self.combined_capture_region['left']
            relative_chart_y = self.chart_region['top'] - self.combined_capture_region['top']
            
            # Crop the combined image to get *only* the chart
            img_chart = img[
                relative_chart_y : relative_chart_y + self.chart_region['height'],
                relative_chart_x : relative_chart_x + self.chart_region['width']
            ]
        else:
            # If no chart region specified, just use the whole image
            img_chart = img

        # 3. Run all analysis on the *chart image* (img_chart)
        
        # Detect candlesticks
        candlesticks = self.detect_candlesticks_enhanced(img_chart)
        
        # Detect trend lines
        trend_lines = self.detect_trend_lines(img_chart)
        
        # Calculate statistics
        bullish_count = sum(1 for c in candlesticks if c['type'] == 'bullish')
        bearish_count = sum(1 for c in candlesticks if c['type'] == 'bearish')
        total_candles = len(candlesticks)

        bullish_pct = (bullish_count / total_candles) * 100 if total_candles else 0.0
        bearish_pct = (bearish_count / total_candles) * 100 if total_candles else 0.0

        # Uptrend override guard (dark-theme color misread)
        N = min(30, total_candles)
        if N >= 12:
            xs = np.array([c['x'] for c in candlesticks[-N:]])
            ys = np.array([c['y'] for c in candlesticks[-N:]])  # lower y = higher price on screen
            slope = np.polyfit(xs, ys, 1)[0]                    # uptrend → negative slope in pixel space
            if bearish_pct > 70 and slope < -0.03:
                bullish_pct, bearish_pct = bearish_pct, bullish_pct

        # Softer threshold → fewer false "neutral"
        if bullish_pct > 55:
            sentiment = 'bullish'
        elif bearish_pct > 55:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        
        # Count line types
        horizontal_lines = sum(1 for l in trend_lines if l['type'] == 'horizontal')
        diagonal_lines = sum(1 for l in trend_lines if l['type'] == 'diagonal')
        # Get chart dimensions for spatial filtering
        chart_height, chart_width = img_chart.shape[:2]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'chart_width': chart_width,  # ← MOVED TO TOP LEVEL (was inside statistics)
            'candlesticks': candlesticks,
            'trend_lines': trend_lines,
            'statistics': {
                'total_candles': total_candles,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'bullish_percentage': bullish_pct,
                'bearish_percentage': bearish_pct,
                'sentiment': sentiment,
                'horizontal_lines': horizontal_lines,
                'diagonal_lines': diagonal_lines
            }
        }
    
    def display_frame(self, img: np.ndarray = None, analysis: Dict = None, window_name: str = "TradingView Analysis"):
        """
        Display frame with optional overlays (only if display_mode is enabled)
        
        Args:
            img: Image to display (uses current_frame if None)
            analysis: Analysis results to overlay
            window_name: Name for the display window
        """
        # SKIP if display mode is disabled
        if not self.display_mode:
            return
        
        if img is None:
            img = self.current_frame
        if img is None:
            return
        
        display_img = img.copy()
        
        # All analysis coordinates (candlesticks, lines) are relative to the chart,
        # so we must offset them to draw on the combined image.
        
        offset_x = 0
        offset_y = 0
        if self.chart_region:
            offset_x = self.chart_region['left'] - self.combined_capture_region['left']
            offset_y = self.chart_region['top'] - self.combined_capture_region['top']

        if analysis:
            # Draw candlesticks
            for candle in analysis.get('candlesticks', []):
                color = (0, 255, 0) if candle['type'] == 'bullish' else (0, 0, 255)
                cv2.rectangle(
                    display_img,
                    (candle['x'] + offset_x, candle['y'] + offset_y),
                    (candle['x'] + candle['width'] + offset_x, candle['y'] + candle['height'] + offset_y),
                    color, 2
                )
            
            # Draw trend lines
            for line in analysis.get('trend_lines', []):
                color = (255, 255, 0) if line['type'] == 'horizontal' else (255, 0, 255)
                cv2.line(
                    display_img,
                    (line['x1'] + offset_x, line['y1'] + offset_y),
                    (line['x2'] + offset_x, line['y2'] + offset_y),
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
            
            if analysis.get('current_price'):
                info_lines.append(f"Price: {analysis['current_price']:.2f}")
            else:
                info_lines.append("Price: N/A")

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
        
        # Clean up MSS instances safely
        for thread_id, sct in list(self._sct_instances.items()):
            try:
                sct.close()
            except Exception as e:
                self.logger.debug(f"Error closing MSS instance: {e}")
        self._sct_instances.clear()
        
        self.logger.info("Screen capture stopped")
    
    @staticmethod
    def select_capture_regions():
        """
        Interactive region selection tool for CHART and PRICE AXIS
        Returns a tuple of (chart_region, price_region)
        """
        print("\n" + "="*70)
        print("CAPTURE REGION SELECTOR (2-STEP)")
        print("="*70)
        print("\nInstructions:")
        print("1. You will be asked to select the CHART area.")
        print("2. Then, you will be asked to select the PRICE AXIS area.")
        print("3. Press ENTER to confirm each selection, ESC to cancel")
        print("\nMake sure TradingView is visible!")
        print("="*70)
        input("\nPress ENTER to continue...")
        
        chart_region = None
        price_region = None
        
        # Capture full screen first
        with mss.mss() as sct:
            monitor = sct.monitors[1] # Use monitor 1 (primary)
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 1. Select CHART Region
        print("\nSTEP 1: Select the CHART Area (the candlesticks)")
        print("Press SPACE or ENTER to confirm.")
        roi_chart = cv2.selectROI("STEP 1: Select CHART Area", img, False)
        
        if roi_chart[2] > 0 and roi_chart[3] > 0:
            chart_region = {
                'left': int(roi_chart[0]) + monitor['left'],
                'top': int(roi_chart[1]) + monitor['top'],
                'width': int(roi_chart[2]),
                'height': int(roi_chart[3])
            }
            print(f"✓ Chart region selected: {chart_region}")
        else:
            print("\n✗ Selection cancelled")
            cv2.destroyAllWindows()
            return None, None
        
        # 2. Select PRICE Region
        print("\nSTEP 2: Select the PRICE AXIS Area (the updating price number)")
        print("Press SPACE or ENTER to confirm.")
        # Re-show the image for the second selection
        roi_price = cv2.selectROI("STEP 2: Select PRICE AXIS Area", img, False)
        
        if roi_price[2] > 0 and roi_price[3] > 0:
            price_region = {
                'left': int(roi_price[0]) + monitor['left'],
                'top': int(roi_price[1]) + monitor['top'],
                'width': int(roi_price[2]),
                'height': int(roi_price[3])
            }
            print(f"✓ Price region selected: {price_region}")
        else:
            print("\n✗ Price region selection cancelled")
        
        cv2.destroyAllWindows()
        
        return chart_region, price_region


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
    
    # Region selection
    (capture_region, price_region) = TradingViewCaptureV2.select_capture_regions()
    
    if not capture_region:
        print("\n✗ No chart region selected, exiting...")
        exit()
    
    if not price_region:
        print("\n⚠ WARNING: No price region selected. OCR will fail.")
    
    print("\n" + "="*70)
    print("ENHANCED SCREEN CAPTURE STARTED")
    print("="*70 + "\n")
    
    # Create capture instance with display enabled
    capture = TradingViewCaptureV2(
        monitor_number=1, 
        capture_region=capture_region,
        price_region=price_region,
        display_mode=True  # Enable visualization for interactive mode
    )
    
    # Start capture
    capture.start_capture_loop(interval=2.0)
    
    print("✓ Capture started! Watch the analysis window...")
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
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        capture.stop_capture()
        cv2.destroyAllWindows()
        print("✓ Done!")