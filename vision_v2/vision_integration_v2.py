"""
Complete Vision Trading Integration V2.0
Integrates enhanced computer vision with your NQ trading bot

NEW FEATURES:
- Fixed threading issues
- Better pattern detection
- Multiple chart patterns
- Support/resistance levels
- Real-time signal generation
- Easy region selection
- Performance tracking
"""
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List
import json
import cv2

# Add parent directory for imports
bot_dir = Path(__file__).parent
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

# Import enhanced vision modules
from vision_v2.screen_capture_v2 import TradingViewCaptureV2
from vision_v2.pattern_recognition_v2 import TradingViewAIV2

# Try to import bot components
try:
    from core.position_manager import position_manager
    from utils.data_handler import data_handler
    from utils.logger import trading_logger
    BOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Could not connect to trading bot: {e}")
    print("Running in ANALYSIS MODE (signals will be logged only)")
    BOT_AVAILABLE = False


class VisionTradingV2:
    """
    Enhanced integration of computer vision with NQ trading bot
    """
    
    def __init__(self,
                 monitor_number: int = 1,
                 capture_region: Dict = None,
                 price_region: Dict = None, # --- MODIFICATION: Added price_region ---
                 capture_interval: float = 2.0,
                 min_confidence: float = 0.75,
                 trading_enabled: bool = False):
        """
        Initialize vision trading system
        
        Args:
            monitor_number: Which monitor to capture
            capture_region: Specific region to capture
            price_region: Specific region for price OCR
            capture_interval: Seconds between captures
            min_confidence: Minimum signal confidence to act on
            trading_enabled: Enable automatic trading
        """
        self.logger = self._setup_logging()
        
        # --- MODIFICATION: Pass price_region to capture class ---
        self.capture = TradingViewCaptureV2(
            monitor_number, 
            capture_region, 
            price_region, 
            display_mode=False
        )
        self.ai = TradingViewAIV2()
        
        # Configuration
        self.capture_interval = capture_interval
        self.min_confidence = min_confidence
        self.trading_enabled = trading_enabled and BOT_AVAILABLE
        
        # Trading state
        self.last_signal_time = None
        self.signal_cooldown = 60  # seconds between signals
        self.active_signals = []
        
        # Performance tracking
        self.session_start = datetime.now()
        self.signals_generated = 0
        self.trades_executed = 0
        
        self.logger.info("Vision Trading V2.0 initialized")
        self.logger.info(f"Trading mode: {'ENABLED' if self.trading_enabled else 'ANALYSIS ONLY'}")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger('VisionTradingV2')
        logger.setLevel(logging.INFO)
        
        # Stop logs from propagating to the root logger (which causes duplicates)
        logger.propagate = False
        
        # Avoid duplicate handlers if this class is initialized multiple times
        if logger.handlers:
            return logger
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        fh = logging.FileHandler(log_dir / 'vision_trading_v2.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def process_analysis(self, analysis: Dict):
        """
        Process frame analysis and take action on signals
        
        Args:
            analysis: Analysis dictionary from vision system
        """
        # Run AI processing
        analysis = self.ai.process_frame(analysis)
        
        signals = analysis.get('signals', [])
        
        if not signals:
            return
        
        self.signals_generated += len(signals)
        
        # Filter high-confidence actionable signals
        actionable = [
            s for s in signals 
            if s['confidence'] >= self.min_confidence 
            and s['signal'] in ['LONG', 'SHORT']
        ]
        
        if not actionable:
            return
        
        # Check cooldown
        now = datetime.now()
        if self.last_signal_time:
            time_since_last = (now - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                self.logger.debug(f"Signal cooldown active ({time_since_last:.0f}s / {self.signal_cooldown}s)")
                return
        
        # Take the highest confidence signal
        best_signal = actionable[0]
        
        self.logger.info("="*70)
        self.logger.info(f"HIGH CONFIDENCE SIGNAL DETECTED:")
        self.logger.info(f"  Pattern: {best_signal['pattern']}")
        self.logger.info(f"  Direction: {best_signal['signal']}")
        self.logger.info(f"  Confidence: {best_signal['confidence']:.1%}")
        
        # Display Entry, TP, SL
        if best_signal.get('entry'):
            self.logger.info(f"  Entry: {best_signal['entry']:.2f}")
            self.logger.info(f"  Stop Loss: {best_signal['stop_loss']:.2f} ({best_signal['risk_points']} pts)")
            self.logger.info(f"  Take Profit: {best_signal['take_profit']:.2f} ({best_signal['reward_points']} pts)")
            self.logger.info(f"  Risk/Reward: 1:{best_signal['risk_reward_ratio']}")
        
        self.logger.info(f"  Reason: {best_signal['reason']}")
        self.logger.info("="*70)
        
        # Execute trade if enabled
        if self.trading_enabled and BOT_AVAILABLE:
            self._execute_trade(best_signal, analysis)
        
        self.last_signal_time = now
        self.active_signals.append({
            'timestamp': now.isoformat(),
            'signal': best_signal
        })
    
    def _execute_trade(self, signal: Dict, analysis: Dict):
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal dictionary
            analysis: Full analysis for context
        """
        try:
            # Create signal format for position_manager
            trade_signal = {
                'strategy': 'vision',
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'price': None,  # Will be filled by position_manager
                'timestamp': datetime.now(),
                'pattern': signal['pattern'],
                'reason': signal['reason']
            }
            
            self.logger.info(f"Executing {signal['signal']} trade via vision system...")
            
            # Send to position_manager
            position_manager.open_position(trade_signal)
            
            self.trades_executed += 1
            self.logger.info(f"✓ Trade executed successfully (Total: {self.trades_executed})")
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}", exc_info=True)
    
    def run(self, duration_minutes: Optional[int] = None):
        """
        Run the vision trading system
        
        Args:
            duration_minutes: How long to run (None = unlimited)
        """
        self.logger.info("="*70)
        self.logger.info("VISION TRADING V2.0 - STARTING")
        self.logger.info("="*70)
        self.logger.info(f"Capture interval: {self.capture_interval}s")
        self.logger.info(f"Min confidence: {self.min_confidence:.0%}")
        self.logger.info(f"Trading: {'ENABLED' if self.trading_enabled else 'ANALYSIS ONLY'}")
        if duration_minutes:
            self.logger.info(f"Duration: {duration_minutes} minutes")
        else:
            self.logger.info("Duration: Unlimited (press Ctrl+C to stop)")
        self.logger.info("="*70)
        
        end_time = None
        if duration_minutes:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        try:
            while True:
                # Check duration
                if end_time and datetime.now() >= end_time:
                    self.logger.info("Duration elapsed - stopping")
                    break
                
                # Capture and analyze
                img = self.capture.capture_screen()
                analysis = self.capture.analyze_frame_enhanced(img)
                
                # Process signals
                self.process_analysis(analysis)
                
                # Wait for next capture
                time.sleep(self.capture_interval)
        
        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error in run loop: {e}", exc_info=True)
        
        finally:
            self._print_summary()
            self._cleanup()
        
        self.logger.info("Vision system stopped")
    
    def _cleanup(self):
        """Clean up resources safely"""
        try:
            # Stop capture thread
            if hasattr(self.capture, 'running') and self.capture.running:
                self.capture.running = False
                
            # Give thread time to finish
            time.sleep(0.5)
            
            # Close windows
            try:
                cv2.destroyAllWindows()
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    
    def _print_summary(self):
        """Print session summary"""
        duration = datetime.now() - self.session_start
        
        self.logger.info("\n" + "="*70)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Signals generated: {self.signals_generated}")
        self.logger.info(f"Trades executed: {self.trades_executed}")
        self.logger.info("="*70)


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("VISION TRADING SYSTEM V2.0")
    print("="*70)
    print("\nEnhancements:")
    print("  ✓ Fixed threading issues")
    print("  ✓ Better candlestick detection") 
    print("  ✓ Multiple chart patterns (engulfing, hammer, stars, double tops/bottoms)")
    print("  ✓ Support/resistance level detection")
    print("  ✓ Enhanced pattern recognition")
    print("  ✓ Real-time signal generation")
    print("="*70)
    
    # Check bot availability
    if BOT_AVAILABLE:
        print("\n✓ Successfully connected to your trading bot")
        print("  - position_manager: Available")
        print("  - data_handler: Available")
    else:
        print("\n⚠ Could not connect to trading bot")
        print("  Running in ANALYSIS MODE (signals logged only)")
    
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)
    
    # --- START MODIFICATION ---
    # Region selection
    print("\nWould you like to:")
    print("  [1] Capture full screen (Price OCR may fail)")
    print("  [2] Select specific regions (Recommended)")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    capture_region = None
    price_region = None
    if choice == '2':
        print("\nOpening region selector...")
        # Now returns two values
        (capture_region, price_region) = TradingViewCaptureV2.select_capture_regions()
        if not capture_region:
            print("Using full screen capture")
        if not price_region:
            print("⚠ WARNING: No price region selected. OCR will likely fail.")
    
    # --- END MODIFICATION ---
    
    # Trading mode
    trading_enabled = False
    if BOT_AVAILABLE:
        print("\n" + "="*70)
        print("TRADING MODE")
        print("="*70)
        print("\n  [1] Analysis Mode (SAFE - signals logged only, no trades)")
        print("  [2] Trading Mode (signals execute trades automatically)")
        
        mode_choice = input("\nChoice (1 or 2): ").strip()
        
        if mode_choice == '2':
            print("\n⚡ WARNING: AUTOMATIC TRADING MODE")
            print("Signals will execute real trades via position_manager")
            confirm = input("Are you SURE? Type 'YES' to confirm: ").strip()
            
            if confirm.upper() == 'YES':
                trading_enabled = True
                print("✓ Trading mode ENABLED")
            else:
                print("✓ Staying in analysis mode")
    
    # Configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    capture_interval = float(input("\nCapture interval (seconds) [default: 2.0]: ").strip() or "2.0")
    min_confidence = float(input("Minimum confidence (0.0-1.0) [default: 0.75]: ").strip() or "0.75")
    
    # Duration
    duration_input = input("\nRun duration (minutes, or 'unlimited'): ").strip().lower()
    duration_minutes = None if duration_input == 'unlimited' else int(duration_input or "60")
    
    # Create and run system
    print("\n" + "="*70)
    print("STARTING SYSTEM")
    print("="*70)
    print("\nMake sure TradingView is visible on your screen!")
    print("Press Ctrl+C or 'q' in the window to stop")
    print("="*70)
    
    input("\nPress ENTER to start...")
    
    # --- MODIFICATION: Pass price_region ---
    system = VisionTradingV2(
        monitor_number=1,
        capture_region=capture_region,
        price_region=price_region,
        capture_interval=capture_interval,
        min_confidence=min_confidence,
        trading_enabled=trading_enabled
    )
    
    system.run(duration_minutes=duration_minutes)


if __name__ == "__main__":
    main()