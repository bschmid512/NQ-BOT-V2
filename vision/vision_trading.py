"""
Integration Module - Connect Computer Vision to Trading Bot
*** UPDATED VERSION WITH YOUR BOT INTEGRATION ***
"""
import sys
from pathlib import Path
import time
from datetime import datetime
import logging
from typing import Dict, Optional

# Add parent directory to path for imports
bot_dir = Path(__file__).parent.parent
sys.path.insert(0, str(bot_dir))

from vision.screen_capture import TradingViewCapture
from vision.pattern_recognition import TradingViewAI

# Import from your existing bot
try:
    from position_manager import position_manager
    from utils.data_handler import data_handler
    from utils.logger import trading_logger
    BOT_AVAILABLE = True
    print("✓ Successfully connected to your trading bot!")
except ImportError as e:
    print(f"Warning: Could not connect to trading bot: {e}")
    print("Running in standalone mode (signals will be logged only)")
    BOT_AVAILABLE = False


class VisionTradingIntegration:
    """
    Integrates computer vision analysis with YOUR NQ trading bot
    """
    
    def __init__(self, 
                 monitor_number: int = 1,
                 capture_region: Dict = None,
                 capture_interval: float = 2.0,
                 min_confidence: float = 0.7):
        """
        Initialize integration
        
        Args:
            monitor_number: Which monitor to capture
            capture_region: Specific region to capture
            capture_interval: Seconds between captures
            min_confidence: Minimum signal confidence to trade
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        log_file = 'logs/vision_trading.log' if Path('logs').exists() else 'vision_trading.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize components
        self.capture = TradingViewCapture(monitor_number, capture_region)
        self.ai = TradingViewAI()
        
        self.capture_interval = capture_interval
        self.min_confidence = min_confidence
        
        # Trading state
        self.enabled = True
        self.last_signal_time = None
        self.signal_cooldown = 60  # seconds between signals
        
        # Stop loss / take profit settings
        self.stop_loss_points = 20   # Points for stop loss
        self.take_profit_points = 40  # Points for take profit
        
        self.logger.info("=" * 70)
        self.logger.info("Vision Trading Integration initialized")
        self.logger.info("=" * 70)
        self.logger.info(f"Bot integration: {'ENABLED' if BOT_AVAILABLE else 'DISABLED (logs only)'}")
        self.logger.info(f"Capture interval: {capture_interval}s")
        self.logger.info(f"Min confidence: {min_confidence:.1%}")
        self.logger.info(f"Stop loss: {self.stop_loss_points} points")
        self.logger.info(f"Take profit: {self.take_profit_points} points")
        self.logger.info("=" * 70)
    
    def on_frame_analyzed(self, analysis: Dict):
        """
        Callback when a frame is analyzed
        
        Args:
            analysis: Frame analysis results
        """
        # Process with AI
        result = self.ai.process_frame(analysis)
        
        # Check if we should trade
        if self.enabled and self._can_generate_signal():
            should_trade, best_signal = self.ai.should_take_trade(self.min_confidence)
            
            if should_trade:
                self._execute_signal(best_signal)
    
    def _can_generate_signal(self) -> bool:
        """Check if enough time has passed since last signal"""
        if self.last_signal_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_signal_time).total_seconds()
        return elapsed >= self.signal_cooldown
    
    def _calculate_stops_targets(self, signal: Dict, entry_price: float) -> tuple:
        """
        Calculate stop loss and take profit for CV signals
        
        Args:
            signal: CV signal
            entry_price: Entry price
            
        Returns:
            (stop_loss, take_profit)
        """
        if signal['signal'] == 'LONG':
            stop_loss = entry_price - self.stop_loss_points
            take_profit = entry_price + self.take_profit_points
        else:  # SHORT
            stop_loss = entry_price + self.stop_loss_points
            take_profit = entry_price - self.take_profit_points
        
        return stop_loss, take_profit
    
    def _execute_signal(self, signal: Dict):
        """
        Execute a trading signal
        
        Args:
            signal: Signal dict with pattern, signal, confidence, reason
        """
        self.logger.info("=" * 70)
        self.logger.info("🎯 TRADING SIGNAL DETECTED")
        self.logger.info("=" * 70)
        self.logger.info(f"Pattern: {signal['pattern']}")
        self.logger.info(f"Direction: {signal['signal']}")
        self.logger.info(f"Confidence: {signal['confidence']:.1%}")
        self.logger.info(f"Reason: {signal['reason']}")
        self.logger.info("=" * 70)
        
        # Update last signal time
        self.last_signal_time = datetime.now()
        
        # If bot is available, execute trade
        if BOT_AVAILABLE:
            try:
                self._send_to_bot(signal)
            except Exception as e:
                self.logger.error(f"Failed to send signal to bot: {e}", exc_info=True)
        else:
            self.logger.info("Bot not available - signal logged only")
            self.logger.info("To enable trading, ensure your bot is running and imports work")
    
    def _send_to_bot(self, signal: Dict):
        """
        Send signal to YOUR trading bot for execution
        
        Args:
            signal: Trading signal from CV
        """
        # Get current price from your data handler
        try:
            latest_bars = data_handler.get_latest_bars(1)
            if latest_bars.empty:
                self.logger.warning("⚠️  No price data available from data_handler")
                self.logger.warning("Make sure your bot has received at least one webhook")
                return
            
            current_price = latest_bars['close'].iloc[-1]
            self.logger.info(f"Current market price: ${current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to get current price: {e}")
            return
        
        # Calculate stops and targets
        stop_loss, take_profit = self._calculate_stops_targets(signal, current_price)
        
        # Convert CV signal to YOUR bot's format
        bot_signal = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': 'computer_vision',
            'signal': signal['signal'],        # 'LONG' or 'SHORT'
            'price': current_price,
            'confidence': signal['confidence'], # 0.7 = 70%
            'weight': signal['confidence'] * 100,  # Convert to percentage
            'reason': f"CV: {signal['pattern']} - {signal['reason']}",
            'target': take_profit,
            'stop': stop_loss
        }
        
        self.logger.info(f"Formatted signal for bot: {bot_signal['signal']}")
        self.logger.info(f"Entry: ${current_price:.2f}")
        self.logger.info(f"Stop: ${stop_loss:.2f} ({-self.stop_loss_points:+.0f} pts)")
        self.logger.info(f"Target: ${take_profit:.2f} ({+self.take_profit_points:+.0f} pts)")
        
        # Send to your position manager
        try:
            self.logger.info("Calling position_manager.open_position()...")
            position = position_manager.open_position(bot_signal, current_price)
            
            if position:
                self.logger.info("=" * 70)
                self.logger.info(f"✅ SUCCESS! Position opened: #{position.id}")
                self.logger.info(f"Direction: {position.direction}")
                self.logger.info(f"Entry: ${position.entry_price:.2f}")
                self.logger.info(f"Stop: ${position.stop_loss:.2f}")
                self.logger.info(f"Target: ${position.take_profit:.2f}")
                self.logger.info("=" * 70)
            else:
                self.logger.warning("=" * 70)
                self.logger.warning("⚠️  Position rejected by position_manager")
                self.logger.warning("Check logs/trades.log for reason")
                self.logger.warning("Common reasons:")
                self.logger.warning("  - Max positions reached")
                self.logger.warning("  - Daily loss limit hit")
                self.logger.warning("  - Circuit breaker active (5 consecutive losses)")
                self.logger.warning("=" * 70)
                
        except Exception as e:
            self.logger.error("=" * 70)
            self.logger.error(f"❌ ERROR opening position: {e}")
            self.logger.error("=" * 70)
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start the vision trading system"""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🚀 STARTING VISION TRADING SYSTEM")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info("IMPORTANT:")
        self.logger.info("  - Make sure TradingView is visible on your screen")
        self.logger.info("  - Chart should be clearly visible and not obscured")
        self.logger.info("  - Keep this window open while trading")
        self.logger.info("")
        
        if BOT_AVAILABLE:
            self.logger.info("✓ Connected to your trading bot")
            if self.enabled:
                self.logger.info("✓ Automatic trading: ENABLED")
                self.logger.info("  Signals will execute trades automatically")
            else:
                self.logger.info("⚠ Automatic trading: DISABLED")
                self.logger.info("  Signals will be logged only (analysis mode)")
        else:
            self.logger.info("⚠ Bot connection: UNAVAILABLE")
            self.logger.info("  Running in analysis mode (logs only)")
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("System is running. Press Ctrl+C to stop.")
        self.logger.info("=" * 70)
        self.logger.info("")
        
        # Start capture loop
        self.capture.start_capture_loop(
            interval=self.capture_interval,
            callback=self.on_frame_analyzed
        )
    
    def stop(self):
        """Stop the vision trading system"""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Stopping Vision Trading System...")
        self.logger.info("=" * 70)
        self.capture.stop_capture()
        self.logger.info("✓ System stopped")
        self.logger.info("Check logs/vision_trading.log for complete session log")
    
    def enable_trading(self):
        """Enable automatic trading"""
        self.enabled = True
        self.logger.info("=" * 70)
        self.logger.info("⚡ Automatic trading ENABLED")
        self.logger.info("=" * 70)
    
    def disable_trading(self):
        """Disable automatic trading (analysis only)"""
        self.enabled = False
        self.logger.info("=" * 70)
        self.logger.info("⏸  Automatic trading DISABLED (analysis mode)")
        self.logger.info("=" * 70)
    
    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold"""
        self.min_confidence = threshold
        self.logger.info(f"Confidence threshold set to {threshold:.1%}")
    
    def set_stop_take_profit(self, stop_points: int, target_points: int):
        """Set stop loss and take profit in points"""
        self.stop_loss_points = stop_points
        self.take_profit_points = target_points
        self.logger.info(f"Updated: Stop={stop_points} pts, Target={target_points} pts")
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        return {
            'enabled': self.enabled,
            'bot_connected': BOT_AVAILABLE,
            'min_confidence': self.min_confidence,
            'signal_history_length': len(self.ai.signal_history),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'stop_loss_points': self.stop_loss_points,
            'take_profit_points': self.take_profit_points
        }


def main():
    """Main entry point for vision trading"""
    print()
    print("=" * 70)
    print("TradingView Computer Vision Trading System")
    print("*** INTEGRATED WITH YOUR NQ BOT ***")
    print("=" * 70)
    print()
    print("This system will:")
    print("  1. Capture your TradingView screen")
    print("  2. Analyze charts with AI")
    print("  3. Detect patterns and generate signals")
    print("  4. Send signals to your position_manager for execution")
    print()
    print("REQUIREMENTS:")
    print("  ✓ TradingView must be visible on your screen")
    print("  ✓ Your main bot should be running (python main.py)")
    print("  ✓ Chart should be clear and not obscured")
    print()
    print("=" * 70)
    print()
    
    # Check if bot is available
    if BOT_AVAILABLE:
        print("✓ Successfully connected to your trading bot")
        print("  - position_manager: Available")
        print("  - data_handler: Available")
        print()
    else:
        print("⚠ Could not connect to trading bot")
        print("  Running in ANALYSIS MODE (signals logged only)")
        print("  Make sure your bot directory structure is correct")
        print()
    
    # Configuration
    try:
        input("Press Enter to continue (or Ctrl+C to cancel)...")
        print()
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return
    
    # Create integration
    integration = VisionTradingIntegration(
        monitor_number=1,
        capture_interval=2.0,  # Analyze every 2 seconds
        min_confidence=0.7      # 70% confidence minimum
    )
    
    # Ask about automatic trading
    if BOT_AVAILABLE:
        print("=" * 70)
        print("TRADING MODE")
        print("=" * 70)
        print()
        print("Choose a mode:")
        print("  [1] Analysis Mode (SAFE - signals logged only, no trades)")
        print("  [2] Trading Mode (signals execute trades automatically)")
        print()
        
        choice = input("Enter 1 or 2: ").strip()
        print()
        
        if choice == '2':
            print("⚡ AUTOMATIC TRADING MODE")
            print("Signals will execute trades via position_manager")
            print()
            confirm = input("Are you sure? Type 'YES' to confirm: ").strip()
            print()
            
            if confirm == 'YES':
                integration.enable_trading()
                print("✓ Automatic trading enabled")
            else:
                integration.disable_trading()
                print("✓ Switched to analysis mode")
        else:
            integration.disable_trading()
            print("✓ Running in ANALYSIS MODE")
            print("Signals will be logged but not executed")
    
    print()
    print("=" * 70)
    
    # Start system
    integration.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Shutdown requested...")
        print("=" * 70)
    
    finally:
        integration.stop()
        
        # Show stats
        print()
        print("Session Statistics:")
        print("-" * 70)
        stats = integration.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("-" * 70)
        print()
        print("Check logs for details:")
        print("  - Vision signals: logs/vision_trading.log")
        if BOT_AVAILABLE:
            print("  - Bot trades: logs/trades.log")
            print("  - Bot system: logs/system.log")
        print()


if __name__ == "__main__":
    main()