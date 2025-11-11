#!/usr/bin/env python3
"""
Enhanced NQ Trading Bot Main Application
Phase 1-4 Implementation: High-Performance Scalping System
"""

import asyncio
import threading
import time
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import enhanced components
from config import (
    OPTIMIZATION_FLAGS, WEBHOOK_PORT, DASHBOARD_PORT,
    REDIS_CONFIG, DATA_PIPELINE_CONFIG
)
from enhanced_data_handler import enhanced_data_handler
from enhanced_strategy_engine import enhanced_strategy_engine
from enhanced_risk_manager import enhanced_risk_manager
from enhanced_performance_monitor import performance_monitor
from enhanced_webhook_server import EnhancedWebhookServer
from enhanced_trading_dashboard import EnhancedTradingDashboard

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    """
    Main trading bot application with all optimizations
    Coordinates all enhanced components for high-performance scalping
    """
    
    def __init__(self):
        self.running = False
        self.components = {}
        self.threads = []
        
        # Performance tracking
        self.start_time = datetime.now(timezone.utc)
        self.total_bars_processed = 0
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        
        logger.info("🚀 Enhanced NQ Trading Bot initializing...")
    
    def initialize_components(self):
        """Initialize all trading bot components"""
        try:
            logger.info("📦 Initializing components...")
            
            # Check Redis availability
            if OPTIMIZATION_FLAGS.get('use_redis_cache', False):
                self._check_redis_availability()
            
            # Initialize webhook server
            if OPTIMIZATION_FLAGS.get('enable_webhook', True):
                self.components['webhook'] = EnhancedWebhookServer()
                logger.info("✅ Webhook server initialized")
            
            # Initialize dashboard
            if OPTIMIZATION_FLAGS.get('enable_dashboard', True):
                self.components['dashboard'] = EnhancedTradingDashboard()
                logger.info("✅ Trading dashboard initialized")
            
            # Start performance monitoring
            if OPTIMIZATION_FLAGS.get('real_time_performance_tracking', True):
                performance_monitor.start_monitoring()
                logger.info("✅ Performance monitoring started")
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def _check_redis_availability(self):
        """Check if Redis is available for caching"""
        try:
            import redis
            r = redis.Redis(**REDIS_CONFIG)
            r.ping()
            logger.info("✅ Redis connection successful")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}")
            logger.warning("Falling back to in-memory caching")
            OPTIMIZATION_FLAGS['use_redis_cache'] = False
            return False
    
    def start_services(self):
        """Start all trading bot services"""
        try:
            logger.info("🚀 Starting trading bot services...")
            
            # Start webhook server
            if 'webhook' in self.components:
                webhook_thread = threading.Thread(
                    target=self.components['webhook'].run,
                    kwargs={'port': WEBHOOK_PORT, 'debug': False},
                    daemon=True
                )
                webhook_thread.start()
                self.threads.append(webhook_thread)
                logger.info(f"✅ Webhook server started on port {WEBHOOK_PORT}")
            
            # Start dashboard
            if 'dashboard' in self.components:
                dashboard_thread = threading.Thread(
                    target=self.components['dashboard'].run,
                    kwargs={'port': DASHBOARD_PORT, 'debug': False},
                    daemon=True
                )
                dashboard_thread.start()
                self.threads.append(dashboard_thread)
                logger.info(f"✅ Dashboard started on port {DASHBOARD_PORT}")
            
            # Start data processing loop
            data_thread = threading.Thread(
                target=self._data_processing_loop,
                daemon=True
            )
            data_thread.start()
            self.threads.append(data_thread)
            logger.info("✅ Data processing loop started")
            
            logger.info("✅ All services started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting services: {e}")
            raise
    
    def _data_processing_loop(self):
        """Main data processing loop"""
        logger.info("🔄 Starting data processing loop...")
        
        while self.running:
            try:
                # Process any pending data
                self._process_market_data()
                
                # Update performance metrics
                if OPTIMIZATION_FLAGS.get('real_time_performance_tracking', True):
                    performance_monitor.update_performance_metrics()
                
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in data processing loop: {e}")
                time.sleep(1)  # Wait before retrying
    
    def _process_market_data(self):
        """Process incoming market data"""
        try:
            # Get latest bars from data handler
            latest_bars = enhanced_data_handler.get_latest_bars(10)
            
            if not latest_bars.empty:
                # Process through strategy engine
                current_bar = latest_bars.iloc[-1]
                bar_data = {
                    'timestamp': current_bar.name.isoformat() if hasattr(current_bar.name, 'isoformat') else str(current_bar.name),
                    'open': float(current_bar.get('open', current_bar.get('close', 0))),
                    'high': float(current_bar.get('high', current_bar.get('close', 0))),
                    'low': float(current_bar.get('low', current_bar.get('close', 0))),
                    'close': float(current_bar.get('close', 0)),
                    'volume': float(current_bar.get('volume', 0))
                }
                
                # Generate signals
                signals = enhanced_strategy_engine.process_new_bar(bar_data)
                
                if signals:
                    self.total_signals_generated += len(signals)
                    self._process_signals(signals, bar_data['close'])
                
                self.total_bars_processed += 1
        
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
    
    def _process_signals(self, signals: list, current_price: float):
        """Process generated trading signals"""
        try:
            for signal in signals:
                # Check if we can open position
                can_open, reason = enhanced_risk_manager.can_open_position(signal, current_price)
                
                if can_open:
                    # Calculate position size
                    df = enhanced_data_handler.get_latest_bars(50)
                    position_size = enhanced_risk_manager.calculate_dynamic_position_size(signal, df)
                    
                    # Open position
                    position = enhanced_risk_manager.open_position(signal, current_price, position_size)
                    
                    if position:
                        self.total_trades_executed += 1
                        logger.info(f"📊 Trade executed: {signal['strategy']} {signal['signal']} @ {current_price:.2f}")
                        
                        # Add to performance tracking
                        performance_monitor.add_trade({
                            'timestamp': datetime.now(timezone.utc),
                            'strategy': signal['strategy'],
                            'signal': signal['signal'],
                            'price': current_price,
                            'size': position_size,
                            'confidence': signal.get('confidence', 0.5)
                        })
                else:
                    logger.debug(f"Signal rejected: {reason}")
        
        except Exception as e:
            logger.error(f"❌ Error processing signals: {e}")
    
    def run(self):
        """Main run method"""
        try:
            logger.info("🚀 Starting Enhanced NQ Trading Bot...")
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Initialize components
            self.initialize_components()
            
            # Start services
            self.start_services()
            
            self.running = True
            
            # Main loop
            logger.info("✅ Trading bot started successfully!")
            logger.info("Press Ctrl+C to stop...")
            
            while self.running:
                try:
                    # Display status every 30 seconds
                    self._display_status()
                    time.sleep(30)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Shutdown requested by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    time.sleep(5)
            
            self.stop()
            
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            self.stop()
            sys.exit(1)
    
    def _display_status(self):
        """Display current system status"""
        try:
            uptime = datetime.now(timezone.utc) - self.start_time
            
            # Get performance metrics
            performance = performance_monitor.get_current_performance()
            strategy_performance = performance_monitor.get_strategy_performance()
            
            print(f"\n📊 Trading Bot Status (Uptime: {str(uptime).split('.')[0]})")
            print("=" * 60)
            print(f"📈 Performance:")
            print(f"   Total P&L: ${performance.get('total_pnl', 0):.2f}")
            print(f"   Win Rate: {performance.get('win_rate', 0):.1%}")
            print(f"   Total Trades: {performance.get('total_trades', 0)}")
            print(f"   Latency: {performance.get('latency_ms', 0):.1f}ms")
            print(f"   Cache Hit Rate: {performance.get('cache_hit_rate', 0):.1%}")
            
            print(f"\n🎯 Strategy Performance:")
            for name, metrics in strategy_performance.items():
                if metrics['total_signals'] > 0:
                    print(f"   {name.upper()}: {metrics['total_signals']} signals, "
                          f"{metrics['win_rate']:.1%} win rate")
            
            print(f"\n🔧 System Health:")
            print(f"   Bars Processed: {self.total_bars_processed}")
            print(f"   Signals Generated: {self.total_signals_generated}")
            print(f"   Trades Executed: {self.total_trades_executed}")
            print(f"   Open Positions: {performance.get('open_positions', 0)}")
            
            # Show any alerts
            alerts = performance_monitor.get_alerts(severity='warning')
            if alerts:
                print(f"\n⚠️  Active Alerts:")
                for alert in alerts[-3:]:  # Show last 3 alerts
                    print(f"   {alert.get('message', 'Unknown alert')}")
            
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error displaying status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}")
        self.running = False
    
    def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("🛑 Stopping Enhanced NQ Trading Bot...")
        
        self.running = False
        
        try:
            # Stop performance monitoring
            if OPTIMIZATION_FLAGS.get('real_time_performance_tracking', True):
                performance_monitor.stop_monitoring()
            
            # Flush all pending data
            logger.info("💾 Flushing pending data...")
            enhanced_data_handler.flush_all_data()
            
            # Close all open positions
            logger.info("📊 Closing open positions...")
            enhanced_risk_manager.force_close_all(0, "SYSTEM_SHUTDOWN")
            
            # Wait for threads to finish
            logger.info("⏳ Waiting for services to stop...")
            for thread in self.threads:
                thread.join(timeout=5)
            
            # Final performance report
            self._display_final_report()
            
            logger.info("✅ Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def _display_final_report(self):
        """Display final performance report"""
        try:
            uptime = datetime.now(timezone.utc) - self.start_time
            performance = performance_monitor.get_current_performance()
            
            print(f"\n🎯 Final Performance Report")
            print("=" * 60)
            print(f"Total Uptime: {str(uptime).split('.')[0]}")
            print(f"Total Bars Processed: {self.total_bars_processed}")
            print(f"Total Signals Generated: {self.total_signals_generated}")
            print(f"Total Trades Executed: {self.total_trades_executed}")
            print(f"Final P&L: ${performance.get('total_pnl', 0):.2f}")
            print(f"Final Win Rate: {performance.get('win_rate', 0):.1%}")
            print(f"Average Latency: {performance.get('latency_ms', 0):.1f}ms")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error displaying final report: {e}")


def main():
    """Main entry point"""
    # Create and run trading bot
    bot = EnhancedTradingBot()
    bot.run()

if __name__ == "__main__":
    main()