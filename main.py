"""
Main Application Entry Point for NQ Trading Bot - COMPLETE VERSION
Integrates: Flask webhook + Dash dashboard + Strategy engine + Position manager
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, redirect
from datetime import datetime
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from config import WEBHOOK_PORT, WEBHOOK_PASSPHRASE
from utils.data_handler import data_handler
from utils.logger import trading_logger
from dashboard.trading_dashboard import TradingDashboard
from strategy_engine import strategy_engine
from position_manager import position_manager
from utils.context_manager import context_manager # <-- 1. IMPORT IT
import io
# Add these lines with your other strategy imports
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.breakout_strategy import BreakoutStrategy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# Create Flask server first
server = Flask(__name__)
logger = trading_logger.webhook_logger

@server.route('/webhook', methods=['POST'])
def receive_webhook():
    """
    Handle incoming webhook from TradingView
    NOW WITH STRATEGY EXECUTION!
    """
    try:
        # Validate content type (allow charset parameter)
        if not request.content_type or not request.content_type.startswith('application/json'):
            logger.warning(f"Invalid content type: {request.content_type}")
            return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
        
        data = request.json
        if not data:
            logger.warning("Empty request body")
            return jsonify({'status': 'error', 'message': 'Empty request body'}), 400
        
        # Verify passphrase for security
        if data.get('passphrase') != WEBHOOK_PASSPHRASE:
            logger.warning(f"Invalid passphrase attempt from {request.remote_addr}")
            return jsonify({'status': 'error', 'message': 'Invalid passphrase'}), 401
        
        # Extract bar data
        bar_data = {
            'timestamp': datetime.now().isoformat(),
            'open': float(data.get('open', data.get('close', 0))),
            'high': float(data.get('high', data.get('close', 0))),
            'low': float(data.get('low', data.get('close', 0))),
            'close': float(data.get('close', 0)),
            'volume': float(data.get('volume', 0))
        }
        
        # Validate data
        if bar_data['close'] == 0:
            logger.warning("Invalid bar data: close price is 0")
            return jsonify({'status': 'error', 'message': 'Invalid price data'}), 400
        
        # Store the bar
        data_handler.add_bar(bar_data)
        
        current_price = bar_data['close']
        current_time = datetime.now()
        
        logger.info(
            f"Bar received: {bar_data['timestamp']} | "
            f"O:{bar_data['open']:.2f} H:{bar_data['high']:.2f} "
            f"L:{bar_data['low']:.2f} C:{current_price:.2f} V:{bar_data['volume']}"
        )
        
        # ⭐⭐⭐ THIS IS THE KEY ADDITION ⭐⭐⭐
        # Update existing positions first (check stops/targets)
        position_manager.update_positions(current_price, current_time)
        
        # <-- 2. GET CONTEXT from our new manager
        current_context = context_manager.get_market_context()
        logger.debug(f"Current Context: {current_context}")
        
        # Run strategy engine to generate new signals
        # <-- 3. PASS CONTEXT to the engine
        signals = strategy_engine.process_new_bar(bar_data, current_context)
        
        # If we got signals and can open positions, do it
        if signals:
            for signal in signals:
                # Check if we can open a new position
                can_open, reason = position_manager.can_open_position()
                
                if can_open:
                    # Open position at current market price
                    position = position_manager.open_position(signal, current_price)
                    
                    if position:
                        logger.info(f"✅ Opened position from {signal['strategy']} signal")
                else:
                    logger.info(f"Signal from {signal['strategy']} rejected: {reason}")
        
        # Get daily stats for response
        daily_stats = position_manager.get_daily_stats()
        
        return jsonify({
            'status': 'success',
            'message': 'Bar data received and processed',
            'timestamp': bar_data['timestamp'],
            'signals_generated': len(signals),
            'open_positions': len(position_manager.get_open_positions()),
            'daily_pnl': daily_stats['pnl'],
            'daily_trades': daily_stats['trades']
        }), 200
        
    except ValueError as e:
        logger.error(f"Value error processing webhook: {e}")
        return jsonify({'status': 'error', 'message': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@server.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
    daily_stats = position_manager.get_daily_stats()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'NQ Trading Bot',
        'open_positions': len(position_manager.get_open_positions()),
        'daily_pnl': daily_stats['pnl'],
        'daily_trades': daily_stats['trades'],
        'daily_win_rate': daily_stats['win_rate']
    }), 200

@server.route('/stats', methods=['GET'])
def get_stats():
    """Return detailed system statistics"""
    try:
        metrics = data_handler.calculate_performance_metrics()
        daily_stats = position_manager.get_daily_stats()
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'overall': metrics,
            'today': daily_stats,
            'open_positions': len(position_manager.get_open_positions())
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@server.route('/positions', methods=['GET'])
def get_positions():
    """Get current open positions"""
    try:
        open_positions = position_manager.get_open_positions()
        
        positions_data = [
            {
                'id': p.id,
                'strategy': p.strategy,
                'direction': p.direction,
                'entry_price': p.entry_price,
                'entry_time': p.entry_time.isoformat(),
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'current_pnl_points': p.mfe if p.direction == 'LONG' else -p.mae
            }
            for p in open_positions
        ]
        
        return jsonify({
            'status': 'success',
            'count': len(positions_data),
            'positions': positions_data
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@server.route('/close-all', methods=['POST'])
def force_close_all():
    """Emergency endpoint to close all positions"""
    try:
        # Get current market price from latest bar
        df = data_handler.get_latest_bars(1)
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No market data available'}), 400
        
        current_price = df['close'].iloc[-1]
        position_manager.force_close_all(current_price, reason="MANUAL_CLOSE")
        
        return jsonify({
            'status': 'success',
            'message': 'All positions closed',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@server.route('/')
def redirect_to_dashboard():
    """Redirect root to dashboard"""
    return redirect('/dashboard/')


class NQTradingBot:
    """Main trading bot application"""
    
    def __init__(self):
        self.logger = trading_logger.system_logger
        self.server = server
        
        # Initialize Dash app with the Flask server
        self.dashboard = TradingDashboard(server=self.server)
        
        # <-- 4. START THE CONTEXT MANAGER THREAD
        try:
            context_manager.start()
            self.logger.info("✅ Market Context Manager thread: ACTIVE")
        except Exception as e:
            self.logger.error(f"Failed to start Context Manager: {e}")
        
        self.logger.info("=" * 60)
        self.logger.info("NQ FUTURES TRADING BOT - COMPLETE VERSION")
        self.logger.info("=" * 60)
        self.logger.info(f"Webhook endpoint: http://0.0.0.0:{WEBHOOK_PORT}/webhook")
        self.logger.info(f"Dashboard URL: http://0.0.0.0:{WEBHOOK_PORT}/dashboard/")
        self.logger.info(f"Health check: http://0.0.0.0:{WEBHOOK_PORT}/health")
        self.logger.info(f"Positions API: http://0.0.0.0:{WEBHOOK_PORT}/positions")
        self.logger.info(f"Stats API: http://0.0.0.0:{WEBHOOK_PORT}/stats")
        self.logger.info("=" * 60)
        self.logger.info("✅ Strategy engine: ACTIVE")
        self.logger.info("✅ Position manager: ACTIVE")
        self.logger.info("✅ Paper trading: ENABLED")
        self.logger.info("=" * 60)
    
    def run(self, host='0.0.0.0', port=WEBHOOK_PORT, debug=False):
        """Start the trading bot application"""
        try:
            self.logger.info("Starting NQ Trading Bot...")
            
            self.logger.info(f"Bot is running on http://{host}:{port}")
            self.logger.info(f"Access dashboard at: http://{host}:{port}/dashboard/")
            self.logger.info(f"Webhook ready at: http://{host}:{port}/webhook")
            
            print("\n" + "=" * 60)
            print("🚀 SUCCESS! NQ Trading Bot is FULLY OPERATIONAL")
            print("=" * 60)
            print(f"Dashboard:   http://localhost:{port}/dashboard/")
            print(f"Webhook:     http://localhost:{port}/webhook")
            print(f"Health:      http://localhost:{port}/health")
            print(f"Positions:   http://localhost:{port}/positions")
            print(f"Stats:       http://localhost:{port}/stats")
            print("=" * 60)
            print("✅ Strategies ACTIVE and will execute on incoming data")
            print("✅ Position management ENABLED")
            print("✅ Paper trading mode: ON")
            print("✅ Market Context (ES/VIX): ACTIVE") # <-- New status
            print("=" * 60)
            print("Press CTRL+C to stop\n")
            
            # Run the Flask server with Dash app integrated
            self.server.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True, # Must be True for context manager
                use_reloader=False
            )
            
        except KeyboardInterrupt:
            # Close all positions before shutting down
            try:
                df = data_handler.get_latest_bars(1)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    position_manager.force_close_all(current_price, reason="SHUTDOWN")
            except:
                pass
            
            # <-- 5. STOP THE CONTEXT MANAGER
            self.logger.info("Stopping Context Manager...")
            context_manager.stop()
            
            self.logger.info("\nShutting down NQ Trading Bot...")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}", exc_info=True)
            
            # <-- 5. STOP THE CONTEXT MANAGER
            self.logger.info("Stopping Context Manager...")
            context_manager.stop()
            
            sys.exit(1)


def main():
    """Main entry point"""
    try:
        bot = NQTradingBot()
        bot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()