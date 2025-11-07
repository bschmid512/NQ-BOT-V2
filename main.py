"""
Main Application Entry Point for NQ Trading Bot
*** FUSION ARCHITECTURE - V3 ***
Integrates: Flask webhook + Dash dashboard + Enhanced Strategy Engine + Fusion Engine
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, redirect
from datetime import datetime
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import io
import threading, time

# --- Core Component Imports (with correct new paths) ---
from config import WEBHOOK_PORT, WEBHOOK_PASSPHRASE
from utils.data_handler import data_handler
from utils.logger import trading_logger
from dashboard.trading_dashboard import TradingDashboard

# --- NEW FUSION IMPORTS ---
# We now import the components that make the decisions
from core.market_context_fusion import market_context_fusion
from core.enhanced_strategy_engine import enhanced_strategy_engine
from core.position_manager import position_manager
# from core.signal_fusion_engine import signal_fusion_engine # Imported by enhanced_strategy_engine
# --- END FUSION IMPORTS ---

# UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Create Flask server first
server = Flask(__name__)

# --- Vision inbox (for fusion) ---
LATEST_VISION = {"data": None, "ts": 0.0}
_VISION_LOCK = threading.Lock()

@server.route('/vision', methods=['POST'])
def vision_ingest():
    payload = request.get_json(force=True, silent=True) or {}
    with _VISION_LOCK:
        LATEST_VISION["data"] = payload
        LATEST_VISION["ts"] = time.time()
    return jsonify({"status": "ok"}), 200

def _get_latest_vision(max_age_sec: float = 10.0):
    with _VISION_LOCK:
        if time.time() - LATEST_VISION["ts"] <= max_age_sec:
            return LATEST_VISION["data"]
    return None

logger = trading_logger.webhook_logger

@server.route('/webhook', methods=['POST'])
def receive_webhook():
    """
    Handle incoming webhook from TradingView
    *** NOW WITH FUSION ENGINE LOGIC ***
    """
    try:
        # ... (Validation for content type, empty data, passphrase is the same) ...
        if not request.content_type or not request.content_type.startswith('application/json'):
            logger.warning(f"Invalid content type: {request.content_type}")
            return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
        data = request.json
        if not data:
            logger.warning("Empty request body")
            return jsonify({'status': 'error', 'message': 'Empty request body'}), 400
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
        
        if bar_data['close'] == 0: # Validate data
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
        
        # === NEW FUSION LOGIC PIPELINE ===

        # 1. Update existing positions first (check stops/targets)
        position_manager.update_positions(current_price, current_time)
        
        # 2. Get historical data for context and strategy analysis
        df = data_handler.get_latest_bars(200)
        
        # 3. Process data through the Enhanced Strategy Engine
        # This single call will:
        #   - Update market_context_fusion
        #   - Get the unified context
        #   - Run all strategies (pullback, momentum, etc.)
        #   - Send all signals to signal_fusion_engine
        #   - Return ONE final, approved signal (or None)
        
        # Note: We pass vision_data=None because the vision loop isn't connected here yet.
        fused_signal = enhanced_strategy_engine.process_new_bar(
            bar_data=bar_data,
            df=df,
            vision_data=_get_latest_vision()
        )
        
        # 4. If we get a final, high-conviction signal, take action
        if fused_signal:
            # Log fused signal for dashboard
            data_handler.append_signal({
                'timestamp': datetime.now().isoformat(),
                'strategy': fused_signal.get('strategy','fusion'),
                'signal': fused_signal.get('direction'),
                'price': current_price,
                'confidence': fused_signal.get('confidence', 0.0)
            })
            # Check if we can open a new position
            can_open, reason = position_manager.can_open_position()
            
            if can_open:
                # --- IMPLEMENTS "SCALE IN/OUT" ---
                # The fusion engine adds a 'size' key based on conviction
                trade_size = fused_signal.get('size', 1) 
                
                position = position_manager.open_position(
                    signal=fused_signal,
                    entry_price=current_price,
                    size=trade_size
                )
                
                if position:
                    # Append OPEN trade row for dashboard
                    data_handler.append_trade({
                        'timestamp': datetime.now().isoformat(),
                        'ticker': 'NQ',
                        'action': 'BUY' if position.direction == 'LONG' else 'SELL',
                        'price': current_price,
                        'size': position.size,
                        'signal': position.strategy,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit,
                        'pnl': 0.0,
                        'status': 'OPEN'
                    })
                    logger.info(f"✅ FUSION TRADE OPENED ({trade_size} contracts) from {fused_signal['strategy']} signal")
            else:
                # Log the *rejection* from position_manager (e.g., max trades)
                # The fusion engine's rejections are logged by the comprehensive_logger
                logger.info(f"Fusion signal from {fused_signal['strategy']} rejected: {reason}")
        
        # === END FUSION LOGIC PIPELINE ===
        
        # Get daily stats for response
        daily_stats = position_manager.get_daily_stats()
        
        return jsonify({
            'status': 'success',
            'message': 'Bar data processed by Fusion Engine',
            'timestamp': bar_data['timestamp'],
            'signal_approved': 1 if fused_signal else 0,
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

# ... (All other routes: /health, /stats, /positions, /close-all, / are IDENTICAL) ...

@server.route('/health', methods=['GET'])
def health_check():
    daily_stats = position_manager.get_daily_stats()
    return jsonify({
        'status': 'healthy', 'timestamp': datetime.now().isoformat(),
        'service': 'NQ Trading Bot (Fusion)', 'open_positions': len(position_manager.get_open_positions()),
        'daily_pnl': daily_stats['pnl'], 'daily_trades': daily_stats['trades'],
        'daily_win_rate': daily_stats['win_rate']
    }), 200

@server.route('/stats', methods=['GET'])
def get_stats():
    try:
        metrics = data_handler.calculate_performance_metrics()
        daily_stats = position_manager.get_daily_stats()
        return jsonify({
            'status': 'success', 'timestamp': datetime.now().isoformat(),
            'overall': metrics, 'today': daily_stats,
            'open_positions': len(position_manager.get_open_positions())
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@server.route('/positions', methods=['GET'])
def get_positions():
    try:
        open_positions = position_manager.get_open_positions()
        positions_data = [p.to_dict() for p in open_positions] # Simplified using to_dict()
        return jsonify({
            'status': 'success', 'count': len(positions_data),
            'positions': positions_data
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@server.route('/close-all', methods=['POST'])
def force_close_all():
    try:
        df = data_handler.get_latest_bars(1)
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No market data available'}), 400
        current_price = df['close'].iloc[-1]
        position_manager.force_close_all(current_price, reason="MANUAL_CLOSE")
        return jsonify({
            'status': 'success', 'message': 'All positions closed',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@server.route('/')
def redirect_to_dashboard():
    return redirect('/dashboard/')


class NQTradingBot:
    """Main trading bot application"""
    
    def __init__(self):
        self.logger = trading_logger.system_logger
        self.server = server
        
        # Initialize Dash app with the Flask server
        self.dashboard = TradingDashboard(server=self.server)
        
        # --- REMOVED OLD CONTEXT MANAGER ---
        # The new fusion engine is passive and doesn't need a separate thread
        # for yfinance data, simplifying the application.
        
        self.logger.info("=" * 60)
        self.logger.info("NQ FUTURES TRADING BOT - HYBRID FUSION SYSTEM")
        self.logger.info("=" * 60)
        self.logger.info(f"Webhook endpoint: http://0.0.0.0:{WEBHOOK_PORT}/webhook")
        self.logger.info(f"Dashboard URL: http://0.0.0.0:{WEBHOOK_PORT}/dashboard/")
        self.logger.info(f"Health check: http://0.0.0.0:{WEBHOOK_PORT}/health")
        self.logger.info("=" * 60)
        self.logger.info("Enhanced Strategy Engine: ACTIVE")
        self.logger.info(" Signal Fusion Engine: ACTIVE")
        self.logger.info(" Position Manager: ACTIVE")
        self.logger.info(" Comprehensive Logger: ACTIVE")
        self.logger.info(" Paper trading: ENABLED")
        self.logger.info("=" * 60)
    
    def run(self, host='0.0.0.0', port=WEBHOOK_PORT, debug=False):
        """Start the trading bot application"""
        try:
            self.logger.info("Starting NQ Trading Bot (Fusion Mode)...")
            
            print("\n" + "=" * 60)
            print("SUCCESS! NQ Trading Bot is FULLY OPERATIONAL (FUSION MODE)")
            print("=" * 60)
            print(f"Dashboard:   http://localhost:{port}/dashboard/")
            print(f"Webhook:     http://localhost:{port}/webhook")
            print(f"Health:      http://localhost:{port}/health")
            print("=" * 60)
            print("Enhanced Strategy Engine ACTIVE")
            print("Signal Fusion Engine will make all final decisions")
            print("Paper trading mode: ON")
            print("=" * 60)
            print("Press CTRL+C to stop\n")
            
            # Run the Flask server with Dash app integrated
            self.server.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True, # Still need threaded for Dash and concurrent requests
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
            
            self.logger.info("\nShutting down NQ Trading Bot...")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}", exc_info=True)
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