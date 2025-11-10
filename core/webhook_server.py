"""
Flask Webhook Receiver for TradingView Alerts
Receives 1-minute NQ futures data and stores in CSV
"""
from flask import Flask, request, jsonify
from datetime import datetime
import json
from config import WEBHOOK_PORT, WEBHOOK_PASSPHRASE, TICKER
from utils.data_handler import data_handler
from utils.logger import trading_logger


class WebhookServer:
    """Flask server to receive TradingView webhooks"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.logger = trading_logger.webhook_logger
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure Flask routes"""
        
        @self.app.route('/webhook', methods=['POST'])
        def receive_webhook():
            """Handle incoming webhook from TradingView"""
            try:
                # Verify content type
                if request.content_type != 'application/json':
                    self.logger.warning(f"Invalid content type: {request.content_type}")
                    return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
                
                # Get JSON data
                data = request.json
                
                if not data:
                    self.logger.warning("Empty request body")
                    return jsonify({'status': 'error', 'message': 'Empty request body'}), 400
                
                # Verify passphrase for security
                if data.get('passphrase') != WEBHOOK_PASSPHRASE:
                    self.logger.warning(f"Invalid passphrase attempt from {request.remote_addr}")
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
                    self.logger.warning("Invalid bar data: close price is 0")
                    return jsonify({'status': 'error', 'message': 'Invalid price data'}), 400
                
                # Store the bar
                data_handler.add_bar(bar_data)
                df = data_handler.get_latest_bars(500)  # this is what you pass into the engine/dashboard

                self.logger.info(
                    f"Bar received: {bar_data['timestamp']} | "
                    f"O:{bar_data['open']:.2f} H:{bar_data['high']:.2f} "
                    f"L:{bar_data['low']:.2f} C:{bar_data['close']:.2f} V:{bar_data['volume']}"
                )
                
                return jsonify({
                    'status': 'success',
                    'message': 'Bar data received',
                    'timestamp': bar_data['timestamp']
                }), 200
                
            except ValueError as e:
                self.logger.error(f"Value error processing webhook: {e}")
                return jsonify({'status': 'error', 'message': f'Invalid data format: {str(e)}'}), 400
            
            except Exception as e:
                self.logger.error(f"Error processing webhook: {e}", exc_info=True)
                trading_logger.log_error("Webhook", e)
                return jsonify({'status': 'error', 'message': 'Internal server error'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'NQ Trading Bot Webhook'
            }), 200
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Return current system statistics"""
            try:
                metrics = data_handler.calculate_performance_metrics()
                return jsonify({
                    'status': 'success',
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }), 200
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=WEBHOOK_PORT, debug=False):
        """Start the Flask server"""
        self.logger.info(f"Starting webhook server on {host}:{port}")
        self.logger.info(f"Webhook URL: http://{host}:{port}/webhook")
        self.logger.info(f"Health check: http://{host}:{port}/health")
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)


# For standalone testing
if __name__ == '__main__':
    server = WebhookServer()
    server.run()
