"""
Enhanced Webhook Server with Async Processing
Optimized for high-performance data ingestion
"""

from flask import Flask, request, jsonify
from datetime import datetime
import json
import threading
import time
from typing import Dict, Any

from config import WEBHOOK_PORT, WEBHOOK_PASSPHRASE, TICKER, OPTIMIZATION_FLAGS
from enhanced_data_handler import enhanced_data_handler
from enhanced_strategy_engine import enhanced_strategy_engine
from enhanced_risk_manager import enhanced_risk_manager
from enhanced_performance_monitor import performance_monitor

# Configure logging
import logging
logger = logging.getLogger(__name__)

class EnhancedWebhookServer:
    """
    Enhanced webhook server with async processing and performance optimization
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        
        logger.info("✅ Enhanced Webhook Server initialized")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/webhook', methods=['POST'])
        def receive_webhook():
            """Handle incoming webhook from TradingView"""
            start_time = time.time()
            
            try:
                self.request_count += 1
                self.last_request_time = datetime.now()
                
                # Verify content type
                if request.content_type != 'application/json':
                    self.error_count += 1
                    logger.warning(f"Invalid content type: {request.content_type}")
                    return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
                
                # Get JSON data
                data = request.json
                
                if not data:
                    self.error_count += 1
                    logger.warning("Empty request body")
                    return jsonify({'status': 'error', 'message': 'Empty request body'}), 400
                
                # Verify passphrase for security
                if data.get('passphrase') != WEBHOOK_PASSPHRASE:
                    self.error_count += 1
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
                    self.error_count += 1
                    logger.warning("Invalid bar data: close price is 0")
                    return jsonify({'status': 'error', 'message': 'Invalid price data'}), 400
                
                # Process bar data asynchronously if enabled
                if OPTIMIZATION_FLAGS.get('enable_async_processing', False):
                    threading.Thread(
                        target=self._process_bar_async,
                        args=(bar_data,),
                        daemon=True
                    ).start()
                else:
                    self._process_bar_sync(bar_data)
                
                # Log successful processing
                processing_time = (time.time() - start_time) * 1000
                logger.info(f"✅ Bar processed in {processing_time:.2f}ms: "
                           f"O:{bar_data['open']:.2f} H:{bar_data['high']:.2f} "
                           f"L:{bar_data['low']:.2f} C:{bar_data['close']:.2f} V:{bar_data['volume']}")
                
                # Add latency measurement
                performance_monitor.add_latency_measurement(processing_time)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Bar data received',
                    'timestamp': bar_data['timestamp'],
                    'processing_time_ms': processing_time
                }), 200
                
            except ValueError as e:
                self.error_count += 1
                logger.error(f"Value error processing webhook: {e}")
                return jsonify({'status': 'error', 'message': f'Invalid data format: {str(e)}'}), 400
            
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error processing webhook: {e}")
                return jsonify({'status': 'error', 'message': 'Internal server error'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'NQ Trading Bot Webhook',
                'request_count': self.request_count,
                'error_count': self.error_count,
                'uptime_seconds': time.time() - getattr(self, 'start_time', time.time())
            }), 200
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get webhook statistics"""
            try:
                return jsonify({
                    'status': 'success',
                    'request_count': self.request_count,
                    'error_count': self.error_count,
                    'error_rate': self.error_count / max(self.request_count, 1),
                    'last_request': self.last_request_time.isoformat() if self.last_request_time else None,
                    'timestamp': datetime.now().isoformat()
                }), 200
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/performance', methods=['GET'])
        def get_performance():
            """Get performance metrics"""
            try:
                metrics = performance_monitor.get_current_performance()
                return jsonify({
                    'status': 'success',
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }), 200
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _process_bar_sync(self, bar_data: Dict[str, Any]):
        """Process bar data synchronously"""
        try:
            # Store the bar
            enhanced_data_handler.add_bar(bar_data)
            
            # Get recent bars for strategy analysis
            df = enhanced_data_handler.get_latest_bars(200)
            
            # Process through strategy engine
            signals = enhanced_strategy_engine.process_new_bar(bar_data, {'df': df})
            
            # Process signals
            self._process_signals(signals, bar_data['close'])
            
        except Exception as e:
            logger.error(f"Error processing bar synchronously: {e}")
    
    def _process_bar_async(self, bar_data: Dict[str, Any]):
        """Process bar data asynchronously"""
        try:
            # Store the bar
            enhanced_data_handler.add_bar(bar_data)
            
            # Process through strategy engine with market context
            context = self._get_market_context(bar_data)
            signals = enhanced_strategy_engine.process_new_bar(bar_data, context)
            
            # Process signals
            self._process_signals(signals, bar_data['close'])
            
        except Exception as e:
            logger.error(f"Error processing bar asynchronously: {e}")
    
    def _get_market_context(self, bar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get market context for signal generation"""
        try:
            # Get recent market data
            df = enhanced_data_handler.get_latest_bars(50)
            
            if df.empty:
                return {}
            
            # Calculate basic context
            current_price = bar_data['close']
            
            # Simple trend detection
            if len(df) >= 20:
                recent_high = df['high'].tail(10).max()
                recent_low = df['low'].tail(10).min()
                
                if current_price > recent_high * 0.998:
                    trend = 'bullish'
                elif current_price < recent_low * 1.002:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'
            
            # Volume analysis
            if 'volume' in df.columns and len(df) > 10:
                avg_volume = df['volume'].tail(10).mean()
                current_volume = bar_data.get('volume', 0)
                volume_ratio = current_volume / max(avg_volume, 1)
            else:
                volume_ratio = 1.0
            
            return {
                'trend': trend,
                'volume_ratio': volume_ratio,
                'current_price': current_price,
                'market_regime': 'trending' if trend != 'neutral' else 'ranging'
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {}
    
    def _process_signals(self, signals: list, current_price: float):
        """Process trading signals"""
        try:
            for signal in signals:
                # Check risk management
                can_open, reason = enhanced_risk_manager.can_open_position(signal, current_price)
                
                if can_open:
                    # Calculate position size
                    df = enhanced_data_handler.get_latest_bars(50)
                    position_size = enhanced_risk_manager.calculate_dynamic_position_size(signal, df)
                    
                    # Open position
                    position = enhanced_risk_manager.open_position(signal, current_price, position_size)
                    
                    if position:
                        logger.info(f"📊 Position opened: {signal['strategy']} {signal['signal']} "
                                   f"@ {current_price:.2f} (size: {position_size})")
                else:
                    logger.debug(f"Signal rejected: {reason}")
        
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
    
    def run(self, host='0.0.0.0', port=WEBHOOK_PORT, debug=False):
        """Start the webhook server"""
        self.start_time = time.time()
        
        logger.info(f"🚀 Starting Enhanced Webhook Server on {host}:{port}")
        logger.info(f"📡 Webhook URL: http://{host}:{port}/webhook")
        logger.info(f"🔍 Health check: http://{host}:{port}/health")
        logger.info(f"📊 Stats: http://{host}:{port}/stats")
        logger.info(f"📈 Performance: http://{host}:{port}/performance")
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)