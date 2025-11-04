"""
Logging utilities for NQ Trading Bot
"""
import logging
from logging.handlers import RotatingFileHandler
import colorlog
from pathlib import Path
from config import LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Optional specific log file name
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    if log_file is None:
        log_file = f'{name}.log'
    file_handler = RotatingFileHandler(
        LOG_DIR / log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class TradingLogger:
    """Centralized logging for trading system"""
    
    def __init__(self):
        self.system_logger = setup_logger('system', 'system.log')
        self.trade_logger = setup_logger('trades', 'trades.log')
        self.strategy_logger = setup_logger('strategy', 'strategy.log')
        self.ml_logger = setup_logger('ml', 'ml.log')
        self.webhook_logger = setup_logger('webhook', 'webhook.log')
    
    def log_trade(self, trade_dict: dict):
        """Log trade execution details"""
        self.trade_logger.info(
            f"TRADE | {trade_dict['timestamp']} | {trade_dict['action']} | "
            f"Price: {trade_dict['price']} | Size: {trade_dict.get('size', 1)} | "
            f"Signal: {trade_dict.get('signal', 'N/A')}"
        )
    
    def log_signal(self, strategy: str, signal: str, confidence: float, price: float):
        """Log strategy signals"""
        self.strategy_logger.info(
            f"SIGNAL | Strategy: {strategy} | {signal} | "
            f"Confidence: {confidence:.2%} | Price: {price}"
        )
    
    def log_ml_prediction(self, model: str, prediction: dict):
        """Log ML model predictions"""
        self.ml_logger.info(
            f"ML | Model: {model} | Direction: {prediction['direction']} | "
            f"Probability: {prediction['probability']:.2%} | "
            f"Target: {prediction.get('target', 'N/A')}"
        )
    
    def log_error(self, component: str, error: Exception):
        """Log errors with full traceback"""
        self.system_logger.error(
            f"ERROR in {component}: {str(error)}",
            exc_info=True
        )
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        self.system_logger.info(
            f"PERFORMANCE | Win Rate: {metrics['win_rate']:.2%} | "
            f"Profit Factor: {metrics['profit_factor']:.2f} | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Max DD: {metrics['max_drawdown']:.2%}"
        )


# Create global logger instance
trading_logger = TradingLogger()
