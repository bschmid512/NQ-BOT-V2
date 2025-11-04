"""
Data Handler for NQ Trading Bot - FIXED VERSION
Manages CSV data ingestion, storage, and retrieval
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Optional, Dict, List
from config import DATA_DIR, LIVE_DATA_FILE, TRADES_FILE, SIGNALS_FILE
from utils.logger import trading_logger


class DataHandler:
    """Handle all data operations for the trading system"""
    
    def __init__(self):
        self.logger = trading_logger.system_logger
        self._ensure_data_directory()
        self._initialize_data_files()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Data directory initialized: {DATA_DIR}")
    
    def _initialize_data_files(self):
        """Initialize CSV files with headers if they don't exist"""
        # Live data file
        if not LIVE_DATA_FILE.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df.to_csv(LIVE_DATA_FILE, index=False)
            self.logger.info(f"Created live data file: {LIVE_DATA_FILE}")
        
        # Trades file
        if not TRADES_FILE.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'ticker', 'action', 'price', 'size', 
                'signal', 'stop_loss', 'take_profit', 'pnl', 'status'
            ])
            df.to_csv(TRADES_FILE, index=False)
            self.logger.info(f"Created trades file: {TRADES_FILE}")
        
        # Signals file
        if not SIGNALS_FILE.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'strategy', 'signal', 'confidence', 
                'price', 'target', 'stop'
            ])
            df.to_csv(SIGNALS_FILE, index=False)
            self.logger.info(f"Created signals file: {SIGNALS_FILE}")
    
    def add_bar(self, bar_data: Dict):
        """
        Add a new price bar from TradingView webhook
        
        Args:
            bar_data: Dict with keys: timestamp, open, high, low, close, volume
        """
        try:
            df = pd.read_csv(LIVE_DATA_FILE)
            
            # Convert timestamp to datetime if it's a string
            if isinstance(bar_data['timestamp'], str):
                bar_data['timestamp'] = pd.to_datetime(bar_data['timestamp'])
            
            # Append new bar
            new_row = pd.DataFrame([bar_data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Keep only last 10,000 bars to manage file size
            if len(df) > 10000:
                df = df.tail(10000)
            
            df.to_csv(LIVE_DATA_FILE, index=False)
            self.logger.debug(f"Added bar: {bar_data['timestamp']} | Close: {bar_data['close']}")
            
        except Exception as e:
            self.logger.error(f"Error adding bar: {e}", exc_info=True)
            trading_logger.log_error("DataHandler.add_bar", e)
    
    def get_latest_bars(self, n: int = 100) -> pd.DataFrame:
        """
        Get the most recent N bars
        
        Args:
            n: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            df = pd.read_csv(LIVE_DATA_FILE)
            
            # FIXED: Use format='mixed' to handle both timestamp formats
            # This handles both: "2024-11-04 14:28:00" and "2024-11-04 14:28:00.626901"
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            
            df = df.sort_values('timestamp').tail(n)
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            self.logger.error(f"Error getting latest bars: {e}")
            return pd.DataFrame()
    
    def get_bars_between(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get bars within a specific time range"""
        try:
            df = pd.read_csv(LIVE_DATA_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
            df = df[mask].set_index('timestamp')
            return df
        except Exception as e:
            self.logger.error(f"Error getting bars between times: {e}")
            return pd.DataFrame()
    
    def get_opening_range_bars(self, date: datetime, or_minutes: int = 15) -> pd.DataFrame:
        """
        Get bars for the opening range period
        
        Args:
            date: Trading date
            or_minutes: Opening range duration in minutes
        """
        et_tz = pytz.timezone('US/Eastern')
        
        # Handle timezone-aware and naive datetimes
        if date.tzinfo is None:
            market_open = et_tz.localize(datetime.combine(date.date(), datetime.strptime('09:30', '%H:%M').time()))
        else:
            market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
        
        or_end = market_open + timedelta(minutes=or_minutes)
        
        return self.get_bars_between(market_open, or_end)
    
    def add_trade(self, trade_data: Dict):
        """
        Record a trade execution
        
        Args:
            trade_data: Dict with trade details
        """
        try:
            df = pd.read_csv(TRADES_FILE)
            new_trade = pd.DataFrame([trade_data])
            df = pd.concat([df, new_trade], ignore_index=True)
            df.to_csv(TRADES_FILE, index=False)
            
            trading_logger.log_trade(trade_data)
            self.logger.info(f"Trade recorded: {trade_data['action']} @ {trade_data['price']}")
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {e}")
            trading_logger.log_error("DataHandler.add_trade", e)
    
    def add_signal(self, signal_data: Dict):
        """
        Record a trading signal
        
        Args:
            signal_data: Dict with signal details
        """
        try:
            df = pd.read_csv(SIGNALS_FILE)
            new_signal = pd.DataFrame([signal_data])
            df = pd.concat([df, new_signal], ignore_index=True)
            df.to_csv(SIGNALS_FILE, index=False)
            
            trading_logger.log_signal(
                signal_data['strategy'],
                signal_data['signal'],
                signal_data['confidence'],
                signal_data['price']
            )
            
        except Exception as e:
            self.logger.error(f"Error adding signal: {e}")
    
    def get_all_trades(self) -> pd.DataFrame:
        """Get all recorded trades"""
        try:
            df = pd.read_csv(TRADES_FILE)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            return df
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return pd.DataFrame()
    
    def get_all_signals(self) -> pd.DataFrame:
        """Get all recorded signals"""
        try:
            df = pd.read_csv(SIGNALS_FILE)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            return df
        except Exception as e:
            self.logger.error(f"Error getting signals: {e}")
            return pd.DataFrame()
    
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Calculate PnL for a specific date"""
        if date is None:
            date = datetime.now()
        
        try:
            df = self.get_all_trades()
            if df.empty:
                return 0.0
            
            df['date'] = df['timestamp'].dt.date
            daily_trades = df[df['date'] == date.date()]
            
            return daily_trades['pnl'].sum() if 'pnl' in daily_trades.columns else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            df = self.get_all_trades()
            
            if df.empty or len(df) < 5:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'total_pnl': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Basic metrics
            winning_trades = df[df['pnl'] > 0]
            losing_trades = df[df['pnl'] < 0]
            
            total_trades = len(df)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            total_pnl = df['pnl'].sum()
            
            # Sharpe ratio (simplified)
            if len(df) > 1:
                returns = df['pnl'].pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cumulative_pnl = df['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / running_max.replace(0, 1)  # Avoid division by zero
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            trading_logger.log_performance(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """Export dataframe to CSV in data directory"""
        try:
            filepath = DATA_DIR / filename
            df.to_csv(filepath, index=False)
            self.logger.info(f"Data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")


# Create global data handler instance
data_handler = DataHandler()