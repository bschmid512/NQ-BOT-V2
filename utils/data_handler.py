from __future__ import annotations

import os, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import pandas as pd
import numpy as np
import pytz

# Import from your config
from config import DATA_DIR, LIVE_DATA_FILE, TRADES_FILE, SIGNALS_FILE
from utils.logger import trading_logger
from storage.sqlite_store import SQLiteStore

class DataHandler:
    """
    SQLite-backed bars (atomic). CSV for trades/signals.
    Complete version with all methods for dashboard.
    """
    
    _instance = None
    
    def __new__(cls, db_path: str = "data/nq_live.db"):
        """Singleton pattern - ensures only one DataHandler instance exists"""
        if cls._instance is None:
            cls._instance = super(DataHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = "data/nq_live.db"):
        # Only initialize once
        if self._initialized:
            return
            
        self.system_logger = trading_logger.system_logger
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_trade_signal_files()
        
        # Use the singleton SQLiteStore
        self.store = SQLiteStore(db_path)
        
        # Import CSV data only once at startup if DB is empty
        try:
            self._maybe_import_csv_to_sqlite()
        except Exception as e:
            self.system_logger.error(f"CSV→SQLite import failed: {e}")
        
        self._initialized = True

    def _init_trade_signal_files(self):
        """Initialize CSV files for trades and signals"""
        if not TRADES_FILE.exists():
            pd.DataFrame(columns=[
                "timestamp","ticker","action","price","size","signal","stop_loss","take_profit","pnl","status",
                "entry_price","exit_price","entry_time","exit_time","r_multiple"
            ]).to_csv(TRADES_FILE, index=False)
        if not SIGNALS_FILE.exists():
            pd.DataFrame(columns=[
                "timestamp","strategy","signal","confidence","price","target","stop"
            ]).to_csv(SIGNALS_FILE, index=False)

    # ---------- Bars (SQLite) ----------
    def add_bar(self, bar_data: dict) -> None:
        """Add a new bar to the database"""
        ts = pd.to_datetime(bar_data.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(ts):
            self.system_logger.warning(f"Invalid timestamp in bar_data: {bar_data.get('timestamp')}")
            return
        
        rec = {
            "timestamp": ts.isoformat(),
            "open":   float(bar_data.get("open",   np.nan)),
            "high":   float(bar_data.get("high",   np.nan)),
            "low":    float(bar_data.get("low",    np.nan)),
            "close":  float(bar_data.get("close",  np.nan)),
            "volume": float(bar_data.get("volume", np.nan)),
        }
        
        try:
            self.store.upsert_bar(rec)
            self.system_logger.debug(
                f"Added bar: {ts.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Close: {rec['close']:.2f} | "
                f"Total rows: {self.store.count()}"
            )
        except Exception as e:
            self.system_logger.error(f"Failed to add bar: {e}")
            raise

    def _rows_to_df(self, rows):
        """Convert database rows to DataFrame with proper timestamp handling"""
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        
        if df.empty:
            return pd.DataFrame()
        
        return df.set_index("timestamp").sort_index()

    def get_latest_bars(self, n: int = 150) -> pd.DataFrame:
        """Get the latest n bars"""
        try:
            rows = self.store.latest(max(200, n + 10))
            df = self._rows_to_df(rows)
            if len(df) > n:
                df = df.tail(n)
            return df
        except Exception as e:
            self.system_logger.error(f"Error getting latest bars: {e}")
            return pd.DataFrame()

    def get_bars_between(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get bars between two timestamps"""
        start = pd.Timestamp(start_time, tz="UTC") if start_time.tzinfo is None else pd.Timestamp(start_time).tz_convert("UTC")
        end   = pd.Timestamp(end_time,   tz="UTC") if end_time.tzinfo is None else pd.Timestamp(end_time).tz_convert("UTC")
        
        try:
            rows = self.store.between(start.isoformat(), end.isoformat())
            return self._rows_to_df(rows)
        except Exception as e:
            self.system_logger.error(f"Error getting bars between timestamps: {e}")
            return pd.DataFrame()

    def get_opening_range_bars(self, date: datetime, or_minutes: int = 15) -> pd.DataFrame:
        """Get bars during the opening range period"""
        et = pytz.timezone("US/Eastern")
        if date.tzinfo is None:
            market_open = et.localize(datetime.combine(date.date(), datetime.strptime("09:30", "%H:%M").time()))
        else:
            market_open = date.astimezone(et).replace(hour=9, minute=30, second=0, microsecond=0)
        or_end = market_open + timedelta(minutes=or_minutes)
        return self.get_bars_between(market_open, or_end)

    def _maybe_import_csv_to_sqlite(self):
        """One-time backfill from CSV if database is empty"""
        if self.store.count() > 0:
            self.system_logger.info(f"Database already has {self.store.count()} bars, skipping CSV import")
            return
        if not LIVE_DATA_FILE.exists():
            self.system_logger.info("No CSV file found to import")
            return
        
        try:
            df = pd.read_csv(LIVE_DATA_FILE)
            if df.empty:
                return
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            
            for idx, row in df.iterrows():
                try:
                    self.store.upsert_bar({
                        "timestamp": row["timestamp"].isoformat(),
                        "open": float(row.get("open", 0) or 0),
                        "high": float(row.get("high", 0) or 0),
                        "low": float(row.get("low", 0) or 0),
                        "close": float(row.get("close", 0) or 0),
                        "volume": float(row.get("volume", 0) or 0),
                    })
                except Exception as e:
                    self.system_logger.error(f"Error importing row {idx}: {e}")
                    continue
            
            self.system_logger.info(f"Successfully imported {len(df)} rows from CSV into SQLite")
        except Exception as e:
            self.system_logger.error(f"CSV import error: {e}")
            raise

    # ---------- Signals (CSV) ----------
    def append_signal(self, row: Dict) -> None:
        """Append a signal to the signals CSV file"""
        try:
            cols = ["timestamp","strategy","signal","confidence","price","target","stop"]
            rec = {
                "timestamp": row.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "strategy": row.get("strategy", "fusion"),
                "signal": (row.get("direction") or row.get("signal") or ""),
                "confidence": float(row.get("confidence", 0.0)),
                "price": float(row.get("price", 0.0)),
                "target": row.get("target"),
                "stop": row.get("stop"),
            }
            header_needed = (not SIGNALS_FILE.exists()) or os.path.getsize(SIGNALS_FILE) == 0
            with open(SIGNALS_FILE, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                if header_needed:
                    w.writeheader()
                w.writerow(rec)
        except Exception as e:
            self.system_logger.error(f"Error appending signal: {e}")

    def get_all_signals(self, minutes: int | None = None) -> pd.DataFrame:
        """Get all signals from CSV, optionally filtered by time"""
        try:
            df = pd.read_csv(SIGNALS_FILE, on_bad_lines="skip")
        except FileNotFoundError:
            return pd.DataFrame()
        
        if df.empty:
            return df
            
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)
            
            # Filter by time if requested
            if minutes is not None:
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
                df = df[df["timestamp"] >= cutoff]
        
        return df.reset_index(drop=True)

    # ---------- Trades (CSV) ----------
    def append_trade(self, row: Dict) -> None:
        """Append a trade to the trades CSV file"""
        try:
            cols = ["timestamp","ticker","action","price","size","signal","stop_loss","take_profit","pnl","status",
                    "entry_price","exit_price","entry_time","exit_time","r_multiple"]
            rec = {
                "timestamp": row.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "ticker": row.get("ticker", "NQ"),
                "action": row.get("action", ""),
                "price": float(row.get("price", 0.0)),
                "size": int(row.get("size", 1)),
                "signal": row.get("signal", ""),
                "stop_loss": row.get("stop_loss", ""),
                "take_profit": row.get("take_profit", ""),
                "pnl": float(row.get("pnl", 0.0)),
                "status": row.get("status", ""),
                "entry_price": row.get("entry_price"),
                "exit_price": row.get("exit_price"),
                "entry_time": row.get("entry_time"),
                "exit_time": row.get("exit_time"),
                "r_multiple": row.get("r_multiple"),
            }
            header_needed = (not TRADES_FILE.exists()) or os.path.getsize(TRADES_FILE) == 0
            with open(TRADES_FILE, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                if header_needed:
                    w.writeheader()
                w.writerow(rec)
        except Exception as e:
            self.system_logger.error(f"Error appending trade: {e}")

    def get_all_trades(self) -> pd.DataFrame:
        """Get all trades from CSV"""
        try:
            df = pd.read_csv(TRADES_FILE, on_bad_lines="skip")
        except FileNotFoundError:
            return pd.DataFrame()
        
        if df.empty:
            return df
            
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # ---------- Performance Metrics ----------
    def calculate_performance_metrics(self) -> Dict:
        """Calculate overall trading performance metrics"""
        try:
            trades_df = self.get_all_trades()
            
            # Default metrics if no trades
            if trades_df.empty:
                return {
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            # Filter only closed trades with valid PnL
            closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
            if 'pnl' in closed_trades.columns:
                closed_trades['pnl'] = pd.to_numeric(closed_trades['pnl'], errors='coerce')
                closed_trades = closed_trades.dropna(subset=['pnl'])
            
            if closed_trades.empty:
                return {
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            # Calculate metrics
            total_pnl = float(closed_trades['pnl'].sum())
            total_trades = len(closed_trades)
            
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            losing_trades = closed_trades[closed_trades['pnl'] < 0]
            
            num_wins = len(winning_trades)
            num_losses = len(losing_trades)
            
            win_rate = num_wins / total_trades if total_trades > 0 else 0.0
            
            gross_profit = float(winning_trades['pnl'].sum()) if num_wins > 0 else 0.0
            gross_loss = float(abs(losing_trades['pnl'].sum())) if num_losses > 0 else 0.0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
            
            avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
            avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
            
            # Calculate max drawdown
            cumulative_pnl = closed_trades['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0
            
            # Calculate Sharpe ratio (simplified - assumes daily returns)
            if len(closed_trades) > 1:
                returns = closed_trades['pnl']
                sharpe_ratio = float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            return {
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'winning_trades': num_wins,
                'losing_trades': num_losses,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            self.system_logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'sharpe_ratio': 0.0
            }


# Create a singleton instance
data_handler = DataHandler()