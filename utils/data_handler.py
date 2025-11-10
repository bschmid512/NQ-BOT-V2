"""
Data Handler for NQ Trading Bot — robust version
Manages CSV data ingestion, storage, and retrieval for dashboard + engine.
"""
from __future__ import annotations

import os
import csv
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import pandas as pd
import numpy as np
import pytz

from config import DATA_DIR, LIVE_DATA_FILE, TRADES_FILE, SIGNALS_FILE
from utils.logger import trading_logger


class DataHandler:
    """Handle all data operations for the trading system."""

    # ─────────────────────────── init / setup ─────────────────────────── #

    def __init__(self):
        self.system_logger = trading_logger.system_logger
        self._ensure_data_directory()
        self._initialize_data_files()

    def _ensure_data_directory(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.system_logger.info(f"Data directory initialized: {DATA_DIR}")

    def _initialize_data_files(self):
        # Live bars
        if not LIVE_DATA_FILE.exists():
            pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            ).to_csv(LIVE_DATA_FILE, index=False)
            self.system_logger.info(f"Created live data file: {LIVE_DATA_FILE}")

        # Trades
        if not TRADES_FILE.exists():
            pd.DataFrame(
                columns=[
                    "timestamp", "ticker", "action", "price", "size",
                    "signal", "stop_loss", "take_profit", "pnl", "status",
                    "entry_price", "exit_price", "entry_time", "exit_time", "r_multiple",
                ]
            ).to_csv(TRADES_FILE, index=False)
            self.system_logger.info(f"Created trades file: {TRADES_FILE}")

        # Signals
        if not SIGNALS_FILE.exists():
            pd.DataFrame(
                columns=["timestamp", "strategy", "signal", "confidence", "price", "target", "stop"]
            ).to_csv(SIGNALS_FILE, index=False)
            self.system_logger.info(f"Created signals file: {SIGNALS_FILE}")

    # ───────────────────────────── bars I/O ───────────────────────────── #

    def add_bar(self, bar_data: dict) -> None:
        """
        Append a single OHLCV bar (from the webhook) into LIVE_DATA_FILE.
        Ensures UTC tz-aware timestamps and a stable schema.
        """
        cols = ["timestamp", "open", "high", "low", "close", "volume"]

        # Load existing or create empty with schema
        if LIVE_DATA_FILE.exists():
            try:
                df = pd.read_csv(LIVE_DATA_FILE)
            except Exception:
                df = pd.DataFrame(columns=cols)
        else:
            df = pd.DataFrame(columns=cols)

        # Normalize incoming payload
        ts = pd.to_datetime(bar_data.get("timestamp"), errors="coerce", utc=True)

        row = {
            "timestamp": ts,
            "open":   float(bar_data.get("open",   np.nan)),
            "high":   float(bar_data.get("high",   np.nan)),
            "low":    float(bar_data.get("low",    np.nan)),
            "close":  float(bar_data.get("close",  np.nan)),
            "volume": float(bar_data.get("volume", np.nan)),
        }

        # Append
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Clean up & enforce UTC tz-aware timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Trim file size
        if len(df) > 10000:
            df = df.tail(10000)

        # Persist
        df.to_csv(LIVE_DATA_FILE, index=False)

        # Log
        try:
            self.system_logger.debug(f"Added bar: {ts} | Close: {row['close']}")
        except Exception:
            pass

    def get_latest_bars(self, n: int = 100) -> pd.DataFrame:
        try:
            df = pd.read_csv(LIVE_DATA_FILE)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            return df.sort_values("timestamp").tail(n).set_index("timestamp")
        except Exception as e:
            self.system_logger.error(f"Error getting latest bars: {e}")
            return pd.DataFrame()

    def get_bars_between(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        try:
            df = pd.read_csv(LIVE_DATA_FILE)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

            # normalize inputs to UTC tz-aware
            start = pd.Timestamp(start_time, tz="UTC") if start_time.tzinfo is None \
                else pd.Timestamp(start_time).tz_convert("UTC")
            end = pd.Timestamp(end_time, tz="UTC") if end_time.tzinfo is None \
                else pd.Timestamp(end_time).tz_convert("UTC")

            m = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            return df.loc[m].set_index("timestamp")
        except Exception as e:
            self.system_logger.error(f"Error getting bars between times: {e}")
            return pd.DataFrame()

    def get_opening_range_bars(self, date: datetime, or_minutes: int = 15) -> pd.DataFrame:
        et = pytz.timezone("US/Eastern")
        if date.tzinfo is None:
            market_open = et.localize(datetime.combine(date.date(), datetime.strptime("09:30", "%H:%M").time()))
        else:
            market_open = date.astimezone(et).replace(hour=9, minute=30, second=0, microsecond=0)
        or_end = market_open + timedelta(minutes=or_minutes)
        return self.get_bars_between(market_open, or_end)

    # ─────────────────────── trades & signals (writers) ─────────────────────── #

    def add_trade(self, trade_data: Dict):
        """Legacy writer: RMW (read/modify/write)."""
        try:
            df = pd.read_csv(TRADES_FILE)
            df = pd.concat([df, pd.DataFrame([trade_data])], ignore_index=True)
            df.to_csv(TRADES_FILE, index=False)
            trading_logger.log_trade(trade_data)
            self.system_logger.info(f"Trade recorded: {trade_data.get('action')} @ {trade_data.get('price')}")
        except Exception as e:
            self.system_logger.error(f"Error adding trade: {e}")
            trading_logger.log_error("DataHandler.add_trade", e)

    def add_signal(self, signal_data: Dict):
        """Legacy writer: RMW (read/modify/write)."""
        try:
            df = pd.read_csv(SIGNALS_FILE)
            df = pd.concat([df, pd.DataFrame([signal_data])], ignore_index=True)
            df.to_csv(SIGNALS_FILE, index=False)
            trading_logger.log_signal(
                signal_data.get("strategy", ""),
                signal_data.get("signal", ""),
                signal_data.get("confidence", 0.0),
                signal_data.get("price", 0.0),
            )
        except Exception as e:
            self.system_logger.error(f"Error adding signal: {e}")

    # New, safe append methods (used by main.py / webhook)

    def append_signal(self, row: Dict) -> None:
        """Append one signal row without RMW; keeps the schema the dashboard expects."""
        try:
            cols = ["timestamp", "strategy", "signal", "confidence", "price", "target", "stop"]
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

    def append_trade(self, row: Dict) -> None:
        """Append one trade row without RMW; keeps the schema the dashboard expects."""
        try:
            cols = [
                "timestamp", "ticker", "action", "price", "size", "signal",
                "stop_loss", "take_profit", "pnl", "status",
                "entry_price", "exit_price", "entry_time", "exit_time", "r_multiple",
            ]
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

    # ───────────────────────────── readers for UI ───────────────────────────── #

    def get_all_trades(self) -> pd.DataFrame:
        """Read trades safely; tolerate bad rows and ensure expected columns."""
        cols = [
            "timestamp", "ticker", "action", "price", "size", "signal", "stop_loss", "take_profit",
            "pnl", "status", "entry_price", "exit_price", "entry_time", "exit_time", "r_multiple",
        ]
        try:
            df = pd.read_csv(TRADES_FILE, on_bad_lines="skip")
        except FileNotFoundError:
            return pd.DataFrame(columns=cols)

        for c in cols:
            if c not in df.columns:
                df[c] = None

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        # numeric coercions
        for c in ["price", "size", "stop_loss", "take_profit", "pnl", "entry_price", "exit_price", "r_multiple"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # text normalization (Series-safe)
        for c in ["ticker", "action", "signal", "status"]:
            if c in df.columns:
                df[c] = df[c].astype("string")
        if "status" in df.columns:
            df["status"] = df["status"].str.upper()

        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)[cols]

    def get_all_signals(self, minutes: int | None = None) -> pd.DataFrame:
        """Read signals for Current Signals panel; optional minute filter."""
        cols = ["timestamp", "strategy", "signal", "confidence", "price", "target", "stop"]
        try:
            df = pd.read_csv(SIGNALS_FILE, dtype={"signal": "string"}, on_bad_lines="skip")
        except FileNotFoundError:
            return pd.DataFrame(columns=cols)

        for c in cols:
            if c not in df.columns:
                df[c] = None

        # Make the series timezone-aware (UTC)
        df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["price"]      = pd.to_numeric(df["price"], errors="coerce")
        df["strategy"]   = df["strategy"].astype("string")
        df["signal"]     = df["signal"].astype("string")

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)

        if minutes:
            # Compare on NAIVE UTC to avoid tz-aware/naive mismatch
            cutoff_naive = pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)
            ts_naive = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            df = df[ts_naive >= cutoff_naive]

        return df.reset_index(drop=True)[cols]

    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        date = date or datetime.now()
        try:
            df = self.get_all_trades()
            if df.empty:
                return 0.0
            df["date"] = df["timestamp"].dt.date
            return float(df.loc[df["date"] == date.date(), "pnl"].sum())
        except Exception as e:
            self.system_logger.error(f"Error calculating daily PnL: {e}")
            return 0.0

    def calculate_performance_metrics(self) -> Dict:
        """Totals for the KPI cards (P&L, Win Rate, Profit Factor, Max DD)."""
        try:
            df = self.get_all_trades()
            if df.empty:
                return {"total_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

            closed = df[df["status"].eq("CLOSED")] if "status" in df.columns else df
            if closed.empty:
                return {"total_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

            pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
            total_pnl = float(pnl.sum())

            wins = int((pnl > 0).sum())
            losses = int((pnl < 0).sum())
            win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else 0.0

            gross_profit = float(pnl[pnl > 0].sum())
            gross_loss = float(-pnl[pnl < 0].sum())
            if gross_loss == 0.0:
                profit_factor = float("inf") if gross_profit > 0 else 0.0
            else:
                profit_factor = gross_profit / gross_loss

            equity = pnl.cumsum()
            run_max = equity.cummax()
            max_drawdown = float(abs((equity - run_max).min())) if not equity.empty else 0.0

            return {
                "total_pnl": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),  # 0..1 (UI formats as %)
                "profit_factor": 0.0 if math.isinf(profit_factor) else round(profit_factor, 2),
                "max_drawdown": round(max_drawdown, 2),
            }
        except Exception as e:
            self.system_logger.error(f"Error calculating metrics: {e}")
            return {"total_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    # ───────────────────────────── utilities ───────────────────────────── #

    def export_to_csv(self, df: pd.DataFrame, filename: str):
        try:
            (DATA_DIR / filename).write_text(df.to_csv(index=False))
            self.system_logger.info(f"Data exported to {DATA_DIR/filename}")
        except Exception as e:
            self.system_logger.error(f"Error exporting CSV: {e}")


# Global singleton
data_handler = DataHandler()
