from __future__ import annotations

import os, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import pandas as pd
import numpy as np
import pytz

from config import DATA_DIR, LIVE_DATA_FILE, TRADES_FILE, SIGNALS_FILE
from utils.logger import trading_logger
from storage.sqlite_store import SQLiteStore

class DataHandler:
    """SQLite-backed bars (atomic). CSV for trades/signals."""
    def __init__(self, db_path: str = "data/nq_live.db"):
        self.system_logger = trading_logger.system_logger
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_trade_signal_files()
        self.store = SQLiteStore(db_path)
        # if DB empty but CSV exists, import once so history is preserved
        try:
            self._maybe_import_csv_to_sqlite()
        except Exception as e:
            self.system_logger.error(f"CSV→SQLite import failed: {e}")

    def _init_trade_signal_files(self):
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
        ts = pd.to_datetime(bar_data.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(ts):
            return
        rec = {
            "timestamp": ts.isoformat(),
            "open":   float(bar_data.get("open",   np.nan)),
            "high":   float(bar_data.get("high",   np.nan)),
            "low":    float(bar_data.get("low",    np.nan)),
            "close":  float(bar_data.get("close",  np.nan)),
            "volume": float(bar_data.get("volume", np.nan)),
        }
        self.store.upsert_bar(rec)
        try:
            self.system_logger.debug(f"Added bar: {ts} | Close: {rec['close']} | rows={self.store.count()}")
        except Exception:
            pass

    def _rows_to_df(self, rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df.set_index("timestamp").sort_index()

    def get_latest_bars(self, n: int = 150) -> pd.DataFrame:
        rows = self.store.latest(max(200, n))
        return self._rows_to_df(rows)

    def get_bars_between(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start = pd.Timestamp(start_time, tz="UTC") if start_time.tzinfo is None else pd.Timestamp(start_time).tz_convert("UTC")
        end   = pd.Timestamp(end_time,   tz="UTC") if end_time.tzinfo is None else pd.Timestamp(end_time).tz_convert("UTC")
        rows = self.store.between(start.isoformat(), end.isoformat())
        return self._rows_to_df(rows)

    def get_opening_range_bars(self, date: datetime, or_minutes: int = 15) -> pd.DataFrame:
        et = pytz.timezone("US/Eastern")
        if date.tzinfo is None:
            market_open = et.localize(datetime.combine(date.date(), datetime.strptime("09:30", "%H:%M").time()))
        else:
            market_open = date.astimezone(et).replace(hour=9, minute=30, second=0, microsecond=0)
        or_end = market_open + timedelta(minutes=or_minutes)
        return self.get_bars_between(market_open, or_end)

    # one-time backfill from CSV if present
    def _maybe_import_csv_to_sqlite(self):
        if self.store.count() > 0:
            return
        if not LIVE_DATA_FILE.exists():
            return
        df = pd.read_csv(LIVE_DATA_FILE)
        if df.empty:
            return
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        for _, row in df.iterrows():
            self.store.upsert_bar({
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row.get("open", 0) or 0),
                "high": float(row.get("high", 0) or 0),
                "low": float(row.get("low", 0) or 0),
                "close": float(row.get("close", 0) or 0),
                "volume": float(row.get("volume", 0) or 0),
            })
        self.system_logger.info(f"Imported {len(df)} rows from CSV into SQLite.")

    # ---------- Trades & signals (CSV append) ----------
    def add_trade(self, trade_data: Dict):
        try:
            df = pd.read_csv(TRADES_FILE)
            df = pd.concat([df, pd.DataFrame([trade_data])], ignore_index=True)
            df.to_csv(TRADES_FILE, index=False)
            trading_logger.log_trade(trade_data)
        except Exception as e:
            self.system_logger.error(f"Error adding trade: {e}")
            trading_logger.log_error("DataHandler.add_trade", e)

    def add_signal(self, signal_data: Dict):
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

    def append_signal(self, row: Dict) -> None:
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

    def append_trade(self, row: Dict) -> None:
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
        try:
            df = pd.read_csv(TRADES_FILE, on_bad_lines="skip")
        except FileNotFoundError:
            return pd.DataFrame()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    def get_all_signals(self, minutes: int | None = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(SIGNALS_FILE, on_bad_lines="skip")
        except FileNotFoundError:
            return pd.DataFrame()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)
        return df.reset_index(drop=True)

data_handler = DataHandler()
