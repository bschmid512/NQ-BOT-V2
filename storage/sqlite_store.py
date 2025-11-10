from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Mapping, Iterable, Tuple
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  ts TEXT PRIMARY KEY,
  open REAL, high REAL, low REAL, close REAL, volume REAL
);
CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars(ts);
"""

class SQLiteStore:
    """Thread-safe SQLite store with proper transaction handling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = "data/nq_live.db"):
        """Singleton pattern to ensure one connection per database"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SQLiteStore, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = "data/nq_live.db"):
        # Only initialize once
        if self._initialized:
            return
            
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection with better settings
        self.conn = sqlite3.connect(
            db_path, 
            check_same_thread=False,
            isolation_level='DEFERRED',  # Use transactions instead of autocommit
            timeout=10.0  # Wait up to 10 seconds if database is locked
        )
        
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
        
        # Create schema
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        
        self._write_lock = threading.Lock()
        self._initialized = True

    def upsert_bar(self, bar: Mapping[str, float | str]) -> None:
        """Insert or update a bar with proper transaction handling"""
        with self._write_lock:
            try:
                self.conn.execute(
                    "INSERT OR REPLACE INTO bars(ts, open, high, low, close, volume) VALUES(?,?,?,?,?,?)",
                    (
                        str(bar["timestamp"]),
                        float(bar["open"]),
                        float(bar["high"]),
                        float(bar["low"]),
                        float(bar["close"]),
                        float(bar["volume"]),
                    ),
                )
                self.conn.commit()  # Explicitly commit the transaction
            except Exception as e:
                self.conn.rollback()
                raise e

    def latest(self, n: int = 1000) -> Iterable[Tuple[str, float, float, float, float, float]]:
        """Get the latest n bars in chronological order"""
        # Use a separate cursor for reading
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT ts, open, high, low, close, volume FROM bars ORDER BY ts DESC LIMIT ?",
                (int(n),),
            )
            rows = cur.fetchall()
            return rows[::-1]  # Reverse to get chronological order
        finally:
            cur.close()

    def between(self, start_iso: str, end_iso: str):
        """Get bars between two timestamps"""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT ts, open, high, low, close, volume FROM bars WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
                (start_iso, end_iso),
            )
            return cur.fetchall()
        finally:
            cur.close()

    def count(self) -> int:
        """Get total number of bars"""
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT COUNT(1) FROM bars")
            r = cur.fetchone()
            return int(r[0]) if r and r[0] is not None else 0
        finally:
            cur.close()
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()