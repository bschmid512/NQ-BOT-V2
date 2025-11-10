from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Mapping, Iterable, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  ts TEXT PRIMARY KEY,
  open REAL, high REAL, low REAL, close REAL, volume REAL
);
CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars(ts);
"""

class SQLiteStore:
    def __init__(self, db_path: str = "data/nq_live.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        # Better concurrency for readers while webhook writes
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        # IMPORTANT: multi-statement schema must use executescript
        self.conn.executescript(SCHEMA)

    def upsert_bar(self, bar: Mapping[str, float | str]) -> None:
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

    def latest(self, n: int = 1000) -> Iterable[Tuple[str, float, float, float, float, float]]:
        cur = self.conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars ORDER BY ts DESC LIMIT ?",
            (int(n),),
        )
        rows = cur.fetchall()
        return rows[::-1]

    def between(self, start_iso: str, end_iso: str):
        cur = self.conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
            (start_iso, end_iso),
        )
        return cur.fetchall()

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(1) FROM bars")
        r = cur.fetchone()
        return int(r[0]) if r and r[0] is not None else 0
