from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


class SessionFilter:
    """
    Session gate for trading.

    - Converts incoming UTC timestamps to New York time before comparing.
    - If allow_globex is True, all times are allowed.
    """

    def __init__(self, allow_globex: bool = True, rth_start: time | None = None, rth_end: time | None = None):
        self.allow_globex = allow_globex
        self.rth_start = rth_start or time(9, 30)  # 09:30
        self.rth_end = rth_end or time(16, 0)     # 16:00

    def in_session(self, ts_utc):
        if self.allow_globex:
            return True
        if ts_utc is None:
            return False
        try:
            # critical: convert from UTC to New York before session check
            ts_local = ts_utc.astimezone(NY)
        except Exception:
            # fallback for naive datetimes (treat as NY local)
            ts_local = ts_utc.replace(tzinfo=NY)
        t = ts_local.time()
        return self.rth_start <= t <= self.rth_end
