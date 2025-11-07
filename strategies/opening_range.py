"""
Opening Range Breakout Strategy — hardened & engine-compatible
- Computes an opening range dict safely
- Accepts df/current_bar/or_data/context
- Returns a consistent signal dict for the fusion engine
"""
from datetime import datetime, time
from typing import Optional, Dict

import pandas as pd
import pytz

from config import STRATEGIES
from utils.logger import trading_logger


class OpeningRangeBreakout:
    """
    Opening Range Breakout Strategy (realistic assumptions)

    OR window: first `or_period` minutes after the session start.
    By default this uses the *calendar day start* as a proxy for session start.
    If you want explicit 9:30–9:45 ET, tell me and I’ll switch the window to that clock.
    """

    def __init__(self, config: Dict = None):
        cfg = (config or STRATEGIES.get('orb', {})) or {}
        self.config = cfg
        self.or_period = int(cfg.get('or_period', 15))
        self.target_pct = float(cfg.get('target_pct', 1.0))
        self.max_sl_points = float(cfg.get('max_sl_points', 50))
        self.min_range_pct = float(cfg.get('min_range_pct', 0.0015))   # 0.15%
        self.max_range_pct = float(cfg.get('max_range_pct', 0.0040))   # 0.40%
        self.weight = float(cfg.get('weight', 0.6))
        self.optimal_days = cfg.get('optimal_days', [0, 2, 4])         # Mon/Wed/Fri by default

        self.logger = trading_logger.strategy_logger

        # Daily state (optional; not required for signal)
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_size: Optional[float] = None
        self.current_date = None
        self.trade_taken_today = False

        self.logger.info(
            f"ORB Strategy initialized: period={self.or_period}m, "
            f"target_pct={self.target_pct:.2f}, max_sl={self.max_sl_points}"
        )

    # ---------- helpers ----------

    def _latest_ts(self, df: Optional[pd.DataFrame], current_bar: Optional[dict]):
        try:
            if isinstance(current_bar, dict) and current_bar.get("timestamp"):
                return pd.to_datetime(current_bar["timestamp"])
            if df is not None and not df.empty:
                return df.index[-1]
        except Exception:
            pass
        return pd.Timestamp.utcnow()

    def _compute_opening_range(self, df: Optional[pd.DataFrame]) -> Optional[Dict]:
        """
        Build OR dict: {'high','low','size','start','end'} or None if not ready.

        Current implementation uses the first `or_period` minutes of the *day* (00:00 – 00:15 as proxy).
        If your feed is RTH (9:30) only, I can switch this to 9:30–9:30+or_period ET explicitly.
        """
        try:
            if df is None or df.empty:
                return None

            # Ensure we have a DatetimeIndex
            idx = df.index
            if not isinstance(idx, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.copy()
                    df.index = pd.to_datetime(df["timestamp"])
                else:
                    return None

            last_ts = df.index[-1]
            # Window: first `or_period` minutes of the current day (proxy for session)
            day_start = last_ts.normalize()
            or_end = day_start + pd.Timedelta(minutes=self.or_period)
            or_window = df[(df.index >= day_start) & (df.index < or_end)]

            if or_window.empty or len(or_window) < 2:
                return None

            or_high = float(or_window["high"].max())
            or_low = float(or_window["low"].min())
            size = max(or_high - or_low, 0.0)

            # Sanity range filters (by close %)
            close_ref = float(df["close"].iloc[-1])
            rng_pct = size / max(1e-9, close_ref)
            if rng_pct < self.min_range_pct or rng_pct > self.max_range_pct:
                return None

            return {
                "high": or_high,
                "low": or_low,
                "size": size,
                "start": day_start,
                "end": or_end,
            }
        except Exception as e:
            try:
                self.logger.error(f"ORB: compute range failed: {e}")
            except Exception:
                pass
            return None

    def _is_after_or_window(self, when: pd.Timestamp) -> bool:
        """Require signals only after OR window completes."""
        try:
            day_start = when.normalize()
            or_end = day_start + pd.Timedelta(minutes=self.or_period)
            return when >= or_end
        except Exception:
            return True

    # ---------- public API ----------

    def reset_daily_state(self, current_date: datetime.date):
        if self.current_date != current_date:
            self.or_high = self.or_low = self.or_size = None
            self.trade_taken_today = False
            self.current_date = current_date

    def should_trade_today(self, day_of_week: int) -> bool:
        return day_of_week in self.optimal_days

    def is_within_trading_hours(self, timestamp: datetime) -> bool:
        """Example RTH check: after OR window up to 16:00 ET."""
        et = pytz.timezone("US/Eastern")
        ts_et = timestamp.astimezone(et) if timestamp.tzinfo else et.localize(timestamp)
        or_end_hour = 9
        or_end_minute = 30 + self.or_period
        if or_end_minute >= 60:
            or_end_hour += 1
            or_end_minute -= 60
        or_end_time = time(or_end_hour, or_end_minute)
        return (ts_et.time() >= or_end_time) and (ts_et.time() < time(16, 0))

    def generate_signal(
        self,
        df: Optional[pd.DataFrame] = None,
        current_bar: Optional[dict] = None,
        or_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Generate an ORB signal dict or None.

        Returns (example):
        {
            "strategy": "orb",
            "direction": "LONG" | "SHORT",
            "signal": "LONG" | "SHORT",   # duplicate for backward-compat
            "price": 12345.0,
            "stop": 12325.0,
            "stop_loss": 12325.0,
            "target": 12395.0,
            "take_profit": 12395.0,
            "confidence": 0.6,
            "reason": "ORB breakout above ...",
            "timestamp": pd.Timestamp(...),
            "or_high": ..., "or_low": ..., "or_size": ...
        }
        """
        try:
            ts = self._latest_ts(df, current_bar)
            # Optional “day filter”
            if not self.should_trade_today(ts.weekday()):
                return None
            # Require after OR window ends
            if not self._is_after_or_window(ts):
                return None

            # Build or_data if not provided/invalid
            if not isinstance(or_data, dict) or not {"high", "low", "size"} <= set(or_data.keys()):
                or_data = self._compute_opening_range(df)
            if not or_data:
                return None

            # Price from current_bar or last close
            price = None
            if isinstance(current_bar, dict):
                price = current_bar.get("close") or current_bar.get("price")
            if price is None and df is not None and not df.empty:
                price = float(df["close"].iloc[-1])
            if price is None:
                return None

            or_high = float(or_data["high"])
            or_low = float(or_data["low"])
            size = float(or_data["size"])

            broke_up = price > or_high
            broke_dn = price < or_low
            if not (broke_up or broke_dn):
                return None

            max_sl = float(self.max_sl_points) if self.max_sl_points is not None else size
            tgt_pct = float(self.target_pct) if self.target_pct is not None else 1.0
            rr_min = 1.5

            if broke_up:
                stop_price = price - min(size, max_sl)
                target_dist = max(size * tgt_pct, rr_min * (price - stop_price))
                target_price = price + target_dist
                return {
                    "strategy": self.config.get("name", "orb"),
                    "direction": "LONG",
                    "signal": "LONG",
                    "price": float(price),
                    "stop": float(stop_price),
                    "stop_loss": float(stop_price),
                    "target": float(target_price),
                    "take_profit": float(target_price),
                    "confidence": self.weight,
                    "reason": f"ORB breakout above {or_high:.2f} (size {size:.2f})",
                    "timestamp": ts,
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_size": size,
                }

            # broke_dn
            stop_price = price + min(size, max_sl)
            target_dist = max(size * tgt_pct, rr_min * (stop_price - price))
            target_price = price - target_dist
            return {
                "strategy": self.config.get("name", "orb"),
                "direction": "SHORT",
                "signal": "SHORT",
                "price": float(price),
                "stop": float(stop_price),
                "stop_loss": float(stop_price),
                "target": float(target_price),
                "take_profit": float(target_price),
                "confidence": self.weight,
                "reason": f"ORB breakdown below {or_low:.2f} (size {size:.2f})",
                "timestamp": ts,
                "or_high": or_high,
                "or_low": or_low,
                "or_size": size,
            }

        except Exception as e:
            try:
                self.logger.error(f"ORB generate_signal error: {e}")
            except Exception:
                pass
            return None


# Global instance used by the engine
orb_strategy = OpeningRangeBreakout()
