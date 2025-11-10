from __future__ import annotations
from datetime import datetime, time, timezone, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import pytz

from config import STRATEGIES
from utils.logger import trading_logger


class OpeningRangeBreakout:
    """
    Opening Range Breakout Strategy (realistic assumptions)

    OR window: first `or_period` minutes after the session start.
    This version uses 9:30 ET + `or_period` by default.
    """

    name = "orb"

    def __init__(self, config: Dict = None):
        cfg = (config or STRATEGIES.get('orb', {})) or {}
        self.config = cfg
        self.or_period = int(cfg.get('or_period', 15))
        self.target_pct = float(cfg.get('target_pct', 1.0))
        self.max_sl_points = float(cfg.get('max_sl_points', 50))
        self.min_range_pct = float(cfg.get('min_range_pct', 0.0015))   # 0.15%
        self.max_range_pct = float(cfg.get('max_range_pct', 0.0040))   # 0.40%
        self.weight = float(cfg.get('weight', 0.6))
        self.optimal_days = cfg.get('optimal_days', [0, 2, 4])         # Mon/Wed/Fri

        self.logger = trading_logger.strategy_logger

        # Daily state (optional; not required for signal)
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_size: Optional[float] = None
        self.current_date = None
        self.trade_taken_today = False

        self._et = pytz.timezone("US/Eastern")
        self.logger.info(
            f"ORB Strategy initialized: period={self.or_period}m, "
            f"target_pct={self.target_pct:.2f}, max_sl={self.max_sl_points}"
        )

    # ---------- helpers (NOW inside the class) ----------

    def _latest_ts(self, df: Optional[pd.DataFrame], current_bar: Optional[dict]):
        try:
            if isinstance(current_bar, dict) and current_bar.get("timestamp") is not None:
                ts = pd.to_datetime(current_bar["timestamp"], errors="coerce", utc=True)
                if pd.notna(ts):
                    return ts
            if df is not None and not df.empty:
                idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(
                    df.get("timestamp"), errors="coerce", utc=True
                )
                idx = idx[~idx.isna()]
                if len(idx):
                    return idx.max()
        except Exception:
            pass
        return pd.Timestamp.now(tz="UTC")

    def _compute_opening_range(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        try:
            if df is None or df.empty:
                return None

            # Ensure tz-aware index
            if isinstance(df.index, pd.DatetimeIndex):
                idx = pd.to_datetime(df.index, errors="coerce", utc=True)
            elif "timestamp" in df.columns:
                idx = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            else:
                return None

            mask = ~idx.isna()
            if not mask.any():
                return None
            df = df.loc[mask].copy()
            df.index = idx[mask]

            last_ts = df.index.max()
            if pd.isna(last_ts):
                return None

            # 9:30 ET -> 9:30 + or_period ET
            day_et = last_ts.tz_convert(self._et).replace(hour=9, minute=30, second=0, microsecond=0)
            or_end_et = day_et + pd.Timedelta(minutes=self.or_period)
            start_utc = day_et.tz_convert("UTC")
            end_utc = or_end_et.tz_convert("UTC")

            window = df[(df.index >= start_utc) & (df.index < end_utc)]
            if window.empty or len(window) < 2:
                return None

            or_high = float(window["high"].max())
            or_low  = float(window["low"].min())
            size    = max(or_high - or_low, 0.0)

            # sanity: keep ranges within expected pct of last close
            close_ref = float(df["close"].iloc[-1])
            rng_pct = size / max(1e-9, close_ref)
            if rng_pct < self.min_range_pct or rng_pct > self.max_range_pct:
                return None

            return {"high": or_high, "low": or_low, "size": size, "start": start_utc, "end": end_utc}

        except Exception as e:
            try:
                self.logger.error(f"ORB: compute range failed (guarded): {e}")
            except Exception:
                pass
            return None

    def _is_after_or_window(self, when: pd.Timestamp) -> bool:
        try:
            when_et = when.tz_convert(self._et)
            or_end = when_et.replace(hour=9, minute=30, second=0, microsecond=0) + pd.Timedelta(minutes=self.or_period)
            return when_et >= or_end
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
        et = self._et
        ts_et = timestamp.tz_convert(et) if isinstance(timestamp, pd.Timestamp) else et.localize(timestamp)
        or_end_time = (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=self.or_period)).time()
        return (ts_et.time() >= or_end_time) and (ts_et.time() < time(16, 0))

    def generate_signal(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Flexible signature so different engines can call it.
        Accepts df/current_bar/or_data/context via kwargs or a context dict in args[0].
        """
        try:
            ctx = args[0] if args and isinstance(args[0], dict) else {}
            df = kwargs.get("df") or kwargs.get("bars") or ctx.get("df") or ctx.get("bars")
            current_bar = kwargs.get("current_bar") or ctx.get("current_bar")

            ts = self._latest_ts(df, current_bar)

            # Optional day filter & post-OR enforcement
            if not self.should_trade_today(ts.weekday()):
                return None
            if not self._is_after_or_window(ts):
                return None

            or_data = kwargs.get("or_data") or ctx.get("or_data")
            if not isinstance(or_data, dict) or not {"high", "low", "size"} <= set(or_data.keys()):
                or_data = self._compute_opening_range(df)
            if not or_data:
                return None

            # price from current_bar or last close
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
                direction = "LONG"
            else:
                stop_price = price + min(size, max_sl)
                target_dist = max(size * tgt_pct, rr_min * (stop_price - price))
                target_price = price - target_dist
                direction = "SHORT"

            return {
                "strategy": self.name,
                "direction": direction,              # for engines using 'direction'
                "signal": direction,                 # for engines using 'signal'
                "price": float(price),
                "stop": float(stop_price),
                "stop_loss": float(stop_price),      # dashboard compatibility
                "target": float(target_price),
                "take_profit": float(target_price),  # dashboard compatibility
                "confidence": float(self.weight),
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
