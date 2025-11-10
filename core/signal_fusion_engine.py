from __future__ import annotations

import logging

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

# optional
try:
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    np = None  # type: ignore

# ---- session filter (make sure this import path matches your repo) ----
try:
    from utils.session_filter import SessionFilter
except Exception:
    # Minimal inline fallback if path differs (won't be used if import above works)
    from datetime import time
    NY = ZoneInfo("America/New_York")

    class SessionFilter:  # type: ignore
        def __init__(self, allow_globex: bool = True, rth_start: time | None = None, rth_end: time | None = None):
            self.allow_globex = allow_globex
            self.rth_start = rth_start or time(9, 30)
            self.rth_end = rth_end or time(16, 0)

        def in_session(self, ts_utc):
            if self.allow_globex:
                return True
            if ts_utc is None:
                return False
            ts_local = ts_utc.astimezone(NY)
            t = ts_local.time()
            return self.rth_start <= t <= self.rth_end


DEFAULT_FUSION_CONFIG: Dict[str, Any] = {
    # debug-friendly defaults (loosened so you can verify pipeline)
    "min_total_weight": 0.35,
    "min_signals_required": 1,
    "trade_cooldown_seconds": 5,
    "require_vision": False,
    "min_atr": 0.0,
    "force_once": False,   # set True to force a single trade for pipeline test
    "allow_globex": True,  # allow trading 24h; set False for RTH-only
}


@dataclass
class SignalFusionEngine:
    logger: Any
    cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FUSION_CONFIG))
    _cooldown_until: Optional[datetime] = None
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # convert UTC → New York before session check; allow Globex by default
        self.session = SessionFilter(allow_globex=self.cfg.get("allow_globex", True))

    # ----------------------------- helpers ----------------------------- #
    def on_cooldown(self) -> bool:
        return self._cooldown_until is not None and datetime.utcnow() < self._cooldown_until

    def start_cooldown(self):
        secs = float(self.cfg.get("trade_cooldown_seconds", 5))
        self._cooldown_until = datetime.utcnow() + timedelta(seconds=secs)

    def _audit(self, msg: str):
        try:
            self.logger.log_info(msg)
        except Exception:
            print(msg)

    def _reject(self, stage: str, reason: str) -> None:
        self._audit(f"[NO_TRADE] {stage}: {reason}")

    # ------------------------------ API -------------------------------- #
    def evaluate_trade_setup(
        self,
        market_context: Dict[str, Any],
        component_signals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Fuse component signals into one actionable trade dict.
        Return None if a gate rejects the setup.

        Expected output keys (downstream UI/PM expect these):
          signal, price, stop_loss, take_profit, weight, reason
        """

        # --- 0) freshness & session gates
        ts = market_context.get("timestamp")            # should be timezone-aware UTC
        price = market_context.get("price")
        bar_age = market_context.get("bar_age_sec")

        if bar_age is not None and bar_age > 180:
            self._reject("freshness", f"bar_age={bar_age:.1f}s > 180s")
            return None

        if not self.session.in_session(ts):
            self._reject("session", f"outside RTH (ts={ts})")
            return None

        if self.on_cooldown():
            self._reject("cooldown", f"until={self._cooldown_until}")
            return None

        # --- 1) collect/weight component signals
        approved: List[Dict[str, Any]] = []
        total_weight = 0.0
        fused_direction: Optional[str] = None

        for s in component_signals or []:
            # accept either {'signal': 'LONG'} or {'direction': 'LONG'}
            sig = s.get("signal") or s.get("direction")
            if sig not in ("LONG", "SHORT"):
                continue
            w = float(s.get("weight", 0.0))
            if w <= 0:
                continue
            s = dict(s)
            s["signal"] = sig
            approved.append(s)
            total_weight += w
            fused_direction = sig  # simple majority / last-wins

        if len(approved) < self.cfg["min_signals_required"]:
            self._reject("fusion", f"signals={len(approved)} < {self.cfg['min_signals_required']}")
            return None

        if total_weight < self.cfg["min_total_weight"]:
            self._reject("fusion", f"weight={total_weight:.2f} < {self.cfg['min_total_weight']:.2f}")
            return None

        # --- 2) preliminary trade object
        fused_result: Dict[str, Any] = {
            "direction": fused_direction or "LONG",
            "entry": price,
            "stop": market_context.get("stop") or (price - 40 if fused_direction == "LONG" else price + 40),
            "targets": market_context.get("targets") or [
                price + 60 if fused_direction == "LONG" else price - 60
            ],
            "weight": total_weight,
            "reason": "fusion_approved",
        }

        # --- 3) normalize keys *before* logging/returning
        # direction → signal
        fused_result["signal"] = fused_result.pop("direction")

        # entry → price
        fused_result["price"] = fused_result.get("entry")

        # targets/target → take_profit  (prefer T2 if present)
        tp = fused_result.get("target")
        if tp is None:
            tlist = fused_result.get("targets") or []
            if isinstance(tlist, list) and tlist:
                tp = tlist[1] if len(tlist) > 1 else tlist[0]
        if tp is None and fused_result.get("price") is not None:
            px = fused_result["price"]
            tp = px + 50 if fused_result["signal"] == "LONG" else px - 50
        fused_result["take_profit"] = tp

        # stop → stop_loss
        fused_result["stop_loss"] = fused_result.get("stop")

        # convenience alias for UI tables
        fused_result["entry"] = fused_result["price"]

        # --- 4) optional one-off forced trade to verify pipeline
        if self.cfg.get("force_once") and not getattr(self, "_forced", False):
            self._forced = True
            self._audit("[FORCE] Emitting one debug trade to verify pipeline")
            forced = {
                "signal": "LONG",
                "price": price,
                "stop_loss": price - 40 if price is not None else None,
                "take_profit": price + 60 if price is not None else None,
                "weight": 0.99,
                "reason": "force_once",
            }
            try:
                self.logger.log_trade_taken(forced, position={}, context=market_context)
            finally:
                self.start_cooldown()
            return forced

        # --- 5) log & cooldown
        try:
            self.logger.log_trade_taken(fused_result, position={}, context=market_context)
        finally:
            self.start_cooldown()

        self.recent_signals.append({
            "timestamp": datetime.utcnow(),
            "direction": fused_result["signal"],
            "weight": fused_result["weight"],
        })
        return fused_result
# ---- compatibility shim for older imports ----
def signal_fusion_engine(logger, cfg=None):
    """
    Backward-compatible factory so code that does:
        from core.signal_fusion_engine import signal_fusion_engine
    keeps working.
    """
    # start with defaults and overlay user cfg
    merged = dict(DEFAULT_FUSION_CONFIG)
    if cfg:
        merged.update(cfg)
    return SignalFusionEngine(logger=logger, cfg=merged)

# Create global instance
signal_fusion_engine = SignalFusionEngine(logger=logging.getLogger('signal_fusion'))