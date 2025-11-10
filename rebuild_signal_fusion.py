#!/usr/bin/env python3
"""
COMPREHENSIVE FIX for signal_fusion_engine.py
Rebuilds the file with correct structure
"""

import os
import shutil
from datetime import datetime

# Path to your file
file_path = r"C:\Users\bschm\OneDrive\Documents\nq_trading_bot\NQ-BOT-V2\core\signal_fusion_engine.py"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    print("Please update the file_path in this script")
    exit(1)

print("🔧 COMPREHENSIVE FIX for signal_fusion_engine.py")
print("="*70)

# Backup first
backup_path = file_path + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(file_path, backup_path)
print(f"✅ Backup created: {backup_path}")

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.splitlines(keepends=True)

print(f"📄 Original file: {len(lines)} lines")
print("\n🔍 Analyzing structure...\n")

# Find problematic lines
issues = []
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith('return ') and not line.startswith(' '):
        # return at column 0 - definitely wrong
        issues.append((i, line.rstrip(), "return at column 0"))
    elif stripped.startswith('return '):
        # Check if it's properly indented (should be at least 8 spaces for method)
        indent = len(line) - len(line.lstrip())
        if indent < 8:
            issues.append((i, line.rstrip(), f"return with only {indent} spaces"))

if issues:
    print("❌ Found issues:")
    for line_num, line_content, issue in issues:
        print(f"   Line {line_num}: {issue}")
        print(f"      Content: '{line_content}'")
    print()

# The file should have proper structure. Let me write the correct version:
correct_content = '''"""
Signal Fusion Engine - Combines multiple strategy signals
"""
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

# ---- session filter ----
try:
    from utils.session_filter import SessionFilter
except Exception:
    # Minimal inline fallback
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
    "min_total_weight": 0.35,
    "min_signals_required": 1,
    "trade_cooldown_seconds": 5,
    "require_vision": False,
    "min_atr": 0.0,
    "force_once": False,
    "allow_globex": True,
}


@dataclass
class SignalFusionEngine:
    logger: Any
    cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FUSION_CONFIG))
    _cooldown_until: Optional[datetime] = None
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.session = SessionFilter(allow_globex=self.cfg.get("allow_globex", True))

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

    def evaluate_trade_setup(
        self,
        market_context: Dict[str, Any],
        component_signals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Fuse component signals into one actionable trade dict.
        Return None if a gate rejects the setup.
        """
        # --- 0) freshness & session gates
        ts = market_context.get("timestamp")
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
            fused_direction = sig

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

        # --- 3) normalize keys
        fused_result["signal"] = fused_result.pop("direction")
        fused_result["price"] = fused_result.get("entry")

        # targets/target → take_profit
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
        fused_result["entry"] = fused_result["price"]

        # --- 4) optional one-off forced trade
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
    Backward-compatible factory function.
    """
    merged = dict(DEFAULT_FUSION_CONFIG)
    if cfg:
        merged.update(cfg)
    return SignalFusionEngine(logger=logger, cfg=merged)


# Create global instance
signal_fusion_engine = SignalFusionEngine(logger=logging.getLogger('signal_fusion'))
'''

# Write the corrected file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(correct_content)

print("✅ File rebuilt with correct structure")
print(f"✅ New file: {len(correct_content.splitlines())} lines")
print(f"✅ Backup saved to: {backup_path}")
print("\n" + "="*70)
print("You can now run your bot:")
print("  python main.py")
print("="*70)
