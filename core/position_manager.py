"""
Position Manager — stable, UTC-safe
Manages open positions, risk limits, and trade execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from config import (
    MAX_OPEN_POSITIONS,
    MAX_DAILY_LOSS,
    MAX_DAILY_TRADES,
    MIN_TIME_BETWEEN_TRADES,
    CONTRACT_SIZE,
    COMMISSION_PER_SIDE,
)


# ---------- Data Model ----------

@dataclass
class Position:
    """Represents an open trading position."""
    id: int
    strategy: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    size: int = 1
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_points: float = 0.0
    exit_reason: Optional[str] = None

    def update(self, current_price: float, current_time: datetime) -> Optional[str]:
        """
        Update position with current price, check for exits.
        Returns exit reason if position should close, None otherwise.
        """
        if self.status != "OPEN":
            return None

        # Check stop loss
        if self.stop_loss is not None:
            if self.direction == "LONG" and current_price <= self.stop_loss:
                return "STOP_LOSS"
            if self.direction == "SHORT" and current_price >= self.stop_loss:
                return "STOP_LOSS"

        # Check take profit
        if self.take_profit is not None:
            if self.direction == "LONG" and current_price >= self.take_profit:
                return "TAKE_PROFIT"
            if self.direction == "SHORT" and current_price <= self.take_profit:
                return "TAKE_PROFIT"

        return None

    def close(self, exit_price: float, exit_time: datetime, reason: str) -> None:
        """Close the position and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = "CLOSED"

        # P&L in points (per contract)
        if self.direction == "LONG":
            self.pnl_points = (exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.pnl_points = (self.entry_price - exit_price) * self.size

        # Convert to dollars using contract multiplier and subtract round-trip commissions
        self.pnl = self.pnl_points * CONTRACT_SIZE
        self.pnl -= (COMMISSION_PER_SIDE * 2 * self.size)

    def to_dict(self) -> Dict:
        """Convert position to dictionary for storage."""
        return {
            "id": self.id,
            "strategy": self.strategy,
            "direction": self.direction,
            "status": self.status,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "pnl": round(self.pnl, 2),
            "pnl_points": round(self.pnl_points, 2),
            "exit_reason": self.exit_reason,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
        }


# ---------- Manager ----------

class PositionManager:
    """Manages all trading positions and risk limits."""

    def __init__(
        self,
        max_positions: int = MAX_OPEN_POSITIONS,
        max_daily_loss: float = MAX_DAILY_LOSS,
    ):
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss

        # Position tracking
        self.positions: Dict[int, Position] = {}
        self.position_counter = 0

        # Daily tracking
        self.current_date = datetime.now(timezone.utc).date()
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time: Optional[datetime] = None

        print(
            f" Position Manager initialized: "
            f"Max={max_positions}, Daily Loss Limit=${max_daily_loss}"
        )

    # ----- Helpers -----

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _ensure_utc(self, dt: datetime) -> datetime:
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    def reset_daily_counters(self) -> None:
        """Reset daily counters if new trading day."""
        today = self._now_utc().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_time = None
            print(f" Daily counters reset for {today}")

    # ----- Risk Gates -----

    def can_open_position(self) -> tuple[bool, str]:
        """Check if we can open a new position."""
        self.reset_daily_counters()

        # Max concurrent positions
        open_positions = [p for p in self.positions.values() if p.status == "OPEN"]
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Daily loss limit (stop trading if PnL is below -MAX_DAILY_LOSS)
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit hit (${self.daily_pnl:.2f})"

        # Max daily trades
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, f"Max daily trades reached ({MAX_DAILY_TRADES})"

        # Minimum time between entries
        if self.last_trade_time is not None:
            elapsed = (self._now_utc() - self.last_trade_time).total_seconds()
            if elapsed < MIN_TIME_BETWEEN_TRADES:
                return False, f"Too soon since last trade ({int(elapsed)}s)"

        return True, "OK"

    # ----- Lifecycle -----

    def open_position(self, signal: Dict, entry_price: float, size: int = 1) -> Optional[Position]:
        """
        Open a new position from a signal.

        signal keys supported:
          - strategy
          - direction ('LONG'|'SHORT')
          - stop | stop_loss
          - target | take_profit
        """
        can_open, reason = self.can_open_position()
        if not can_open:
            print(f" Position rejected: {reason}")
            return None

        stop = signal.get("stop", signal.get("stop_loss"))
        target = signal.get("target", signal.get("take_profit"))

        # Create new position
        self.position_counter += 1
        position = Position(
            id=self.position_counter,
            strategy=signal.get("strategy", "unknown"),
            direction=signal.get("direction", "LONG"),
            entry_price=float(entry_price),
            entry_time=self._now_utc(),
            stop_loss=float(stop) if stop is not None else None,
            take_profit=float(target) if target is not None else None,
            size=int(size),
        )
        self.positions[position.id] = position
        self.daily_trades += 1
        self.last_trade_time = self._now_utc()

        stop_str = f"{position.stop_loss:.2f}" if position.stop_loss is not None else "None"
        target_str = f"{position.take_profit:.2f}" if position.take_profit is not None else "None"

        print(f" POSITION OPENED #{position.id} | {position.direction} @ {position.entry_price:.2f}")
        print(f"   Stop: {stop_str} | Target: {target_str} | Strategy: {position.strategy}")

        # Log OPEN trade for dashboard
        try:
            from utils.data_handler import data_handler
            data_handler.append_trade({
                "timestamp": position.entry_time.isoformat(),
                "ticker": "NQ",
                "action": "BUY" if position.direction == "LONG" else "SELL",
                "price": position.entry_price,
                "size": position.size,
                "signal": position.strategy,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "pnl": 0.0,
                "status": "OPEN",
                # explicit entry/exit fields for the UI table
                "entry_price": position.entry_price,
                "exit_price": None,
                "entry_time": position.entry_time.isoformat(),
                "exit_time": None,
                "r_multiple": None,
            })
        except Exception as e:
            print(f" Failed to append OPEN trade: {e}")

        return position

    def update_positions(self, current_price: float, current_time: datetime) -> None:
        """Update all open positions with current market price."""
        current_time = self._ensure_utc(current_time)
        for position in [p for p in self.positions.values() if p.status == "OPEN"]:
            exit_reason = position.update(current_price, current_time)
            if exit_reason:
                self._close_position(position, current_price, current_time, exit_reason)

    def _close_position(self, position: Position, exit_price: float, exit_time: datetime, reason: str) -> Dict:
        """Close a position and record the trade."""
        exit_time = self._ensure_utc(exit_time)
        position.close(float(exit_price), exit_time, reason)

        # R multiple (risk per contract in points)
        r_points: Optional[float] = None
        if position.stop_loss is not None:
            if position.direction == "LONG":
                risk = position.entry_price - position.stop_loss
            else:
                risk = position.stop_loss - position.entry_price
            r_points = (position.pnl_points / position.size) / risk if risk > 0 else None

        # Append CLOSED trade to dashboard CSV
        try:
            from utils.data_handler import data_handler
            data_handler.append_trade({
                "timestamp": exit_time.isoformat(),
                "ticker": "NQ",
                "action": "SELL" if position.direction == "LONG" else "BUY",
                "price": float(exit_price),
                "size": position.size,
                "signal": position.strategy,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "pnl": position.pnl,
                "status": "CLOSED",
                "entry_price": position.entry_price,
                "exit_price": float(exit_price),
                "entry_time": position.entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "r_multiple": None if r_points is None else round(r_points, 2),
            })
        except Exception as e:
            print(f" Failed to append CLOSED trade: {e}")

        # Update daily P&L
        self.daily_pnl += position.pnl

        # Log
        print(f" POSITION CLOSED #{position.id} | {position.direction}")
        print(f"   Entry: {position.entry_price:.2f} → Exit: {float(exit_price):.2f}")
        print(f"   P&L: ${position.pnl:.2f} ({position.pnl_points:+.2f} pts)")
        print(f"   Reason: {reason} | Strategy: {position.strategy}")

        return position.to_dict()

    # ----- Queries -----

    def get_open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.status == "OPEN"]

    def get_daily_stats(self) -> Dict:
        """Get statistics for current trading day."""
        self.reset_daily_counters()
        closed_today = [
            p
            for p in self.positions.values()
            if p.status == "CLOSED" and p.exit_time and p.exit_time.date() == self.current_date
        ]

        if not closed_today:
            return {
                "trades": 0,
                "pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        wins = [p for p in closed_today if p.pnl > 0]
        losses = [p for p in closed_today if p.pnl <= 0]

        return {
            "trades": len(closed_today),
            "pnl": sum(p.pnl for p in closed_today),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(closed_today)) if closed_today else 0.0,
            "avg_win": (sum(p.pnl for p in wins) / len(wins)) if wins else 0.0,
            "avg_loss": (sum(p.pnl for p in losses) / len(losses)) if losses else 0.0,
            "largest_win": max((p.pnl for p in wins), default=0.0),
            "largest_loss": min((p.pnl for p in losses), default=0.0),
        }

    def force_close_all(self, current_price: float, reason: str = "FORCED_CLOSE") -> List[Dict]:
        """Force close all open positions (e.g., EOD)."""
        open_positions = self.get_open_positions()
        if not open_positions:
            return []

        print(f" Force closing {len(open_positions)} positions: {reason}")
        out: List[Dict] = []
        for p in open_positions:
            out.append(self._close_position(p, current_price, self._now_utc(), reason))
        return out


# Create a global instance (import this where needed)
position_manager = PositionManager()
