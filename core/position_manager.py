"""
Position Manager - FIXED VERSION
Manages open positions, risk limits, and trade execution
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from config import (
    MAX_OPEN_POSITIONS, MAX_DAILY_LOSS, MAX_DAILY_TRADES,
    MIN_TIME_BETWEEN_TRADES, CONTRACT_SIZE, COMMISSION_PER_SIDE
)


@dataclass
class Position:
    """Represents an open trading position"""
    id: int
    strategy: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    size: int = 1
    status: str = 'OPEN'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_points: float = 0.0
    exit_reason: Optional[str] = None
    
    def update(self, current_price: float, current_time: datetime) -> Optional[str]:
        """
        Update position with current price, check for exits
        Returns exit reason if position should close, None otherwise
        """
        if self.status != 'OPEN':
            return None
        
        # Check stop loss
        if self.stop_loss is not None:
            if self.direction == 'LONG' and current_price <= self.stop_loss:
                return 'STOP_LOSS'
            elif self.direction == 'SHORT' and current_price >= self.stop_loss:
                return 'STOP_LOSS'
        
        # Check take profit
        if self.take_profit is not None:
            if self.direction == 'LONG' and current_price >= self.take_profit:
                return 'TAKE_PROFIT'
            elif self.direction == 'SHORT' and current_price <= self.take_profit:
                return 'TAKE_PROFIT'
        
        return None
    
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the position and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = 'CLOSED'
        
        # Calculate P&L
        if self.direction == 'LONG':
            self.pnl_points = (exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.pnl_points = (self.entry_price - exit_price) * self.size
        
        # Convert to dollars
        self.pnl = self.pnl_points * CONTRACT_SIZE
        
        # Subtract commissions (both sides)
        self.pnl -= (COMMISSION_PER_SIDE * 2 * self.size)
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for storage"""
        return {
            'id': self.id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'strategy': self.strategy,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pnl': round(self.pnl, 2),
            'pnl_points': round(self.pnl_points, 2),
            'status': self.status,
            'exit_reason': self.exit_reason
        }


class PositionManager:
    """Manages all trading positions and risk limits"""
    
    def __init__(self, max_positions: int = MAX_OPEN_POSITIONS, 
                 max_daily_loss: float = MAX_DAILY_LOSS):
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        
        # Position tracking
        self.positions: Dict[int, Position] = {}
        self.position_counter = 0
        
        # Daily tracking
        self.current_date = datetime.now().date()
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        
        print(f"✅ Position Manager initialized: Max={max_positions}, Daily Loss Limit=${max_daily_loss}")
    
    def reset_daily_counters(self):
        """Reset daily counters if new trading day"""
        today = datetime.now().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_time = None
            print(f"📅 Daily counters reset for {today}")
    
    def can_open_position(self) -> tuple[bool, str]:
        """Check if we can open a new position"""
        self.reset_daily_counters()
        
        # Check max positions
        open_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            return False, f"Daily loss limit hit (${self.daily_pnl:.2f})"
        
        # Check max daily trades
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, f"Max daily trades reached ({MAX_DAILY_TRADES})"
        
        # Check time between trades
        if self.last_trade_time is not None:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < MIN_TIME_BETWEEN_TRADES:
                return False, f"Too soon since last trade ({int(time_since_last)}s)"
        
        return True, "OK"
    
    def open_position(self, signal: Dict, entry_price: float, size: int = 1) -> Optional[Position]:
        """
        Open a new position from a signal
        
        Args:
            signal: Signal dictionary from strategy or fusion engine
            entry_price: Actual entry price
            size: Number of contracts
            
        Returns:
            Position object if opened, None if rejected
        """
        can_open, reason = self.can_open_position()
        
        if not can_open:
            print(f"⛔ Position rejected: {reason}")
            return None
        
        # Create new position
        self.position_counter += 1
        position = Position(
            id=self.position_counter,
            strategy=signal.get('strategy', 'unknown'),
            direction=signal.get('direction', 'LONG'),
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=signal.get('stop'),
            take_profit=signal.get('target'),
            size=size
        )
        
        self.positions[position.id] = position
        self.daily_trades += 1
        self.last_trade_time = datetime.now()
        
        stop_str = f"{position.stop_loss:.2f}" if position.stop_loss else "None"
        target_str = f"{position.take_profit:.2f}" if position.take_profit else "None"
        
        print(f"🟢 POSITION OPENED #{position.id} | {position.direction} @ {entry_price:.2f}")
        print(f"   Stop: {stop_str} | Target: {target_str} | Strategy: {position.strategy}")
        
        return position
    
    def update_positions(self, current_price: float, current_time: datetime):
        """Update all open positions with current market price"""
        open_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        
        for position in open_positions:
            exit_reason = position.update(current_price, current_time)
            
            if exit_reason:
                self._close_position(position, current_price, current_time, exit_reason)
    
    def _close_position(self, position: Position, exit_price: float, 
                       exit_time: datetime, reason: str):
        """Close a position and record the trade"""
        position.close(exit_price, exit_time, reason)
        
        # Update daily P&L
        self.daily_pnl += position.pnl
        
        # Log the trade
        print(f"🔴 POSITION CLOSED #{position.id} | {position.direction}")
        print(f"   Entry: {position.entry_price:.2f} → Exit: {exit_price:.2f}")
        print(f"   P&L: ${position.pnl:.2f} ({position.pnl_points:+.2f} pts)")
        print(f"   Reason: {reason} | Strategy: {position.strategy}")
        
        # Store in data handler (will be imported in main.py to avoid circular import)
        return position.to_dict()
    
    def get_open_positions(self) -> List[Position]:
        """Get all currently open positions"""
        return [p for p in self.positions.values() if p.status == 'OPEN']
    
    def get_daily_stats(self) -> Dict:
        """Get statistics for current trading day"""
        self.reset_daily_counters()
        
        closed_today = [p for p in self.positions.values() 
                       if p.status == 'CLOSED' and p.exit_time.date() == self.current_date]
        
        if not closed_today:
            return {
                'trades': 0,
                'pnl': 0.0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        wins = [p for p in closed_today if p.pnl > 0]
        losses = [p for p in closed_today if p.pnl <= 0]
        
        return {
            'trades': len(closed_today),
            'pnl': sum(p.pnl for p in closed_today),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed_today) if closed_today else 0.0,
            'avg_win': sum(p.pnl for p in wins) / len(wins) if wins else 0.0,
            'avg_loss': sum(p.pnl for p in losses) / len(losses) if losses else 0.0,
            'largest_win': max([p.pnl for p in wins]) if wins else 0.0,
            'largest_loss': min([p.pnl for p in losses]) if losses else 0.0
        }
    
    def force_close_all(self, current_price: float, reason: str = "FORCED_CLOSE"):
        """Force close all open positions (for EOD, emergencies, etc.)"""
        open_positions = self.get_open_positions()
        
        if not open_positions:
            return []
        
        print(f"⚠️ Force closing {len(open_positions)} positions: {reason}")
        
        closed_trades = []
        for position in open_positions:
            trade_dict = self._close_position(position, current_price, datetime.now(), reason)
            closed_trades.append(trade_dict)
        
        return closed_trades


# Create global instance
position_manager = PositionManager()