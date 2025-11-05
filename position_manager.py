"""
Position Manager for NQ Trading Bot
Tracks open positions, manages exits, calculates P&L (paper trading)
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from utils.logger import trading_logger
from utils.data_handler import data_handler


class Position:
    """Represents a single trading position"""
    
    def __init__(self, signal: Dict, entry_price: float, position_id: int):
        self.id = position_id
        self.strategy = signal['strategy']
        self.direction = signal['signal']  # LONG or SHORT
        self.entry_price = entry_price
        self.entry_time = signal['timestamp']
        self.size = 1  # Number of contracts
        self.stop_loss = signal['stop']
        self.take_profit = signal['target']
        self.confidence = signal['confidence']
        
        # Status tracking
        self.status = 'OPEN'
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_points = 0.0
        
        # Max favorable/adverse excursion
        self.mfe = 0.0  # Max Favorable Excursion
        self.mae = 0.0  # Max Adverse Excursion
    
    def update(self, current_price: float, current_time: datetime) -> Optional[str]:
        """
        Update position and check for exit conditions
        
        Returns:
            Exit reason if position should close, None otherwise
        """
        if self.status != 'OPEN':
            return None
        

        if self.direction == 'LONG':
                    points = current_price - self.entry_price
                    
                    # Check stop loss
                    if self.stop_loss is not None and current_price <= self.stop_loss:
                        return 'STOP_LOSS'
                    
                    # Check take profit
                    if self.take_profit is not None and current_price >= self.take_profit:
                        return 'TAKE_PROFIT'
                    
        else:  # SHORT
                    points = self.entry_price - current_price
                    
                    # Check stop loss
                    if self.stop_loss is not None and current_price >= self.stop_loss:
                        return 'STOP_LOSS'
                    
                    # Check take profit
                    if self.take_profit is not None and current_price <= self.take_profit:
                        return 'TAKE_PROFIT'
        
        # Update MFE/MAE
        if points > self.mfe:
            self.mfe = points
        if points < self.mae:
            self.mae = points
        
        # Time-based exit (max 2 hours in trade)
        if (current_time - self.entry_time).total_seconds() > 7200:
            return 'TIME_EXIT'
        
        return None
    
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the position and calculate final P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = 'CLOSED'
        
        # Calculate P&L
        if self.direction == 'LONG':
            self.pnl_points = exit_price - self.entry_price
        else:  # SHORT
            self.pnl_points = self.entry_price - exit_price
        
        # Convert to dollars ($20 per point × points × size)
        self.pnl = self.pnl_points * 20 * self.size
        
        # Account for commission ($5 round trip per contract)
        commission = 5.00 * self.size
        self.pnl -= commission
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for storage"""
        return {
            'timestamp': self.entry_time,
            'ticker': 'NQ',
            'action': 'BUY' if self.direction == 'LONG' else 'SELL',
            'price': self.entry_price,
            'size': self.size,
            'signal': self.strategy,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pnl': self.pnl,
            'status': self.status,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'exit_reason': self.exit_reason,
            'pnl_points': self.pnl_points,
            'confidence': self.confidence,
            'mfe': self.mfe,
            'mae': self.mae
        }


class PositionManager:
    """
    Manages all trading positions for paper trading
    Tracks P&L, enforces risk limits, handles exits
    """
    
    def __init__(self, max_positions: int = 1, max_daily_loss: float = -1000):
        self.logger = trading_logger.trade_logger
        self.positions: Dict[int, Position] = {}
        self.position_counter = 0
        
        # Risk parameters
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = datetime.now().date()
        
        self.logger.info(f"Position Manager initialized: Max positions={max_positions}, Max daily loss=${max_daily_loss}")
    
    def reset_daily_counters(self):
        """Reset daily P&L and trade counters"""
        today = datetime.now().date()
        if today != self.current_date:
            self.logger.info(f"Daily reset: Previous P&L=${self.daily_pnl:.2f}, Trades={self.daily_trades}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = today
    
    def can_open_position(self) -> tuple[bool, str]:
        """
        Check if we can open a new position
        
        Returns:
            (bool, str): (can_open, reason_if_not)
        """
        self.reset_daily_counters()
        
        # Check max positions limit
        open_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached: {self.max_positions}"
        
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            return False, f"Daily loss limit hit: ${self.daily_pnl:.2f}"
        
        # Check if we've had too many consecutive losses (circuit breaker)
        recent_trades = self._get_recent_closed_positions(5)
        if len(recent_trades) >= 5:
            losing_streak = all(p.pnl < 0 for p in recent_trades)
            if losing_streak:
                return False, "5 consecutive losses - circuit breaker activated"
        
        return True, "OK"
    
    def open_position(self, signal: Dict, entry_price: float) -> Optional[Position]:
        """
        Open a new position from a signal
        
        Args:
            signal: Signal dictionary from strategy
            entry_price: Actual entry price (current market price)
            
        Returns:
            Position object if opened, None if rejected
        """
        can_open, reason = self.can_open_position()
        
        if not can_open:
            self.logger.warning(f"Position rejected: {reason}")
            return None
        
        # Create new position
        self.position_counter += 1
        position = Position(signal, entry_price, self.position_counter)
        self.positions[position.id] = position
        
        self.daily_trades += 1
        
        stop_str = f"{position.stop_loss:.2f}" if position.stop_loss is not None else "None"
        target_str = f"{position.take_profit:.2f}" if position.take_profit is not None else "None"
        
        self.logger.info(
            f"🔵 POSITION OPENED #{position.id} | {position.direction} @ {entry_price:.2f} | "
            f"Stop: {stop_str} | Target: {target_str} | "
            f"Strategy: {position.strategy}"
        )
        
        return position
    
    def update_positions(self, current_price: float, current_time: datetime):
        """
        Update all open positions with current market price
        Check for exit conditions
        """
        open_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        
        for position in open_positions:
            exit_reason = position.update(current_price, current_time)
            
            if exit_reason:
                self._close_position(position, current_price, current_time, exit_reason)
    
    def _close_position(self, position: Position, exit_price: float, exit_time: datetime, reason: str):
        """Close a position and record the trade"""
        position.close(exit_price, exit_time, reason)
        
        # Update daily P&L
        self.daily_pnl += position.pnl
        
        # Log the trade
        self.logger.info(
            f"🔴 POSITION CLOSED #{position.id} | {position.direction} | "
            f"Entry: {position.entry_price:.2f} → Exit: {exit_price:.2f} | "
            f"P&L: ${position.pnl:.2f} ({position.pnl_points:+.2f} pts) | "
            f"Reason: {reason} | Strategy: {position.strategy}"
        )
        
        # Store trade in database
        data_handler.add_trade(position.to_dict())
        
        # Check if we hit daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            self.logger.warning(f"⚠️  DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}")
    
    def _get_recent_closed_positions(self, n: int) -> List[Position]:
        """Get N most recent closed positions"""
        closed = [p for p in self.positions.values() if p.status == 'CLOSED']
        closed.sort(key=lambda x: x.exit_time, reverse=True)
        return closed[:n]
    
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
                'win_rate': 0.0
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
            return
        
        self.logger.warning(f"Force closing {len(open_positions)} positions: {reason}")
        
        for position in open_positions:
            self._close_position(position, current_price, datetime.now(), reason)


# Create global position manager instance
position_manager = PositionManager(max_positions=1, max_daily_loss=-1000)
