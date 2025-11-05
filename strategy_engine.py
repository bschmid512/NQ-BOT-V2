"""
Strategy Execution Engine for NQ Trading Bot - FIXED VERSION
Critical timezone bug resolved - trades will now execute!
"""
import pandas as pd
from datetime import datetime, time
import pytz
from typing import List, Dict, Optional
from utils.logger import trading_logger
from utils.data_handler import data_handler
from strategies.opening_range import orb_strategy
from strategies.mean_reversion import mean_reversion_strategy
from strategies.trend_following import trend_following_strategy  # ← ADD THIS
from strategies.breakout import breakout_strategy              # ← ADD THIS
class StrategyEngine:
    """
    Main strategy execution engine
    Runs all enabled strategies and aggregates signals
    """
    
    def __init__(self):
        self.logger = trading_logger.strategy_logger
        self.strategies = {
            'orb': orb_strategy,
            'mean_reversion': mean_reversion_strategy,
            'trend_following': trend_following_strategy,  # ← ADD THIS
            'breakout': breakout_strategy                 # ← ADD THIS
        }
        
        # High-impact economic events (update monthly)
        self.high_impact_dates = [
            datetime(2024, 11, 7),   # FOMC
            datetime(2024, 11, 13),  # CPI
            datetime(2024, 11, 15),  # Retail Sales
            datetime(2024, 12, 6),   # NFP
            datetime(2024, 12, 11),  # CPI
            datetime(2024, 12, 18),  # FOMC
        ]
        
        self.logger.info("Strategy Engine initialized")
        self.logger.info(f"Active strategies: {list(self.strategies.keys())}")
    
    def should_trade_now(self, timestamp: datetime, context: Dict) -> tuple[bool, str]:
        """
        Determine if we should trade based on time/session/events/context
        
        Returns:
            (bool, str): (should_trade, reason_if_not)
        """
        et_tz = pytz.timezone('US/Eastern')
        
        # ⭐ FIX: Proper time handling - this was the critical bug!
        if timestamp.tzinfo is None:
            et_time = et_tz.localize(timestamp)
        else:
            et_time = timestamp.astimezone(et_tz)
        
        current_hour = et_time.hour
        current_date = et_time.date()  # ← THIS LINE IS MISSING!
        # Check for high-impact economic events
        for event_date in self.high_impact_dates:
            if event_date.date() == current_date:
                return False, f"High-impact economic event: {event_date.date()}"
        
        # Session filters - MORE PERMISSIVE NOW
        # Avoid Asian session (6pm-2am ET) - low volume, wide spreads
        if 18 <= current_hour or current_hour < 2:
            return False, "Asian session - low volume"
        
        # Avoid early European session (2am-6am ET) - very thin
        if 2 <= current_hour < 6:
            return False, "Early European session - too thin"
            
        # ⭐ RELAXED CONTEXT CHECK - Only block on extreme VIX
        #if context.get('vix_status') == 'SPIKING':
         #   self.logger.warning(f"VIX is spiking but allowing CHOP trades")
            # We'll still allow trading, just log the warning
        
        # Allow late European + US pre-market (6am-9:30am ET) - ORB setup period
        if 6 <= current_hour < 9:
            return True, "European/Pre-market session"
        
        # Allow US regular session (9:30am-4pm ET) - best trading
        if 9 <= current_hour < 16:
            return True, "US regular session"
        
        # Allow limited after-hours (4pm-6pm ET) - earnings reactions
        if 16 <= current_hour < 18:
            return True, "After-hours session"
        
        return False, "Outside trading hours"
    
    def process_new_bar(self, bar_data: Dict, context: Dict) -> List[Dict]:
        """
        Process a new price bar through all strategies
        
        Args:
            bar_data: Latest OHLCV bar
            context: Latest market context (from ContextManager)
            
        Returns:
            List of generated signals
        """
        signals = []
        
        try:
            # Check if we should trade
            timestamp = pd.to_datetime(bar_data['timestamp'])
            
            # Pass context to should_trade_now
            should_trade, reason = self.should_trade_now(timestamp, context)
            
            if not should_trade:
                self.logger.debug(f"Not trading: {reason}")
                return signals
            
            # Get recent bars for strategy analysis
            df = data_handler.get_latest_bars(200)  # Get more bars
            
            self.logger.info(f"Processing bar at {timestamp} | Close: {bar_data['close']:.2f} | Bars: {len(df)}")
            
            # Run each enabled strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    if not strategy.config.get('enabled', False):
                        continue
                    
                    self.logger.debug(f"Running {strategy_name} strategy...")
                    
                    # Special handling for ORB strategy
                    if strategy_name == 'orb':
                        signal = self._process_orb_strategy(strategy, df, timestamp, context)
                    
                    # Standard processing for other strategies
                    else:
                        signal = strategy.generate_signal(df, context)
                    
                    if signal:
                        signals.append(signal)
                        self.logger.info(f"✅ {strategy_name.upper()} generated signal: {signal['signal']} @ {signal['price']:.2f}")
                        
                        # Store signal
                        data_handler.add_signal(signal)
                    
                except Exception as e:
                    self.logger.error(f"Error running {strategy_name} strategy: {e}", exc_info=True)
            
            if not signals:
                self.logger.debug("No signals generated from any strategy")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in strategy engine: {e}", exc_info=True)
            return signals
    
    def _process_orb_strategy(self, strategy, df: pd.DataFrame, timestamp: datetime, context: Dict) -> Optional[Dict]:
        """
        Special processing for Opening Range Breakout strategy
        Handles daily state reset and OR calculation
        
        ⭐⭐⭐ FIXED: Timezone handling bug resolved ⭐⭐⭐
        """
        # Reset daily state if new day
        strategy.reset_daily_state(timestamp.date())
        
        et_tz = pytz.timezone('US/Eastern')
        
        # ⭐ FIXED: Properly handle timezone conversion
        if timestamp.tzinfo is None:
            et_time = et_tz.localize(timestamp)
        else:
            et_time = timestamp.astimezone(et_tz)
        
        current_time = et_time.time()  # Now et_time is a datetime, so .time() works
        
        # Check if we're in the opening range period (9:30-9:45 AM ET)
        or_start = time(9, 30)
        or_end = time(9, 30 + strategy.or_period)
        
        if or_start <= current_time < or_end:
            # We're still building the opening range
            self.logger.debug(f"Building opening range: {current_time}")
            return None
        
        # After OR period, check if we have OR data
        if strategy.or_high is None:
            # Calculate OR from morning bars
            or_bars = data_handler.get_opening_range_bars(timestamp, strategy.or_period)
            
            if not or_bars.empty:
                or_data = strategy.calculate_opening_range(or_bars)
                
                if or_data:
                    strategy.or_high = or_data['high']
                    strategy.or_low = or_data['low']
                    strategy.or_size = or_data['size']
                    
                    self.logger.info(
                        f"Opening Range set: High={or_data['high']:.2f}, "
                        f"Low={or_data['low']:.2f}, Size={or_data['size']:.2f}"
                    )
                else:
                    self.logger.debug("Opening range rejected by filters")
                    return None
            else:
                self.logger.debug("No opening range data available yet")
                return None
        
        # Generate signal based on OR breakout
        current_bar = df.iloc[-1]
        
        # Pass context to the strategy.
        return strategy.generate_signal(current_bar, or_data=None, context=context)
    
    def get_active_signals(self) -> pd.DataFrame:
        """Get all signals from current session"""
        signals = data_handler.get_all_signals()
        
        if signals.empty:
            return signals
        
        # Filter to today's signals
        today = datetime.now().date()
        signals['date'] = pd.to_datetime(signals['timestamp']).dt.date
        today_signals = signals[signals['date'] == today]
        
        return today_signals.sort_values('timestamp', ascending=False)
    
    def get_strategy_performance(self, strategy_name: str) -> Dict:
        """Get performance metrics for a specific strategy"""
        signals = data_handler.get_all_signals()
        
        if signals.empty:
            return {'signals': 0, 'win_rate': 0, 'avg_confidence': 0}
        
        strategy_signals = signals[signals['strategy'] == strategy_name]
        
        return {
            'total_signals': len(strategy_signals),
            'long_signals': len(strategy_signals[strategy_signals['signal'] == 'LONG']),
            'short_signals': len(strategy_signals[strategy_signals['signal'] == 'SHORT']),
            'avg_confidence': strategy_signals['confidence'].mean() if len(strategy_signals) > 0 else 0
        }


# Create global strategy engine instance
strategy_engine = StrategyEngine()