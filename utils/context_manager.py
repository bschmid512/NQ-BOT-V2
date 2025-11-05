"""
Market Context Manager
Runs in a background thread to fetch and analyze context data (ES, VIX)
to provide a "why" for the strategy engine.

** VERSION 1.1 - LESS STRICT FILTERS **
"""
import yfinance as yf
import pandas as pd
import threading
import time
from typing import Dict, Any
from utils.logger import trading_logger # Assuming your logger is here

class ContextManager:
    
    def __init__(self, update_interval: int = 30):
        """
        Initializes the ContextManager.
        
        Args:
            update_interval (int): How often (in seconds) to fetch new data.
        """
        self.logger = trading_logger.system_logger
        self.tickers = {
            'es': yf.Ticker("ES=F"),  # S&P 500 Futures
            'vix': yf.Ticker("^VIX")   # VIX Index
        }
        self.update_interval = update_interval
        
        # This state is shared and updated by the background thread
        self.market_state = {
            'es_trend': 'NEUTRAL',
            'vix_status': 'CALM',
            'last_updated': None
        }
        
        # Thread-safe access
        self.lock = threading.Lock()
        
        self.running = False
        self.thread = None
        self.logger.info("ContextManager initialized (v1.1 - Less Strict).")

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetches the latest 1-day data for context tickers"""
        data = {}
        try:
            # We fetch 2 days of 1m data to ensure we have recent price action
            # yfinance is often delayed, so we get 1m data as a proxy
            data['es'] = self.tickers['es'].history(period='2d', interval='1m')
            data['vix'] = self.tickers['vix'].history(period='2d', interval='1m')
        except Exception as e:
            self.logger.warning(f"[ContextManager] yfinance fetch failed: {e}")
        return data

    def analyze_context(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyzes the fetched data and updates the market_state.
        This is where you define the "why".
        """
        new_state = {}
        
        try:
            # 1. Analyze ES (S&P 500) Trend
            if 'es' in data and not data['es'].empty:
                es_data = data['es']['Close']
                # Simple 10-period vs 30-period moving average
                ema_fast = es_data.ewm(span=10).mean().iloc[-1]
                ema_slow = es_data.ewm(span=30).mean().iloc[-1]
                
                # --- MODIFIED LOGIC ---
                # We will be in "CHOP" more often, allowing more signals
                buffer = 0.0010  # Was 0.0005. We now need a 0.1% move to be "trending"
                
                if ema_fast > ema_slow * (1 + buffer):
                    new_state['es_trend'] = 'STRONG_UP'
                elif ema_fast < ema_slow * (1 - buffer):
                    new_state['es_trend'] = 'STRONG_DOWN'
                else:
                    new_state['es_trend'] = 'CHOP'
            
            # 2. Analyze VIX (Volatility)
            if 'vix' in data and not data['vix'].empty:
                vix_data = data['vix']['Close']
                vix_current = vix_data.iloc[-1]
                vix_avg = vix_data.rolling(30).mean().iloc[-1] # 30-min avg
                
                # --- MODIFIED LOGIC ---
                # VIX needs to spike 15% now (was 10%)
                if vix_current > vix_avg * 1.15: # Was 1.10
                    new_state['vix_status'] = 'SPIKING'
                elif vix_current < 15: # Arbitrary "calm" level
                    new_state['vix_status'] = 'CALM'
                else:
                    new_state['vix_status'] = 'ELEVATED'

            new_state['last_updated'] = pd.Timestamp.now()
            return new_state

        except Exception as e:
            self.logger.error(f"[ContextManager] Context analysis failed: {e}", exc_info=True)
            return {}

    def run_loop(self):
        """The main loop for the background thread."""
        self.running = True
        self.logger.info("ContextManager thread started.")
        while self.running:
            try:
                # 1. Fetch
                raw_data = self.fetch_data()
                
                # 2. Analyze
                if raw_data:
                    new_state = self.analyze_context(raw_data)
                    
                    # 3. Update state (thread-safe)
                    if new_state:
                        with self.lock:
                            self.market_state.update(new_state)
                        self.logger.debug(f"[ContextManager] Context updated: {self.market_state}")
                
                # 4. Sleep
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"[ContextManager] ContextManager loop error: {e}", exc_info=True)
                time.sleep(60) # Longer sleep on error

    def start(self):
        """Starts the background thread."""
        if not self.running:
            self.thread = threading.Thread(target=self.run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the background thread."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info("ContextManager thread stopped.")

    def get_market_context(self) -> Dict[str, Any]:
        """Public method for the StrategyEngine to get the latest state."""
        with self.lock:
            return self.market_state.copy()

# Create a single global instance
context_manager = ContextManager(update_interval=30)