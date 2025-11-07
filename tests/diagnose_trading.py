#!/usr/bin/env python3
"""
Diagnostic Script for NQ Trading Bot
Identifies why trades are not being taken
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datetime import datetime, time
import pytz

from utils.data_handler import data_handler
from strategies.opening_range import orb_strategy
from strategies.mean_reversion import mean_reversion_strategy
from utils.context_manager import context_manager
from position_manager import position_manager


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_data_availability():
    """Check if we have data flowing"""
    print_section("1. DATA AVAILABILITY CHECK")
    
    df = data_handler.get_latest_bars(200)
    
    if df.empty:
        print("❌ NO DATA AVAILABLE")
        print("   → Make sure main.py is running")
        print("   → Check if TradingView alert is active")
        print("   → Run: python test_webhook.py")
        return False
    else:
        print(f"✅ Data Available: {len(df)} bars")
        print(f"   Latest bar: {df.index[-1]}")
        print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
        print(f"   Current price: {df['close'].iloc[-1]:.2f}")
        return True


def check_market_context():
    """Check market context from ContextManager"""
    print_section("2. MARKET CONTEXT CHECK")
    
    context = context_manager.get_market_context()
    
    if not context or not context.get('last_updated'):
        print("⚠️  Context Manager not running or no data")
        print("   ES Trend: UNKNOWN")
        print("   VIX Status: UNKNOWN")
        print("   → This might block some trades due to filters")
        return context
    else:
        print(f"✅ Context Manager Active")
        print(f"   Last Updated: {context['last_updated']}")
        print(f"   ES Trend: {context.get('es_trend', 'UNKNOWN')}")
        print(f"   VIX Status: {context.get('vix_status', 'UNKNOWN')}")
        
        # Check if context is blocking trades
        if context.get('es_trend') in ['STRONG_DOWN', 'STRONG_UP']:
            print(f"   ⚠️  Strong trend detected - may block mean reversion trades")
        if context.get('vix_status') == 'SPIKING':
            print(f"   ⚠️  VIX spiking - ALL trades blocked!")
        
        return context


def check_orb_strategy():
    """Check ORB strategy status"""
    print_section("3. OPENING RANGE BREAKOUT (ORB) STRATEGY CHECK")
    
    print(f"Strategy Enabled: {orb_strategy.config.get('enabled', False)}")
    print(f"OR Period: {orb_strategy.or_period} minutes")
    print(f"Current Date: {orb_strategy.current_date}")
    print(f"Trade Taken Today: {orb_strategy.trade_taken_today}")
    
    if orb_strategy.or_high is not None:
        print(f"\n✅ Opening Range SET:")
        print(f"   OR High: {orb_strategy.or_high:.2f}")
        print(f"   OR Low: {orb_strategy.or_low:.2f}")
        print(f"   OR Size: {orb_strategy.or_size:.2f} points")
        
        # Check current price vs OR
        df = data_handler.get_latest_bars(1)
        if not df.empty:
            current_price = df['close'].iloc[-1]
            print(f"\n   Current Price: {current_price:.2f}")
            
            if current_price > orb_strategy.or_high:
                print(f"   📈 Price is ABOVE OR high (+{current_price - orb_strategy.or_high:.2f} pts)")
                print(f"   → Should generate LONG signal if not blocked")
            elif current_price < orb_strategy.or_low:
                print(f"   📉 Price is BELOW OR low (-{orb_strategy.or_low - current_price:.2f} pts)")
                print(f"   → Should generate SHORT signal if not blocked")
            else:
                print(f"   ⏸️  Price inside OR range (waiting for breakout)")
    else:
        print(f"\n⚠️  Opening Range NOT SET")
        
        # Check current time
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        current_time = now.time()
        
        or_end_minute = 30 + orb_strategy.or_period
        or_end = time(9, or_end_minute if or_end_minute < 60 else or_end_minute - 60)
        
        if current_time < time(9, 30):
            print(f"   → Too early: Market opens at 9:30 AM ET (currently {current_time})")
        elif current_time < or_end:
            print(f"   → Building OR: Period ends at {or_end} ET (currently {current_time})")
        else:
            print(f"   → OR period passed but not calculated")
            print(f"   → Check if data from 9:30-9:45 AM ET is available")
            
            # Try to get OR bars
            or_bars = data_handler.get_opening_range_bars(now, orb_strategy.or_period)
            if or_bars.empty:
                print(f"   ❌ No data for OR period found")
            else:
                print(f"   ✅ Found {len(or_bars)} bars for OR period")
                or_data = orb_strategy.calculate_opening_range(or_bars)
                if or_data is None:
                    print(f"   ❌ OR rejected by filters (range too small/large)")


def check_mean_reversion():
    """Check Mean Reversion strategy status"""
    print_section("4. MEAN REVERSION STRATEGY CHECK")
    
    print(f"Strategy Enabled: {mean_reversion_strategy.config.get('enabled', False)}")
    
    df = data_handler.get_latest_bars(50)
    if df.empty or len(df) < 20:
        print("❌ Insufficient data for mean reversion")
        return
    
    # Add indicators
    df = mean_reversion_strategy.add_indicators(df)
    current = df.iloc[-1]
    
    print(f"\nCurrent Market Conditions:")
    print(f"   Price: {current['close']:.2f}")
    print(f"   RSI: {current.get('rsi', 0):.1f}")
    print(f"   ADX: {current.get('adx', 0):.1f}")
    
    if 'bb_upper' in df.columns:
        print(f"   BB Upper: {current['bb_upper']:.2f}")
        print(f"   BB Middle: {current['bb_middle']:.2f}")
        print(f"   BB Lower: {current['bb_lower']:.2f}")
        
        # Check proximity to bands
        if current['close'] >= current['bb_upper']:
            print(f"   📈 Price at/above upper BB → Potential SHORT")
        elif current['close'] <= current['bb_lower']:
            print(f"   📉 Price at/below lower BB → Potential LONG")
        else:
            dist_to_upper = current['bb_upper'] - current['close']
            dist_to_lower = current['close'] - current['bb_lower']
            print(f"   ⏸️  Price between bands (↑{dist_to_upper:.2f} pts to upper, ↓{dist_to_lower:.2f} pts to lower)")
    
    # Check filters
    if current.get('adx', 0) > 25:
        print(f"\n   ⚠️  TRENDING market (ADX={current['adx']:.1f}) → Mean reversion disabled")
    else:
        print(f"\n   ✅ RANGING market (ADX={current['adx']:.1f}) → Mean reversion enabled")


def check_position_manager():
    """Check position manager status"""
    print_section("5. POSITION MANAGER CHECK")
    
    can_open, reason = position_manager.can_open_position()
    
    if can_open:
        print("✅ Position Manager: Ready to open positions")
    else:
        print(f"❌ Position Manager: Cannot open positions")
        print(f"   Reason: {reason}")
    
    open_positions = position_manager.get_open_positions()
    print(f"\nOpen Positions: {len(open_positions)}")
    
    daily_stats = position_manager.get_daily_stats()
    print(f"Today's Stats:")
    print(f"   Trades: {daily_stats['trades']}")
    print(f"   P&L: ${daily_stats['pnl']:.2f}")
    print(f"   Win Rate: {daily_stats['win_rate']*100:.1f}%")


def check_recent_signals():
    """Check if signals are being generated"""
    print_section("6. RECENT SIGNALS CHECK")
    
    signals = data_handler.get_all_signals()
    
    if signals.empty:
        print("❌ NO SIGNALS GENERATED")
        print("   → Strategies are running but not finding opportunities")
        print("   → Check strategy filters and market conditions above")
    else:
        today = datetime.now().date()
        signals['date'] = pd.to_datetime(signals['timestamp']).dt.date
        today_signals = signals[signals['date'] == today]
        
        print(f"✅ Total Signals Ever: {len(signals)}")
        print(f"   Today's Signals: {len(today_signals)}")
        
        if len(today_signals) > 0:
            print(f"\nLatest Signals Today:")
            for _, sig in today_signals.tail(5).iterrows():
                print(f"   {sig['timestamp']} | {sig['strategy'].upper()} | "
                      f"{sig['signal']} @ {sig['price']:.2f}")


def run_full_diagnostic():
    """Run complete diagnostic check"""
    print("\n" + "=" * 70)
    print("  NQ TRADING BOT - DIAGNOSTIC REPORT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Run all checks
    has_data = check_data_availability()
    
    if not has_data:
        print("\n" + "=" * 70)
        print("DIAGNOSIS: NO DATA - Bot not receiving price data")
        print("=" * 70)
        return
    
    context = check_market_context()
    check_orb_strategy()
    check_mean_reversion()
    check_position_manager()
    check_recent_signals()
    
    # Final diagnosis
    print_section("DIAGNOSIS & RECOMMENDATIONS")
    
    # Check if context is blocking
    if context and context.get('vix_status') == 'SPIKING':
        print("🔴 CRITICAL: VIX spiking - ALL trades blocked by context filter")
        print("   → Wait for volatility to normalize")
        print("   → OR temporarily disable VIX filter in context_manager.py")
    
    # Check if ORB is set up
    if orb_strategy.config.get('enabled'):
        if orb_strategy.or_high is None:
            et_tz = pytz.timezone('US/Eastern')
            now = datetime.now(et_tz)
            if now.time() > time(9, 45):
                print("⚠️  WARNING: ORB enabled but OR not set (after 9:45 AM ET)")
                print("   → Check if data from 9:30-9:45 AM ET was received")
                print("   → OR may have been rejected (range too small/large)")
    
    # Check error logs
    print("\n📋 Check error logs for more details:")
    print("   tail -f logs/strategy.log")
    print("   tail -f logs/system.log")
    
    print("\n" + "=" * 70)
    print("Diagnostic complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        run_full_diagnostic()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled.")
    except Exception as e:
        print(f"\n❌ Error running diagnostic: {e}")
        import traceback
        traceback.print_exc()
