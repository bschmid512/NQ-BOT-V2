#!/usr/bin/env python3
"""
NQ Trading Bot - Fix Validation Script
Tests all critical components after fixes have been applied
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def test_imports():
    """Test all critical imports"""
    print_header("TESTING IMPORTS")
    
    tests = []
    
    # Test config
    try:
        from config import (
            FUSION_CONFIG, STRATEGIES, MAX_POSITION_SIZE,
            MAX_DAILY_LOSS, COMMISSION_PER_SIDE, CONTRACT_SIZE
        )
        print_success("config.py imports")
        tests.append(True)
    except Exception as e:
        print_error(f"config.py imports: {e}")
        tests.append(False)
    
    # Test data handler
    try:
        from utils.data_handler import data_handler
        print_success("data_handler imports")
        tests.append(True)
    except Exception as e:
        print_error(f"data_handler imports: {e}")
        tests.append(False)
    
    # Test position manager
    try:
        from core.position_manager import position_manager
        print_success("position_manager imports")
        tests.append(True)
    except Exception as e:
        print_error(f"position_manager imports: {e}")
        tests.append(False)
    
    # Test fusion engine
    try:
        from core.signal_fusion_engine import signal_fusion_engine
        print_success("signal_fusion_engine imports")
        tests.append(True)
    except Exception as e:
        print_error(f"signal_fusion_engine imports: {e}")
        tests.append(False)
    
    # Test enhanced strategy engine
    try:
        from core.enhanced_strategy_engine import enhanced_strategy_engine
        print_success("enhanced_strategy_engine imports")
        tests.append(True)
    except Exception as e:
        print_error(f"enhanced_strategy_engine imports: {e}")
        tests.append(False)
    
    # Test market context fusion
    try:
        from core.market_context_fusion import market_context_fusion
        print_success("market_context_fusion imports")
        tests.append(True)
    except Exception as e:
        print_error(f"market_context_fusion imports: {e}")
        tests.append(False)
    
    return all(tests)

def test_position_manager_fix():
    """Test that position manager uses config values"""
    print_header("TESTING POSITION MANAGER FIX")
    
    try:
        from config import COMMISSION_PER_SIDE, CONTRACT_SIZE
        from core.position_manager import Position
        
        # Create a test position
        test_position = Position(
            id=1,
            strategy='test',
            direction='LONG',
            entry_price=16500.0,
            entry_time=datetime.now(),
            stop_loss=16450.0,
            take_profit=16600.0,
            size=2
        )
        
        # Close it
        test_position.close(16550.0, datetime.now(), 'TEST')
        
        # Calculate expected P&L
        points = (16550.0 - 16500.0) * 2  # 100 points
        expected_pnl = points * CONTRACT_SIZE - (COMMISSION_PER_SIDE * 2 * 2)
        
        # Check if commission calculation is correct
        if abs(test_position.pnl - expected_pnl) < 0.01:
            print_success(f"Position P&L calculation correct: ${test_position.pnl:.2f}")
            print(f"   Points: {points}, Commission: ${COMMISSION_PER_SIDE * 2 * 2:.2f}")
            return True
        else:
            print_error(f"Position P&L incorrect!")
            print(f"   Expected: ${expected_pnl:.2f}, Got: ${test_position.pnl:.2f}")
            return False
            
    except Exception as e:
        print_error(f"Position manager test failed: {e}")
        return False

def test_fusion_config():
    """Test fusion configuration"""
    print_header("TESTING FUSION CONFIGURATION")
    
    try:
        from config import FUSION_CONFIG, STRATEGIES
        
        print(f"Fusion Config:")
        for key, value in FUSION_CONFIG.items():
            print(f"   {key}: {value}")
        
        print(f"\nActive Strategies:")
        for name, config in STRATEGIES.items():
            if config.get('enabled', False):
                print(f"   ✅ {name}: weight={config.get('weight', 0)}, conf={config.get('min_confidence', 0)}")
            else:
                print(f"   ⭕ {name}: disabled")
        
        # Check critical values
        tests = []
        
        if FUSION_CONFIG.get('min_signals_required', 0) >= 2:
            print_success(f"min_signals_required: {FUSION_CONFIG['min_signals_required']}")
            tests.append(True)
        else:
            print_error("min_signals_required should be >= 2")
            tests.append(False)
        
        if FUSION_CONFIG.get('min_total_weight', 0) >= 50:
            print_success(f"min_total_weight: {FUSION_CONFIG['min_total_weight']}")
            tests.append(True)
        else:
            print_warning(f"min_total_weight is low: {FUSION_CONFIG.get('min_total_weight', 0)}")
            tests.append(True)  # Warning, not error
        
        return all(tests)
        
    except Exception as e:
        print_error(f"Fusion config test failed: {e}")
        return False

def test_data_handler():
    """Test data handler basic functionality"""
    print_header("TESTING DATA HANDLER")
    
    try:
        from utils.data_handler import data_handler
        
        # Test adding a bar
        test_bar = {
            'timestamp': datetime.now().isoformat(),
            'open': 16500.0,
            'high': 16510.0,
            'low': 16490.0,
            'close': 16505.0,
            'volume': 1000.0
        }
        
        data_handler.add_bar(test_bar)
        print_success("Can add bars to data handler")
        
        # Test retrieving bars
        df = data_handler.get_latest_bars(10)
        if df is not None:
            print_success(f"Can retrieve bars: {len(df)} bars available")
        else:
            print_warning("No bars available yet (expected if fresh install)")
        
        return True
        
    except Exception as e:
        print_error(f"Data handler test failed: {e}")
        return False

def test_fusion_engine():
    """Test fusion engine basic functionality"""
    print_header("TESTING FUSION ENGINE")
    
    try:
        from core.signal_fusion_engine import signal_fusion_engine
        
        # Create mock signals
        signals = [
            {
                'strategy': 'momentum',
                'direction': 'LONG',
                'price': 16500.0,
                'stop': 16450.0,
                'target': 16600.0,
                'weight': 45,
                'confidence': 0.80
            },
            {
                'strategy': 'pullback',
                'direction': 'LONG',
                'price': 16500.0,
                'stop': 16460.0,
                'target': 16580.0,
                'weight': 35,
                'confidence': 0.75
            }
        ]
        
        # Mock context
        context = {
            'current_price': 16500.0,
            'market_regime': 'trending',
            'vision_sentiment': 'bullish',
            'vision_available': True
        }
        
        # Test fusion
        result = signal_fusion_engine.evaluate_trade_setup(signals, context)
        
        if result:
            print_success("Fusion engine approved the setup")
            print(f"   Direction: {result.get('signal', result.get('direction'))}")
            print(f"   Weight: {result.get('weight', 0)}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Size: {result.get('size', 1)} contracts")
        else:
            print_warning("Fusion engine rejected the setup (may be expected)")
        
        return True
        
    except Exception as e:
        print_error(f"Fusion engine test failed: {e}")
        return False

def test_strategies():
    """Test strategy imports"""
    print_header("TESTING STRATEGIES")
    
    tests = []
    
    strategies_to_test = [
        ('strategies.opening_range', 'orb_strategy'),
        ('strategies.momentum_continuation_strategy', 'MomentumContinuationStrategy'),
        ('strategies.pullback_entry_strategy', 'PullbackEntryStrategy'),
        ('strategies.mean_reversion', 'mean_reversion_strategy'),
    ]
    
    for module_path, object_name in strategies_to_test:
        try:
            module = __import__(module_path, fromlist=[object_name])
            obj = getattr(module, object_name, None)
            if obj:
                print_success(f"{object_name}")
                tests.append(True)
            else:
                print_warning(f"{object_name} not found in module")
                tests.append(False)
        except Exception as e:
            print_error(f"{object_name}: {e}")
            tests.append(False)
    
    return all(tests)

def run_all_tests():
    """Run all validation tests"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{'NQ TRADING BOT - FIX VALIDATION'.center(70)}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"\n{YELLOW}Running comprehensive tests...{RESET}\n")
    
    results = {
        'Imports': test_imports(),
        'Position Manager Fix': test_position_manager_fix(),
        'Fusion Configuration': test_fusion_config(),
        'Data Handler': test_data_handler(),
        'Fusion Engine': test_fusion_engine(),
        'Strategies': test_strategies()
    }
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        if passed_test:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{BLUE}{'─'*70}{RESET}")
    if passed == total:
        print(f"{GREEN}✅ ALL TESTS PASSED ({passed}/{total}){RESET}")
        print(f"\n{GREEN}Your bot is ready to run!{RESET}")
        print(f"\nNext steps:")
        print(f"1. Start the bot: {YELLOW}python main.py{RESET}")
        print(f"2. Test webhook: {YELLOW}python test_webhook.py{RESET}")
        print(f"3. Monitor logs: {YELLOW}tail -f logs/system.log{RESET}")
    else:
        print(f"{YELLOW}⚠️  {passed}/{total} TESTS PASSED{RESET}")
        print(f"\n{YELLOW}Some tests failed. Review the errors above.{RESET}")
        print(f"You may need to:")
        print(f"1. Check that all files are in the correct directories")
        print(f"2. Install missing dependencies: pip install -r requirements.txt")
        print(f"3. Review the fix documentation")
    
    print(f"{BLUE}{'─'*70}{RESET}\n")
    
    return passed == total

if __name__ == '__main__':
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{RED}❌ Tests cancelled by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{RED}❌ Error during testing: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
