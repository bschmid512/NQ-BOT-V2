#!/usr/bin/env python3
"""
Test Script for NQ Trading Bot
Simulates TradingView webhook data for testing
"""
import requests
import json
import time
from datetime import datetime, timedelta
import random
import sys


class WebhookTester:
    """Test webhook endpoint with simulated data"""
    
    def __init__(self, base_url='http://localhost:8050', passphrase='change_this_secure_passphrase'):
        self.base_url = base_url
        self.passphrase = passphrase
        self.session = requests.Session()
        
    def test_health(self):
        """Test health check endpoint"""
        print("Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"✅ Health Check: {response.status_code}")
            print(f"   Response: {response.json()}")
            return True
        except Exception as e:
            print(f"❌ Health Check Failed: {e}")
            return False
    
    def send_bar(self, bar_data):
        """Send a single bar to webhook"""
        try:
            payload = {
                'passphrase': self.passphrase,
                **bar_data
            }
            
            response = self.session.post(
                f"{self.base_url}/webhook",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            return response.status_code, response.json()
            
        except Exception as e:
            print(f"Error sending bar: {e}")
            return None, None
    
    def generate_realistic_bar(self, prev_close=16500, volatility=10):
        """Generate realistic OHLCV bar"""
        # Random walk with mean reversion
        change = random.gauss(0, volatility)
        open_price = prev_close + change
        
        # Generate high/low with realistic spread
        range_size = abs(random.gauss(0, volatility * 0.5))
        high = open_price + range_size
        low = open_price - range_size
        
        # Close somewhere in the range
        close = random.uniform(low, high)
        
        # Volume with some randomness
        volume = random.randint(500, 2000)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        }
    
    def simulate_market_data(self, num_bars=50, delay=1.0):
        """Simulate live market data stream"""
        print(f"\n📊 Simulating {num_bars} bars with {delay}s delay...")
        print("=" * 60)
        
        current_price = 16500
        success_count = 0
        
        for i in range(num_bars):
            # Generate bar
            bar = self.generate_realistic_bar(prev_close=current_price)
            current_price = bar['close']
            
            # Send to webhook
            status, response = self.send_bar(bar)
            
            if status == 200:
                success_count += 1
                print(f"✅ Bar {i+1}/{num_bars} | "
                      f"O:{bar['open']:.2f} H:{bar['high']:.2f} "
                      f"L:{bar['low']:.2f} C:{bar['close']:.2f} "
                      f"V:{bar['volume']}")
            else:
                print(f"❌ Bar {i+1}/{num_bars} | Failed: {response}")
            
            # Delay before next bar
            time.sleep(delay)
        
        print("\n" + "=" * 60)
        print(f"Simulation complete: {success_count}/{num_bars} bars sent successfully")
        return success_count == num_bars
    
    def test_opening_range_scenario(self):
        """Simulate an opening range breakout scenario"""
        print("\n🎯 Testing Opening Range Breakout Scenario...")
        print("=" * 60)
        
        base_price = 16500
        
        # 1. Create 15-minute opening range (9:30-9:45)
        print("Simulating opening range (15 bars)...")
        or_high = base_price + 20
        or_low = base_price - 20
        
        for i in range(15):
            bar = {
                'timestamp': datetime.now().isoformat(),
                'open': random.uniform(or_low, or_high),
                'high': or_high + random.uniform(-5, 0),
                'low': or_low + random.uniform(0, 5),
                'close': random.uniform(or_low, or_high),
                'volume': random.randint(800, 1200)
            }
            status, _ = self.send_bar(bar)
            print(f"  Bar {i+1}/15: {status}")
            time.sleep(0.5)
        
        # 2. Breakout above OR high
        print(f"\nSimulating breakout above OR high ({or_high:.2f})...")
        for i in range(3):
            bar = {
                'timestamp': datetime.now().isoformat(),
                'open': or_high + (i * 5),
                'high': or_high + (i * 5) + 10,
                'low': or_high + (i * 5) - 3,
                'close': or_high + (i * 5) + 7,
                'volume': random.randint(1500, 2500)
            }
            status, _ = self.send_bar(bar)
            print(f"  Breakout bar {i+1}: C={bar['close']:.2f}")
            time.sleep(0.5)
        
        print("\n✅ ORB scenario complete")
        print("Check dashboard for signal generation!")
    
    def test_invalid_requests(self):
        """Test error handling"""
        print("\n🧪 Testing Error Handling...")
        print("=" * 60)
        
        # Test wrong passphrase
        print("1. Testing wrong passphrase...")
        payload = {
            'passphrase': 'wrong_passphrase',
            'close': 16500,
            'volume': 1000
        }
        response = self.session.post(f"{self.base_url}/webhook", json=payload)
        print(f"   Expected 401: Got {response.status_code} ✅")
        
        # Test missing data
        print("2. Testing missing data...")
        payload = {
            'passphrase': self.passphrase
        }
        response = self.session.post(f"{self.base_url}/webhook", json=payload)
        print(f"   Expected 400: Got {response.status_code} ✅")
        
        # Test invalid content type
        print("3. Testing invalid content type...")
        response = self.session.post(
            f"{self.base_url}/webhook",
            data="not json",
            headers={'Content-Type': 'text/plain'}
        )
        print(f"   Expected 400: Got {response.status_code} ✅")
        
        print("\n✅ Error handling tests complete")


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("  NQ TRADING BOT - WEBHOOK TESTER")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = 'http://localhost:8050'
    
    if len(sys.argv) > 2:
        passphrase = sys.argv[2]
    else:
        passphrase = 'change_this_secure_passphrase'
    
    print(f"\nTarget URL: {url}")
    print(f"Passphrase: {passphrase}\n")
    
    # Initialize tester
    tester = WebhookTester(base_url=url, passphrase=passphrase)
    
    # Run tests
    print("\n🔍 Running Tests...\n")
    
    # 1. Health check
    if not tester.test_health():
        print("\n❌ Server not responding. Make sure the bot is running:")
        print("   python main.py")
        sys.exit(1)
    
    time.sleep(1)
    
    # 2. Error handling tests
    tester.test_invalid_requests()
    time.sleep(1)
    
    # 3. Ask user which test to run
    print("\n" + "=" * 60)
    print("Select test mode:")
    print("1. Simulate realistic market data (50 bars)")
    print("2. Test Opening Range Breakout scenario")
    print("3. Both")
    print("=" * 60)
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            tester.simulate_market_data(num_bars=50, delay=1.0)
        elif choice == '2':
            tester.test_opening_range_scenario()
        elif choice == '3':
            tester.simulate_market_data(num_bars=50, delay=0.5)
            time.sleep(2)
            tester.test_opening_range_scenario()
        else:
            print("Invalid choice")
            sys.exit(1)
        
        print("\n✅ All tests complete!")
        print(f"\n📊 View results at: {url}/dashboard/")
        
    except KeyboardInterrupt:
        print("\n\n❌ Tests cancelled by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
