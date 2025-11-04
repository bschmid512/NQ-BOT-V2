# csv_data_feeder.py - SAFE VERSION
import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime

# Configuration
CSV_PATH = Path(r"C:\Users\bschm\OneDrive\Documents\nq_trading_bot\NQ-BOT-V2\data\nq_live_data.csv")
WEBHOOK_URL = "http://localhost:8050/webhook"
REFRESH_INTERVAL = 5  # seconds between updates
MAX_ROWS_TO_PROCESS = 100  # Only process last 100 rows initially

def send_signal_to_dashboard(row_data):
    """Send a signal to the dashboard webhook"""
    payload = {
        "timestamp": row_data.get('time', datetime.now().isoformat()),
        "symbol": "NQ",
        "action": "BUY" if row_data.get('signal', 'LONG') == 'LONG' else "SELL",
        "price": float(row_data.get('close', 0)),
        "quantity": 1,
        "strategy": row_data.get('strategy', 'CSV_Feed'),
        "stop_loss": float(row_data.get('sl', 0)),
        "take_profit": float(row_data.get('tp', 0)),
        "confidence": float(row_data.get('quality_score', 0.5))
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"✅ Sent: {payload['action']} @ {payload['price']:.2f}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def stream_csv_data():
    """Stream CSV data to dashboard - SAFE VERSION"""
    print(f"Starting CSV data feeder (SAFE MODE)...")
    print(f"CSV file: {CSV_PATH}")
    print(f"Webhook: {WEBHOOK_URL}")
    print(f"Refresh interval: {REFRESH_INTERVAL}s")
    print(f"Max initial rows: {MAX_ROWS_TO_PROCESS}")
    print("-" * 60)
    
    # Start from the END of the file (most recent data)
    last_processed_index = None
    
    while True:
        try:
            # Check if file exists
            if not CSV_PATH.exists():
                print(f"⚠️ CSV file not found: {CSV_PATH}")
                time.sleep(5)
                continue
            
            # Read ONLY the last N rows (not the entire file!)
            df = pd.read_csv(
                CSV_PATH, 
                on_bad_lines='skip', 
                engine='python',
                nrows=None  # Read all first time to get total count
            )
            
            total_rows = len(df)
            print(f"📊 CSV has {total_rows:,} total rows")
            
            # Only process the LAST 100 rows on first run
            if last_processed_index is None:
                df = df.tail(MAX_ROWS_TO_PROCESS)
                last_processed_index = total_rows - MAX_ROWS_TO_PROCESS
                print(f"✅ Starting from row {last_processed_index:,} (last {MAX_ROWS_TO_PROCESS} rows)")
            else:
                # On subsequent runs, only get NEW rows
                new_row_count = total_rows - last_processed_index
                
                if new_row_count > 0:
                    df = df.tail(new_row_count)
                    print(f"\n📊 Found {new_row_count} new rows")
                else:
                    # No new data
                    print(".", end="", flush=True)
                    time.sleep(REFRESH_INTERVAL)
                    continue
            
            # Process rows with signals
            signals_sent = 0
            
            for idx, row in df.iterrows():
                # Check if this row has a signal
                if 'long_sig' in row and row['long_sig'] == 1:
                    row_dict = row.to_dict()
                    row_dict['signal'] = 'LONG'
                    row_dict['sl'] = row.get('sl_long', row['close'] - 5)
                    row_dict['tp'] = row.get('tp_long', row['close'] + 10)
                    row_dict['strategy'] = row.get('strategy', 'CSV_Feed')
                    
                    if send_signal_to_dashboard(row_dict):
                        signals_sent += 1
                    time.sleep(0.1)
                
                if 'short_sig' in row and row['short_sig'] == 1:
                    row_dict = row.to_dict()
                    row_dict['signal'] = 'SHORT'
                    row_dict['sl'] = row.get('sl_short', row['close'] + 5)
                    row_dict['tp'] = row.get('tp_short', row['close'] - 10)
                    row_dict['strategy'] = row.get('strategy', 'CSV_Feed')
                    
                    if send_signal_to_dashboard(row_dict):
                        signals_sent += 1
                    time.sleep(0.1)
            
            if signals_sent > 0:
                print(f"✅ Sent {signals_sent} signals to dashboard")
            
            # Update last processed index
            last_processed_index = total_rows
            
            # Wait before next check
            time.sleep(REFRESH_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping CSV data feeder...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    print("=" * 60)
    print("CSV DATA FEEDER FOR NQ TRADING DASHBOARD")
    print("=" * 60)
    print("\nThis script feeds data from your CSV file to the dashboard")
    print("Press Ctrl+C to stop\n")
    print("=" * 60)
    
    stream_csv_data()