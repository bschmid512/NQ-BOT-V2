"""Example: send a vision snapshot to main.py /vision endpoint."""
import requests
from datetime import datetime

def post_vision(host="http://127.0.0.1", port=5000):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "statistics": {"sentiment": "bullish", "confidence": 0.78},
        "patterns": ["triangle", "bull_flag"],
        "support_resistance": {"support": [24950.0], "resistance": [25020.0]}
    }
    url = f"{host}:{port}/vision"
    r = requests.post(url, json=payload, timeout=2)
    print("POST /vision ->", r.status_code, r.text)

if __name__ == "__main__":
    post_vision()
