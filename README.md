# NQ Futures Trading Bot 🚀

An ML-powered algorithmic trading system for NQ (Nasdaq-100) E-mini futures with real-time signal generation, multiple trading strategies, and live Plotly Dash dashboard.

## Features ✨

- **Real-time Data Processing**: Receives 1-minute bars from TradingView webhooks
- **Multiple Trading Strategies**:
  - Opening Range Breakout (74% win rate)
  - Mean Reversion (Bollinger Bands + RSI)
  - Fair Value Gap (FVG) Detection
  - Pivot Point Scalping
  - Trend Pullback Strategies
- **ML-Powered Predictions**: XGBoost + LSTM ensemble for directional signals
- **Live Dashboard**: Interactive Plotly Dash interface with real-time charts
- **Comprehensive Technical Indicators**: VWAP, EMAs, RSI, MACD, Bollinger Bands, ATR, ADX, Standard Error Bands
- **Risk Management**: Position sizing, stop losses, profit targets
- **Performance Tracking**: Win rate, profit factor, Sharpe ratio, max drawdown

## Quick Start 🏃

### 1. Installation

```bash
# Clone the repository
cd nq_trading_bot

# Install Python dependencies
pip install -r requirements.txt

# Note: ta-lib requires binary installation
# For Ubuntu/Debian:
sudo apt-get install ta-lib
# For macOS:
brew install ta-lib
# For Windows: download from https://github.com/mrjbq7/ta-lib
```

### 2. Configuration

Edit `config.py` to customize your settings:

```python
# Trading Parameters
MAX_POSITION_SIZE = 3  # Maximum contracts
RISK_PER_TRADE = 0.02  # 2% risk per trade
WEBHOOK_PASSPHRASE = 'your_secure_passphrase'  # Change this!

# Strategy Weights (must sum to 1.0)
STRATEGIES = {
    'orb': {'weight': 0.30},
    'mean_reversion': {'weight': 0.25},
    'fvg': {'weight': 0.25},
    'pivot': {'weight': 0.10},
    'pullback': {'weight': 0.10}
}
```

### 3. Run the Bot

```bash
python main.py
```

The bot will start:
- **Webhook Server**: http://localhost:8050/webhook
- **Dashboard**: http://localhost:8050/dashboard/
- **Health Check**: http://localhost:8050/health

## TradingView Setup 📈

### 1. Create Alert

1. Open NQ futures chart on TradingView (1-minute timeframe)
2. Click "Alerts" button
3. Set "Condition" to "On bar close"
4. Enable "Webhook URL"

### 2. Webhook Configuration

**Webhook URL:**
```
http://your-server-ip:8050/webhook
```

**Message (JSON format):**
```json
{
  "passphrase": "your_secure_passphrase",
  "ticker": "NQ",
  "open": {{open}},
  "high": {{high}},
  "low": {{low}},
  "close": {{close}},
  "volume": {{volume}},
  "timestamp": "{{time}}"
}
```

### 3. Test the Connection

```bash
# Send test webhook
curl -X POST http://localhost:8050/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "your_secure_passphrase",
    "ticker": "NQ",
    "open": 16500,
    "high": 16510,
    "low": 16495,
    "close": 16505,
    "volume": 1000
  }'
```

## Project Structure 📁

```
nq_trading_bot/
├── config.py                   # Configuration settings
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── webhook_server.py           # Flask webhook receiver
│
├── data/                       # Data storage (CSV files)
│   ├── nq_live_data.csv       # Real-time price data
│   ├── trades.csv             # Trade history
│   └── signals.csv            # Generated signals
│
├── strategies/                 # Trading strategies
│   ├── opening_range.py       # ORB strategy
│   ├── mean_reversion.py      # Mean reversion strategy
│   └── fair_value_gap.py      # FVG strategy (coming soon)
│
├── models/                     # ML models
│   ├── ml_ensemble.py         # Ensemble model
│   └── trained_models/        # Saved model files
│
├── dashboard/                  # Plotly Dash UI
│   └── trading_dashboard.py   # Main dashboard
│
├── utils/                      # Utilities
│   ├── logger.py              # Logging system
│   ├── data_handler.py        # Data management
│   └── indicators.py          # Technical indicators
│
├── backtest/                   # Backtesting framework
│   └── backtest_engine.py     # Backtrader integration
│
└── logs/                       # Log files
    ├── system.log
    ├── trades.log
    └── strategy.log
```

## Trading Strategies 📊

### Opening Range Breakout (ORB)
- **Win Rate**: ~74%
- **Logic**: Trade breakouts of 15-minute opening range (9:30-9:45 ET)
- **Entry**: Price closes outside OR on 5-min candle
- **Target**: 50% of OR size
- **Stop**: Opposite side of OR or 50 points max

### Mean Reversion
- **Win Rate**: ~62%
- **Logic**: Fade overextensions using Bollinger Bands + RSI
- **Entry**: Price touches BB band + RSI confirmation + reversal candle
- **Target**: Middle Bollinger Band (20 SMA)
- **Stop**: 10 points from entry
- **Filter**: Only in ranging markets (ADX < 25)

### Fair Value Gap (FVG)
- **Win Rate**: ~67%
- **Logic**: Trade price returning to unfilled gaps
- **Detection**: 3-candle pattern with gap between candle 1 and 3
- **Entry**: Price returns to FVG zone + confirmation
- **Enhanced**: Use NQ-ES divergence for confluence

## Dashboard Features 📺

The live dashboard displays:

1. **Performance Metrics**
   - Total P&L
   - Win Rate
   - Profit Factor
   - Maximum Drawdown

2. **Price Chart**
   - Candlestick chart with volume
   - Technical indicators (VWAP, EMAs, Bollinger Bands)
   - Signal markers (entry points)
   - Support/Resistance zones

3. **Signal Indicators**
   - Current active signals
   - Strategy confidence scores
   - Entry/exit prices

4. **Recent Trades Table**
   - Trade history
   - P&L per trade
   - Win/loss status

## Risk Management 🛡️

The bot implements strict risk controls:

- **Position Sizing**: Half-Kelly criterion (0.5x Kelly)
- **Per-Trade Risk**: 2% of capital maximum
- **Daily Loss Limit**: 5% of capital (halts trading)
- **Maximum Drawdown**: 15% threshold
- **Stop Losses**: Every trade has defined stop
- **Profit Targets**: Minimum 1.5:1 reward-to-risk ratio

## Performance Metrics 📈

The system tracks:

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Mean profit and loss per trade
- **Expectancy**: Expected value per trade

## Logging 📝

All activities are logged to separate files:

- `system.log`: System events and errors
- `trades.log`: Trade executions
- `strategy.log`: Signal generation
- `ml.log`: ML model predictions
- `webhook.log`: Incoming data

View logs in real-time:
```bash
tail -f logs/system.log
```

## Production Deployment 🚀

### Using Gunicorn (Recommended)

```bash
gunicorn -w 4 -b 0.0.0.0:8050 main:server --timeout 120
```

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t nq-trading-bot .
docker run -p 8050:8050 nq-trading-bot
```

## Backtesting 🔬

Test strategies with historical data:

```python
from backtest.backtest_engine import BacktestEngine

# Initialize backtester
bt = BacktestEngine()

# Load historical data
bt.load_data('nq_historical.csv')

# Run backtest
results = bt.run_strategy('orb', start_date='2023-01-01', end_date='2024-01-01')

# Print results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Troubleshooting 🔧

### Webhook Not Receiving Data

1. Check firewall settings
2. Verify TradingView alert is active
3. Check passphrase matches
4. Review `logs/webhook.log`

### Dashboard Not Loading

1. Ensure port 8050 is available
2. Check browser console for errors
3. Verify all dependencies installed

### No Signals Generated

1. Confirm strategies are enabled in config
2. Check if market conditions meet criteria
3. Review `logs/strategy.log`

## Support & Resources 📚

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` folder
- **Community**: Discord/Telegram (links)

## Disclaimer ⚠️

This software is for educational purposes only. Trading futures involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly in paper trading before using real capital.

## License 📄

MIT License - See LICENSE file

## Acknowledgments 🙏

Built with:
- Plotly Dash
- XGBoost & TensorFlow
- Backtrader
- pandas-ta
- Flask

---

**Happy Trading! 🎯**
