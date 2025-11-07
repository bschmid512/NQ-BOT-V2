# Deployment Checklist - Hybrid Trading Bot

## 📋 Pre-Deployment Checklist

### 1. Configuration ✓
- [ ] Changed `WEBHOOK_PASSPHRASE` in `config.py`
- [ ] Set appropriate `MAX_POSITION_SIZE` (start with 1)
- [ ] Set conservative `MAX_DAILY_LOSS` (e.g., -$500)
- [ ] Verified `PAPER_TRADING = True`
- [ ] Updated `HIGH_IMPACT_DATES` for current month
- [ ] Set reasonable `MAX_DAILY_TRADES` (10-15)

### 2. Strategy Configuration ✓
- [ ] Reviewed strategy weights in `config.py`
- [ ] Confirmed `min_signals_required` (start with 2)
- [ ] Set `min_total_weight` appropriately (60-70)
- [ ] Vision weight multiplier configured (1.1-1.3)

### 3. Testing ✓
- [ ] Ran `python test_system.py` successfully
- [ ] All tests passed
- [ ] Health endpoint responding
- [ ] Webhook receiving test data
- [ ] Statistics calculating correctly
- [ ] Logs generating in `/logs` directory

### 4. Data Directories ✓
- [ ] `/data` directory exists
- [ ] `/logs` directory exists
- [ ] CSV files initialized (bars, trades, signals)
- [ ] Log files created (decisions, opportunities, fusion)

### 5. Dependencies ✓
- [ ] Python 3.8+ installed
- [ ] `flask` installed
- [ ] `pandas` installed
- [ ] `numpy` installed
- [ ] `pytz` installed
- [ ] (Optional) Vision system dependencies

## 🚀 Deployment Steps

### Step 1: Initial System Check

```bash
# Start the bot
python main.py

# Expected output:
# 🚀 NQ HYBRID TRADING BOT - INITIALIZED
# 📡 Webhook endpoint: http://0.0.0.0:8050/webhook
# System is running. Waiting for TradingView webhooks...
```

### Step 2: Test Health

In another terminal:
```bash
curl http://localhost:8050/health

# Expected: {"status": "healthy", ...}
```

### Step 3: Run System Tests

```bash
python test_system.py

# Expected: All tests pass
```

### Step 4: Configure TradingView Alert

1. Open TradingView
2. Open NQ futures chart (1-minute timeframe)
3. Create alert:
   - **Condition**: "On bar close" every 1 minute
   - **Alert name**: "NQ Hybrid Bot Feed"
   - **Webhook URL**: `http://YOUR_SERVER:8050/webhook`
   - **Message**:
   ```json
   {
     "passphrase": "your_actual_passphrase",
     "ticker": "NQ",
     "open": {{open}},
     "high": {{high}},
     "low": {{low}},
     "close": {{close}},
     "volume": {{volume}}
   }
   ```

### Step 5: Monitor First Hour

Watch for:
- ✅ Bars being received (check terminal output)
- ✅ Market context updates printing
- ✅ Strategy evaluations running
- ✅ Fusion decisions logged
- ⚠️ Any errors or warnings

```bash
# Monitor logs in real-time:
tail -f logs/decisions.jsonl
tail -f logs/fusion_decisions.jsonl
```

### Step 6: Verify Data Collection

After 30 minutes:
```bash
# Check bars collected
wc -l data/nq_live_bars.csv
# Should have ~30 lines (one per minute)

# Check logs
ls -lh logs/
# All log files should have content
```

## 📊 First Week Monitoring

### Daily Tasks

**Morning (before market open):**
- [ ] Check bot is running
- [ ] Review previous day's performance
- [ ] Check for any errors in logs
- [ ] Verify data files are growing

**During Market Hours:**
- [ ] Monitor for signals being generated
- [ ] Watch fusion approval decisions
- [ ] Check positions are being managed correctly
- [ ] Verify stops/targets are reasonable

**End of Day:**
- [ ] Review daily report: `curl http://localhost:8050/report`
- [ ] Analyze missed opportunities
- [ ] Check trade statistics
- [ ] Save log files (optional backup)

### Key Metrics to Watch

**Fusion Engine:**
```bash
curl http://localhost:8050/stats | jq '.fusion_engine'
```
- Approval rate: 15-30% is healthy
- Too high (>50%) = too aggressive
- Too low (<10%) = too conservative

**Strategy Performance:**
```bash
curl http://localhost:8050/stats | jq '.overall_performance'
```
- Win rate: Target 55%+
- Profit factor: Target 1.5+
- Max drawdown: Keep under 15%

**Missed Opportunities:**
```bash
# Check opportunity log
cat logs/missed_opportunities.jsonl | tail -20
```
- If missing lots of moves: Lower `min_total_weight`
- If taking too many bad trades: Raise `min_total_weight`

## 🔧 Tuning Guide

### If Bot is Too Conservative (Not Trading Enough)

1. **Lower fusion requirements**:
   ```python
   # In config.py:
   FUSION_CONFIG = {
       'min_signals_required': 2,  # Keep at 2
       'min_total_weight': 50,     # Lower from 60 to 50
   }
   ```

2. **Increase strategy weights**:
   ```python
   'momentum': {'weight': 40},  # Up from 35
   'pullback': {'weight': 40},  # Up from 35
   ```

3. **Lower confidence thresholds**:
   ```python
   'min_confidence': 0.65,  # Down from 0.75
   ```

### If Bot is Too Aggressive (Overtrading)

1. **Raise fusion requirements**:
   ```python
   FUSION_CONFIG = {
       'min_signals_required': 3,  # Up from 2 (need more agreement)
       'min_total_weight': 70,     # Up from 60
   }
   ```

2. **Increase cooldown periods**:
   ```python
   MIN_TIME_BETWEEN_TRADES = 300  # 5 minutes
   ```

3. **Add convergence requirement**:
   ```python
   # Only trade if 3+ strategies agree
   'min_signals_required': 3,
   ```

### If Missing Big Moves

1. **Check momentum strategy is enabled**:
   ```python
   'momentum': {'enabled': True}
   ```

2. **Lower momentum threshold**:
   ```python
   'momentum_threshold': 0.002,  # Down from 0.003
   ```

3. **Review missed opportunity logs**:
   ```bash
   grep "missed_rally" logs/missed_opportunities.jsonl
   ```

## ⚠️ Safety Checks

### Before Increasing Position Size

- [ ] At least 100 trades in paper trading
- [ ] Win rate above 50%
- [ ] Profit factor above 1.3
- [ ] Max drawdown under 20%
- [ ] Understand why losses occurred
- [ ] Confident in risk management

### Before Going Live

- [ ] **Minimum 2 weeks paper trading**
- [ ] Profitable across different market conditions
- [ ] Reviewed ALL closed trades
- [ ] Understand fusion logic completely
- [ ] Have emergency stop plan
- [ ] Risk per trade is acceptable loss
- [ ] Broker API configured correctly

### Daily Safety Limits

Never exceed these in one day:
- Max positions: START with 1, max 3
- Daily loss: -$1000 (adjust based on account size)
- Max trades: 15-20

## 🆘 Emergency Procedures

### If System Goes Haywire

1. **Stop new trades immediately**:
   ```python
   # Set in config.py:
   MAX_DAILY_TRADES = 0  # Prevents new entries
   ```

2. **Force close all positions**:
   ```bash
   # In Python console:
   from position_manager import position_manager
   position_manager.force_close_all(current_price, "EMERGENCY_STOP")
   ```

3. **Review logs**:
   ```bash
   tail -100 logs/decisions.jsonl
   tail -100 logs/fusion_decisions.jsonl
   ```

4. **Identify root cause** before restarting

### If Webhook Stops Receiving Data

1. Check TradingView alert is active
2. Verify network connectivity
3. Check passphrase matches
4. Review Flask logs for errors

### If Strategies Stop Generating Signals

1. Check strategy enabled flags
2. Verify sufficient data (need 50+ bars)
3. Check if cooldown periods too long
4. Review market conditions (maybe legitimately no setups)

## 📈 Performance Benchmarks

### Week 1 Targets (Paper Trading)
- Trades: 10-30
- Win Rate: >45%
- Profit Factor: >1.0
- Max Drawdown: <25%
- System Uptime: 95%+

### Month 1 Targets (Paper Trading)
- Trades: 50-150
- Win Rate: >50%
- Profit Factor: >1.3
- Max Drawdown: <20%
- System Uptime: 98%+

### Before Live Trading
- Trades: 100+
- Win Rate: >53%
- Profit Factor: >1.5
- Max Drawdown: <15%
- Consistent across multiple weeks

## 📝 Notes

- **Start small**: 1 contract only
- **Stay patient**: Good setups are rare
- **Trust the system**: It's designed to be selective
- **Review daily**: Learn from every trade
- **Adjust gradually**: Small tweaks, big impact

## ✅ Final Pre-Launch Checklist

Day before going live:
- [ ] Paper trading profitable for 2+ weeks
- [ ] All safety limits tested
- [ ] Emergency procedures documented
- [ ] Account funded with appropriate capital
- [ ] Risk per trade is 1-2% of account
- [ ] Broker API credentials verified
- [ ] Backup plan if system fails
- [ ] Phone alerts configured
- [ ] Trading journal ready
- [ ] Mental preparation complete

---

**Remember**: The bot is a tool, not a money printer. Proper risk management and monitoring are essential for success.
