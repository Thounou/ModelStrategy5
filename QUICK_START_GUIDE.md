# 🚀 Quick Start Guide - US30m MACD ML Trading Bot

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (2 minutes)

```bash
pip install pandas numpy MetaTrader5 scipy lightgbm scikit-learn joblib
```

### Step 2: First Time Setup (10-15 minutes)

1. **Open MetaTrader 5** on your computer
2. **Run the bot**:
   ```bash
   python ModelStrategy5.py
   ```

3. **In the menu, execute these commands in order**:
   ```
   1  ← Connect to MT5
   6  ← Complete Three-Phase ML Analysis (downloads data 2019-2025, trains model)
   10 ← Save ML Models
   ```

⏰ Training takes 10-15 minutes. You'll see:
- Data download progress
- Feature engineering (58 features)
- Model training with cross-validation
- Performance results for Train/Val/Test phases

### Step 3: Daily Trading (30 seconds)

1. **Open MetaTrader 5**
2. **Run the bot**:
   ```bash
   python ModelStrategy5.py
   ```

3. **In the menu**:
   ```
   1  ← Connect to MT5
   11 ← Load ML Models (instant)
   12 ← Start Autonomous Trading ⭐
   ```

That's it! The bot is now running autonomously. 🎉

## 📊 What Happens During Autonomous Trading?

Every 5 minutes, the bot:
1. ✅ Fetches latest market data
2. ✅ Calculates MACD histogram
3. ✅ Detects zero crossings
4. ✅ Predicts next turning point using ML
5. ✅ Checks risk management rules
6. ✅ Opens position if signal confirmed (BUY on zero→positive, SELL on zero→negative)
7. ✅ Monitors exit conditions (ML predicted max/min, 15% drop, 3 declining bars)
8. ✅ Closes position when exit condition met
9. ✅ Logs everything to `logs/` directory

## 🎯 Trading Signals Explained

### 🟢 BUY Signal (Go Long)
**Triggers when ALL conditions met:**
- ✅ MACD histogram crosses from zero to positive
- ✅ ML predicts maximum ahead
- ✅ ML confidence > 60%
- ✅ Not at max concurrent positions (3)
- ✅ Not at max drawdown (25%)

**Position opened:** Dynamic size 2-5% of equity based on confidence

### 🔴 SELL Signal (Go Short)
**Triggers when ALL conditions met:**
- ✅ MACD histogram crosses from zero to negative
- ✅ ML predicts minimum ahead
- ✅ ML confidence > 60%
- ✅ Not at max concurrent positions (3)
- ✅ Not at max drawdown (25%)

**Position opened:** Dynamic size 2-5% of equity based on confidence

### 📤 EXIT Signals (Close Position)

**For LONG positions, exit when:**
1. ML predicts maximum within 2 bars, OR
2. Histogram drops 15% from its peak, OR
3. Histogram declining for 3 consecutive bars, OR
4. Opposite SELL signal appears

**For SHORT positions, exit when:**
1. ML predicts minimum within 2 bars, OR
2. Histogram rises 15% from its trough, OR
3. Histogram rising for 3 consecutive bars, OR
4. Opposite BUY signal appears

## 🛡️ Safety Features

**Automatic Protection:**
- ⛔ **Max Drawdown**: Stops trading at 25% account drawdown
- 🎯 **Position Limits**: Maximum 3 concurrent positions
- 📉 **ATR Stops**: Stop loss at 2x ATR distance
- 🎲 **Confidence Filter**: Only trades with >60% ML confidence

**Manual Controls:**
- Press `Ctrl+C` to stop autonomous trading anytime
- All positions automatically closed on exit
- Use menu option 14 to manually close all positions

## 📈 Monitoring Your Bot

### Real-time Console Output
```
🤖 Starting autonomous trading... Press Ctrl+C to stop

Balance: $10,000.00 | Equity: $10,000.00 | Positions: 0 | Signal: 0 | Confidence: 0.00
MACD Hist: 0.0234 | ATR: 125.50 | Price: 42,345.20

🚀 BUY signal detected! Opening long position: 0.05 lots
   Confidence: 75% | MACD Hist: 0.0234
```

### Log Files
Check `logs/trading_bot_YYYYMMDD_HHMMSS.log` for detailed activity:
```
2025-01-15 14:30:00 - TradingBot - INFO - Trading iteration 1
2025-01-15 14:30:00 - TradingBot - INFO - Balance: $10,000.00 | Equity: $10,000.00
2025-01-15 14:30:00 - TradingBot - INFO - 🚀 BUY signal - Opening long position: 0.05 lots
```

## 🎮 Useful Menu Options

### Before Trading
- **Option 7**: Preview live signals without trading
- **Option 8**: Analyze market cycles with FFT
- **Option 9**: See which features are most important
- **Option 15**: Check if market is open

### During Trading
- **Option 13**: Show current account status and open positions
- **Option 14**: Manually close all positions
- **Ctrl+C**: Stop autonomous trading

### After Trading
- Review logs in `logs/` directory
- **Option 16**: Backtest specific periods to analyze performance

## 🔍 Example Trading Session

```bash
$ python ModelStrategy5.py

# First, connect
Enter your choice: 1
✓ Successfully connected to MT5

# Load your trained models
Enter your choice: 11
✓ Models loaded successfully

# Start trading
Enter your choice: 12
🤖 Starting autonomous trading... Press Ctrl+C to stop

Balance: $10,000.00 | Equity: $10,000.00 | Positions: 0
MACD Hist: 0.0123 | ATR: 120.00 | Price: 42,100.00
Waiting for next 5-minute bar...

[5 minutes later]
Balance: $10,000.00 | Equity: $10,000.00 | Positions: 0
MACD Hist: 0.0156 | ATR: 121.50 | Price: 42,125.00

🚀 BUY signal detected! Opening long position: 0.05 lots
   Confidence: 72% | MACD Hist: 0.0156
   
Balance: $9,995.00 | Equity: $9,998.50 | Positions: 1
MACD Hist: 0.0189 | ATR: 122.00 | Price: 42,150.00
Waiting for next 5-minute bar...

[Later...]
📤 Closing position: ML predicts maximum approaching
✓ Position closed. PnL: +$87.50

Balance: $10,080.00 | Equity: $10,080.00 | Positions: 0
```

## ⚠️ Important Reminders

1. **Always test on DEMO first** before using real money
2. **Monitor the bot** - don't leave it completely unattended
3. **Check logs daily** to understand what the bot is doing
4. **Start small** - use minimum position sizes initially
5. **Retrain monthly** for best performance (Option 6)
6. **Verify your broker supports US30m** on 5-minute timeframe

## 🆘 Troubleshooting

**Bot won't connect?**
```
✗ MT5 initialization failed
→ Solution: Make sure MetaTrader 5 is running and you're logged in
```

**No trades happening?**
```
→ Check: Is confidence threshold met? (needs >60%)
→ Check: Are there zero crossings happening?
→ Check: Option 7 to see current signals
→ Check: Is market open? Option 15
```

**Positions opening but immediately closing?**
```
→ Normal! Exit conditions are strict (ML predicted turning point)
→ Review logs to see specific exit reasons
→ Consider adjusting PEAK_DROP_PCT if too sensitive
```

## 📞 Need Help?

1. **Check README.md** for detailed documentation
2. **Review logs** in `logs/` directory for error messages
3. **Test on demo account** to understand behavior
4. **Use Option 16** to backtest and validate strategy

---

## ✅ Checklist Before Going Live

- [ ] Tested on demo account for at least 1 week
- [ ] Reviewed and understood all log files
- [ ] Verified US30m symbol available on broker
- [ ] Checked account has sufficient balance (recommended: $1,000+)
- [ ] Set up monitoring schedule (check every few hours)
- [ ] Saved backup of trained models
- [ ] Read and accepted all risk warnings in README.md

---

**🎯 You're Ready! Good luck and trade responsibly!** 🚀

Remember: The bot makes decisions, but YOU are ultimately responsible for your trading account.
