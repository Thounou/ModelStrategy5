# 🤖 Autonomous US30m MACD Trading Bot with Machine Learning

## 📋 Overview

This is a sophisticated autonomous trading bot that trades the **US30m** (Dow Jones mini) index using **MACD histogram zero-crossing detection** combined with **Machine Learning predictions** for turning point anticipation.

### 🎯 Trading Strategy

The bot implements a precise MACD-based strategy:

**ENTRY SIGNALS:**
- **BUY (Long)**: When MACD histogram crosses from zero to positive AND ML predicts a maximum ahead
- **SELL (Short)**: When MACD histogram crosses from zero to negative AND ML predicts a minimum ahead

**EXIT SIGNALS:**
- **Exit Long**: When ML predicts maximum approaching (within 2 bars) OR histogram drops 15% from peak OR histogram declining for 3 consecutive bars
- **Exit Short**: When ML predicts minimum approaching (within 2 bars) OR histogram rises 15% from trough OR histogram rising for 3 consecutive bars

## 🚀 Key Features

### Machine Learning System
- **LightGBM** with quantile regression (5 models for uncertainty estimation)
- **FFT Analysis**: 3 features for frequency domain analysis
- **Hilbert Transform**: 4 features for phase and amplitude detection
- **Rolling Statistics**: 42 features across multiple timeframes
- **Turning Point Prediction**: ML predicts when MACD will hit max/min BEFORE it happens
- **Walk-forward validation** for realistic performance estimation

### Risk Management
- **Dynamic Position Sizing**: 2-5% of equity per trade based on ML confidence
- **ATR-based Stop Loss**: 2x ATR stop distance
- **Maximum Drawdown Protection**: Automatic shutdown at 25% drawdown
- **Concurrent Position Limits**: Maximum 1-3 positions simultaneously
- **Real-time Drawdown Monitoring**

### Data Coverage
- **Training**: 2019-2023 (5 years)
- **Validation**: 2024 (1 year)
- **Testing**: January-June 2025 (6 months)
- **Automatic MT5 Data Download**

### Technical Specifications
- **Symbol**: US30m
- **Timeframe**: 5 minutes
- **MACD Parameters**: Fast=12, Slow=26, Signal=9
- **Platform**: MetaTrader 5
- **Supported Environments**: Windows, VPS

## 📊 Feature Engineering (58 Total Features)

### 1. FFT Features (3)
- Dominant frequency
- Dominant power
- Spectral entropy

### 2. Hilbert Transform Features (4)
- Instantaneous amplitude
- Instantaneous phase
- Instantaneous frequency
- Phase derivative

### 3. Rolling Statistics (42)
For each lookback period (5, 10, 20, 50, 100 bars):
- MACD MA, Std Dev, Skewness, Kurtosis
- Rate of Change
- Z-score
- Returns and Volatility

### 4. MACD Features (6)
- MACD line
- Signal line
- Histogram
- Bandpass filtered MACD
- Zero crossing indicators
- Turning point predictions

### 5. Additional Features (3)
- Cycle stage labels
- Time features (hour, day of week)
- Volume

## 🔧 Installation

### Prerequisites
```bash
Python 3.8+
MetaTrader 5 Terminal
Active MT5 Trading Account
```

### Install Dependencies
```bash
pip install pandas numpy MetaTrader5 scipy lightgbm scikit-learn joblib
```

## ⚙️ Configuration

MT5 credentials are configured in the `Config` class:

```python
LOGIN = 222566231
PASSWORD = "1234Narra#"
SERVER = "Exness-MT5Real30"
SYMBOL = "US30m"
TIMEFRAME = mt5.TIMEFRAME_M5
```

Risk parameters:
```python
MIN_RISK_PCT = 0.02  # 2% per trade
MAX_RISK_PCT = 0.05  # 5% per trade
MAX_DRAWDOWN_PCT = 0.25  # 25% max drawdown
MAX_CONCURRENT_POSITIONS = 3
ATR_PERIOD = 14
PEAK_DROP_PCT = 0.15  # Exit at 15% drop from peak
DECLINE_BARS = 3  # Exit after 3 declining bars
ML_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence
```

## 🎮 Usage

### Running the Bot

```bash
python ModelStrategy5.py
```

### Interactive Menu Options

**[CONNECTION]**
1. Connect to MT5

**[MACHINE LEARNING]**
2. Train ML Model (Full Pipeline)
3. Run ML Training Phase (2019-2023)
4. Run ML Validation Phase (2024)
5. Run ML Testing Phase (2025)
6. Complete Three-Phase ML Analysis

**[ANALYSIS]**
7. Generate Live ML Trading Signals
8. Analyze FFT Frequency Spectrum
9. Show Feature Importance

**[MODEL MANAGEMENT]**
10. Save ML Models
11. Load ML Models

**[TRADING]**
12. **Start Autonomous Trading** ⭐
13. Show Account Status
14. Close All Positions

**[SYSTEM]**
15. Check Market Status
16. Backtest Custom Period
17. Exit

## 📈 Typical Workflow

### First Time Setup:

1. **Connect to MT5** (Option 1)
2. **Train ML Model** (Option 2 or 6 for full 3-phase analysis)
   - Downloads data from 2019 to mid-2025
   - Engineers 58 features
   - Trains 5 quantile models + 1 main model
   - Performs walk-forward validation
3. **Save ML Models** (Option 10)

### Daily Trading:

1. **Connect to MT5** (Option 1)
2. **Load ML Models** (Option 11)
3. **Check Live Signals** (Option 7) - Optional preview
4. **Start Autonomous Trading** (Option 12) ⭐

### Monitoring:

- Bot checks signals every 5 minutes
- Logs all activity to `logs/` directory
- Displays account status every hour
- Auto-stops at 25% drawdown

## 🔬 Machine Learning Model

### Architecture
- **Algorithm**: LightGBM Gradient Boosting
- **Objective**: Quantile Regression + Standard Regression
- **Quantiles**: [0.1, 0.25, 0.5, 0.75, 0.9]
- **Validation**: 12-fold Time Series Walk-Forward Cross-Validation

### Training Process
1. Download historical data (2019-2025)
2. Calculate MACD indicators
3. Engineer 58 advanced features
4. Create turning point labels
5. Train 6 models (5 quantile + 1 main)
6. Validate on out-of-sample data
7. Test on 2025 data

### Prediction Outputs
- **Next Turning Point**: Maximum (1) or Minimum (-1)
- **Distance to Turning Point**: Number of bars ahead
- **Confidence**: Based on quantile spread
- **Cycle Stage**: Current MACD phase (0-4)

## 📊 Performance Metrics

The bot tracks and displays:
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Percentage gain/loss
- **Final Equity**: Account balance after period
- **Average Confidence**: Mean ML confidence of trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

## 🚨 Risk Warnings

⚠️ **IMPORTANT DISCLAIMERS:**

1. **Past performance does not guarantee future results**
2. **Trading involves substantial risk of loss**
3. **Only trade with capital you can afford to lose**
4. **Test thoroughly on demo accounts first**
5. **Monitor the bot regularly - automation doesn't eliminate risk**
6. **The 25% max drawdown protection is a safety net, not a guarantee**
7. **Market conditions can change rapidly**

## 📁 File Structure

```
/workspace/
├── ModelStrategy5.py       # Main trading bot
├── README.md              # This file
├── /models/               # Saved ML models
│   ├── lgbm_main.txt
│   ├── lgbm_quantile_*.txt
│   └── scaler.pkl
└── /logs/                 # Trading logs
    └── trading_bot_*.log
```

## 🔍 How It Works

### Signal Generation Process:

1. **Data Collection**: Fetch last 60 days of 5-minute bars from MT5
2. **MACD Calculation**: Compute MACD line, signal, and histogram
3. **Feature Engineering**: Calculate 58 advanced features
4. **ML Prediction**: Predict next turning point and distance
5. **Zero Crossing Detection**: Identify histogram zero crossings
6. **Signal Confirmation**: Combine zero crossing + ML prediction
7. **Confidence Check**: Ensure confidence > 60%
8. **Position Sizing**: Calculate lots based on risk & confidence
9. **Trade Execution**: Open position via MT5
10. **Exit Monitoring**: Check 3 exit conditions every bar
11. **Position Management**: Close when exit condition met

### Exit Logic Priority:

1. **ML Exit Signal**: Turning point within 2 bars (highest priority)
2. **Peak Drop**: Histogram drops 15% from peak
3. **Declining Bars**: Histogram declining for 3 consecutive bars
4. **Opposite Signal**: New signal in opposite direction
5. **Max Drawdown**: 25% account drawdown reached

## 🛠️ Troubleshooting

### Common Issues:

**"Not connected to MT5"**
- Ensure MetaTrader 5 is running
- Check credentials in Config class
- Verify internet connection

**"Insufficient data for training"**
- Bot automatically handles NaN values
- Ensure MT5 has historical data available
- Check date range settings

**"Model not trained"**
- Run Option 2 or 6 first
- Or load existing models with Option 11

**Positions not opening**
- Check account balance
- Verify symbol "US30m" exists on your broker
- Check ML confidence threshold
- Ensure max positions not reached

## 📞 Support & Maintenance

### Model Retraining
Retrain the model periodically (monthly recommended):
- Option 6: Complete Three-Phase ML Analysis
- Option 10: Save updated models

### Performance Monitoring
Review logs in `/logs/` directory for:
- Trade execution details
- Signal confidence levels
- Exit reasons
- Error messages

## 🎓 Advanced Usage

### Custom Backtesting
Use Option 16 to test specific periods:
```
Enter start date (YYYY-MM-DD): 2024-01-01
Enter end date (YYYY-MM-DD): 2024-12-31
```

### Feature Importance Analysis
Use Option 9 to see which features drive decisions:
- Top 20 most important features
- Visual importance bars
- Current feature values

### FFT Spectrum Analysis
Use Option 8 to identify market cycles:
- Dominant cycle periods
- Spectral entropy
- Frequency power distribution

## 📝 Version History

**Version 5.0 (Current)**
- ✅ MACD histogram zero-crossing detection
- ✅ ML turning point prediction
- ✅ Advanced risk management (2-5% sizing, 25% drawdown)
- ✅ Smart exit strategy (ML + histogram decline + % from peak)
- ✅ 58 engineered features (FFT + Hilbert + Rolling stats)
- ✅ Walk-forward validation
- ✅ Comprehensive logging
- ✅ Autonomous trading with real-time monitoring

## 📜 License

This trading bot is provided as-is for educational and research purposes.

## ⚡ Quick Start Command Summary

```bash
# First time setup
python ModelStrategy5.py
# Then in menu: 1 → 2 → 10

# Daily trading
python ModelStrategy5.py
# Then in menu: 1 → 11 → 12

# Monitoring: Check logs/ directory for detailed trading activity
```

---

**🎯 Remember**: The key to success is proper testing, risk management, and continuous monitoring. Start with demo accounts and small position sizes!

**📊 Trade Responsibly. Monitor Constantly. Adapt Continuously.**
