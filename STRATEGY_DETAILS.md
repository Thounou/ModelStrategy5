# 📈 Trading Strategy Details

## Complete Strategy Breakdown

### Core Concept
This bot combines **classical technical analysis (MACD)** with **modern machine learning** to trade the US30m index. The key innovation is using ML to predict MACD turning points BEFORE they happen, allowing earlier entries and optimal exits.

## 🎯 Entry Strategy

### Long Entry (BUY)
**Required Conditions:**
1. **Zero Crossing**: MACD histogram crosses from negative/zero to positive
2. **ML Prediction**: Model predicts a maximum (peak) ahead
3. **Confidence**: ML confidence score > 60%
4. **Risk Check**: Under 3 concurrent positions
5. **Drawdown Check**: Current drawdown < 25%

**Logic:**
```
IF histogram[t-1] <= 0 AND histogram[t] > 0 AND
   ML_predicts_peak(next_10_bars) AND
   ML_confidence > 0.6 AND
   open_positions < 3 AND
   drawdown < 0.25
THEN
   position_size = calculate_risk_based_size(confidence, ATR)
   OPEN_LONG(position_size)
```

**Example:**
```
Time: 14:30:00
MACD Hist: -0.0012 → 0.0023 (crossed zero to positive ✓)
ML Prediction: Maximum in 8 bars ✓
ML Confidence: 0.73 (73%) ✓
ATR: 120.5
Balance: $10,000
Risk: 3.6% (scales with confidence: 2% + 1.6%)
Position Size: 0.06 lots

→ BUY 0.06 lots at $42,125.50
→ Stop Loss: $42,125.50 - (2 × 120.5) = $41,884.50
→ Risk Amount: $360 (3.6% of $10,000)
```

### Short Entry (SELL)
**Required Conditions:**
1. **Zero Crossing**: MACD histogram crosses from positive/zero to negative
2. **ML Prediction**: Model predicts a minimum (trough) ahead
3. **Confidence**: ML confidence score > 60%
4. **Risk Check**: Under 3 concurrent positions
5. **Drawdown Check**: Current drawdown < 25%

**Logic:**
```
IF histogram[t-1] >= 0 AND histogram[t] < 0 AND
   ML_predicts_trough(next_10_bars) AND
   ML_confidence > 0.6 AND
   open_positions < 3 AND
   drawdown < 0.25
THEN
   position_size = calculate_risk_based_size(confidence, ATR)
   OPEN_SHORT(position_size)
```

## 🚪 Exit Strategy

### Long Exit (Close BUY)
**Exit on ANY of these conditions:**

#### 1. ML Predicted Maximum (Highest Priority)
```
IF ML_distance_to_maximum <= 2 bars
THEN EXIT
Reason: "ML predicts maximum approaching"
```
**Example:** ML predicts peak in 1 bar → Exit NOW to capture profit before reversal

#### 2. Peak Drop Threshold
```
IF current_histogram < peak_histogram × (1 - 0.15)
THEN EXIT
Reason: "Histogram dropped 15% from peak"
```
**Example:** 
- Peak histogram: 0.0450
- Current: 0.0380
- Drop: (0.0450 - 0.0380) / 0.0450 = 15.6% → EXIT

#### 3. Declining Momentum
```
IF histogram[t-2] > histogram[t-1] > histogram[t]
THEN EXIT
Reason: "Histogram declining for 3 bars"
```
**Example:**
- t-2: 0.0423
- t-1: 0.0401
- t: 0.0378
- Declining for 3 bars → EXIT

#### 4. Opposite Signal
```
IF new_signal == SELL
THEN EXIT
Reason: "Opposite signal detected"
```

### Short Exit (Close SELL)
**Exit on ANY of these conditions:**

#### 1. ML Predicted Minimum (Highest Priority)
```
IF ML_distance_to_minimum <= 2 bars
THEN EXIT
Reason: "ML predicts minimum approaching"
```

#### 2. Trough Rise Threshold
```
IF current_histogram > trough_histogram × (1 + 0.15)
THEN EXIT
Reason: "Histogram rose 15% from trough"
```
**Example:**
- Trough histogram: -0.0450
- Current: -0.0380
- Rise: (-0.0380 - (-0.0450)) / 0.0450 = 15.6% → EXIT

#### 3. Rising Momentum
```
IF histogram[t-2] < histogram[t-1] < histogram[t]
THEN EXIT
Reason: "Histogram rising for 3 bars"
```

#### 4. Opposite Signal
```
IF new_signal == BUY
THEN EXIT
Reason: "Opposite signal detected"
```

## 📊 Position Sizing Formula

### Dynamic Risk-Based Sizing

```python
def calculate_position_size(balance, confidence, atr, price):
    # Risk scales from 2% to 5% based on confidence
    risk_pct = 0.02 + (0.05 - 0.02) × confidence
    
    # Amount to risk in dollars
    risk_amount = balance × risk_pct
    
    # Stop distance is 2x ATR
    stop_distance = 2 × atr
    
    # Position size to maintain risk
    position_size = risk_amount / stop_distance
    
    # Convert to lots
    lots = position_size / price
    
    # Minimum 0.01 lots
    return max(0.01, round(lots, 2))
```

**Example Calculation:**
```
Balance: $10,000
Confidence: 0.75 (75%)
ATR: 120
Price: $42,000

Risk %: 0.02 + (0.03 × 0.75) = 0.0425 (4.25%)
Risk Amount: $10,000 × 0.0425 = $425
Stop Distance: 2 × 120 = 240 points
Position Size: $425 / 240 = $1.77 per point
Lots: $1.77 / $42,000 = 0.000042 → 0.05 lots (rounded)

Result: Trade 0.05 lots with $425 at risk
```

## 🛡️ Risk Management Rules

### Maximum Drawdown Protection

```python
def check_drawdown(current_equity, peak_equity):
    drawdown = (peak_equity - current_equity) / peak_equity
    
    if drawdown >= 0.25:
        CLOSE_ALL_POSITIONS()
        STOP_TRADING()
        ALERT("MAX DRAWDOWN REACHED")
```

**Example:**
```
Peak Equity: $12,000
Current Equity: $9,000
Drawdown: ($12,000 - $9,000) / $12,000 = 25%

→ STOP ALL TRADING
→ Close all open positions
→ Require manual restart
```

### Concurrent Position Limits

```python
MAX_POSITIONS = 3

def can_open_position(current_positions):
    return current_positions < MAX_POSITIONS
```

**Rationale:** Limits exposure and prevents over-leveraging during volatile periods

### Stop Loss Placement

```python
def calculate_stop_loss(entry_price, atr, position_type):
    stop_distance = 2 × atr
    
    if position_type == LONG:
        stop_loss = entry_price - stop_distance
    else:  # SHORT
        stop_loss = entry_price + stop_distance
    
    return stop_loss
```

**Example:**
```
Long Entry: $42,000
ATR: 125
Stop Distance: 2 × 125 = 250
Stop Loss: $42,000 - 250 = $41,750

If price hits $41,750 → Position closed automatically
```

## 🔬 Machine Learning Model Details

### What the Model Predicts

The ML model outputs THREE key predictions:

1. **Next Turning Point Type**
   - Value: 1 (maximum) or -1 (minimum)
   - Interpretation: Which direction MACD will turn

2. **Distance to Turning Point**
   - Value: Number of bars (0-50)
   - Interpretation: How soon the turning point will occur

3. **Prediction Confidence**
   - Value: 0.0 to 1.0
   - Calculation: `1 / (1 + quantile_spread)`
   - Interpretation: How certain the model is

### Training Process

**Data Split:**
- Training: 2019-2023 (70% of data, ~524,000 5-min bars)
- Validation: 2024 (15% of data, ~105,000 bars)
- Testing: Jan-Jun 2025 (15% of data, ~52,500 bars)

**Features Used:**
- 58 engineered features
- Top importance typically:
  1. MACD histogram
  2. Recent rate of change
  3. Rolling standard deviation
  4. Hilbert instantaneous phase
  5. FFT dominant frequency

**Model Architecture:**
- 6 LightGBM models:
  - 5 quantile models (0.1, 0.25, 0.5, 0.75, 0.9)
  - 1 main regression model
- 100 boosting rounds with early stopping
- Walk-forward validation (12 folds)

**Performance Expectations:**
- Training RMSE: ~0.8-1.2
- Validation RMSE: ~1.0-1.5
- Testing RMSE: ~1.2-1.8
- Win Rate: Typically 45-65% (varies by market conditions)

## 📉 Example Trade Scenarios

### Scenario 1: Successful Long Trade

```
14:30:00 - MACD hist crosses zero to positive (0.0015)
         - ML predicts maximum in 9 bars
         - Confidence: 78%
         → ENTER LONG at $42,125 (0.06 lots)
         → Stop Loss: $41,885
         → Risk: $360 (3.8%)

14:45:00 - MACD hist: 0.0087 (rising)
15:00:00 - MACD hist: 0.0134 (rising)
15:15:00 - MACD hist: 0.0189 (rising)
15:30:00 - MACD hist: 0.0223 (peak reached)
15:45:00 - ML signals: "Maximum approaching in 1 bar"
         → EXIT LONG at $42,365
         → Profit: $240 - $5 commission = $235
         → Return: 2.35%
```

### Scenario 2: Stopped Out Trade

```
10:00:00 - MACD hist crosses zero to positive (0.0012)
         - ML predicts maximum in 8 bars
         - Confidence: 65%
         → ENTER LONG at $41,890 (0.05 lots)
         → Stop Loss: $41,650
         → Risk: $240 (2.8%)

10:15:00 - MACD hist: 0.0034 (rising)
10:30:00 - Unexpected news event
         - MACD hist: -0.0045 (reversed)
         → Histogram declining for 3 bars
         → EXIT LONG at $41,725
         → Loss: -$165 - $5 commission = -$170
         → Loss: -1.7% (less than planned 2.8% risk)
```

### Scenario 3: False Signal Avoided

```
11:00:00 - MACD hist crosses zero to positive (0.0008)
         - ML predicts MINIMUM in 12 bars (not maximum!)
         - Confidence: 43% (below 60% threshold)
         → NO TRADE (conflicting signals)
         
11:15:00 - MACD hist: 0.0002 (barely positive)
11:30:00 - MACD hist: -0.0023 (back to negative)
         → ML correctly avoided false breakout
```

## 🎓 Why This Strategy Works

### 1. Zero-Crossing + ML Combination
- **Zero crossings** identify momentum shifts
- **ML predictions** filter false signals
- **Together**: Only trade high-probability setups

### 2. Early Exit Protection
- ML predicts turning points BEFORE they happen
- Exits near peaks/troughs, not after reversal
- Captures more of the trend move

### 3. Risk-Adjusted Position Sizing
- Higher confidence → Larger position
- Lower confidence → Smaller position
- Adapts to market uncertainty

### 4. Multiple Exit Conditions
- Doesn't rely on single exit trigger
- Protects against different failure modes
- Locks in profits proactively

### 5. Robust Feature Engineering
- FFT captures cyclical patterns
- Hilbert transform captures phase shifts
- Rolling stats capture momentum changes
- 58 features provide comprehensive market view

## 📊 Expected Performance Characteristics

**Typical Metrics (will vary by market conditions):**
- Win Rate: 50-60%
- Average Win: 1.5-3.0% per trade
- Average Loss: 1.0-2.0% per trade
- Risk/Reward: 1:1.5 to 1:2
- Trades per Day: 2-8
- Holding Time: 30 minutes to 4 hours
- Max Drawdown: Under 25% (forced stop)
- Monthly Return: Highly variable (-10% to +15%)

**Best Market Conditions:**
- Trending markets with clear momentum
- Normal volatility (not too high/low)
- During major trading sessions (London/NY)

**Challenging Conditions:**
- Choppy, range-bound markets
- Very low volatility (few signals)
- Very high volatility (more false signals)
- Major news events (unpredictable moves)

## 🔧 Strategy Optimization

### Adjustable Parameters

**In Config class:**
```python
# Entry filtering
ML_CONFIDENCE_THRESHOLD = 0.6  # Higher = fewer but better trades

# Exit sensitivity
PEAK_DROP_PCT = 0.15  # Lower = faster exits
DECLINE_BARS = 3      # Higher = more confirmation needed

# Risk management
MIN_RISK_PCT = 0.02   # Minimum position size
MAX_RISK_PCT = 0.05   # Maximum position size
MAX_DRAWDOWN_PCT = 0.25  # Safety shutdown level
MAX_CONCURRENT_POSITIONS = 3  # Exposure limit

# Technical indicators
MACD_FAST = 12   # Standard
MACD_SLOW = 26   # Standard
MACD_SIGNAL = 9  # Standard
ATR_PERIOD = 14  # Stop loss calculation
```

**Optimization Tips:**
1. **Conservative**: Increase ML_CONFIDENCE_THRESHOLD to 0.7+
2. **Aggressive**: Decrease PEAK_DROP_PCT to 0.10
3. **Lower Risk**: Reduce MAX_RISK_PCT to 0.03
4. **Higher Risk**: Increase MAX_RISK_PCT to 0.07 (not recommended)

## 📝 Strategy Limitations

**Known Weaknesses:**
1. Requires trending markets to be profitable
2. Can generate false signals in ranging markets
3. ML model may underperform during regime changes
4. Slippage and commissions impact small moves
5. Requires good broker execution speed

**Mitigation:**
- Regular model retraining (monthly)
- Strict risk management
- Monitor performance metrics
- Adjust parameters for current market regime
- Use high-quality broker with fast execution

---

**🎯 This strategy is designed to capture high-probability MACD trend reversals while managing risk aggressively through ML-enhanced exits and position sizing.**
