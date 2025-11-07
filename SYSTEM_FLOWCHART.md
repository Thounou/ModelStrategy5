# 📊 System Flowchart - Autonomous US30m MACD ML Trading Bot

## 🔄 Complete Trading Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                      START AUTONOMOUS TRADING                    │
│                 (Every 5 minutes, continuously)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONNECT TO MT5 & GET DATA                     │
│  • Fetch last 60 days of 5-minute bars                         │
│  • Check account balance and equity                             │
│  • Get current open positions                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CALCULATE TECHNICAL INDICATORS                  │
│  • MACD Line = EMA(12) - EMA(26)                               │
│  • Signal Line = EMA(9) of MACD                                │
│  • Histogram = MACD - Signal                                   │
│  • ATR(14) for stop loss calculation                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ENGINEER 58 ADVANCED ML FEATURES                    │
│                                                                  │
│  FFT Features (3):           Hilbert Features (4):             │
│  • Dominant frequency        • Instantaneous amplitude         │
│  • Dominant power           • Instantaneous phase              │
│  • Spectral entropy         • Instantaneous frequency          │
│                             • Phase derivative                 │
│  Rolling Stats (42):                                           │
│  • MACD MA, Std, Skew, Kurt (5 periods × 4 = 20)             │
│  • MACD Rate of Change (5 periods = 5)                        │
│  • MACD Z-scores (5 periods = 5)                              │
│  • Returns (5 periods = 5)                                     │
│  • Volatility (5 periods = 5)                                 │
│  • Trend indicators (2)                                        │
│                                                                  │
│  Additional Features (9):                                       │
│  • MACD, Signal, Histogram                                     │
│  • Bandpass filtered MACD                                      │
│  • Close price, Volume                                         │
│  • Hour, Day of week                                           │
│  • Cycle stage                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            DETECT ZERO CROSSINGS & TURNING POINTS                │
│                                                                  │
│  Zero Crossing Detection:                                       │
│  • histogram[t-1] ≤ 0 AND histogram[t] > 0  → Zero to Positive │
│  • histogram[t-1] ≥ 0 AND histogram[t] < 0  → Zero to Negative │
│                                                                  │
│  Turning Point Prediction:                                      │
│  • Use scipy.signal.find_peaks() to detect peaks/troughs       │
│  • For each bar, predict: next turning point type (±1)         │
│  • Calculate: distance to next turning point (bars)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             ML PREDICTION WITH UNCERTAINTY ESTIMATION            │
│                                                                  │
│  Input: 58 features for current bar                            │
│                                                                  │
│  LightGBM Models (6):                                           │
│  ├─ Quantile 0.1 model  ─────┐                                │
│  ├─ Quantile 0.25 model ─────┤                                │
│  ├─ Quantile 0.5 model  ─────┼─→ Prediction Spread           │
│  ├─ Quantile 0.75 model ─────┤                                │
│  ├─ Quantile 0.9 model  ─────┘                                │
│  └─ Main regression model ────→ Primary Prediction             │
│                                                                  │
│  Output:                                                        │
│  • Next turning point type: 1 (max) or -1 (min)               │
│  • Distance to turning point: 0-50 bars                        │
│  • Confidence: 1 / (1 + spread)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GENERATE TRADING SIGNAL                     │
│                                                                  │
│  BUY Signal (1):                                                │
│    IF zero_crossing == 1 (zero to positive)                    │
│    AND next_turning_point == 1 (predicts maximum)              │
│    AND confidence > 0.6                                         │
│    THEN signal = BUY                                            │
│                                                                  │
│  SELL Signal (-1):                                              │
│    IF zero_crossing == -1 (zero to negative)                   │
│    AND next_turning_point == -1 (predicts minimum)             │
│    AND confidence > 0.6                                         │
│    THEN signal = SELL                                           │
│                                                                  │
│  HOLD Signal (0):                                               │
│    IF confidence < 0.6 OR conflicting signals                  │
│    THEN signal = HOLD                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │                 │
          ┌─────────▼────────┐  ┌────▼─────────┐
          │  HAVE POSITIONS? │  │ NO POSITIONS │
          │       YES        │  │              │
          └─────────┬────────┘  └────┬─────────┘
                    │                │
                    │                ▼
                    │   ┌────────────────────────────────┐
                    │   │     CHECK ENTRY CONDITIONS      │
                    │   │                                 │
                    │   │  ✓ Signal = 1 or -1?           │
                    │   │  ✓ Confidence > 0.6?           │
                    │   │  ✓ Open positions < 3?         │
                    │   │  ✓ Drawdown < 25%?             │
                    │   └──────┬──────────────────────────┘
                    │          │
                    │          ▼
                    │   ┌────────────────────────────────┐
                    │   │   CALCULATE POSITION SIZE       │
                    │   │                                 │
                    │   │  risk_pct = 2% + (3% × conf)   │
                    │   │  risk_amt = balance × risk_pct │
                    │   │  stop_dist = 2 × ATR           │
                    │   │  lots = risk_amt / stop_dist   │
                    │   └──────┬──────────────────────────┘
                    │          │
                    │          ▼
                    │   ┌────────────────────────────────┐
                    │   │      OPEN POSITION (MT5)        │
                    │   │                                 │
                    │   │  IF signal == 1:               │
                    │   │    → mt5.ORDER_TYPE_BUY        │
                    │   │  IF signal == -1:              │
                    │   │    → mt5.ORDER_TYPE_SELL       │
                    │   └──────┬──────────────────────────┘
                    │          │
                    │          └──────────┐
                    │                     │
                    ▼                     │
          ┌─────────────────────┐        │
          │  CHECK EXIT CONDITIONS        │ 
          │                              │
          │  For LONG positions:         │
          │  1. ML predicts max in ≤2 bars?   │
          │  2. Hist dropped 15% from peak?    │
          │  3. Hist declining for 3 bars?     │
          │  4. Opposite SELL signal?          │
          │                              │
          │  For SHORT positions:        │
          │  1. ML predicts min in ≤2 bars?    │
          │  2. Hist rose 15% from trough?     │
          │  3. Hist rising for 3 bars?        │
          │  4. Opposite BUY signal?           │
          └────────┬────────────┘        │
                   │                     │
                   ▼                     │
            ┌──────────────┐             │
            │ EXIT NEEDED? │             │
            └──┬────────┬──┘             │
               │ YES    │ NO             │
               │        │                │
               ▼        │                │
     ┌─────────────────┐│                │
     │ CLOSE POSITION  ││                │
     │    (MT5)        ││                │
     │                 ││                │
     │ • Log PnL       ││                │
     │ • Update equity ││                │
     │ • Clear tracking││                │
     └─────────────────┘│                │
                        │                │
                        └────────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  CHECK RISK LIMITS   │
                     │                      │
                     │  Current Drawdown:   │
                     │  (peak - current)    │
                     │  ─────────────────   │
                     │       peak           │
                     └──────┬───────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Drawdown     │
                     │   ≥ 25%?     │
                     └──┬───────┬───┘
                   YES  │       │ NO
                        │       │
                        ▼       │
              ┌─────────────────┤
              │ ⚠️ MAX DRAWDOWN │
              │   REACHED!      │
              │                 │
              │ • Close all     │
              │ • Stop trading  │
              │ • Alert user    │
              └─────────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   LOG ALL ACTIVITY   │
                     │                      │
                     │ • Console output     │
                     │ • File logs/...log   │
                     │ • Timestamp          │
                     │ • All decisions      │
                     └──────────────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   WAIT 5 MINUTES     │
                     │  (Next M5 candle)    │
                     └──────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │  Loop back to START     │
                  │  (unless Ctrl+C or      │
                  │   max drawdown reached) │
                  └─────────────────────────┘
```

## 🔑 Key Decision Points

### Entry Logic
```
ENTRY = Zero_Crossing ∧ ML_Prediction ∧ Confidence ∧ Risk_Check

Where:
  Zero_Crossing = histogram crosses zero
  ML_Prediction = model predicts correct turning point
  Confidence = ML confidence > 60%
  Risk_Check = positions < 3 AND drawdown < 25%
```

### Exit Logic
```
EXIT = ML_Exit ∨ Peak_Drop ∨ Momentum_Decline ∨ Opposite_Signal

Where:
  ML_Exit = turning point predicted in ≤2 bars
  Peak_Drop = histogram drops 15% from peak
  Momentum_Decline = histogram declining for 3 bars
  Opposite_Signal = new signal in opposite direction
```

### Position Sizing
```
risk_percentage = 2% + (3% × confidence)
risk_amount = account_balance × risk_percentage
stop_distance = 2 × ATR(14)
position_size = risk_amount / stop_distance
lots = max(0.01, round(position_size / price, 2))
```

## 📊 Data Flow

```
MT5 Historical Data
        │
        ▼
  OHLCV Dataframe
        │
        ▼
  MACD Calculation
        │
        ├──────────────────┐
        ▼                  ▼
  Feature Engine    Zero Crossing Detection
        │                  │
        ├──────────────────┤
        ▼                  │
  58 Features              │
        │                  │
        ▼                  │
  ML Models (6)            │
        │                  │
        ├──────────────────┘
        ▼
  Predictions + Confidence
        │
        ▼
  Signal Generation
        │
        ├────────┬─────────┐
        ▼        ▼         ▼
      BUY      SELL      HOLD
        │        │
        └────┬───┘
             ▼
      Risk Manager
             │
             ▼
      Position Sizer
             │
             ▼
      MT5 Order
             │
             ▼
      Trade Execution
             │
             ▼
      Position Monitor
             │
             ▼
      Exit Manager
             │
             ▼
      Close Position
             │
             ▼
      Performance Log
```

## 🎯 System Components Interaction

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE                         │
│            (Interactive Menu - 17 Options)               │
└────────┬─────────────────────────────────────────┬───────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                      ┌────────────────┐
│  CONFIGURATION  │                      │   LOGGING      │
│                 │                      │   SYSTEM       │
│ • MT5 Creds    │◄─────────────────────┤                │
│ • Parameters   │                      │ • File logs    │
│ • Risk Rules   │                      │ • Console      │
└────────┬────────┘                      │ • Alerts       │
         │                               └────────────────┘
         ▼                                         ▲
┌──────────────────┐                              │
│  MT5 CONNECTION  │                              │
│                  │──────────────────────────────┤
│ • Initialize     │                              │
│ • Login          │                              │
│ • Get data       │                              │
└────────┬─────────┘                              │
         │                                         │
         ▼                                         │
┌──────────────────────────────────────┐          │
│      FEATURE ENGINE                  │          │
│                                      │          │
│ ┌──────────┐  ┌──────────┐         │          │
│ │   FFT    │  │ Hilbert  │         │          │
│ │ Analysis │  │Transform │         │          │
│ └────┬─────┘  └────┬─────┘         │          │
│      │             │                │          │
│      └──────┬──────┘                │          │
│             │                       │          │
│      ┌──────▼──────┐                │          │
│      │   Rolling   │                │          │
│      │ Statistics  │                │          │
│      └──────┬──────┘                │          │
│             │                       │          │
│      ┌──────▼──────────┐            │          │
│      │  58 Features    │            │          │
│      └─────────────────┘            │          │
└────────┬───────────────────────────┘          │
         │                                        │
         ▼                                        │
┌────────────────────────┐                       │
│   ML TRADING MODEL     │                       │
│                        │                       │
│ ┌──────────────────┐   │                       │
│ │  LightGBM (×6)   │   │                       │
│ │                  │   │                       │
│ │ • 5 Quantile     │───┼───────────────────────┤
│ │ • 1 Main         │   │                       │
│ └──────────────────┘   │                       │
│                        │                       │
│ Output:                │                       │
│ • Turning point       │                       │
│ • Distance            │                       │
│ • Confidence          │                       │
└────────┬───────────────┘                       │
         │                                        │
         ▼                                        │
┌────────────────────────┐                       │
│   SIGNAL GENERATOR     │                       │
│                        │                       │
│ • Zero crossings      │───────────────────────┤
│ • ML predictions      │                       │
│ • Confidence filter   │                       │
└────────┬───────────────┘                       │
         │                                        │
         ▼                                        │
┌────────────────────────┐                       │
│    RISK MANAGER        │                       │
│                        │                       │
│ • Drawdown monitor    │───────────────────────┤
│ • Position limits     │                       │
│ • Position sizing     │                       │
│ • ATR stops           │                       │
└────────┬───────────────┘                       │
         │                                        │
         ▼                                        │
┌────────────────────────┐                       │
│   POSITION MANAGER     │                       │
│                        │                       │
│ • Open positions      │───────────────────────┤
│ • Exit monitoring     │                       │
│ • Trade execution     │                       │
└────────┬───────────────┘                       │
         │                                        │
         ▼                                        │
┌────────────────────────┐                       │
│   MT5 ORDER SYSTEM     │                       │
│                        │                       │
│ • Buy orders          │───────────────────────┘
│ • Sell orders         │
│ • Close orders        │
└───────────────────────┘
```

## 🔄 Training Pipeline

```
Historical Data (2019-2025)
          │
          ▼
┌──────────────────────┐
│  Data Preprocessing  │
│  • MACD calculation  │
│  • ATR calculation   │
│  • Data cleaning     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Feature Engineering  │
│  • 58 features       │
│  • NaN handling      │
│  • Forward/back fill │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Label Creation      │
│  • Turning points    │
│  • Cycle stages      │
│  • Distances         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Data Split         │
│                      │
│  Train: 2019-2023    │
│  Val:   2024         │
│  Test:  2025 (half)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Walk-Forward Validation     │
│                              │
│  ┌────┐ ┌────┐ ┌────┐       │
│  │Fold│→│Fold│→│Fold│→ ...  │
│  │ 1  │ │ 2  │ │ 3  │       │
│  └────┘ └────┘ └────┘       │
│                              │
│  12 folds total              │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Train 6 Models      │
│                      │
│  For each quantile:  │
│  • LightGBM fit      │
│  • Early stopping    │
│  • Validation score  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Model Evaluation    │
│  • MAE               │
│  • RMSE              │
│  • Feature import.   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Save Models        │
│  • lgbm_*.txt        │
│  • scaler.pkl        │
└──────────────────────┘
```

---

**This flowchart represents the complete end-to-end autonomous trading system!** 🚀
