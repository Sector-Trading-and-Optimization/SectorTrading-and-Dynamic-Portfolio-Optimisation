# SectorTrading-and-Dynamic-Portfolio-Optimisation

# Quantitative Trading Strategy Framework

A comprehensive Python-based quantitative trading system that implements sector-specific stock selection, technical analysis signals, portfolio optimization, backtesting, and risk management for Indian equity markets.

## 🏗️ Project Overview

This framework provides an end-to-end solution for systematic trading strategy development and evaluation, featuring:

- **Multi-sector stock selection** with custom scoring algorithms
- **Technical analysis signals** tailored to different sectors
- **Portfolio optimization** using portfolio optimization techniques
- **Historical backtesting** with realistic transaction costs
- **Risk management** with drawdown and volatility monitoring

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance Metrics](#performance-metrics)
- [Risk Management](#risk-management)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Stock Selection
- **FMCG Sector**: Momentum and Sharpe ratio-based scoring
- **Technology Sector**: 6-month momentum, RSI, and P/E ratio analysis
- **Banking Sector**: Inherits tech scoring with SMA crossover signals

### Technical Indicators
- **Bollinger Bands** for FMCG stocks
- **RSI (Relative Strength Index)** for technology stocks
- **SMA Crossover** for banking stocks

### Portfolio Optimization
- **Mean-Variance Optimization**: Risk-adjusted return maximization
- **Minimum Variance**: Risk minimization approach
- **Risk Parity**: Equal risk contribution allocation

### Backtesting Engine
- Realistic transaction costs and slippage
- Position sizing based on optimal weights
- Benchmark comparison (NIFTY 50)
- Detailed trade logging

### Risk Management
- Real-time drawdown monitoring
- Volatility assessment
- Risk level scoring system
- Automated risk alerts

### Core Components

| Module | Description |
|--------|-------------|
| `main_pipeline.py` | Orchestrates the entire trading workflow. The final function to call all classes and member functions |
| `stock_selection_module.py` | FMCG, Tech, and Banking sector analysis and algorithmic stock selection |
| `backtester_RiskManager_module.py` | Backtesting engine and risk assessment |
| `optimizer_module.py` | Portfolio optimization strategies |

## 🚀 Installation

### Prerequisites

```bash
pip install yfinance pandas numpy matplotlib cvxpy ta seaborn
```

### Required Libraries

- **yfinance**: Stock data retrieval
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **cvxpy**: Convex optimization
- **ta**: Technical analysis indicators
- **seaborn**: Statistical visualization

## ⚡ Quick Start

```python
# Run the complete trading pipeline
python main_pipeline.py
```

This will execute the full workflow:
1. Select top stocks from each sector
2. Generate trading signals
3. Optimize portfolio weights
4. Run backtesting simulation
5. Perform risk assessment

## 🔧 Components

### 1. Stock Selection (`stock_selection_module.py`)

#### FMCG Class
```python
fmcg = FMCG(["HINDUNILVR.NS", "NESTLEIND.NS"], start_date, end_date)
top_fmcg_stock = fmcg.score()
```

**Scoring Formula:**
```
Score = 0.35 × Total_Return + 0.5 × Sharpe_Ratio - 0.15 × Volatility
```

#### Tech Class
```python
tech = Tech(["INFY.NS", "TCS.NS"], start_date, end_date)
top_tech_stock = tech.score()
```

**Scoring Formula:**
```
Score = 0.4 × 6M_Momentum + 0.3 × RSI_Score - 0.2 × Volatility - 0.1 × PE_Penalty
```

#### Banking Class
```python
banking = Banking(["HDFCBANK.NS", "ICICIBANK.NS"], start_date, end_date)
top_bank_stock = banking.score()
```

### 2. Technical Signals

#### Bollinger Bands (FMCG)
- **Buy Signal**: Price crosses above upper band
- **Sell Signal**: Price crosses below lower band
- **Parameters**: 20-period MA, 2 standard deviations

#### RSI (Technology)
- **Buy Signal**: RSI crosses above 30 (oversold recovery)
- **Sell Signal**: RSI crosses below 70 (overbought correction)
- **Parameters**: 14-period RSI

#### SMA Crossover (Banking)
- **Buy Signal**: Short SMA crosses above Long SMA
- **Sell Signal**: Short SMA crosses below Long SMA
- **Parameters**: 20-period and 50-period SMAs

### 3. Portfolio Optimization (`optimizer_module.py`)

#### Available Strategies

```python
optimizer = Optimizer(returns_data)

# Mean-Variance Optimization
weights_mv, _ = optimizer.mean_variance(risk_aversion=1.0)

# Minimum Variance
weights_min, _ = optimizer.min_variance()

# Risk Parity
weights_rp, _ = optimizer.risk_parity()
```

### 4. Backtesting Engine (`backtester_RiskManager_module.py`)

```python
backtester = Backtester(prices, signals, weights, capital=1000000)
equity_curve, benchmark, trade_log = backtester.run(
    cost=0.0015,      # 0.15% transaction cost
    slip=0.001,       # 0.1% slippage
    bench='^NSEI'     # NIFTY 50 benchmark
)
```

#### Key Features
- **Position Sizing**: Based on optimized weights
- **Rebalancing**: Triggered by trading signals
- **Cost Modeling**: Realistic transaction costs and slippage
- **Performance Tracking**: Equity curve vs benchmark

### 5. Risk Management

```python
risk_manager = RiskManager(equity_curve, max_dd=0.2, vol_th=0.25)
risk_assessment = risk_manager.assess()

print(f"Risk Level: {risk_assessment['Level']}")
print(f"Recommended Action: {risk_assessment['Action']}")
```

#### Risk Scoring System
- **High Risk (Score ≥ 5)**: Reduce positions
- **Moderate Risk (3 ≤ Score < 5)**: Monitor and hedge
- **Normal Risk (Score < 3)**: Continue current strategy

## 📊 Usage Examples

### Basic Workflow

```python
from stock_selection_module import FMCG, Tech, Banking
from backtester_RiskManager_module import Backtester, RiskManager
from optimizer_module import Optimizer

# 1. Stock Selection
start_date, end_date = '2020-01-01', '2023-12-31'

fmcg = FMCG(["HINDUNILVR.NS", "NESTLEIND.NS"], start_date, end_date)
tech = Tech(["INFY.NS", "TCS.NS"], start_date, end_date)
banking = Banking(["HDFCBANK.NS", "ICICIBANK.NS"], start_date, end_date)

selected_stocks = [fmcg.score(), tech.score(), banking.score()]

# 2. Portfolio Optimization
optimizer = Optimizer(returns_data)
weights, strategy_name = optimizer.mean_variance()

# 3. Backtesting
backtester = Backtester(prices, signals, dict(zip(selected_stocks, weights)))
equity_curve, benchmark, logs = backtester.run()

# 4. Performance Analysis
metrics = backtester.metrics()
backtester.print_summary()

# 5. Risk Assessment
risk_manager = RiskManager(equity_curve)
risk_assessment = risk_manager.assess()
```

### Custom Signal Generation

```python
# Generate custom Bollinger Band signals for FMCG
price_data = fetch_price_data(['HINDUNILVR.NS'], start_date, end_date)[0]
ma, upper_band, lower_band, signals = FMCG.bollinger_signal(
    price_data['HINDUNILVR.NS'], 
    window=20, 
    num_std=2
)

# Visualize signals
FMCG.plot(price_data['HINDUNILVR.NS'], ma, upper_band, lower_band, signals, 'HINDUNILVR')
```

## ⚙️ Configuration

### Default Parameters

```python
# Backtesting Parameters
TRANSACTION_COST = 0.0015  # 0.15%
SLIPPAGE = 0.001          # 0.1%
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs

# Risk Management Thresholds
MAX_DRAWDOWN = 0.20       # 20%
VOLATILITY_THRESHOLD = 0.25  # 25%

# Technical Indicator Parameters
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SMA_SHORT = 20
SMA_LONG = 50
```

### Customization

You can modify parameters by passing them to the respective functions:

```python
# Custom Bollinger Bands
ma, ub, lb, signals = FMCG.bollinger_signal(prices, window=25, num_std=2.5)

# Custom RSI thresholds
rsi_signals = Tech.rsi_signal(prices, window=21, threshold=25)

# Custom optimization risk aversion
weights, _ = optimizer.mean_variance(risk_aversion=0.5)
```

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

### Portfolio Metrics
- **Total Return**: Cumulative portfolio return
- **Annualized Return**: Geometric mean annual return
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline

### Risk Metrics
- **Current Drawdown**: Real-time drawdown level
- **Rolling Volatility**: 30-day rolling volatility
- **Risk Score**: Composite risk assessment score

### Benchmark Comparison
- **Excess Return**: Portfolio return - Benchmark return
- **Information Ratio**: Excess return / Tracking error
- **Beta**: Portfolio sensitivity to benchmark

## 🛡️ Risk Management

### Risk Assessment Framework

The `RiskManager` class provides automated risk monitoring:

```python
risk_manager = RiskManager(equity_curve, max_dd=0.20, vol_th=0.25)
assessment = risk_manager.assess()

# Risk levels and recommended actions
if assessment['Level'] == 'High':
    print("⚠️ High Risk Detected - Consider reducing positions")
elif assessment['Level'] == 'Moderate':
    print("⚡ Moderate Risk - Monitor closely and consider hedging")
else:
    print("✅ Normal Risk - Continue current strategy")
```

### Risk Flags
- **High Drawdown**: Maximum drawdown exceeds threshold
- **High Volatility**: Current volatility above threshold
- **Current Drawdown**: Real-time drawdown monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new functionality
- Update documentation for API changes
- Ensure backward compatibility

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always consult with qualified financial advisors before making investment decisions.

---

**Happy Trading! 📈**
