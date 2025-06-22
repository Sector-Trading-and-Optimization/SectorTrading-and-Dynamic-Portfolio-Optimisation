# config.py

# >>> Data sources & date ranges
START_DATE = "2019-04-01"
END_DATE   = "2022-03-31"

# >>> Sector tickers
FMCG_TICKERS   = ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS"]
TECH_TICKERS   = ["INFY.NS", "TCS.NS", "WIPRO.NS"]
BANK_TICKERS   = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]

# >>> Backtest settings
INITIAL_CAPITAL   = 1_000_000
TRANSACTION_COST  = 0.001
SLIPPAGE          = 0.0005
BENCHMARK_TICKER  = "^NSEI"

# >>> Optimization hyper‑parameters
RISK_AVERSION     = 1.0

# >>> Rebalance frequency, stop‑loss levels, etc.
REBALANCE_FREQ    = "M"      # monthly
STOP_LOSS_LEVEL   = 0.05     # 5%
