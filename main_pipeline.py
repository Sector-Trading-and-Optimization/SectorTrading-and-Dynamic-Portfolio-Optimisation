from stock_selection_module import FMCG, Tech, Banking
from sector_base import fetch_price_data
from optimizer_module import Optimizer
from backtester_RiskManager_module import Backtester, RiskManager
from ta.momentum import RSIIndicator
import pandas as pd

start, end = '2020-01-01', '2023-12-31'

f = FMCG(["HINDUNILVR.NS", "NESTLEIND.NS"], start, end)
t = Tech(["INFY.NS", "TCS.NS"], start, end)
b = Banking(["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"], start, end)

top_f, top_t, top_b = f.score(), t.score(), b.score()
print(f"FMCG: {top_f} | Tech: {top_t} | Bank: {top_b}")

tickers = [top_f, top_t, top_b]
prices_all, _ = fetch_price_data(tickers, start, end)

# === FMCG Bollinger ===
f_price = prices_all[top_f].dropna()
ma, ub, lb, sig_f = FMCG.bollinger_signal(f_price)
FMCG.plot(f_price, ma, ub, lb, sig_f, top_f)
fmcg_signal_df = pd.DataFrame({top_f: sig_f})

# === Tech RSI ===
t_price = prices_all[top_t].dropna()
rsi = RSIIndicator(t_price).rsi()
buy_r = (rsi.shift(1) < 30) & (rsi > 30)
sell_r = (rsi.shift(1) > 70) & (rsi < 70)
Tech.plot(t_price, rsi, buy_r, sell_r, top_t)
tech_signal = pd.Series(0, index=t_price.index)
tech_signal[buy_r] = 1
tech_signal[sell_r] = -1
tech_signal_df = pd.DataFrame({top_t: tech_signal})

# === Banking SMA ===
b_price = prices_all[top_b].dropna()
sma_s, sma_l, signal_df = Banking.sma_crossover_signal(b_price.to_frame())
sig_b = signal_df[top_b]
buy_b = (sig_b == 1) & (sig_b.shift(1) != 1)
sell_b = (sig_b == -1) & (sig_b.shift(1) != -1)
Banking.plot(b_price, sma_s[top_b], sma_l[top_b], buy_b, sell_b, top_b)
bank_signal = pd.Series(0, index=b_price.index)
bank_signal[buy_b] = 1
bank_signal[sell_b] = -1
bank_signal_df = pd.DataFrame({top_b: bank_signal})

# === Combine all signals ===
signals_all = pd.concat([fmcg_signal_df, tech_signal_df, bank_signal_df], axis=1)
signals_all = signals_all.reindex(prices_all.index).fillna(0).astype(int)
print("✅ Combined signals created. Shape:", signals_all.shape)

# === Optimization ===
 # Portfolio optimization
selected_tickers = tickers
returns_all = _._get_returns(selected_tickers)

print("Portfolio Optimization")
optimizer = Optimizer(returns_all)

# Compare different optimization strategies
methods, weights_list, returns_annual, risks_annual, sharpe_ratios = optimizer.plot_portfolio_comparison()

# Select the best strategy based on Sharpe ratio
best_idx = sharpe_ratios.index(max(sharpe_ratios))
best_method = methods[best_idx]
best_weights = weights_list[best_idx]
best_return = returns_annual[best_idx]
best_risk = risks_annual[best_idx]
best_sharpe = sharpe_ratios[best_idx]

# Create weights dictionary
weights_dict = dict(zip(selected_tickers, best_weights))

print(f"\n🎯 Selected Strategy: {best_method}")
print(f"📊 Expected Annual Return: {best_return:.2%}")
print(f"📊 Expected Annual Risk: {best_risk:.2%}")
print(f"📊 Expected Sharpe Ratio: {best_sharpe:.3f}")
print("\n💼 Optimal Portfolio Weights:")
for ticker, weight in weights_dict.items():
    print(f"  {ticker}: {weight:.2%}")

# === Backtesting ===
bt = Backtester(prices_all, signals_all, weights_dict)
ec, bc, log = bt.run(cost=0.0015, slip=0.001, bench='^NSEI')

# bt.print_trades()
bt.print_summary(strategy_name='Mean-Variance')

rm = RiskManager(ec)
ra = rm.assess()
print(f"\nRisk Level: {ra['Level']}, Score: {ra['Score']}, Action: {ra['Action']}")
if ra['Flags']:
    print("Flags:", ', '.join(ra['Flags']))
rm.plot()