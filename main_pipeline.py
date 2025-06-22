from sector_base import fetch_price_data

from stock_selection_module import FMCG, Tech, Banking
from optimizer_module import Optimizer
from backtester_RiskManager_module import Backtester, RiskManager

from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np

def main ():
    start, end = '2020-01-01', '2023-12-31'

    # 1) Stock selection
    f = FMCG(["HINDUNILVR.NS", "NESTLEIND.NS"], start, end)
    t = Tech(["INFY.NS", "TCS.NS"], start, end)
    b = Banking(["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"], start, end)

    top_f, top_t, top_b = f.score(), t.score(), b.score()
    print(f"FMCG: {top_f} | Tech: {top_t} | Bank: {top_b}")

    tickers = [top_f, top_t, top_b]

    # 2) Fetch prices & returns
    prices_all, returns_all = fetch_price_data(tickers, start, end)

    # 3) Generate signals
    # --- FMCG Bollinger
    f_price = prices_all[top_f].dropna()
    ma, ub, lb, sig_f = FMCG.bollinger_signal(f_price)
    FMCG.plot(f_price, ma, ub, lb, sig_f, top_f)
    fmcg_signal_df = pd.DataFrame({top_f: sig_f})

    # --- Tech RSI
    t_price = prices_all[top_t].dropna()
    rsi = RSIIndicator(t_price).rsi()
    buy_r = (rsi.shift(1) < 30) & (rsi > 30)
    sell_r = (rsi.shift(1) > 70) & (rsi < 70)
    Tech.plot(t_price, rsi, buy_r, sell_r, top_t)
    tech_signal = pd.Series(0, index=t_price.index)
    tech_signal[buy_r] = 1
    tech_signal[sell_r] = -1
    tech_signal_df = pd.DataFrame({top_t: tech_signal})

    # --- Banking SMA
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

    signals_all = pd.concat([fmcg_signal_df, tech_signal_df, bank_signal_df], axis=1)
    signals_all = signals_all.reindex(prices_all.index).fillna(0).astype(int)
    print("Combined signals created. Shape:", signals_all.shape)

    # 4) Portfolio optimization – compare all strategies
    print("\nPortfolio Optimization")
    optimizer = Optimizer(returns_all)

    strategies = {
        'Mean-Variance': optimizer.mean_variance(),
        'Minimum-Variance': optimizer.min_variance(),
        'Risk-Parity'  : optimizer.risk_parity()
    }

    best_sharpe = -np.inf
    for name, (w, _) in strategies.items():
        port_ret  = w @ optimizer.mu * 252
        port_risk = np.sqrt(w @ optimizer.cov @ w) * np.sqrt(252)
        sharpe    = port_ret / port_risk if port_risk > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_method = name
            best_weights = w

    weights_dict = dict(zip(tickers, best_weights))
    print(f"🎯 Selected Strategy: {best_method} (Sharpe={best_sharpe:.3f})")
    for tk, wt in weights_dict.items():
        print(f"  {tk}: {wt:.2%}")

    optimizer.plot_portfolio_comparison()

    # 5) Backtesting
    bt = Backtester(prices_all, signals_all, weights_dict)
    ec, bc, log = bt.run(cost=0.0015, slip=0.001, bench='^NSEI')
    bt.print_summary(strategy_name=best_method)

    # 6) Risk evaluation
    rm = RiskManager(ec)
    ra = rm.assess()
    print(f"\nRisk Level: {ra['Level']}, Score: {ra['Score']}, Action: {ra['Action']}")
    if ra['Flags']:
        print("Flags:", ', '.join(ra['Flags']))
    rm.plot()

if __name__ == '__main__':
    main()