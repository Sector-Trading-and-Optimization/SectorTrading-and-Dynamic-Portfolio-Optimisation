import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

def fetch_price_data(tickers, start, end):
    prices = pd.DataFrame()
    for t in tickers:
        try:
            data = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            prices[t] = data['Close']
        except:
            continue
    prices.dropna(axis=1, inplace=True)
    returns = prices.pct_change().dropna()
    return prices, returns

def fetch_pe_ratios(tickers):
    return {t: yf.Ticker(t).info.get('trailingPE', np.nan) for t in tickers}

# === Sector Base Class ===
class Sector:
    def __init__(self, tickers, start, end):
        self.tickers, self.start, self.end = tickers, start, end
        self.prices, self.returns = fetch_price_data(tickers, start, end)
        self.pe = fetch_pe_ratios(tickers)

class FMCG(Sector):
    def score(self):
        best, best_score = None, -np.inf
        for t, r in self.returns.items():
            m, v = (r + 1).prod() - 1, r.std()
            s = r.mean() / v if v else 0
            score = 0.35 * m + 0.5 * s - 0.15 * v
            if score > best_score:
                best, best_score = t, score
        return best

    @staticmethod
    def bollinger_signal(price, w=20, n=2):
        ma, std = price.rolling(w).mean(), price.rolling(w).std()
        upper, lower = ma + n * std, ma - n * std
        sig = pd.Series(0, index=price.index)
        sig[price > upper] = 1
        sig[price < lower] = -1
        return ma, upper, lower, sig

    @staticmethod
    def plot(price, ma, ub, lb, sig, label):
        plt.figure(figsize=(14, 7))
        plt.plot(price, label='Price'), plt.plot(ma, '--'), plt.plot(ub, ':'), plt.plot(lb, ':')
        plt.scatter(sig[sig == 1].index, price[sig == 1], c='g', marker='^', label='Buy')
        plt.scatter(sig[sig == -1].index, price[sig == -1], c='r', marker='v', label='Sell')
        plt.title(f"{label} - Bollinger Band", fontsize=14), plt.legend(), plt.grid(True)
        plt.show()

class Tech(Sector):
    def score(self):
        best, best_score = None, -np.inf
        for t in self.prices.columns:
            p = self.prices[t]
            if len(p) < 126: continue
            m6 = p.iloc[-1] / p.iloc[-126] - 1
            v = self.returns[t].std()
            rs = RSIIndicator(p).rsi().iloc[-1]
            pe = self.pe.get(t, np.nan)
            penalty = pe / 50 if not np.isnan(pe) and pe < 100 else 1
            score = 0.4 * m6 + 0.3 * (rs / 100) - 0.2 * v - 0.1 * penalty
            if score > best_score:
                best, best_score = t, score
        return best

    @staticmethod
    def rsi_signal(prices_df, window=14, threshold=30):
        signals = pd.DataFrame(index=prices_df.index)
        for stock in prices_df.columns:
            rsi = RSIIndicator(prices_df[stock], window=window).rsi()
            sig = ((rsi.shift(1) < threshold) & (rsi > threshold)).astype(int)
            signals[stock] = sig
        return signals

    @staticmethod
    def plot(price, rsi, buy, sell, label):
        fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        axs[0].plot(price, label='Price')
        axs[0].scatter(price[buy].index, price[buy], c='g', marker='^', label='Buy')
        axs[0].scatter(price[sell].index, price[sell], c='r', marker='v', label='Sell')
        axs[0].set_title(f"{label} - Price + RSI Signals"), axs[0].legend(), axs[0].grid(True)

        axs[1].plot(rsi, label='RSI')
        axs[1].axhline(30, ls='--', c='green'), axs[1].axhline(70, ls='--', c='red')
        axs[1].legend(), axs[1].grid(True)
        plt.tight_layout(), plt.show()

class Banking(Tech):
    @staticmethod
    def sma_crossover_signal(prices_df, short=20, long=50):
        sma_s = prices_df.rolling(short).mean()
        sma_l = prices_df.rolling(long).mean()
        signal = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
        signal[sma_s > sma_l] = 1
        signal[sma_s < sma_l] = -1
        return sma_s, sma_l, signal

    @staticmethod
    def plot(price, s, l, buy, sell, label):
        plt.figure(figsize=(14, 7))
        plt.plot(price, label='Price'), plt.plot(s, '--', label='SMA Short'), plt.plot(l, '--', label='SMA Long')
        plt.scatter(price[buy].index, price[buy], c='g', marker='^', label='Buy')
        plt.scatter(price[sell].index, price[sell], c='r', marker='v', label='Sell')
        plt.title(f"{label} - SMA Crossover", fontsize=14), plt.legend(), plt.grid(True)
        plt.show()


def main():
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


if __name__ == "__main__":
    main()

