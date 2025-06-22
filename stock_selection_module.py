import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from sector_base import Sector

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
