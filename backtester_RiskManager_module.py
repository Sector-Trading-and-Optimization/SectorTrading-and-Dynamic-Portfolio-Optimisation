import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Backtester:
    def __init__(self, prices, signals, weights, capital=1e6):
        ticks = set(prices.columns) & set(signals.columns) & set(weights)
        if not ticks:
            raise ValueError("No common tickers.")
        self.prices = prices[list(ticks)].dropna()
        self.signals = signals.reindex(self.prices.index)[list(ticks)].fillna(0)
        self.weights = weights
        self.capital = capital
        self.positions = dict.fromkeys(ticks, 0)
        self.log = []

    def run(self, cost=1e-3, slip=5e-4, bench='^NSEI'):
        data = yf.download(bench,
                           start=self.prices.index[0],
                           end=self.prices.index[-1],
                           auto_adjust=True, progress=False)
        bench_ret = data['Close'].pct_change().reindex(self.prices.index).fillna(0)
        equity = []
        for i in range(1, len(self.prices)):
            date = self.prices.index[i]
            value = self.capital + sum(
                self.positions[t] * self.prices[t].iat[i] for t in self.positions)
            equity.append(value)

            sig = self.signals.iloc[i-1]
            active = sig[sig == 1].index
            trades = {}

            if active.any():
                w = pd.Series({t: self.weights[t] for t in active})
                w /= w.sum()
                targets = value * w
                for t in active:
                    p = self.prices[t].iat[i]
                    d = targets[t] / p - self.positions[t]
                    if abs(d) > 1e-3:
                        exec_p = p * (1 + slip * np.sign(d))
                        fee = abs(d) * exec_p * cost
                        self.capital -= d * exec_p + fee
                        self.positions[t] = targets[t] / p
                        trades[t] = (d, exec_p, fee)

            self.log.append((date, value, trades))

        self.equity_curve = pd.Series(equity, index=self.prices.index[1:])
        self.benchmark_curve = (1 + bench_ret.iloc[1:]).cumprod() * self.capital
        return self.equity_curve, self.benchmark_curve, self.log

    def print_trades(self):
        for date, val, trades in self.log:
            if not trades:
                continue
            print(f"\n\U0001F4C6 {date.strftime('%Y-%m-%d')}")
            print(f"Portfolio Value: ₹{val:,.2f}")
            print("Executed Trades:")
            total_cost = 0
            for t, (d, p, f) in trades.items():
                action = 'BUY' if d > 0 else 'SELL'
                print(f"  {t}: {action} {abs(d):.2f} shares @ ₹{p:.2f}")
                total_cost += f
            print(f"Total Transaction Cost: ₹{total_cost:.2f}")
            print("-" * 50)

    def metrics(self):
        r = self.equity_curve.pct_change().dropna()
        tot = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        ann = (1 + tot) ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        dd = (self.equity_curve / self.equity_curve.cummax() - 1).min()
        bench_ret = self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1
        return {
            'Final Portfolio Value': self.equity_curve.iloc[-1],
            'Total Return': tot,
            'Benchmark Return': bench_ret,
            'Excess Return': tot - bench_ret,
            'Annualized Return': ann,
            'Annualized Volatility': vol,
            'Sharpe Ratio': ann / vol if vol else np.nan,
            'Max Drawdown': dd
        }

    def print_summary(self, strategy_name="Strategy", risk_info=None):
        m = self.metrics()
        fpv = float(m["Final Portfolio Value"])
        tr  = float(m["Total Return"])
        sr  = float(m["Sharpe Ratio"])
        mdd = float(m["Max Drawdown"])

        print("\n" + "="*60)
        print("\u2705 Analysis Complete!")
        print(f"⏰ Finished at: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("\n📋 Executive Summary:")
        print(f"   • Portfolio Strategy: {strategy_name}")
        print(f"   • Final Value: ₹{fpv:,.2f}")
        print(f"   • Total Return: {tr:.2%}")
        print(f"   • Sharpe Ratio: {sr:.3f}")
        print(f"   • Max Drawdown: {mdd:.2%}")

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.equity_curve, label='Portfolio')
        plt.plot(self.benchmark_curve, label='Benchmark')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()


class RiskManager:
    def __init__(self, equity_curve, max_dd=0.2, vol_th=0.25):
        self.ec = equity_curve
        self.max_dd = max_dd
        self.vol_th = vol_th

    def assess(self):
        r = self.ec.pct_change().dropna()
        curr_dd = (self.ec.iloc[-1] - self.ec.cummax().iloc[-1]) / self.ec.cummax().iloc[-1]
        max_dd = (self.ec / self.ec.cummax() - 1).min()
        curr_vol = (r.tail(30).std() * np.sqrt(252)) if len(r) >= 30 else (r.std() * np.sqrt(252))
        flags, score = [], 0
        if max_dd < -self.max_dd:
            flags.append(f"High drawdown: {max_dd:.1%}"); score += 3
        if curr_vol > self.vol_th:
            flags.append(f"High volatility: {curr_vol:.1%}"); score += 2
        if curr_dd < -0.1:
            flags.append(f"Current drawdown: {curr_dd:.1%}"); score += 2
        level = 'High' if score >= 5 else 'Moderate' if score >= 3 else 'Normal'
        action = {'High':'Reduce positions','Moderate':'Monitor & hedge','Normal':'Continue'}[level]
        return {'Level': level, 'Score': score, 'Action': action, 'Flags': flags}

    def plot(self):
        r = self.ec.pct_change().dropna()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(r.rolling(30).std() * np.sqrt(252)); axs[0].set_title('Rolling Volatility')
        draw = (self.ec - self.ec.cummax()) / self.ec.cummax()
        axs[1].fill_between(draw.index, draw, 0, alpha=0.3); axs[1].set_title('Drawdown')
        plt.tight_layout(); plt.show()
