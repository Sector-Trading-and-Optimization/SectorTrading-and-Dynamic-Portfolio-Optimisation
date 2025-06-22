import yfinance as yf
import pandas as pd
import numpy as np
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

class Sector:
    def __init__(self, tickers, start, end):
        self.tickers, self.start, self.end = tickers, start, end
        self.prices, self.returns = fetch_price_data(tickers, start, end)
        self.pe = fetch_pe_ratios(tickers)

