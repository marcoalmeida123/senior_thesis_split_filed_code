# === File: data_loader.py ===
import yfinance as yf
import pandas as pd

def load_spy_data(start="2005-01-01", end="2024-12-31"):
    data = yf.download("SPY", start=start, end=end)
    data['Return'] = data['Close'].pct_change()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data = data.dropna()
    return data
