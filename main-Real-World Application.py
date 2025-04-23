# crypto_analysis.py

"""
Real-World Application:
Helps traders, investors, and analysts make informed decisions by providing:
- Trend insights
- Risk assessments
- Price predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

def fetch_crypto_data(symbol='BTC-USD', start='2022-01-01', end='2023-01-01'):
    data = yf.download(symbol, start=start, end=end)
    return data['Close']

def plot_trend(data):
    data.plot(title='Cryptocurrency Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.show()

def predict_prices(data, steps=30):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def main():
    print("Fetching data...")
    data = fetch_crypto_data()
    print("Plotting trend...")
    plot_trend(data)

    print("Forecasting future prices...")
    forecast = predict_prices(data)
    print("Predicted prices for next 30 days:")
    print(forecast)

if __name__ == "__main__":
    main()
