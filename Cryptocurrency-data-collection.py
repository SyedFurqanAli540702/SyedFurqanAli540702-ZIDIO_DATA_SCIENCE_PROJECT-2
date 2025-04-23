
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Step 1: Data Collection from Yahoo Finance ---
def fetch_crypto_data(symbol='BTC-USD', start='2020-01-01', end='2024-12-31'):
    data = yf.download(symbol, start=start, end=end)
    data = data[['Close']]
    data.dropna(inplace=True)
    return data

# --- Step 2: ARIMA Model ---
def run_arima(data):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    print("ARIMA Summary:")
    print(model_fit.summary())
    return model_fit

# --- Step 3: Prophet Model ---
def run_prophet(data):
    df = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    model.plot(forecast)
    return forecast

# --- Step 4: LSTM Model ---
def run_lstm(data):
    sequence = data.values
    n = len(sequence)
    X, y = [], []
    for i in range(60, n):
        X.append(sequence[i-60:i])
        y.append(sequence[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)
    return model

if __name__ == "__main__":
    crypto_data = fetch_crypto_data()
    arima_model = run_arima(crypto_data)
    prophet_forecast = run_prophet(crypto_data)
    lstm_model = run_lstm(crypto_data)
