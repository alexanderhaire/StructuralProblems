from flask import Flask, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as tradeapi
from joblib import Parallel, delayed
import logging
import threading
import time
import yaml
import os

app = Flask(__name__)

# Alpaca API credentials
API_KEY = 'PK2FNELWQ9GB1F7NMTW8'
API_SECRET = 'TGovWCFLkry0Y5qd0NCsXLQQfXjGn7tO2QFXsdK4'
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Configure logging
logging.basicConfig(filename="trading_service.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Load configuration from a YAML file
def load_config(config_file='config.yml'):
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    else:
        logging.warning(f"{config_file} not found, using default configuration.")
        return {}  # Return an empty or default configuration

config = load_config()
tickers = config.get("tickers", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
xgb_model = XGBClassifier(use_label_encoder=False)  # Initialize without fitting
scaler = StandardScaler()  # Initialize scaler

def get_historical_data(ticker, start_date="2010-01-01", end_date="2024-01-01"):
    logging.info(f"Fetching historical data for {ticker}.")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        logging.error(f"No historical data found for {ticker}.")
        return pd.DataFrame()

    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    data['Momentum'] = data['Close'].diff(10)
    data['RSI'] = compute_rsi(data['Close'])

    data.dropna(inplace=True)
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def call_lstm_service(features):
    logging.info("Calling LSTM service for prediction.")
    try:
        response = requests.post('http://service_a:5001/predict_lstm', json={'data': features})
        return response.json().get('prediction', None)
    except Exception as e:
        logging.error(f"Error calling LSTM service: {e}")
        return None

def place_limit_order(symbol, qty, side, limit_price):
    try:
        api.submit_order(
            symbol=symbol,
            qty=abs(qty),
            side=side,
            type='limit',
            limit_price=str(limit_price),
            time_in_force='gtc'
        )
        logging.info(f"Limit order placed: {side} {abs(qty)} shares of {symbol} at {limit_price}")
    except Exception as e:
        logging.error(f"Error placing order: {e}")

def process_ticker(ticker):
    logging.info(f"Processing ticker {ticker}.")
    try:
        bars = api.get_bars(ticker, timeframe='1Min', limit=200).df
        if bars.empty:
            logging.error(f"No real-time data available for {ticker}.")
            return

        df = pd.DataFrame(bars.reset_index())
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        df['Momentum'] = df['Close'].diff(10)
        df['RSI'] = compute_rsi(df['Close'])
        df.dropna(inplace=True)

        features = ['Momentum', '50_MA', 'RSI']
        X_new = df[features].tail(1)

        lstm_prediction = call_lstm_service(X_new.values.flatten().tolist())
        X_scaled = scaler.transform(X_new)

        xgb_prediction = xgb_model.predict(X_scaled)[0]
        final_prediction = (xgb_prediction + lstm_prediction) / 2 if lstm_prediction is not None else xgb_prediction
        decision = 1 if final_prediction > 0.5 else 0

        handle_decision(decision, ticker)
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}")

def handle_decision(decision, ticker):
    current_price = get_current_price(ticker)
    limit_price = round(current_price * (1.001 if decision == 1 else 0.999), 2)

    try:
        position = api.get_position(ticker)
        current_qty = int(position.qty)
        side = 'buy' if decision == 1 else 'sell'

        if (decision == 1 and current_qty <= 0) or (decision == 0 and current_qty >= 0):
            place_limit_order(ticker, 10, side, limit_price)
    except Exception as e:
        logging.warning(f"Could not retrieve position for {ticker}: {e}")
        place_limit_order(ticker, 10, 'buy' if decision == 1 else 'sell', limit_price)

def get_current_price(ticker):
    bars = api.get_bars(ticker, timeframe='1Min', limit=1).df
    return bars['close'].iloc[-1] if not bars.empty else None

def run_hybrid_strategy():
    while True:
        try:
            Parallel(n_jobs=-1)(delayed(process_ticker)(ticker) for ticker in tickers)
            logging.info(f"Portfolio Value: {api.get_account().portfolio_value}")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error in hybrid strategy execution: {e}")

def retrain_xgb_model():
    global xgb_model, scaler
    while True:
        for ticker in tickers:
            df = get_historical_data(ticker)
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            features = ['Momentum', '50_MA', 'RSI']
            X = df[features]
            y = df['Target']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            xgb_model.fit(X_scaled, y)
            logging.info(f"XGBoost model retrained for {ticker}.")
        time.sleep(3600)

if __name__ == '__main__':
    threading.Thread(target=run_hybrid_strategy, daemon=True).start()
    threading.Thread(target=retrain_xgb_model, daemon=True).start()
    app.run(host='0.0.0.0', port=5002)
