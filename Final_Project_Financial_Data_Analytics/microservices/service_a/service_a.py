# File: service_a.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import threading
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration values
FLASK_PORT = 5001  # Match this with Docker Compose port mapping
LSTM_TIMESTEPS = 6
LSTM_FEATURES = 1
LSTM_UNITS = 100
LSTM_OUTPUT_UNITS = 1
LSTM_DROPOUT_RATE = 0.2
LSTM_OPTIMIZER = 'adam'
TRAINING_SAMPLES = 100
TRAINING_EPOCHS = 10
TRAINING_BATCH_SIZE = 64
RETRAIN_INTERVAL = 3600  # Retrain every 1 hour

# Define the input shape for LSTM
input_shape = (LSTM_TIMESTEPS, LSTM_FEATURES)

# Initialize Flask app
app = Flask(__name__)

# Global LSTM model variable
lstm_model = None

# LSTM model creation function
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(LSTM_DROPOUT_RATE))
    model.add(LSTM(LSTM_UNITS // 2, return_sequences=False))
    model.add(Dropout(LSTM_DROPOUT_RATE))
    model.add(Dense(LSTM_OUTPUT_UNITS, activation='sigmoid'))
    model.compile(optimizer=LSTM_OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to retrain the LSTM model periodically
def retrain_lstm_model():
    global lstm_model
    while True:
        logging.info("Retraining LSTM model...")
        # Generating random data for demonstration purposes
        X_train = np.random.random((TRAINING_SAMPLES, input_shape[0], input_shape[1]))
        y_train = np.random.randint(0, 2, TRAINING_SAMPLES)
        lstm_model.fit(X_train, y_train, epochs=TRAINING_EPOCHS, batch_size=TRAINING_BATCH_SIZE, verbose=1)
        logging.info("LSTM model retrained!")
        time.sleep(RETRAIN_INTERVAL)

# Prediction endpoint
@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    global lstm_model
    data = request.json.get('data')
    if not data:
        logging.error("No input data received for prediction.")
        return jsonify({'error': 'No input data provided'}), 400
    try:
        input_data = np.array(data).reshape((1, len(data), 1))  # Reshape data for LSTM
        prediction = lstm_model.predict(input_data)
        logging.info(f"Prediction made: {prediction[0][0]}")
        return jsonify({'prediction': float(prediction[0][0])})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Main execution block
if __name__ == '__main__':
    lstm_model = create_lstm_model(input_shape)  # Initialize the LSTM model
    threading.Thread(target=retrain_lstm_model, daemon=True).start()  # Start retraining in a separate thread
    app.run(host='0.0.0.0', port=FLASK_PORT)  # Flask runs on port 5001 as per Docker Compose
