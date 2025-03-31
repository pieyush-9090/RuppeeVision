from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the latest dataset
df = pd.read_csv("newdataset.csv")
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df = df.pivot(index="date", columns="currency_name", values="value")
df.bfill(inplace=True)

currencies = ['US Dollar', 'Euro', 'Japanese Yen', 'Pound Sterling']
models = {}
scalers = {}

# Load models and scalers
for currency in currencies:
    models[currency] = tf.keras.models.load_model(f"lstm_{currency.lower().replace(' ', '_')}.h5")
    scalers[currency] = joblib.load(f"scaler_{currency}.pkl")

# Prediction Function
def predict_future_rates(days):
    future_predictions = {currency: [] for currency in currencies}
    start_date = datetime.datetime.today()  # Start predictions from today's date
    seq_length = 30
    
    for currency in currencies:
        scaler = scalers[currency]
        model = models[currency]
        
        # Get latest 30 days of data for the currency
        last_30_days = df[[currency]].values[-seq_length:]
        last_30_days_scaled = scaler.transform(last_30_days)
        
        # Predict for the specified number of days
        for _ in range(days):
            input_seq = last_30_days_scaled.reshape(1, seq_length, 1)
            predicted_scaled = model.predict(input_seq)[0][0]
            predicted_value = scaler.inverse_transform([[predicted_scaled]])[0][0]
            
            # Append the predicted value
            future_predictions[currency].append(predicted_value)
            
            # Update last_30_days_scaled for next prediction
            last_30_days_scaled = np.roll(last_30_days_scaled, -1)
            last_30_days_scaled[-1] = predicted_scaled
    
    return future_predictions

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        days = int(data.get("days", 7))  # Default to 7 days if not provided
        
        if days < 1 or days > 30:
            return jsonify({"error": "Days must be between 1 and 30"}), 400
        
        predictions = predict_future_rates(days)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)