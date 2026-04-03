import pandas as pd
import joblib
from flask import Flask, jsonify, render_template

# --- 1. Initialization ---
app = Flask(__name__)

# --- 2. Load Model & Data ---
try:
    # Load your trained model
    model = joblib.load('nowcast.pkl')
except FileNotFoundError:
    print("FATAL ERROR: 'nowcast.pkl' model file not found.")
    model = None

try:
    # Load your historical data
    # We use this for the charts and to get the "latest" row for prediction
    data = pd.read_csv('power_hourly_clean.csv', 
                       parse_dates=['time'], 
                       index_col='time')
except FileNotFoundError:
    print("FATAL ERROR: 'power_hourly_clean.csv' data file not found.")
    data = pd.DataFrame()

# --- 3. Define API Endpoints ---

@app.route('/api/latest_risk')
def get_latest_risk():
    """
    API Endpoint to get the ML model's latest prediction.
    This is what the "Risk Assessment" card will use.
    """
    if data.empty or model is None:
        return jsonify({"error": "Server not ready. Model or data missing."}), 500

    # Get the most recent row of data
    # We add .copy() to safely add new columns
    latest_data = data.iloc[[-1]].copy()
    
    # --- Feature Engineering ---
    # Create 'hour' and 'month' columns from the data
    # This MUST match what you did in your notebook
    latest_data['hour'] = latest_data['HR']
    latest_data['month'] = latest_data['MO']

    # --- !! IMPORTANT !! ---
    # This 'features' list MUST match your model's training
    # Based on the error, it needs 'hour' and 'month'
    features = ['T2M', 'RH2M', 'WS10M', 'hour', 'month'] 
    
    # Verify these features exist in your data
    if not all(col in latest_data.columns for col in features):
         return jsonify({"error": f"Data is missing required features. Need: {features}"}), 500

    # Create the feature vector [X] for prediction
    X_latest = latest_data[features]
    
    # Make the prediction
    predicted_risk = model.predict(X_latest)[0]
    
    # Define risk thresholds (you can tune these)
    level = "Low"
    if predicted_risk > 50: level = "Moderate"
    if predicted_risk > 75: level = "High"
    if predicted_risk > 90: level = "Severe"

    # Format the output (just like your 'latest_risk.json' sample)
    output = {
        "predicted_value": round(predicted_risk, 2),
        "level": level,
        "lat": latest_data['LAT'].iloc[0],
        "lon": latest_data['LON'].iloc[0],
        "timestamp": latest_data.index[0].isoformat()
    }
    return jsonify(output)


@app.route('/api/historical_data')
def get_historical_data():
    """
    API Endpoint to get data for the charts.
    Returns the last 24 hours of data.
    """
    if data.empty:
        return jsonify({"error": "No data available"}), 500
        
    # Get last 24 hours of data (without deprecated DataFrame.last)
    last_timestamp = data.index.max()
    start_timestamp = last_timestamp - pd.Timedelta(hours=24)
    historical = data.loc[data.index >= start_timestamp].reset_index()
    
    # Return as JSON (in 'records' format, which Chart.js likes)
    return jsonify(historical.to_dict(orient='records'))


# --- 4. Define Main Page Route ---

@app.route('/')
def home():
    """
    Serves the main dashboard page.
    """
    # Flask will look for 'index.html' in the 'templates' folder
    return render_template('index.html') 

# --- 5. Run the App ---
if __name__ == '__main__':
    print("Starting StratoCast server...")
    print("Access at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
