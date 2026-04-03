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

# --- Load Satellite Datasets ---
try:
    insat_data = pd.read_csv('datasets/INSAT/GPI_with_predictions.csv')
    print(f"INSAT data loaded: {len(insat_data)} records")
except FileNotFoundError:
    print("WARNING: INSAT dataset not found.")
    insat_data = pd.DataFrame()

try:
    megha_data = pd.read_csv('datasets/MEGHATROPIQUES/meghatropiques_with_predictions_xgboost.csv')
    print(f"MEGHATROPIQUES data loaded: {len(megha_data)} records")
except FileNotFoundError:
    print("WARNING: MEGHATROPIQUES dataset not found.")
    megha_data = pd.DataFrame()

try:
    nisar_data = pd.read_csv('datasets/NISAR/nisar_pixel_offset_predictions_ensemble.csv')
    print(f"NISAR data loaded: {len(nisar_data)} records")
except FileNotFoundError:
    print("WARNING: NISAR dataset not found.")
    nisar_data = pd.DataFrame()

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


# --- INSAT Dashboard Routes ---

@app.route('/dashboard/insat')
def insat_dashboard():
    """INSAT-3D GPI Dashboard"""
    return render_template('insat_dashboard.html')

@app.route('/api/insat/data')
def get_insat_data():
    """API endpoint for INSAT GPI data"""
    if insat_data.empty:
        return jsonify({"error": "INSAT data not available"}), 500
    
    # Return all data (limited to 5000 points for performance)
    sample_data = insat_data.head(5000).to_dict(orient='records')
    return jsonify(sample_data)

@app.route('/api/insat/stats')
def get_insat_stats():
    """API endpoint for INSAT statistics"""
    if insat_data.empty:
        return jsonify({"error": "INSAT data not available"}), 500
    
    stats = {
        "total_records": len(insat_data),
        "avg_gpi": float(insat_data['GPI'].mean()),
        "avg_predicted_gpi": float(insat_data['predicted_GPI'].mean()),
        "max_gpi": float(insat_data['GPI'].max()),
        "high_risk_count": int((insat_data['predicted_GPI'] > 0.8).sum()),
        "moderate_risk_count": int(((insat_data['predicted_GPI'] > 0.5) & (insat_data['predicted_GPI'] <= 0.8)).sum()),
        "low_risk_count": int((insat_data['predicted_GPI'] <= 0.5).sum()),
        "lat_range": [float(insat_data['latitude'].min()), float(insat_data['latitude'].max())],
        "lon_range": [float(insat_data['longitude'].min()), float(insat_data['longitude'].max())]
    }
    return jsonify(stats)

@app.route('/api/insat/hazard_zones')
def get_insat_hazard_zones():
    """API endpoint for INSAT hazard polygons"""
    if insat_data.empty:
        return jsonify({"error": "INSAT data not available"}), 500
    
    # Define hazard zones based on predicted GPI thresholds
    high_risk = insat_data[insat_data['predicted_GPI'] > 0.8][['latitude', 'longitude', 'predicted_GPI']]
    moderate_risk = insat_data[(insat_data['predicted_GPI'] > 0.5) & (insat_data['predicted_GPI'] <= 0.8)][['latitude', 'longitude', 'predicted_GPI']]
    
    hazard_zones = {
        "high_risk": high_risk.head(1000).to_dict(orient='records'),
        "moderate_risk": moderate_risk.head(1000).to_dict(orient='records')
    }
    return jsonify(hazard_zones)


# --- MEGHATROPIQUES Dashboard Routes ---

@app.route('/dashboard/meghatropiques')
def megha_dashboard():
    """MEGHATROPIQUES Dashboard"""
    return render_template('megha_dashboard.html')

@app.route('/api/meghatropiques/data')
def get_megha_data():
    """API endpoint for MEGHATROPIQUES data"""
    if megha_data.empty:
        return jsonify({"error": "MEGHATROPIQUES data not available"}), 500
    
    # Filter out missing data flags (65535)
    valid_data = megha_data[megha_data['ensemble_mean_pred'] < 60000]
    sample_data = valid_data.head(5000).to_dict(orient='records')
    return jsonify(sample_data)

@app.route('/api/meghatropiques/stats')
def get_megha_stats():
    """API endpoint for MEGHATROPIQUES statistics"""
    if megha_data.empty:
        return jsonify({"error": "MEGHATROPIQUES data not available"}), 500
    
    # Filter valid data
    valid_data = megha_data[megha_data['ensemble_mean_pred'] < 60000]
    
    stats = {
        "total_records": len(megha_data),
        "valid_records": len(valid_data),
        "avg_xgb_pred": float(valid_data['xgb_rain_pred'].mean()) if len(valid_data) > 0 else 0,
        "avg_rf_pred": float(valid_data['rf_rain_pred'].mean()) if len(valid_data) > 0 else 0,
        "avg_ensemble": float(valid_data['ensemble_mean_pred'].mean()) if len(valid_data) > 0 else 0,
        "lat_range": [float(megha_data['latitude'].min()), float(megha_data['latitude'].max())],
        "lon_range": [float(megha_data['longitude'].min()), float(megha_data['longitude'].max())]
    }
    return jsonify(stats)


# --- NISAR Dashboard Routes ---

@app.route('/dashboard/nisar')
def nisar_dashboard():
    """NISAR Pixel Offset Dashboard"""
    return render_template('nisar_dashboard.html')

@app.route('/api/nisar/data')
def get_nisar_data():
    """API endpoint for NISAR data"""
    if nisar_data.empty:
        return jsonify({"error": "NISAR data not available"}), 500
    
    sample_data = nisar_data.head(1000).to_dict(orient='records')
    return jsonify(sample_data)

@app.route('/api/nisar/stats')
def get_nisar_stats():
    """API endpoint for NISAR statistics"""
    if nisar_data.empty:
        return jsonify({"error": "NISAR data not available"}), 500
    
    stats = {
        "total_records": len(nisar_data),
        "avg_true_value": float(nisar_data['true_value'].mean()),
        "avg_rf_pred": float(nisar_data['rf_pred'].mean()),
        "avg_et_pred": float(nisar_data['et_pred'].mean()),
        "avg_ensemble": float(nisar_data['ensemble_pred'].mean()),
        "rf_accuracy": float(1 - abs(nisar_data['true_value'] - nisar_data['rf_pred']).mean()),
        "et_accuracy": float(1 - abs(nisar_data['true_value'] - nisar_data['et_pred']).mean()),
        "ensemble_accuracy": float(1 - abs(nisar_data['true_value'] - nisar_data['ensemble_pred']).mean())
    }
    return jsonify(stats)


# --- 5. Run the App ---
if __name__ == '__main__':
    print("Starting StratoCast server...")
    print("Access at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
