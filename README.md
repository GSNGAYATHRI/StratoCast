# StratoCast App

StratoCast is a Flask dashboard for weather-based risk nowcasting.  
It serves a web UI and two JSON APIs:

- `GET /api/latest_risk`: Predicts current risk from the latest weather row using a trained model.
- `GET /api/historical_data`: Returns the last 24 hours of weather data for charts.

## Project Structure

- `app.py` - Flask backend and API routes.
- `templates/index.html` - Dashboard UI (Tailwind, Leaflet, Chart.js).
- `power_hourly_clean.csv` - Historical weather input data.
- `nowcast.pkl` - Trained model file used for risk prediction.
- `development/` - Additional generated data/assets from development.
- `datasets/` - Satellite datasets and ML models:
  - `INSAT/` - INSAT-3D GPI data, models, and visualizations
  - `MEGHATROPIQUES/` - Megha-Tropiques precipitation data and ensemble models
  - `NISAR/` - NISAR pixel offset data and ML predictions

## Requirements

- Python 3.10+ (recommended)
- Dependencies in `requirements.txt`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open: `http://127.0.0.1:5000`

## Notes

- Ensure `nowcast.pkl` and `power_hourly_clean.csv` remain in the project root.
- If API responses show server-not-ready errors, verify model/data files exist and load correctly.
- Raw satellite data files (.h5, .hdf5) are excluded from the repository due to size. They are available in the source datasets:
  - INSAT: `3RIMG_11NOV2025_0015_L2G_GPI_V01R00.h5`
  - MEGHATROPIQUES: `MT1MADS_*.h5` 
  - NISAR: `NISAR_L1_PR_RIFG_*.h5`
