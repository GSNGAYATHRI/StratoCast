import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ======================================
# 1. LOAD CLEANED MEGHA-TROPIQUES DATA
# ======================================

csv_path = "meghatropiques_clean.csv"
df = pd.read_csv(csv_path)

print("Loaded:", df.shape)
print(df.head())

# We'll predict rain_rate
target_col = "rain_rate"

# Features to use (acquisition_time is a string → skip for now)
feature_cols = ["latitude", "longitude", "rain_flag", "surface_flag"]

X = df[feature_cols]
y = df[target_col]

# ======================================
# 2. TRAIN–TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# 3. XGBOOST REGRESSOR
# ======================================

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
)

print("\nTraining XGBoost...")
xgb_model.fit(X_train, y_train)

xgb_pred_test = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred_test)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_pred_test)

print(f"\nXGBoost performance:")
print(f"  MSE : {xgb_mse:.4f}")
print(f"  RMSE: {xgb_rmse:.4f}")
print(f"  R²  : {xgb_r2:.4f}")

# ======================================
# 4. RANDOM FOREST REGRESSOR
# ======================================

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42,
)

print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)

rf_pred_test = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred_test)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred_test)

print(f"\nRandom Forest performance:")
print(f"  MSE : {rf_mse:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  R²  : {rf_r2:.4f}")

# ======================================
# 5. PREDICT ON FULL DATA & SAVE CSV
# ======================================

df["xgb_rain_pred"] = xgb_model.predict(X)
df["rf_rain_pred"] = rf_model.predict(X)

# Optional: average of both models (simple ensemble)
df["ensemble_mean_pred"] = 0.5 * (df["xgb_rain_pred"] + df["rf_rain_pred"])

out_csv = "meghatropiques_with_predictions.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved predictions → {out_csv}")