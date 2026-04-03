import pandas as pd

df = pd.read_csv("nisar_pixel_offsets_clean.csv")
print("Rows:", len(df))
print(df.head())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Loading CSV...")
df = pd.read_csv("nisar_pixel_offsets_clean.csv")
print("Loaded rows:", len(df))
print(df.head())

# Drop any NaNs just in case
df = df.dropna(subset=["slant_range_offset", "correlation_peak"])
print("Rows after dropna:", len(df))

# Optional: limit size for speed (e.g., 100k samples max)
MAX_SAMPLES = 100000
if len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=42)
    print(f"Sampled down to {MAX_SAMPLES} rows for training.")

# Features and target
X = df[["correlation_peak"]]           # feature(s)
y = df["slant_range_offset"]          # target

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", len(X_train), "Test size:", len(X_test))

# ---------------- Random Forest only ----------------
rf = RandomForestRegressor(
    n_estimators=150,      # lighter than 300
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

print("Training RandomForest...")
rf.fit(X_train, y_train)
print("Finished training RandomForest.")

print("Predicting...")
rf_pred = rf.predict(X_test)

# sklearn version compatible RMSE (no 'squared' kwarg)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
r2_rf = r2_score(y_test, rf_pred)

print("\n--- Model Performance (Random Forest) ---")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

# Save predictions
results = pd.DataFrame({
    "true_value": y_test.values,
    "rf_pred": rf_pred
})
results.to_csv("nisar_pixel_offset_predictions_rf.csv", index=False)

print("\nSaved → nisar_pixel_offset_predictions_rf.csv")

from sklearn.ensemble import ExtraTreesRegressor

# ... same loading + split as above ...

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

et = ExtraTreesRegressor(
    n_estimators=100,      # keep this small initially
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

print("Training RandomForest...")
rf.fit(X_train, y_train)
print("Finished training RandomForest.")

print("Training ExtraTrees...")
et.fit(X_train, y_train)
print("Finished training ExtraTrees.")

print("Predicting...")
rf_pred = rf.predict(X_test)
et_pred = et.predict(X_test)
ensemble_pred = (rf_pred + et_pred) / 2

rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
rmse_et = np.sqrt(mean_squared_error(y_test, et_pred))
rmse_en = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print("\n--- Model Performance ---")
print(f"RF RMSE:        {rmse_rf:.4f}")
print(f"ExtraTrees RMSE:{rmse_et:.4f}")
print(f"Ensemble RMSE:  {rmse_en:.4f}")

from sklearn.metrics import r2_score
print("\nR² Scores:")
print("RF:", r2_score(y_test, rf_pred))
print("ET:", r2_score(y_test, et_pred))
print("Ensemble:", r2_score(y_test, ensemble_pred))

results = pd.DataFrame({
    "true_value": y_test.values,
    "rf_pred": rf_pred,
    "et_pred": et_pred,
    "ensemble_pred": ensemble_pred
})
results.to_csv("nisar_pixel_offset_predictions_ensemble.csv", index=False)

print("\nSaved → nisar_pixel_offset_predictions_ensemble.csv")


