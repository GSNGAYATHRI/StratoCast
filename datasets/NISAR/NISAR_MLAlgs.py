import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

# Load cleaned data
df = pd.read_csv("nisar_pixel_offsets_clean.csv")

# Drop rows with NaN (should already be clean)
df = df.dropna()

# Define ML inputs
X = df[["correlation_peak"]]      # feature
y = df["slant_range_offset"]      # target variable 1 (you can switch)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Initialize models
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

et = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

# Train models
rf.fit(X_train, y_train)
et.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
et_pred = et.predict(X_test)

# Ensemble prediction
ensemble_pred = (rf_pred + et_pred) / 2

# Evaluate
rmse_rf = mean_squared_error(y_test, rf_pred, squared=False)
rmse_et = mean_squared_error(y_test, et_pred, squared=False)
rmse_ens = mean_squared_error(y_test, ensemble_pred, squared=False)

print("Random Forest RMSE:", rmse_rf)
print("Extra Trees RMSE:", rmse_et)
print("Ensemble RMSE:", rmse_ens)

# Save predictions
result_df = pd.DataFrame({
    "true_offset": y_test.values,
    "rf_pred": rf_pred,
    "et_pred": et_pred,
    "ensemble_pred": ensemble_pred
})

result_df.to_csv("nisar_pixel_offset_predictions.csv", index=False)

print("Saved -> nisar_pixel_offset_predictions.csv")