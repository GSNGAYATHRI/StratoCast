import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Loading cleaned dataset...")
clean = pd.read_csv("nisar_pixel_offsets_clean.csv")

print("Dataset shape:", clean.shape)
print("Columns:", clean.columns.tolist())

# 1. Features and target
X = clean[["correlation_peak"]].values      # adjust if you have more features
y = clean["slant_range_offset"].values

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# 3. XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist"
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation (no 'squared' kwarg)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- XGBoost Model Performance ---")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 6. Save predictions
results = pd.DataFrame({
    "true_value": y_test,
    "xgb_pred": y_pred
})
results.to_csv("nisar_pixel_offset_predictions_xgb.csv", index=False)

print("\nSaved → nisar_pixel_offset_predictions_xgb.csv")
