import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ======================================
# 1. LOAD CLEANED DATA FROM CSV
# ======================================

csv_path = "GPI_clean.csv"
df = pd.read_csv(csv_path)

print("Loaded dataframe:", df.shape)
print(df.head())

feature_cols = ["latitude", "longitude", "time"]
target_col = "GPI"

X = df[feature_cols]
y = df[target_col]

# ======================================
# 2. TRAIN–TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# 3. DEFINE & TRAIN XGBOOST MODEL
# ======================================

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)

# ======================================
# 4. EVALUATE ON TEST SET
# ======================================

y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"\nTest MSE : {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²  : {r2:.4f}")

# ======================================
# 5. PREDICT GPI FOR ALL GRID POINTS
# ======================================

df["predicted_GPI"] = model.predict(X)
df["residual"] = df["GPI"] - df["predicted_GPI"]

# ======================================
# 6. SAVE TO NEW CSV
# ======================================

output_csv = "GPI_with_predictions.csv"
df.to_csv(output_csv, index=False)

print(f"\nSaved predictions to: {output_csv}")
