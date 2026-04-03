import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv("meghatropiques_clean.csv")
print("Loaded dataset:", df.shape)
print(df.head())

# Target variable
target = "rain_rate"

# Features to use
features = ["latitude", "longitude", "rain_flag", "surface_flag"]

X = df[features]
y = df[target]

# ============================================================
# 2. TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 3. RANDOM FOREST MODEL
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
rf_pred_test = rf.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred_test)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred_test)

print("\n📌 Random Forest Performance:")
print(f"  MSE : {rf_mse:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  R²  : {rf_r2:.4f}")

# ============================================================
# 4. EXTRA TREES MODEL (Ensemble learner)
# ============================================================

et = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

print("\nTraining Extra Trees...")
et.fit(X_train, y_train)
et_pred_test = et.predict(X_test)

et_mse = mean_squared_error(y_test, et_pred_test)
et_rmse = np.sqrt(et_mse)
et_r2 = r2_score(y_test, et_pred_test)

print("\n📌 Extra Trees Performance:")
print(f"  MSE : {et_mse:.4f}")
print(f"  RMSE: {et_rmse:.4f}")
print(f"  R²  : {et_r2:.4f}")

# ============================================================
# 5. ENSEMBLE PREDICTION (RF + ET)
# ============================================================

ensemble_pred_test = (rf_pred_test + et_pred_test) / 2

ens_mse = mean_squared_error(y_test, ensemble_pred_test)
ens_rmse = np.sqrt(ens_mse)
ens_r2 = r2_score(y_test, ensemble_pred_test)

print("\n📌 Ensemble (RF + ExtraTrees) Performance:")
print(f"  MSE : {ens_mse:.4f}")
print(f"  RMSE: {ens_rmse:.4f}")
print(f"  R²  : {ens_r2:.4f}")

# ============================================================
# 6. SAVE PREDICTIONS TO CSV
# ============================================================

df["rf_pred"] = rf.predict(X)
df["et_pred"] = et.predict(X)
df["ensemble_pred"] = 0.5 * (df["rf_pred"] + df["et_pred"])

out_csv = "megha_rf_ensemble_predictions.csv"
df.to_csv(out_csv, index=False)

print(f"\n✅ Saved predictions → {out_csv}")

print("\n📊 Saved feature importance plot → feature_importance_rf_et.png")
