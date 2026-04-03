import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# USER OPTION → Choose model
# ---------------------------------------------------------
USE_ENSEMBLE = False   # Set to True if you want ensemble_mean_pred instead of XGBoost

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv("meghatropiques_with_predictions_xgboost.csv")
print(df.head())
print(df.shape)

# ---------------------------------------------------------
# Column names
# ---------------------------------------------------------
lat_col = "latitude"
lon_col = "longitude"
true_col = "rain_rate"

pred_col = "ensemble_mean_pred" if USE_ENSEMBLE else "xgb_rain_pred"

# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[lat_col, lon_col, true_col, pred_col])

# ---------------------------------------------------------
# Compute error metrics
# ---------------------------------------------------------
df["error"] = df[pred_col] - df[true_col]

mae = np.mean(np.abs(df["error"]))
rmse = np.sqrt(np.mean(df["error"]**2))
bias = np.mean(df["error"])

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Bias: {bias:.4f}")

# ---------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.suptitle("Megha-Tropiques Rain Rate Prediction Dashboard", fontsize=18, fontweight='bold')

# ---------------------------------------------------------
# A) Spatial True Rain Rate
# ---------------------------------------------------------
ax = axes[0, 0]
sc = ax.scatter(df[lon_col], df[lat_col], c=df[true_col], cmap="Blues", s=8)
ax.set_title("A) True Rain Rate (mm/hr)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.colorbar(sc, ax=ax, label="Rain Rate (mm/hr)")

# ---------------------------------------------------------
# B) Spatial Predicted Rain Rate
# ---------------------------------------------------------
ax = axes[0, 1]
sc = ax.scatter(df[lon_col], df[lat_col], c=df[pred_col], cmap="viridis", s=8)
ax.set_title(f"B) Predicted Rain Rate ({'Ensemble' if USE_ENSEMBLE else 'XGBoost'})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.colorbar(sc, ax=ax, label="Predicted Rain Rate (mm/hr)")

# ---------------------------------------------------------
# C) True vs Predicted Scatter Plot
# ---------------------------------------------------------
ax = axes[1, 0]
sns.scatterplot(x=df[true_col], y=df[pred_col], ax=ax, s=8, alpha=0.6)
max_val = max(df[true_col].max(), df[pred_col].max())
ax.plot([0, max_val], [0, max_val], 'r--', label="Ideal = 1:1")
ax.set_title("C) True vs Predicted Rain Rate")
ax.set_xlabel("True Rain Rate (mm/hr)")
ax.set_ylabel("Predicted Rain Rate (mm/hr)")
ax.legend()

# ---------------------------------------------------------
# D) Error Heatmap (spatial)
# ---------------------------------------------------------
ax = axes[1, 1]
sc = ax.scatter(df[lon_col], df[lat_col], c=df["error"], cmap="coolwarm", s=8)
ax.set_title("D) Spatial Error Map (Prediction - True)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.colorbar(sc, ax=ax, label="Error (mm/hr)")

# ---------------------------------------------------------
# Save figure
# ---------------------------------------------------------
output_name = "MT_Ensemble_Dashboard.png" if USE_ENSEMBLE else "MT_XGBoost_Dashboard.png"
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_name, dpi=300)
plt.show()

print(f"\nSaved dashboard → {output_name}")
