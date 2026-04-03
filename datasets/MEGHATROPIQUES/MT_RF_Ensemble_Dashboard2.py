import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# 1. LOAD DATA
# ============================================================

# RF + ExtraTrees + Ensemble file
df = pd.read_csv("megha_rf_ensemble_predictions.csv")

print(df.head())
print(df.shape)
print("Columns:", df.columns.tolist())

# Column names
lat_col = "latitude"
lon_col = "longitude"
true_col = "rain_rate"

# Use ensemble predictions here
pred_col = "ensemble_mean_pred"   # <-- key change; NOT 'xgb_rain_pred'

# Clean infinities and NaNs
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[lat_col, lon_col, true_col, pred_col])

# Extract arrays
lat = df[lat_col].values
lon = df[lon_col].values
true_rr = df[true_col].values
pred_rr = df[pred_col].values

# Error (ensemble - true)
err = pred_rr - true_rr

# Simple metrics
mae = np.mean(np.abs(err))
rmse = np.sqrt(np.mean(err**2))
bias = np.mean(err)
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"Bias = {bias:.4f}")

# ============================================================
# 2. DASHBOARD LAYOUT (2 x 3, all linear scales)
# ============================================================

fig = plt.figure(figsize=(18, 10))
fig.suptitle("Megha-Tropiques — RF + ExtraTrees Ensemble Dashboard",
             fontsize=18, fontweight="bold", y=0.98)

# ------------------------------------------------------------
# A) True Rain Rate Spatial Map
# ------------------------------------------------------------
ax1 = fig.add_subplot(2, 3, 1)
sc1 = ax1.scatter(lon, lat, c=true_rr,
                  cmap="Blues", s=8, edgecolor="none")
cb1 = plt.colorbar(sc1, ax=ax1)
cb1.set_label("Rain Rate (mm/hr)")

ax1.set_title("A) True Rain Rate Field", fontsize=12, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.grid(True, alpha=0.3)

# ------------------------------------------------------------
# B) Ensemble Predicted Rain Rate Spatial Map
# ------------------------------------------------------------
ax2 = fig.add_subplot(2, 3, 2)
sc2 = ax2.scatter(lon, lat, c=pred_rr,
                  cmap="viridis", s=8, edgecolor="none")
cb2 = plt.colorbar(sc2, ax=ax2)
cb2.set_label("Ensemble Predicted Rain (mm/hr)")

ax2.set_title("B) Ensemble Predicted Rain Rate", fontsize=12, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.grid(True, alpha=0.3)

# ------------------------------------------------------------
# C) True vs Ensemble Scatter Plot
# ------------------------------------------------------------
ax3 = fig.add_subplot(2, 3, 3)

ax3.scatter(true_rr, pred_rr,
            s=5, alpha=0.5, color="purple")
min_val = min(true_rr.min(), pred_rr.min())
max_val = max(true_rr.max(), pred_rr.max())
ax3.plot([min_val, max_val], [min_val, max_val],
         "r--", linewidth=1.2, label="Ideal = 1:1")

ax3.set_title("C) True vs Ensemble Predicted Rain", fontsize=12, fontweight="bold")
ax3.set_xlabel("True Rain Rate (mm/hr)")
ax3.set_ylabel("Ensemble Predicted (mm/hr)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ------------------------------------------------------------
# D) Spatial Error Map (Ensemble − True)
# ------------------------------------------------------------
ax4 = fig.add_subplot(2, 3, 4)

if np.any(~np.isnan(err)):
    vmax = np.max(np.abs(err))
else:
    vmax = 1.0

sc4 = ax4.scatter(lon, lat, c=err,
                  cmap="coolwarm", s=8, edgecolor="none",
                  norm=Normalize(vmin=-vmax, vmax=vmax))
cb4 = plt.colorbar(sc4, ax=ax4)
cb4.set_label("Error (mm/hr)")

ax4.set_title("D) Spatial Error Map (Ensemble − True)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")
ax4.grid(True, alpha=0.3)

# ------------------------------------------------------------
# E) Rain Rate Distribution (Linear Scale)
# ------------------------------------------------------------
ax5 = fig.add_subplot(2, 3, 5)

ax5.hist(true_rr, bins=40, alpha=0.7,
         color="steelblue", edgecolor="black")
ax5.hist(pred_rr, bins=40, alpha=0.5,
         color="orange", edgecolor="black")

ax5.set_title("E) Rain Rate Distribution (Linear Scale)", fontsize=12, fontweight="bold")
ax5.set_xlabel("Rain Rate (mm/hr)")
ax5.set_ylabel("Count")
ax5.legend(["True", "Ensemble Predicted"])
ax5.grid(True, alpha=0.3)

# ------------------------------------------------------------
# F) Error Distribution Histogram
# ------------------------------------------------------------
ax6 = fig.add_subplot(2, 3, 6)

ax6.hist(err, bins=40, alpha=0.7,
         color="crimson", edgecolor="black")

ax6.set_title("F) Error Distribution (Ensemble − True)", fontsize=12, fontweight="bold")
ax6.set_xlabel("Error (mm/hr)")
ax6.set_ylabel("Count")
ax6.grid(True, alpha=0.3)

# ------------------------------------------------------------
# Final layout
# ------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("MT_RF_Ensemble_Dashboard2.png", dpi=300)
plt.show()

print("\nSaved dashboard → MT_RF_Ensemble_Dashboard2.png")
