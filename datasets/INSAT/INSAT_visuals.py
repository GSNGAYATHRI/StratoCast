import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. LOAD PREDICTIONS CSV
# ===============================

df = pd.read_csv("GPI_with_predictions.csv")
print(df.head())

# Extract sorted coordinate grids
lats = np.sort(df["latitude"].unique())
lons = np.sort(df["longitude"].unique())

H, W = len(lats), len(lons)

# Pivot to 2D grids
GPI_true = df.pivot(index="latitude", columns="longitude", values="GPI").values
GPI_pred = df.pivot(index="latitude", columns="longitude", values="predicted_GPI").values
GPI_err  = GPI_true - GPI_pred

# ===============================
# 2. PLOT DASHBOARD
# ===============================

plt.rcParams.update({"font.size": 10})

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.35)

# -------------------------------------
# A) TRUE GPI HEATMAP
# -------------------------------------
axA = fig.add_subplot(gs[0,0])
imA = axA.imshow(GPI_true, cmap="turbo", origin="lower",
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()])
axA.set_title("A) True GPI (INSAT)", fontsize=12, fontweight="bold")
axA.set_xlabel("Longitude")
axA.set_ylabel("Latitude")
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

# -------------------------------------
# B) PREDICTED GPI HEATMAP
# -------------------------------------
axB = fig.add_subplot(gs[0,1])
imB = axB.imshow(GPI_pred, cmap="turbo", origin="lower",
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()])
axB.set_title("B) Predicted GPI (XGBoost)", fontsize=12, fontweight="bold")
axB.set_xlabel("Longitude")
axB.set_ylabel("Latitude")
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

# -------------------------------------
# C) RESIDUAL MAP (ERROR FIELD)
# -------------------------------------
axC = fig.add_subplot(gs[0,2])
imC = axC.imshow(GPI_err, cmap="coolwarm", origin="lower",
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                 vmin=-np.max(abs(GPI_err)), vmax=np.max(abs(GPI_err)))
axC.set_title("C) Error Map (True - Predicted)", fontsize=12, fontweight="bold")
axC.set_xlabel("Longitude")
axC.set_ylabel("Latitude")
fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)

# -------------------------------------
# D) TRUE vs PREDICTED SCATTER
# -------------------------------------
axD = fig.add_subplot(gs[1,0])
sns.scatterplot(x=df["GPI"], y=df["predicted_GPI"], s=8, alpha=0.4, ax=axD)
axD.plot([df["GPI"].min(), df["GPI"].max()],
         [df["GPI"].min(), df["GPI"].max()],
         'r--', linewidth=1)
axD.set_title("D) Scatter: True vs Predicted GPI", fontsize=12, fontweight="bold")
axD.set_xlabel("True GPI")
axD.set_ylabel("Predicted GPI")
axD.grid(True, alpha=0.3)

# -------------------------------------
# E) DISTRIBUTION HISTOGRAM
# -------------------------------------
axE = fig.add_subplot(gs[1,1])
sns.kdeplot(df["GPI"], label="True GPI", fill=True, linewidth=1.5, ax=axE)
sns.kdeplot(df["predicted_GPI"], label="Predicted GPI", fill=True, linewidth=1.5, ax=axE)
axE.set_title("E) Distribution of True vs Predicted GPI", fontsize=12, fontweight="bold")
axE.set_xlabel("GPI Value")
axE.legend()

# -------------------------------------
# F) ERROR DISTRIBUTION
# -------------------------------------
axF = fig.add_subplot(gs[1,2])
sns.histplot(df["residual"], bins=40, kde=True, color="purple", ax=axF)
axF.set_title("F) Prediction Error Distribution", fontsize=12, fontweight="bold")
axF.set_xlabel("Residual (True - Predicted)")

# -------------------------------------
# G) CORRELATION OF LAT/LON GRADIENTS (Spatial)
# -------------------------------------
# Compute gradients in latitude & longitude
dlat = np.gradient(GPI_true, axis=0)
dlon = np.gradient(GPI_true, axis=1)

spatial_df = pd.DataFrame({
    "GPI": GPI_true.flatten(),
    "dGPI_dLat": dlat.flatten(),
    "dGPI_dLon": dlon.flatten(),
})

corr = spatial_df.corr()

axG = fig.add_subplot(gs[2,:])
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axG)
axG.set_title("G) Spatial Correlation Matrix (Gradient-Based)", fontsize=12, fontweight="bold")

# -------------------------------------
# FINISH
# -------------------------------------

fig.suptitle("INSAT GPI Data Dashboard (Cleaned + XGBoost Predictions)",
             fontsize=16, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("INSAT_GPI_Dashboard.png", dpi=300)
plt.show()
