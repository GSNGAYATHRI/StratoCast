import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("megha_rf_ensemble_predictions.csv")
print(df.head())
print(df.shape)

# Column names (adjust here if different)
lat_col = "latitude"
lon_col = "longitude"
true_col = "rain_rate"
ens_col = "ensemble_pred"

# Drop rows with missing values in critical columns
df = df.dropna(subset=[lat_col, lon_col, true_col, ens_col])

# Create error column
df["ensemble_error"] = df[ens_col] - df[true_col]

# ============================================================
# 2. STYLE
# ============================================================

plt.rcParams.update({"font.size": 10})
sns.set_style("whitegrid")

# ============================================================
# 3. DASHBOARD (3 × 2 panels)
# ============================================================

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# ------------------------------------------------------------
# PANEL A — True Rain Scatter Map
# ------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
scA = axA.scatter(
    df[lon_col], df[lat_col],
    c=df[true_col],
    cmap="turbo",
    s=5,
    alpha=0.6
)
axA.set_title("A) True Rain Rate (Megha-Tropiques)", fontsize=12, fontweight="bold")
axA.set_xlabel("Longitude")
axA.set_ylabel("Latitude")
cbarA = fig.colorbar(scA, ax=axA, fraction=0.046, pad=0.04)
cbarA.set_label("Rain Rate (mm/hr)")

# ------------------------------------------------------------
# PANEL B — Ensemble Predicted Rain Scatter Map
# ------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
scB = axB.scatter(
    df[lon_col], df[lat_col],
    c=df[ens_col],
    cmap="turbo",
    s=5,
    alpha=0.6
)
axB.set_title("B) Ensemble Predicted Rain", fontsize=12, fontweight="bold")
axB.set_xlabel("Longitude")
axB.set_ylabel("Latitude")
cbarB = fig.colorbar(scB, ax=axB, fraction=0.046, pad=0.04)
cbarB.set_label("Predicted Rain (mm/hr)")

# ------------------------------------------------------------
# PANEL C — Spatial Error Map
# ------------------------------------------------------------
axC = fig.add_subplot(gs[1, 0])
vmax = np.nanmax(np.abs(df["ensemble_error"])) if df["ensemble_error"].notna().any() else 1.0
scC = axC.scatter(
    df[lon_col], df[lat_col],
    c=df["ensemble_error"],
    cmap="coolwarm",
    s=5,
    alpha=0.7,
    vmin=-vmax,
    vmax=vmax
)
axC.set_title("C) Error Map (Ensemble − True)", fontsize=12, fontweight="bold")
axC.set_xlabel("Longitude")
axC.set_ylabel("Latitude")
cbarC = fig.colorbar(scC, ax=axC, fraction=0.046, pad=0.04)
cbarC.set_label("Error (mm/hr)")

# ------------------------------------------------------------
# PANEL D — True vs Ensemble Scatter
# ------------------------------------------------------------
axD = fig.add_subplot(gs[1, 1])
sns.scatterplot(
    x=df[true_col],
    y=df[ens_col],
    s=8,
    alpha=0.4,
    ax=axD
)
min_val = min(df[true_col].min(), df[ens_col].min())
max_val = max(df[true_col].max(), df[ens_col].max())
axD.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
axD.set_title("D) True vs Ensemble Prediction", fontsize=12, fontweight="bold")
axD.set_xlabel("True Rain Rate (mm/hr)")
axD.set_ylabel("Predicted Rain Rate (mm/hr)")
axD.grid(True, alpha=0.3)

# ------------------------------------------------------------
# PANEL E — Rain Rate Distribution (linear scale)
# ------------------------------------------------------------
axE = fig.add_subplot(gs[2, 0])
sns.kdeplot(
    df[true_col],
    fill=True,
    label="True Rain",
    linewidth=2,
    alpha=0.5,
    ax=axE,
)
sns.kdeplot(
    df[ens_col],
    fill=True,
    label="Ensemble Predicted",
    linewidth=2,
    alpha=0.5,
    ax=axE,
)
axE.set_title("E) Rain Rate Distribution (linear scale)", fontsize=12, fontweight="bold")
axE.set_xlabel("Rain Rate (mm/hr)")
axE.legend()

# ------------------------------------------------------------
# PANEL F — Error Distribution Histogram
# ------------------------------------------------------------
axF = fig.add_subplot(gs[2, 1])
sns.histplot(
    df["ensemble_error"],
    bins=50,
    kde=True,
    color="purple",
    ax=axF
)
axF.set_title("F) Ensemble Error Distribution", fontsize=12, fontweight="bold")
axF.set_xlabel("Error (mm/hr)")
axF.grid(True, alpha=0.3)

# ============================================================
# 4. FINALIZE FIGURE
# ============================================================

fig.suptitle(
    "Megha-Tropiques — RF + ExtraTrees Ensemble Dashboard",
    fontsize=16,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("MT_RF_Ensemble_Dashboard.png", dpi=300)
plt.show()

print("Saved dashboard → MT_RF_Ensemble_Dashboard.png")
