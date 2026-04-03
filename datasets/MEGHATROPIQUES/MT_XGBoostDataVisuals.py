import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD CLEAN MEGHA-TROPIQUES CSV
# ============================================================

df = pd.read_csv("meghatropiques_clean.csv")
print(df.head())
print(df.shape)

# Extract unique sorted coordinate axes
lats = np.sort(df["latitude"].unique())
lons = np.sort(df["longitude"].unique())

H, W = len(lats), len(lons)

# Pivot into a 2D grid for rainfall
rain_grid = df.pivot(index="latitude", columns="longitude", values="rain_rate").values

# Missing data mask
missing_mask = np.isnan(rain_grid)

# ============================================================
# 2. SET STYLES
# ============================================================

plt.rcParams.update({"font.size": 10})
sns.set_style("whitegrid")

# ============================================================
# 3. CREATE DASHBOARD (3×2 panels)
# ============================================================

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

# ------------------------------------------------------------
# PANEL A — Rain Rate Heatmap
# ------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
imA = axA.imshow(rain_grid, cmap="turbo", origin="lower",
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                 aspect="auto")
axA.set_title("A) Rain Rate Field (Megha-Tropiques)", fontsize=12, fontweight="bold")
axA.set_xlabel("Longitude")
axA.set_ylabel("Latitude")
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

# ------------------------------------------------------------
# PANEL B — Scatter Map (Rainfall Points)
# ------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
sc = axB.scatter(df["longitude"], df["latitude"],
                 c=df["rain_rate"], cmap="turbo", s=5, alpha=0.6)
axB.set_title("B) Rainfall Scatter Map", fontsize=12, fontweight="bold")
axB.set_xlabel("Longitude")
axB.set_ylabel("Latitude")
fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)

# ------------------------------------------------------------
# PANEL C — Rain Rate Histogram (Log Scale)
# ------------------------------------------------------------
axC = fig.add_subplot(gs[1, 0])
sns.histplot(df["rain_rate"], bins=50, log_scale=True,
             color="darkblue", alpha=0.8, ax=axC)
axC.set_title("C) Rain Rate Distribution (log-scale)", fontsize=12, fontweight="bold")
axC.set_xlabel("Rain Rate (mm/hr)")
axC.grid(True, alpha=0.3)

# ------------------------------------------------------------
# PANEL D — Rain Rate vs Rain Flag (Quality Flag)
# ------------------------------------------------------------
axD = fig.add_subplot(gs[1, 1])
sns.boxplot(x=df["rain_flag"], y=df["rain_rate"], ax=axD)
axD.set_title("D) Rain Rate vs Rain Flag", fontsize=12, fontweight="bold")
axD.set_xlabel("Rain Flag")
axD.set_ylabel("Rain Rate (mm/hr)")

# ------------------------------------------------------------
# PANEL E — Rain Rate vs Surface Flag
# ------------------------------------------------------------
axE = fig.add_subplot(gs[2, 0])
sns.boxplot(x=df["surface_flag"], y=df["rain_rate"], ax=axE)
axE.set_title("E) Rain Rate vs Surface Flag", fontsize=12, fontweight="bold")
axE.set_xlabel("Surface Flag (0=Ocean, 1=Land)")
axE.set_ylabel("Rain Rate (mm/hr)")

# ------------------------------------------------------------
# PANEL F — Missing Data Map
# ------------------------------------------------------------
axF = fig.add_subplot(gs[2, 1])
imF = axF.imshow(missing_mask, cmap="Reds", origin="lower",
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                 aspect="auto")
axF.set_title("F) Missing Data Map (1 = Missing)", fontsize=12, fontweight="bold")
axF.set_xlabel("Longitude")
axF.set_ylabel("Latitude")
fig.colorbar(imF, ax=axF, fraction=0.046, pad=0.04)

# ------------------------------------------------------------
# FINALIZE FIGURE
# ------------------------------------------------------------

fig.suptitle(
    "Megha-Tropiques — Cleaned Dataset Dashboard",
    fontsize=16,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("MT_Clean_Dashboard.png", dpi=300)
plt.show()

print("Saved dashboard → MT_Clean_Dashboard.png")
