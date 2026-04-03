import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. Load predictions (very light)
# =====================================
print("Loading predictions CSV...")
df = pd.read_csv("nisar_pixel_offset_predictions_rf.csv")
print("Loaded predictions:", df.shape)
print("Columns:", df.columns.tolist())

# Assume two columns: [true, pred]
# Adjust these if your column names differ
true = df.iloc[:, 0].values   # e.g. "true_value"
pred = df.iloc[:, 1].values   # e.g. "rf_pred"

# Residuals
residuals = true - pred

# =====================================
# 2. Create a lightweight 3-panel figure
# =====================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax1, ax2, ax3 = axes

# -------------------------------------
# Panel 1: True vs Predicted (sampled)
# -------------------------------------
# Subsample for speed (e.g. max 5000 points)
max_points = 5000
if len(true) > max_points:
    idx = np.random.choice(len(true), size=max_points, replace=False)
    true_s = true[idx]
    pred_s = pred[idx]
else:
    true_s = true
    pred_s = pred

ax1.scatter(true_s, pred_s, s=5, alpha=0.4)
min_val = min(true_s.min(), pred_s.min())
max_val = max(true_s.max(), pred_s.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
ax1.set_title("True vs Predicted", fontsize=12)
ax1.set_xlabel("True Value")
ax1.set_ylabel("Predicted Value")

# -------------------------------------
# Panel 2: Residual Histogram
# -------------------------------------
ax2.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=1)
ax2.set_title("Residuals Histogram", fontsize=12)
ax2.set_xlabel("Residual (True - Pred)")
ax2.set_ylabel("Count")

# -------------------------------------
# Panel 3: True vs Predicted Distributions (1D)
# -------------------------------------
# Use simple histograms (no KDE for speed)
ax3.hist(true, bins=40, alpha=0.6, label="True", edgecolor="black")
ax3.hist(pred, bins=40, alpha=0.6, label="Predicted", edgecolor="black")
ax3.set_title("Distribution: True vs Pred", fontsize=12)
ax3.set_xlabel("Value")
ax3.set_ylabel("Count")
ax3.legend()

# -------------------------------------
# Final layout
# -------------------------------------
plt.tight_layout()
plt.savefig("NISAR_RF_Model_Dashboard_Lite.png", dpi=200)
plt.show()

print("\nSaved → NISAR_RF_Model_Dashboard_Lite.png")
