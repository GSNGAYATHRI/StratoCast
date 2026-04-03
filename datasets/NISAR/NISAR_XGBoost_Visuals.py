import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. Load XGBoost predictions
# =====================================
print("Loading XGBoost prediction CSV...")
df = pd.read_csv("nisar_pixel_offset_predictions_xgb.csv")
print("Loaded predictions:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# If your columns are named differently, adjust these two lines.
# Assuming columns: ["true_value", "xgb_pred"]
true = df.iloc[:, 0].values   # or df["true_value"].values
pred = df.iloc[:, 1].values   # or df["xgb_pred"].values

# Residuals = True - Predicted
residuals = true - pred

# =====================================
# 2. Create light 3-panel figure
# =====================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax1, ax2, ax3 = axes

# -------------------------------------
# Panel 1: True vs Predicted (sampled)
# -------------------------------------
max_points = 5000  # limit points for speed
n = len(true)
if n > max_points:
    idx = np.random.choice(n, size=max_points, replace=False)
    true_s = true[idx]
    pred_s = pred[idx]
else:
    true_s = true
    pred_s = pred

ax1.scatter(true_s, pred_s, s=5, alpha=0.4)
min_val = min(true_s.min(), pred_s.min())
max_val = max(true_s.max(), pred_s.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
ax1.set_title("XGBoost: True vs Predicted", fontsize=12)
ax1.set_xlabel("True Value")
ax1.set_ylabel("Predicted Value")

# -------------------------------------
# Panel 2: Residual Histogram
# -------------------------------------
ax2.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=1)
ax2.set_title("Residuals Histogram (True - Pred)", fontsize=12)
ax2.set_xlabel("Residual")
ax2.set_ylabel("Count")

# -------------------------------------
# Panel 3: True vs Predicted Distributions
# -------------------------------------
ax3.hist(true, bins=40, alpha=0.6, label="True", edgecolor="black")
ax3.hist(pred, bins=40, alpha=0.6, label="XGBoost Pred", edgecolor="black")
ax3.set_title("Value Distribution: True vs XGBoost", fontsize=12)
ax3.set_xlabel("Value")
ax3.set_ylabel("Count")
ax3.legend()

# -------------------------------------
# Final layout
# -------------------------------------
plt.tight_layout()
plt.savefig("NISAR_XGB_Model_Dashboard_Lite.png", dpi=200)
plt.show()

print("\nSaved → NISAR_XGB_Model_Dashboard_Lite.png")
