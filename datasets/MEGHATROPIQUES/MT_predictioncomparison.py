import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("meghatropiques_with_predictions.csv")

plt.figure(figsize=(5,5))
plt.scatter(df["rain_rate"], df["xgb_rain_pred"], s=3, alpha=0.4)
plt.plot([df["rain_rate"].min(), df["rain_rate"].max()],
         [df["rain_rate"].min(), df["rain_rate"].max()],
         "r--", linewidth=1)
plt.xlabel("True Rain Rate")
plt.ylabel("XGBoost Predicted Rain Rate")
plt.title("True vs Predicted Rain (XGBoost)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
