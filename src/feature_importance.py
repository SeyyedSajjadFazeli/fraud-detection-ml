import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/fraud_model.pkl")

feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
importances = model.feature_importances_

fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(fi["Feature"][:15], fi["Importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
