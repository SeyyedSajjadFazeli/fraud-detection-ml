import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from preprocessing import load_and_preprocess

model = joblib.load("models/fraud_model.pkl")
X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")

y_proba = model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
