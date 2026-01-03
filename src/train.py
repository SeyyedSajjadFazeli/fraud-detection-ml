from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=1,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_res, y_res)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(model, "models/fraud_model.pkl")
print("✅ Model saved")
