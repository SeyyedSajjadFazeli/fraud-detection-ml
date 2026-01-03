import shap
import joblib
import pandas as pd
from src.preprocessing import load_and_preprocess

model = joblib.load("models/fraud_model.pkl")

X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
