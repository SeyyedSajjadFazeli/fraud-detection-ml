import sys
import os
import joblib
import numpy as np
from fastapi import FastAPI

# Fix path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: dict):
    """
    Expected JSON:
    {
        "features": [time, v1, v2, ..., v28, amount]
    }
    """
    features = transaction["features"]
    data = np.array([features])

    prob = model.predict_proba(data)[0][1]

    return {
        "fraud_probability": float(prob),
        "prediction": "Fraud" if prob >= 0.5 else "Normal"
    }
