# ======================================
# FIX PYTHON PATH (CRITICAL FOR STREAMLIT)
# ======================================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================
# IMPORTS
# ======================================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from src.fraud_generator import generate_fraud_samples
from src.preprocessing import load_and_preprocess

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# ======================================
# LOAD MODEL
# ======================================
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

# ======================================
# TITLE
# ======================================
st.title("💳 Credit Card Fraud Detection System")
st.write("End-to-End Machine Learning | Streamlit Dashboard")

# ======================================
# THRESHOLD SLIDER
# ======================================
st.subheader("⚖️ Decision Threshold")

threshold = st.slider(
    "Classification Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.info(f"Current Threshold: {threshold}")

# ======================================
# MANUAL INPUT
# ======================================
st.subheader("🧾 Manual Transaction Input")

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

st.markdown("### PCA Features (V1 - V28)")
features = []

cols = st.columns(4)
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        features.append(st.number_input(f"V{i}", value=0.0))

input_data = np.array([[time] + features + [amount]])

if st.button("🔍 Predict Transaction"):
    prob = model.predict_proba(input_data)[0][1]
    st.markdown(f"### Fraud Probability: **{prob:.4f}**")

    if prob >= threshold:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Normal Transaction")

# ======================================
# AUTO FRAUD GENERATOR
# ======================================
st.divider()
st.subheader("🤖 Auto-Generated Fraud Transactions")

if st.button("⚙️ Generate Fraud Transactions"):
    fraud_df = generate_fraud_samples(n_samples=5)

    probs = model.predict_proba(fraud_df)[:, 1]

    fraud_df["Fraud_Probability"] = probs
    fraud_df["Prediction"] = np.where(
        probs >= threshold,
        "Fraud",
        "Normal"
    )

    st.dataframe(fraud_df, width="stretch")

# ======================================
# CONFUSION MATRIX
# ======================================
st.divider()
st.subheader("📊 Confusion Matrix (Test Set)")

@st.cache_data
def get_test_data():
    _, X_test, _, y_test = load_and_preprocess("data/creditcard.csv")
    return X_test, y_test

X_test, y_test = get_test_data()
y_test_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= threshold).astype(int)

cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# ======================================
# FOOTER
# ======================================
st.divider()
st.caption(
    "Built with Python, XGBoost, Streamlit | "
    "Production-Ready Fraud Detection System"
)
