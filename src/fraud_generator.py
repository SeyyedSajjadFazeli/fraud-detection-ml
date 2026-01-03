import numpy as np
import pandas as pd

def generate_fraud_samples(n_samples=5, random_state=42):
    np.random.seed(random_state)

    data = {}

    # Time
    data["Time"] = np.random.uniform(40000, 60000, n_samples)

    # V1 تا V28 (الگوی آماری تقلب)
    for i in range(1, 29):
        if i in [3, 10, 14, 17]:
            data[f"V{i}"] = np.random.uniform(-7, -3, n_samples)
        else:
            data[f"V{i}"] = np.random.uniform(-3, 3, n_samples)

    # Amount
    data["Amount"] = np.random.uniform(500, 3000, n_samples)

    return pd.DataFrame(data)
