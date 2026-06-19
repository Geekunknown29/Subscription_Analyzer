import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# Load Test Data
# =========================

test_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv"
)

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# =========================
# Load Scaler
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\scaler_v3.pkl",
    "rb"
) as file:
    scaler = pickle.load(file)

# =========================
# Load Model
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\logistic_regression_v3.pkl",
    "rb"
) as file:
    lr_v3 = pickle.load(file)

# =========================
# Scale Test Data
# =========================

X_test_scaled = scaler.transform(X_test)

# =========================
# Subscription Probabilities
# =========================

y_prob = lr_v3.predict_proba(X_test_scaled)[:, 1]

# =========================
# Threshold Analysis
# =========================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([
        threshold,
        tp,
        tn,
        fp,
        fn,
        accuracy,
        precision,
        recall,
        f1
    ])

# =========================
# Results Table
# =========================

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print(results_df)

# =========================
# Save Results
# =========================

results_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis saved successfully.")
