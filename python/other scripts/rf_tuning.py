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
# Load Random Forest Model
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\random_forest_v2.pkl",
    "rb"
) as file:
    rf = pickle.load(file)

# =========================
# Subscription Probabilities
# =========================

y_prob = rf.predict_proba(X_test)[:, 1]

# =========================
# Threshold Analysis
# =========================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"Threshold = {threshold}")

    print(f"TP = {tp}")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")

    print(f"Accuracy  = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1 Score  = {f1:.4f}")

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
# Save Results
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

results_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\rf_threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis saved successfully.")
