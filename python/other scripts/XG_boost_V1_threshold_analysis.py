# XGBoost V1
# Threshold Analysis
# Goal:
# Compare XGBoost against Logistic Regression and Random Forest
# Find the best threshold for subscriber detection

import pandas as pd
import pickle

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# Load Train Data
# =========================

train_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\train_data.csv"
)

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

# =========================
# XGBoost V1
# =========================

xgb_v1 = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_v1.fit(X_train, y_train)

print("XGBoost V1 trained successfully")

# =========================
# Save Model
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\xgboost_v1.pkl",
    "wb"
) as file:
    pickle.dump(xgb_v1, file)

print("Model saved successfully")

# =========================
# Load Test Data
# =========================

test_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv"
)

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# =========================
# Probability Predictions
# =========================

y_prob = xgb_v1.predict_proba(X_test)[:, 1]

# =========================
# Threshold Analysis
# =========================

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
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
# Save Threshold Results
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
    r"D:\Projects\Omdena\bank+marketing\bank\xgboost_threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis saved successfully.")
