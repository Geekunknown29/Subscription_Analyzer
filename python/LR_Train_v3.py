import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
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
# Scale Features
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# =========================
# Logistic Regression V3
# =========================

lr_v3 = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight="balanced"
)

lr_v3.fit(X_train_scaled, y_train)

print("Balanced Logistic Regression trained successfully")

# =========================
# Save Model
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\logistic_regression_v3.pkl",
    "wb"
) as file:
    pickle.dump(lr_v3, file)

# =========================
# Save Scaler
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\scaler_v3.pkl",
    "wb"
) as file:
    pickle.dump(scaler, file)

# =========================
# Load Test Data
# =========================

test_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv"
)

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# =========================
# Scale Test Data
# =========================

X_test_scaled = scaler.transform(X_test)

# =========================
# Predictions
# =========================



# Probability of Subscription

y_prob = lr_v3.predict_proba(X_test_scaled)[:, 1]

# =========================
# Metrics
# =========================

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Build Analysis DataFrame
# =========================

analysis_df = test_df.copy()

analysis_df["actual"] = y_test
analysis_df["predicted"] = y_pred
analysis_df["probability_yes"] = y_prob

# =========================
# TP / TN / FP / FN
# =========================

tp_df = analysis_df[
    (analysis_df["actual"] == 1) &
    (analysis_df["predicted"] == 1)
]

tn_df = analysis_df[
    (analysis_df["actual"] == 0) &
    (analysis_df["predicted"] == 0)
]

fp_df = analysis_df[
    (analysis_df["actual"] == 0) &
    (analysis_df["predicted"] == 1)
]

fn_df = analysis_df[
    (analysis_df["actual"] == 1) &
    (analysis_df["predicted"] == 0)
]

# =========================
# Save Files
# =========================

tp_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\true_positives.csv",
    index=False
)

tn_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\true_negatives.csv",
    index=False
)

fp_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\false_positives.csv",
    index=False
)

fn_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\false_negatives.csv",
    index=False
)

print("\nFiles Saved Successfully")

print("TP:", len(tp_df))
print("TN:", len(tn_df))
print("FP:", len(fp_df))
print("FN:", len(fn_df))
