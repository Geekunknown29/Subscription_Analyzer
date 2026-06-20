# XGBoost V1
# Goal:
# Compare XGBoost against Logistic Regression and Random Forest
# Default threshold = 0.5
# Threshold tuning will be performed after training

import pandas as pd
import pickle

from xgboost import XGBClassifier

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
# Predictions
# =========================

y_pred = xgb_v1.predict(X_test)

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
