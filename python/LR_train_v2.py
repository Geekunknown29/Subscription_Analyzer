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
# Logistic Regression V2
# =========================

lr_v2 = LogisticRegression(
    max_iter=5000,
    random_state=42
)

lr_v2.fit(X_train_scaled, y_train)

print("Scaled Logistic Regression trained successfully")

# =========================
# Save Model
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\logistic_regression_v2.pkl",
    "wb"
) as file:
    pickle.dump(lr_v2, file)

print("Model saved successfully")

# =========================
# Save Scaler
# =========================

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\scaler.pkl",
    "wb"
) as file:
    pickle.dump(scaler, file)

print("Scaler saved successfully")

# =========================
# Load Test Data
# =========================

test_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv"
)

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# IMPORTANT:
# Use transform(), NOT fit_transform()

X_test_scaled = scaler.transform(X_test)

# =========================
# Predictions
# =========================

y_pred = lr_v2.predict(X_test_scaled)

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
