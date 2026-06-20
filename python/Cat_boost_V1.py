# Final CatBoost Model
# Uses best hyperparameters and threshold = 0.7
# Saves final model and 5 representative customer predictions

import pandas as pd
import pickle
import os

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# Create Folders if Needed
# =========================

os.makedirs(
    r"D:\Projects\bankmind-Arpit\models",
    exist_ok=True
)

os.makedirs(
    r"D:\Projects\bankmind-Arpit\results",
    exist_ok=True
)

# =========================
# Load Train Data
# =========================

train_df = pd.read_csv(
    r"D:\Projects\bankmind-Arpit\data\train_data.csv"
)

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

# =========================
# Final CatBoost
# =========================

cat_final = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    class_weights=[1, 7],
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42,
    verbose=0
)

cat_final.fit(
    X_train,
    y_train
)

print("Final CatBoost trained successfully")

# =========================
# Save Model
# =========================

with open(
    r"D:\Projects\bankmind-Arpit\models\final_catboost.pkl",
    "wb"
) as file:
    pickle.dump(cat_final, file)

print("Model saved successfully")

# =========================
# Load Test Data
# =========================

test_df = pd.read_csv(
    r"D:\Projects\bankmind-Arpit\data\test_data.csv"
)

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# =========================
# Predict Probabilities
# =========================

y_prob = cat_final.predict_proba(X_test)[:, 1]

# =========================
# Apply Threshold
# =========================

THRESHOLD = 0.7

y_pred = (
    y_prob >= THRESHOLD
).astype(int)

# =========================
# Metrics
# =========================

print("\nThreshold =", THRESHOLD)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Build Prediction Table
# =========================

results_df = X_test.copy()

results_df["Actual"] = y_test.values
results_df["Probability"] = y_prob
results_df["Prediction"] = y_pred

# =========================
# Select 5 Customers
# =========================

# 2 strongest YES predictions

strong_yes = (
    results_df[results_df["Prediction"] == 1]
    .sort_values(
        by="Probability",
        ascending=False
    )
    .head(2)
)

# 2 strongest NO predictions

strong_no = (
    results_df[results_df["Prediction"] == 0]
    .sort_values(
        by="Probability",
        ascending=True
    )
    .head(2)
)

# 1 borderline customer closest to threshold

borderline = (
    results_df.assign(
        Distance=abs(
            results_df["Probability"] - THRESHOLD
        )
    )
    .sort_values(
        by="Distance"
    )
    .head(1)
    .drop(
        columns=["Distance"]
    )
)

# =========================
# Combine Customers
# =========================

sample_customers = pd.concat(
    [
        strong_yes,
        strong_no,
        borderline
    ]
)

# =========================
# Save Sample Predictions
# =========================

sample_customers.to_csv(
    r"D:\Projects\bankmind-Arpit\results\sample_predictions.csv",
    index=True
)

print("\n5 sample customers saved successfully.")
print(
    "File saved at: "
    r"D:\Projects\bankmind-Arpit\results\sample_predictions.csv"
)

# =========================
# Preview Customers
# =========================

print("\nSelected Customers:")
print(
    sample_customers[
        [
            "Actual",
            "Probability",
            "Prediction"
        ]
    ]
)
