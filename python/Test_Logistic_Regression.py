import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load test data

test_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv"
)

# Separate features and target

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# Load saved model

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\logistic_regression.pkl",
    "rb"
) as file:
    lr = pickle.load(file)

# Predictions

y_pred = lr.predict(X_test)

# Metrics

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
