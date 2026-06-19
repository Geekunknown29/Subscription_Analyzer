import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load train data

train_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\train_data.csv"
)

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

# Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Random Forest trained successfully")

# Save model

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\random_forest.pkl",
    "wb"
) as file:
    pickle.dump(rf, file)

print("Model saved successfully")

# Load test data

test_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv"
)

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# Predictions

y_pred = rf.predict(X_test)

# Metrics

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
