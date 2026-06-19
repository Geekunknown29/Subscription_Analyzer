import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

# Load training data

train_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\train_data.csv"
)

# Separate features and target

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

# Create model

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model

lr.fit(X_train, y_train)

print("Logistic Regression trained successfully")

# Save model

model_path = (
    r"D:\Projects\Omdena\bank+marketing\bank\logistic_regression.pkl"
)

with open(model_path, "wb") as file:
    pickle.dump(lr, file)

print(f"Model saved successfully:\n{model_path}")
