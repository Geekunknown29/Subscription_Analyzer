import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\bank_processed.csv"
)

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Recombine X and y

train_df = X_train.copy()
train_df["y"] = y_train

test_df = X_test.copy()
test_df["y"] = y_test

# Save files

train_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\train_data.csv",
    index=False
)

test_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\test_data.csv",
    index=False
)

print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

print("Files Saved Successfully")
