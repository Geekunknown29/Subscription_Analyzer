import pandas as pd

df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\bank-full.csv",
    sep=";"
)

# Data Preparation
model_df = df.copy()

if "age_group" in model_df.columns:
    model_df.drop("age_group", axis=1, inplace=True)

model_df["y"] = model_df["y"].map({
    "yes": 1,
    "no": 0
})

model_df = pd.get_dummies(
    model_df,
    drop_first=True,
    dtype=int
)

# Save processed dataset
output_path = r"D:\Projects\Omdena\bank+marketing\bank\bank_processed.csv"

model_df.to_csv(
    output_path,
    index=False
)

print(f"Processed dataset saved to:\n{output_path}")

X = model_df.drop("y", axis=1)
y = model_df["y"]

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

print("\nFirst 5 rows of X:")
print(X.head())
