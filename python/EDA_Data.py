# EDA for Bank Marketing Dataset
# Generates all plots and statistics required for Track B
# Saves results to results/eda/

import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Create Results Folder
# =========================

os.makedirs(
    r"D:\Projects\bankmind-Arpit\results\eda",
    exist_ok=True
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    r"D:\Projects\bankmind-Arpit\data\bank-full.csv",
    sep=";"
)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# =========================
# Missing Values
# =========================

print("\nMissing Values:")
print(df.isnull().sum())

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

missing_df.to_csv(
    r"D:\Projects\bankmind-Arpit\results\eda\missing_values.csv",
    index=False
)

# =========================
# Class Distribution
# =========================

print("\nClass Distribution:")
print(df["y"].value_counts())

print("\nClass Distribution Percentage:")
print(
    df["y"].value_counts(normalize=True) * 100
)

class_dist = (
    df["y"]
    .value_counts(normalize=True)
    * 100
)

plt.figure(figsize=(6, 4))
class_dist.plot(kind="bar")
plt.title("Class Distribution (%)")
plt.ylabel("Percentage")
plt.tight_layout()

plt.savefig(
    r"D:\Projects\bankmind-Arpit\results\eda\class_distribution.png"
)

plt.close()

# =========================
# Job Subscription Rate
# =========================

job_rate = (
    df.groupby("job")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nSubscription Rate By Job:")
print(job_rate)

job_rate.to_csv(
    r"D:\Projects\bankmind-Arpit\results\eda\job_subscription_rate.csv"
)

plt.figure(figsize=(10, 5))
job_rate.plot(kind="bar")
plt.title("Subscription Rate by Job")
plt.ylabel("Subscription Rate (%)")
plt.tight_layout()

plt.savefig(
    r"D:\Projects\bankmind-Arpit\results\eda\job_subscription_rate.png"
)

plt.close()

# =========================
# Balance vs Subscription
# =========================

balance_analysis = (
    df.groupby("y")["balance"]
    .mean()
)

print("\nAverage Balance by Subscription:")
print(balance_analysis)

balance_analysis.to_csv(
    r"D:\Projects\bankmind-Arpit\results\eda\balance_subscription.csv"
)

plt.figure(figsize=(6, 4))
balance_analysis.plot(kind="bar")
plt.title("Average Balance by Subscription")
plt.ylabel("Average Balance")
plt.tight_layout()

plt.savefig(
    r"D:\Projects\bankmind-Arpit\results\eda\balance_subscription.png"
)

plt.close()

# =========================
# Age Groups
# =========================

df["age_group"] = pd.cut(
    df["age"],
    bins=[18, 30, 45, 60, 100],
    labels=[
        "18-30",
        "31-45",
        "46-60",
        "60+"
    ]
)

age_rate = (
    df.groupby("age_group")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
)

print("\nSubscription Rate by Age Group:")
print(age_rate)

age_rate.to_csv(
    r"D:\Projects\bankmind-Arpit\results\eda\age_group_subscription.csv"
)

plt.figure(figsize=(6, 4))
age_rate.plot(kind="bar")
plt.title("Subscription Rate by Age Group")
plt.ylabel("Subscription Rate (%)")
plt.tight_layout()

plt.savefig(
    r"D:\Projects\bankmind-Arpit\results\eda\age_group_subscription.png"
)

plt.close()

# =========================
# Housing Loan Analysis
# =========================

housing_rate = (
    df.groupby("housing")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
)

print("\nSubscription Rate by Housing Loan:")
print(housing_rate)

housing_rate.to_csv(
    r"D:\Projects\bankmind-Arpit\results\eda\housing_subscription.csv"
)

plt.figure(figsize=(6, 4))
housing_rate.plot(kind="bar")
plt.title("Subscription Rate by Housing Loan")
plt.ylabel("Subscription Rate (%)")
plt.tight_layout()

plt.savefig(
    r"D:\Projects\bankmind-Arpit\results\eda\housing_subscription.png"
)

plt.close()

print("\nEDA completed successfully.")
print(
    r"Results saved in D:\Projects\bankmind-Arpit\results\eda"
)
