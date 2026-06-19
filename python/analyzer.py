import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\bank-full.csv",
    sep=";"
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["y"].value_counts())

print("\nClass Distribution (%):")
print(df["y"].value_counts(normalize=True) * 100)

# Q1 - Which job types have the highest subscription rate?

job_sub_rate = (
    df.groupby("job")["y"]
      .apply(lambda x: (x == "yes").mean() * 100)
      .sort_values(ascending=False)
)

print("\nSubscription Rate by Job:")
print(job_sub_rate)

job_sub_rate.plot(kind="bar", figsize=(10, 5))
plt.title("Subscription Rate by Job Category")
plt.ylabel("Subscription Rate (%)")
plt.xlabel("Job")
plt.tight_layout()
plt.show()

# Q2 - Is there a pattern between account balance and likelihood to subscribe?

balance_analysis = df.groupby("y")["balance"].mean()

print("\nAverage Balance by Subscription Status:")
print(balance_analysis)

balance_analysis.plot(kind="bar", figsize=(6, 4))
plt.title("Average Balance by Subscription Status")
plt.ylabel("Average Balance")
plt.xlabel("Subscription")
plt.tight_layout()
plt.show()

# Q3 - How does subscription rate differ across age groups?

df["age_group"] = pd.cut(
    df["age"],
    bins=[18, 30, 45, 60, 100],
    labels=["18-30", "31-45", "46-60", "60+"]
)

age_sub_rate = (
    df.groupby("age_group", observed=False)["y"]
      .apply(lambda x: (x == "yes").mean() * 100)
)

print("\nSubscription Rate by Age Group:")
print(age_sub_rate)

age_sub_rate.plot(kind="bar", figsize=(6, 4))
plt.title("Subscription Rate by Age Group")
plt.ylabel("Subscription Rate (%)")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()

# Q4 - Does having an existing housing loan affect subscription rate?

housing_sub_rate = (
    df.groupby("housing")["y"]
      .apply(lambda x: (x == "yes").mean() * 100)
)

print("\nSubscription Rate by Housing Loan:")
print(housing_sub_rate)

housing_sub_rate.plot(kind="bar", figsize=(6, 4))
plt.title("Housing Loan vs Subscription Rate")
plt.ylabel("Subscription Rate (%)")
plt.xlabel("Housing Loan")
plt.tight_layout()
plt.show()
