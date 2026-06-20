# CatBoost Final Model Feature Importance
# Train final CatBoost model and identify most important features.
# Used for business insights and final project explanation.

import pandas as pd

from catboost import CatBoostClassifier

# =========================
# Load Training Data
# =========================

train_df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\train_data.csv"
)

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

# =========================
# Final CatBoost Model
# =========================

model = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    class_weights=[1, 7],
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42,
    verbose=0
)

model.fit(
    X_train,
    y_train
)

print("Final CatBoost model trained successfully")

# =========================
# Feature Importance
# =========================

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Most Important Features:\n")
print(feature_importance.head(20))

# =========================
# Save Results
# =========================

feature_importance.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\catboost_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully")
