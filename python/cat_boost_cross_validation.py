# Final CatBoost Model + 5-Fold Cross Validation
# Uses best CatBoost parameters and threshold = 0.7
# Saves final model and cross-validation results

import pandas as pd
import numpy as np
import pickle

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# Load Train Data
# =========================

df = pd.read_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\train_data.csv"
)

X = df.drop("y", axis=1)
y = df["y"]

# =========================
# Final Parameters
# =========================

THRESHOLD = 0.7

CAT_PARAMS = {
    "iterations": 500,
    "depth": 5,
    "learning_rate": 0.03,
    "class_weights": [1, 7],
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": 42,
    "verbose": 0
}

# =========================
# Train Final Model
# =========================

final_model = CatBoostClassifier(**CAT_PARAMS)

final_model.fit(
    X,
    y
)

with open(
    r"D:\Projects\Omdena\bank+marketing\bank\final_catboost.pkl",
    "wb"
) as file:
    pickle.dump(final_model, file)

print("Final CatBoost model saved successfully")

# =========================
# 5 Fold Cross Validation
# =========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

tp_scores = []
tn_scores = []
fp_scores = []
fn_scores = []

fold = 1

for train_idx, val_idx in skf.split(X, y):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = CatBoostClassifier(**CAT_PARAMS)

    model.fit(
        X_train,
        y_train
    )

    y_prob = model.predict_proba(X_val)[:, 1]

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    tp_scores.append(tp)
    tn_scores.append(tn)
    fp_scores.append(fp)
    fn_scores.append(fn)

    results.append([
        f"Fold {fold}",
        tp,
        tn,
        fp,
        fn,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n" + "=" * 50)
    print(f"Fold {fold}")

    print(f"TP = {tp}")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")

    print(f"Accuracy  = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1 Score  = {f1:.4f}")

    fold += 1

# =========================
# Mean Results
# =========================

results.append([
    "MEAN",
    np.mean(tp_scores),
    np.mean(tn_scores),
    np.mean(fp_scores),
    np.mean(fn_scores),
    np.mean(accuracy_scores),
    np.mean(precision_scores),
    np.mean(recall_scores),
    np.mean(f1_scores)
])

results.append([
    "STD",
    np.std(tp_scores),
    np.std(tn_scores),
    np.std(fp_scores),
    np.std(fn_scores),
    np.std(accuracy_scores),
    np.std(precision_scores),
    np.std(recall_scores),
    np.std(f1_scores)
])

results_df = pd.DataFrame(
    results,
    columns=[
        "Fold",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

results_df.to_csv(
    r"D:\Projects\Omdena\bank+marketing\bank\cat_5_fold_final.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL 5-FOLD CROSS VALIDATION RESULTS")
print("=" * 60)

print(f"\nMean TP        : {np.mean(tp_scores):.2f}")
print(f"Mean TN        : {np.mean(tn_scores):.2f}")
print(f"Mean FP        : {np.mean(fp_scores):.2f}")
print(f"Mean FN        : {np.mean(fn_scores):.2f}")

print(f"\nMean Accuracy  : {np.mean(accuracy_scores):.4f}")
print(f"Mean Precision : {np.mean(precision_scores):.4f}")
print(f"Mean Recall    : {np.mean(recall_scores):.4f}")
print(f"Mean F1 Score  : {np.mean(f1_scores):.4f}")

print("\ncat_5_fold_final.csv saved successfully")
