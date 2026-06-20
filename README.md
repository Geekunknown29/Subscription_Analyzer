# BankMind - Term Deposit Subscription Analyzer

BankMind is a machine learning project built on the Bank Marketing dataset. The goal is to predict whether a customer is likely to subscribe to a term deposit, while handling the strong class imbalance in the data.

The project compares a required baseline Logistic Regression model with stronger tree-based models, including Random Forest, XGBoost, and CatBoost. The final selected model is CatBoost because it gives a better balance between precision and recall for identifying real subscribers.

## Project Structure

```text
bankmind-Arpit/
├── data/
│   ├── bank-full.csv
│   ├── bank_processed.csv
│   ├── train_data.csv
│   └── test_data.csv
├── eda/
│   └── EDA charts and CSV summaries
├── models/
│   ├── final_catboost_main_model.pkl
│   ├── logistic_regression_v3.pkl
│   ├── scaler_v3.pkl
│   └── other models (experimentation)/
├── python/
│   ├── EDA_Data.py
│   ├── LR_Train_v3.py
│   ├── Cat_boost_V1.py
│   ├── cat_boost_cross_validation.py
│   ├── feature_importance_cat_boost_v2.py
│   └── other scripts/
├── results/
│   ├── cat_sample_predictions.csv
│   ├── lr_sample_predictions.csv
│   └── other result files
├── explanation.md
└── README.md
```

## Requirements

Use Python 3.10 or newer.

Install Git LFS before cloning or pulling the repository because the model files and experiment archive are stored with LFS:

```bash
git lfs install
git clone https://github.com/Geekunknown29/bankmind-Arpit.git
cd bankmind-Arpit
git lfs pull
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install pandas scikit-learn catboost xgboost matplotlib
```

## Important Path Note

Some training scripts were originally written on a local Windows machine and contain absolute paths such as:

```text
D:\Projects\bankmind-Arpit
D:\Projects\Omdena\bank+marketing\bank
```

If you clone this repository into a different folder, open the script you want to run and replace those hardcoded paths with your own local project path. For example:

```text
YOUR_CLONE_PATH\data\train_data.csv
YOUR_CLONE_PATH\data\test_data.csv
YOUR_CLONE_PATH\models
YOUR_CLONE_PATH\results
```

The required CSV files and trained model files are already included in the repository, so you do not need to recreate the dataset split unless you want to.

## How To Run The Project

### 1. Explore The Dataset

Run the EDA script:

```bash
python python/EDA_Data.py
```

This analyzes the raw dataset and creates charts/summary files for class imbalance, job category subscription rate, balance, age groups, housing loan behavior, and missing values.

By default, the script writes EDA outputs to `results/eda/`. In the final organized repository, the checked-in EDA outputs are stored in `eda/`, including files such as:

```text
eda/class_distribution.png
eda/job_subscription_rate.csv
eda/job_subscription_rate.png
```

### 2. Train And Test The Baseline Logistic Regression Model

Run:

```bash
python python/LR_Train_v3.py
```

This script:

- loads `data/train_data.csv`
- trains a balanced Logistic Regression model
- applies threshold `0.7`
- evaluates the model on `data/test_data.csv`
- saves the model and scaler
- creates 5 sample customer predictions

Expected outputs:

```text
models/logistic_regression_v3.pkl
models/scaler_v3.pkl
results/lr_sample_predictions.csv
```

The terminal prints accuracy, precision, recall, F1 score, confusion matrix, and classification report.

### 3. Train And Test The Final CatBoost Model

Run:

```bash
python python/Cat_boost_V1.py
```

This is the final model workflow. It:

- loads `data/train_data.csv`
- trains CatBoost with class weights `[1, 7]`
- applies threshold `0.7`
- evaluates the model on `data/test_data.csv`
- saves the final model
- creates 5 representative sample predictions

Expected outputs:

```text
models/final_catboost.pkl
results/sample_predictions.csv
```

In the final organized repository, the main CatBoost model is stored as:

```text
models/final_catboost_main_model.pkl
```

### 4. Run CatBoost Cross Validation

Run:

```bash
python python/cat_boost_cross_validation.py
```

This checks the CatBoost model more robustly using cross validation and saves the cross-validation results.

Expected output:

```text
results/cat_5_fold_final.csv
```

### 5. Check CatBoost Feature Importance

Run:

```bash
python python/feature_importance_cat_boost_v2.py
```

This identifies which features are most important to the final tree-based model. In this project, `duration` was the most important feature.

Expected output:

```text
results/catboost_feature_importance.csv
```

## Running Experimental Models

The Random Forest and XGBoost scripts are kept for comparison and experimentation:

```text
python/other scripts/random_forest_v2.py
python/other scripts/rf_tuning.py
python/other scripts/XG_boost_V1.py
python/other scripts/XG_boost_V1_threshold_analysis.py
```

Their saved model/result files are organized under:

```text
models/other models (experimentation)/
results/other results/
```

These models were useful for comparison, but CatBoost was selected as the final model.

## How To Test The Project Quickly

After installing dependencies and fixing local paths if needed, the easiest test is:

```bash
python python/Cat_boost_V1.py
```

If the setup is correct, you should see metrics printed in the terminal and a sample prediction CSV generated in `results/`.

You can also inspect the already generated prediction file:

```text
results/cat_sample_predictions.csv
```

Each row contains customer features, the actual class, predicted subscription probability, and final prediction.

## Evaluation Approach

The dataset is heavily imbalanced:

```text
yes: 5,289 customers, 11.70%
no: 39,922 customers, 88.30%
```

Because of this imbalance, accuracy alone is not enough. A model can get high accuracy by mostly predicting `no`, but that would miss many real subscribers. This project focuses more on precision, recall, and F1 score.

The final model uses a threshold of `0.7` to reduce unnecessary calls while still identifying likely subscribers.

## Final Model

Final selected model:

```text
CatBoostClassifier
```

Main model file:

```text
models/final_catboost_main_model.pkl
```

Main explanation file:

```text
explanation.md

The explanation file contains the project reasoning, class imbalance discussion, EDA insights, feature importance interpretation, metric choice, and sample prediction analysis.
```


RESULTS :
| Model               | Accuracy | Precision | Recall |     F1 |
| ------------------- | -------: | --------: | -----: | -----: |
| Logistic Regression |   84.60% |    41.86% | 81.38% | 55.28% |
| XGBoost             |   89.73% |    54.40% | 75.43% | 63.21% |
| CatBoost            |   89.73% |    54.40% | 75.43% | 63.21% |


EDA :
<img width="1000" height="500" alt="job_subscription_rate" src="https://github.com/user-attachments/assets/d600a25d-84e1-4d1d-b5cc-fba4d967e5f1" />
<img width="600" height="400" alt="housing_subscription" src="https://github.com/user-attachments/assets/72da0851-f2aa-4a31-8b7a-86fdeb4be6ef" />
<img width="600" height="400" alt="class_distribution" src="https://github.com/user-attachments/assets/d6b41f40-63bc-4b1c-b49c-6c641d3d3d60" />
<img width="600" height="400" alt="age_group_subscription" src="https://github.com/user-attachments/assets/db37b082-d4e9-42c2-ae4d-c33f863fd2c3" />
