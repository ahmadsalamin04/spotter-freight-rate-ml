from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from prepare_data import prepare_features


BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "train_test.csv",
    parse_dates=["date"]
)

X = prepare_features(df)
y = df["posted_rate"]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

folds = [
    ("2025-08-01", "2025-09-01", "August"),
    ("2025-09-01", "2025-10-01", "September"),
    ("2025-10-01", "2025-11-01", "October"),
]

results = []

print("\n CATBOOST MODEL TRAINING ")

# 4. Train and evaluate each fold
for start, end, month_name in folds:

    train_mask = df["date"] < start

    valid_mask = (
        (df["date"] >= start)
        & (df["date"] < end)
    )

    X_train = X.loc[train_mask]
    X_valid = X.loc[valid_mask]

    y_train = y.loc[train_mask]
    y_valid = y.loc[valid_mask]

    model = CatBoostRegressor(
        iterations=900,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function="RMSE",
        cat_features=categorical_features,
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    predictions = np.maximum(predictions, 1.0)

    mae = mean_absolute_error(y_valid, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    errors = np.abs(
        y_valid.to_numpy() - predictions
    )

    p95_error = np.percentile(errors, 95)

    results.append({
        "Month": month_name,
        "CatBoost_MAE": mae,
        "CatBoost_RMSE": rmse,
        "P95_Absolute_Error": p95_error,
    })

    print(f"\n {month_name.upper()} ")
    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_valid)}")
    print(f"CatBoost MAE: ${mae:.2f}")
    print(f"CatBoost RMSE: ${rmse:.2f}")
    print(f"95th percentile error: ${p95_error:.2f}")

results_df = pd.DataFrame(results)

print("\n===== CATBOOST VALIDATION RESULTS =====")
print(results_df.round(2).to_string(index=False))

print("\nAverage CatBoost MAE:",
      round(results_df["CatBoost_MAE"].mean(), 2))

print("Average CatBoost RMSE:",
      round(results_df["CatBoost_RMSE"].mean(), 2))

baseline_path = BASE_DIR / "baseline_results.csv"

if baseline_path.exists():

    baseline = pd.read_csv(baseline_path)

    comparison = baseline[
        ["Month", "Ridge_MAE", "Ridge_RMSE"]
    ].merge(
        results_df[
            ["Month", "CatBoost_MAE", "CatBoost_RMSE"]
        ],
        on="Month",
    )

    print("\n RIDGE VS CATBOOST ")
    print(comparison.round(2).to_string(index=False))

    print("\nAverage Ridge MAE:",
          round(comparison["Ridge_MAE"].mean(), 2))

    print("Average CatBoost MAE:",
          round(comparison["CatBoost_MAE"].mean(), 2))

results_df.to_csv(
    BASE_DIR / "catboost_results.csv",
    index=False
)

print("\nCatBoost training completed successfully!")
