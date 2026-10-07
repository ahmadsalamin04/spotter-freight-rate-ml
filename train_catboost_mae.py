from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    "route"
]

folds = [
    ("2025-08-01", "2025-09-01", "August"),
    ("2025-09-01", "2025-10-01", "September"),
    ("2025-10-01", "2025-11-01", "October"),
]

results = []

print("\n ROBUST CATBOOST TRAINING ")

for start, end, month in folds:

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
        loss_function="MAE",
        eval_metric="MAE",
        cat_features=categorical_features,
        random_seed=42,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(
        model.predict(X_valid), 1.0
    )

    mae = mean_absolute_error(
        y_valid, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    errors = np.abs(
        y_valid.to_numpy() - predictions
    )

    p95 = np.percentile(errors, 95)

    results.append({
        "Month": month,
        "Robust_MAE": mae,
        "Robust_RMSE": rmse,
        "Robust_P95_Error": p95
    })

    print(f"\n {month.upper()} ")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"95th percentile error: ${p95:.2f}")

results_df = pd.DataFrame(results)

print("\n ROBUST MODEL RESULTS ")
print(results_df.round(2).to_string(index=False))

# Compare all three models
ridge = pd.read_csv(
    BASE_DIR / "baseline_results.csv"
)

catboost = pd.read_csv(
    BASE_DIR / "catboost_results.csv"
)

comparison = (
    ridge[["Month", "Ridge_MAE", "Ridge_RMSE"]]
    .merge(
        catboost[["Month", "CatBoost_MAE", "CatBoost_RMSE"]],
        on="Month"
    )
    .merge(
        results_df[["Month", "Robust_MAE", "Robust_RMSE"]],
        on="Month"
    )
)

print("\n FINAL MODEL COMPARISON ")
print(comparison.round(2).to_string(index=False))

print("\n AVERAGE ERRORS ")

for column in [
    "Ridge_MAE",
    "CatBoost_MAE",
    "Robust_MAE",
    "Ridge_RMSE",
    "CatBoost_RMSE",
    "Robust_RMSE"
]:
    print(f"{column}: ${comparison[column].mean():.2f}")

results_df.to_csv(
    BASE_DIR / "catboost_mae_results.csv",
    index=False
)

print("\nRobust model evaluation completed successfully!")
