from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from prepare_data import prepare_features


BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "train_test.csv",
    parse_dates=["date"]
)

X = prepare_features(df)
y = df["posted_rate"]

numeric_features = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "days_since_start",
    "weight_missing",
    "weight_was_negative",
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]

selected_features = numeric_features + categorical_features

X = X[selected_features]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features,
    ),
])

folds = [
    ("2025-08-01", "2025-09-01", "August"),
    ("2025-09-01", "2025-10-01", "September"),
    ("2025-10-01", "2025-11-01", "October"),
]

results = []

print("\n BASELINE MODEL TRAINING ")

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

    median_prediction = np.full(
        len(y_valid),
        y_train.median()
    )

    baseline_mae = mean_absolute_error(
        y_valid, median_prediction
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0, solver="lsqr")),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    predictions = np.maximum(predictions, 1.0)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    results.append({
        "Month": month_name,
        "Median_MAE": baseline_mae,
        "Ridge_MAE": mae,
        "Ridge_RMSE": rmse,
    })

    print(f"\n {month_name.upper()} ")
    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_valid)}")
    print(f"Historical median MAE: ${baseline_mae:.2f}")
    print(f"Ridge Regression MAE: ${mae:.2f}")
    print(f"Ridge Regression RMSE: ${rmse:.2f}")

results_df = pd.DataFrame(results)

print("\n MODEL COMPARISON ")
print(results_df.round(2).to_string(index=False))

print("\nAverage Median Baseline MAE:",
      round(results_df["Median_MAE"].mean(), 2))

print("Average Ridge Regression MAE:",
      round(results_df["Ridge_MAE"].mean(), 2))

print("Average Ridge Regression RMSE:",
      round(results_df["Ridge_RMSE"].mean(), 2))

results_df.to_csv(
    BASE_DIR / "baseline_results.csv",
    index=False
)

print("\nBaseline training completed successfully!")
