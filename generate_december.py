from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prepare_data import prepare_features


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print("\n DECEMBER PREDICTION MODEL ")

train = pd.read_csv(
    DATA_DIR / "train_test.csv",
    parse_dates=["date"]
)

december_path = DATA_DIR / "december_chart_inputs.csv"
december = pd.read_csv(december_path)

print("Training records:", len(train))
print("December records:", len(december))


expected_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

assert list(december.columns) == expected_columns
assert len(december) == 31

dates = pd.to_datetime(december["date"], errors="raise")

expected_dates = pd.date_range(
    "2025-12-01", "2025-12-31", freq="D"
)

assert dates.is_unique
assert set(dates) == set(expected_dates)

assert december["pickup"].eq("Lexington").all()
assert december["delivery"].eq("Fort Wayne").all()
assert december["equipment"].eq("Dry Van").all()

assert np.isclose(
    pd.to_numeric(december["distance"]), 360.0
).all()

assert np.isclose(
    pd.to_numeric(december["weight"]), 32000.0
).all()

print("December input validation passed!")


feature_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "weight_missing",
    "weight_was_negative",
    "route",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "days_since_start",
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

X = prepare_features(train)[feature_columns]
y = train["posted_rate"]

X_december = prepare_features(december)[feature_columns]

assert list(X.columns) == list(X_december.columns)


def create_model():
    return CatBoostRegressor(
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


train_mask = train["date"] < "2025-10-01"

valid_mask = (
    (train["date"] >= "2025-10-01")
    & (train["date"] < "2025-11-01")
)

validation_model = create_model()

validation_model.fit(
    X.loc[train_mask],
    y.loc[train_mask]
)

validation_predictions = np.maximum(
    validation_model.predict(X.loc[valid_mask]),
    1.0
)

october_mae = mean_absolute_error(
    y.loc[valid_mask],
    validation_predictions
)

october_rmse = np.sqrt(
    mean_squared_error(
        y.loc[valid_mask],
        validation_predictions
    )
)

print("\n DECEMBER MODEL VALIDATION ")
print(f"October MAE: ${october_mae:.2f}")
print(f"October RMSE: ${october_rmse:.2f}")


print("\nTraining final December model...")

model = create_model()
model.fit(X, y)

print("December model training completed!")


predictions = model.predict(X_december)

assert np.isfinite(predictions).all()

december["predicted_rate"] = np.maximum(
    predictions, 1.0
)

assert december["predicted_rate"].notna().all()
assert (december["predicted_rate"] > 0).all()


backup_path = (
    DATA_DIR / "december_chart_inputs_original.csv"
)

if not backup_path.exists():
    original = pd.read_csv(december_path)
    original.to_csv(backup_path, index=False)


december.to_csv(
    december_path,
    index=False,
    columns=expected_columns
)

pd.DataFrame([{
    "Model": "December reduced-feature CatBoost MAE",
    "October_MAE": october_mae,
    "October_RMSE": october_rmse,
}]).to_csv(
    BASE_DIR / "december_model_metrics.csv",
    index=False
)


print("\n DECEMBER PREDICTIONS ")

print(
    december[
        ["date", "predicted_rate"]
    ].to_string(index=False)
)

print("\n PREDICTION STATISTICS ")

print(
    december["predicted_rate"]
    .describe()
    .round(2)
)

print("\nTotal December predictions:", len(december))

print("\nMissing predictions:",
      december["predicted_rate"].isna().sum())

print("\nDecember predictions generated successfully!")
