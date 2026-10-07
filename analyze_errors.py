from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from prepare_data import prepare_features

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "train_test.csv",
    parse_dates=["date"]
)

train = df[df["date"] < "2025-10-01"].copy()
valid = df[
    (df["date"] >= "2025-10-01")
    & (df["date"] < "2025-11-01")
].copy()

numeric_cols = [
    "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon",
    "distance", "weight",
    "market_index", "quote_signal",
    "month", "day_of_week",
    "days_since_start",
    "weight_missing",
    "weight_was_negative",
]

categorical_cols = [
    "pickup", "delivery", "equipment"
]

features = numeric_cols + categorical_cols

X_train = prepare_features(train)[features]
X_valid = prepare_features(valid)[features]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=10.0, solver="lsqr")),
])

model.fit(X_train, train["posted_rate"])

predictions = np.maximum(
    model.predict(X_valid), 1.0
)

results = valid[
    ["load_id", "date", "pickup", "delivery",
     "distance", "equipment", "posted_rate"]
].copy()

results["predicted_rate"] = predictions
results["absolute_error"] = np.abs(
    results["posted_rate"] - results["predicted_rate"]
)

print("\n OCTOBER ERROR ANALYSIS ")

print("MAE:", round(
    mean_absolute_error(
        results["posted_rate"],
        results["predicted_rate"]
    ), 2
))

print("Median absolute error:",
      round(results["absolute_error"].median(), 2))

print("95th percentile error:",
      round(results["absolute_error"].quantile(0.95), 2))

print("99th percentile error:",
      round(results["absolute_error"].quantile(0.99), 2))

print("Errors above $1,000:",
      (results["absolute_error"] > 1000).sum())

print("\n 10 LARGEST ERRORS ")

print(
    results.nlargest(
        10, "absolute_error"
    ).to_string(index=False)
)

results.to_csv(
    BASE_DIR / "october_error_analysis.csv",
    index=False
)

print("\nError analysis completed successfully!")
