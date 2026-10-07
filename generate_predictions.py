from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from prepare_data import prepare_features

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print("\n FINAL MODEL TRAINING ")

train = pd.read_csv(DATA_DIR / "train_test.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
template = pd.read_csv(
    DATA_DIR / "validation_predictions_template.csv"
)

print("Training records:", len(train))
print("Validation records:", len(validation))

X_train = prepare_features(train)
X_validation = prepare_features(validation)

y_train = train["posted_rate"]

assert list(X_train.columns) == list(X_validation.columns)
assert y_train.notna().all()
assert (y_train > 0).all()

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

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

print("\nTraining final CatBoost model...")

model.fit(X_train, y_train)

print("Final model training completed!")

print("\nGenerating validation predictions...")

predictions = model.predict(X_validation)

if not np.isfinite(predictions).all():
    raise ValueError("Model generated invalid predictions")

predictions = np.maximum(predictions, 1.0)

assert len(validation) == 12000
assert len(template) == 12000

assert validation["load_id"].is_unique
assert template["load_id"].is_unique

assert set(validation["load_id"]) == set(
    template["load_id"]
)

prediction_map = pd.Series(
    predictions,
    index=validation["load_id"]
)

submission = template[["load_id"]].copy()

submission["predicted_rate"] = (
    submission["load_id"].map(prediction_map)
)

assert list(submission.columns) == [
    "load_id",
    "predicted_rate"
]

assert len(submission) == 12000
assert submission["predicted_rate"].notna().all()
assert np.isfinite(submission["predicted_rate"]).all()
assert (submission["predicted_rate"] > 0).all()

expected_ids = {
    f"TE-{i:06d}" for i in range(1, 12001)
}

assert set(submission["load_id"]) == expected_ids

output_path = BASE_DIR / "validation_predictions.csv"

submission.to_csv(output_path, index=False)

print("\n PREDICTION SUMMARY ")
print("Total predictions:", len(submission))

print("\nPredicted rate statistics:")
print(submission["predicted_rate"].describe().round(2))

print("\nFirst five predictions:")
print(submission.head().to_string(index=False))

print("\nMissing predictions:",
      submission["predicted_rate"].isna().sum())

print("\nOutput saved to:", output_path)

print("\nFinal predictions generated successfully!")
