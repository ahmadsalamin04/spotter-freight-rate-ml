from pathlib import Path
import pandas as pd


def prepare_features(df):
    
    data = df.copy()

    data = data.drop(
        columns=["load_id", "posted_rate"],
        errors="ignore"
    )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    if data["date"].isna().any():
        raise ValueError("Invalid dates found in dataset")

    data["weight_missing"] = (
        data["weight"].isna().astype(int)
    )

    data["weight_was_negative"] = (
        (data["weight"] < 0).astype(int)
    )

    data["weight"] = data["weight"].abs()

    for col in ["pickup", "delivery", "equipment"]:
        data[col] = data[col].fillna("Unknown")
        data[col] = data[col].astype(str)

    data["route"] = (
        data["pickup"] + " -> " + data["delivery"]
    )

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    start_date = pd.Timestamp("2025-01-01")
    data["days_since_start"] = (
        data["date"] - start_date
    ).dt.days

    data = data.drop(columns=["date"])

    return data


if __name__ == "__main__":

    project_folder = Path(__file__).resolve().parent

    train = pd.read_csv(
        project_folder / "data" / "train_test.csv"
    )

    validation = pd.read_csv(
        project_folder / "data" / "validation.csv"
    )

    X_train = prepare_features(train)
    X_validation = prepare_features(validation)

    y_train = train["posted_rate"]

    assert list(X_train.columns) == list(X_validation.columns)
    assert "posted_rate" not in X_train.columns
    assert "load_id" not in X_train.columns
    assert y_train.notna().all()
    assert (y_train > 0).all()

    print("\n DATA PREPARATION RESULTS ")

    print("\nTraining features shape:")
    print(X_train.shape)

    print("\nValidation features shape:")
    print(X_validation.shape)

    print("\nFeatures used for modeling:")
    print(X_train.columns.tolist())

    print("\nMissing values after preparation:")
    print(X_train.isnull().sum())

    print("\nNegative weights after preparation:")
    print((X_train["weight"] < 0).sum())

    print("\nFirst five prepared records:")
    print(X_train.head().to_string(index=False))

    print("\nData preparation completed successfully!")
