from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "train_test.csv",
    parse_dates=["date"]
)

assert df["date"].notna().all(), "Missing dates found"
assert df["posted_rate"].notna().all(), "Missing target rates"
assert (df["posted_rate"] > 0).all(), "Invalid rates"

df = df.sort_values("date").reset_index(drop=True)

folds = [
    ("2025-08-01", "2025-09-01", "August"),
    ("2025-09-01", "2025-10-01", "September"),
    ("2025-10-01", "2025-11-01", "October"),
]

print("\n TIME-BASED VALIDATION ")

for start, end, month_name in folds:

    train = df[df["date"] < start].copy()

    valid = df[
        (df["date"] >= start)
        & (df["date"] < end)
    ].copy()

    assert len(train) > 0
    assert len(valid) > 0

    assert train["date"].max() < valid["date"].min()

    assert set(train["load_id"]).isdisjoint(
        set(valid["load_id"])
    )

    print(f"\n {month_name.upper()} FOLD ")
    print("Training records:", len(train))
    print("Validation records:", len(valid))

    print("Training dates:",
          train["date"].min().date(),
          "to",
          train["date"].max().date())

    print("Validation dates:",
          valid["date"].min().date(),
          "to",
          valid["date"].max().date())

    print("Training average rate:",
          round(train["posted_rate"].mean(), 2))

    print("Validation average rate:",
          round(valid["posted_rate"].mean(), 2))

    print("Chronological split verified!")

print("\nAll validation splits created successfully!")
