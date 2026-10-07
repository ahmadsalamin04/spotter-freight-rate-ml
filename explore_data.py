from pathlib import Path
import pandas as pd

data_path = Path(__file__).resolve().parent / "data" / "train_test.csv"

df = pd.read_csv(data_path, parse_dates=["date"])

pd.set_option("display.max_columns", None)

print("\n DATASET SIZE ")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n FIRST FIVE ROWS ")
print(df.head().to_string(index=False))

print("\n COLUMN INFORMATION ")
df.info()

print("\n MISSING VALUES ")
print(df.isnull().sum().to_string())

print("\n DUPLICATE LOAD IDS ")
print(df["load_id"].duplicated().sum())

print("\n DATA QUALITY CHECKS ")
print("Negative weights:", (df["weight"] < 0).sum())
print("Non-positive distances:", (df["distance"] <= 0).sum())
print("Non-positive rates:", (df["posted_rate"] <= 0).sum())

print("\n DATE RANGE ")
print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())

print("\n EQUIPMENT TYPES ")
print(df["equipment"].value_counts(dropna=False))

print("\n FREIGHT RATE STATISTICS ")
print(df["posted_rate"].describe().round(2))

print("\n NUMERICAL STATISTICS ")
print(df.describe().round(2).to_string())

print("\nDataset exploration completed successfully")
