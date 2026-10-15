import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/heart.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

print(f"Total rows: {len(df)}")
print(f"Unique rows: {len(df.drop_duplicates())}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Number of occurrences for every unique record
duplicate_counts = (
    df.value_counts()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\n" + "=" * 60)
print("RECORD FREQUENCY")
print("=" * 60)

print(duplicate_counts["count"].value_counts().sort_index())

print("\n" + "=" * 60)
print("TOP 20 MOST REPEATED RECORDS")
print("=" * 60)

print(duplicate_counts.head(20).to_string(index=False))

# Target distribution before removing duplicates
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION - ORIGINAL")
print("=" * 60)

print(df["target"].value_counts())
print(
    (df["target"].value_counts(normalize=True) * 100)
    .round(2)
)

# Target distribution after removing duplicates
unique_df = df.drop_duplicates()

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION - UNIQUE RECORDS")
print("=" * 60)

print(unique_df["target"].value_counts())
print(
    (unique_df["target"].value_counts(normalize=True) * 100)
    .round(2)
)

# Save unique dataset
output_path = Path("data/heart_unique.csv")
unique_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("UNIQUE DATASET SAVED")
print("=" * 60)

print(f"Path: {output_path}")
print(f"Rows: {len(unique_df)}")
print(f"Columns: {len(unique_df.columns)}")