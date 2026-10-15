import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("data/heart.csv")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(f"File       : {DATA_PATH}")
print(f"Rows       : {df.shape[0]}")
print(f"Columns    : {df.shape[1]}")


# --------------------------------------------------
# Column information
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

print(df.dtypes)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Missing": missing,
    "Percentage": (missing / len(df) * 100).round(2)
})

print(missing_table)


# --------------------------------------------------
# Duplicate records
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print(f"Duplicate rows: {df.duplicated().sum()}")


# --------------------------------------------------
# Numerical summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all").transpose())


# --------------------------------------------------
# Unique values
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    unique_values = df[column].unique()

    print(f"\n{column}")
    print(f"Number of unique values: {df[column].nunique()}")

    if df[column].nunique() <= 20:
        print(f"Values: {unique_values}")


# --------------------------------------------------
# First 5 rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# --------------------------------------------------
# Final status
# --------------------------------------------------

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)