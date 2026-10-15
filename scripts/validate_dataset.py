import pandas as pd
from pathlib import Path

from sdv.metadata import Metadata

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("data/heart_unique.csv")
METADATA_PATH = Path("models/heart_metadata.json")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("HEART DISEASE DATASET VALIDATION")
print("=" * 70)

print(f"Dataset : {DATA_PATH}")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# --------------------------------------------------
# Expected columns
# --------------------------------------------------

expected_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


# --------------------------------------------------
# Check columns
# --------------------------------------------------

missing_columns = set(expected_columns) - set(df.columns)
extra_columns = set(df.columns) - set(expected_columns)

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

if extra_columns:
    print(f"Warning: Extra columns found: {extra_columns}")


# --------------------------------------------------
# Check missing values
# --------------------------------------------------

missing_values = df.isnull().sum()

if missing_values.sum() > 0:
    print("\nMissing values detected:")
    print(missing_values[missing_values > 0])
    raise ValueError(
        "Dataset contains missing values."
    )

print("\nMissing values: NONE")


# --------------------------------------------------
# Check duplicates
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate rows found in cleaned dataset."
    )


# --------------------------------------------------
# Check data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# Validate categorical columns
# --------------------------------------------------

categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "target",
]

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]


print("\nCategorical columns:")
for column in categorical_columns:
    print(
        f"{column:10} -> "
        f"{sorted(df[column].unique().tolist())}"
    )


print("\nNumerical columns:")
for column in numerical_columns:
    print(
        f"{column:10} -> "
        f"min={df[column].min()}, "
        f"max={df[column].max()}"
    )


# --------------------------------------------------
# Target validation
# --------------------------------------------------

target_values = sorted(df["target"].unique().tolist())

if target_values != [0, 1]:
    raise ValueError(
        f"Unexpected target values: {target_values}"
    )

print("\nTarget values: [0, 1]")


# --------------------------------------------------
# Create SDV metadata
# --------------------------------------------------

print("\n" + "=" * 70)
print("CREATING SDV METADATA")
print("=" * 70)

metadata = Metadata.detect_from_dataframe(
    data=df
)

# Explicitly update semantic types
for column in categorical_columns:
    metadata.update_column(
        column_name=column,
        sdtype="categorical"
    )

for column in numerical_columns:
    metadata.update_column(
        column_name=column,
        sdtype="numerical"
    )


# --------------------------------------------------
# Validate metadata
# --------------------------------------------------

metadata.validate()

print("SDV metadata validation: PASSED")


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

METADATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

metadata.save_to_json(
    filepath=METADATA_PATH
)

print(f"Metadata saved: {METADATA_PATH}")


# --------------------------------------------------
# Display metadata
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL METADATA")
print("=" * 70)

print(metadata)


# --------------------------------------------------
# Final status
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATASET VALIDATION COMPLETE")
print("=" * 70)