import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

REAL_PATH = Path("data/heart_unique.csv")
SYNTHETIC_PATH = Path("synthetic_data/heart_synthetic.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

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


# ============================================================
# NUMERICAL COMPARISON
# ============================================================

print("=" * 75)
print("NUMERICAL DISTRIBUTION COMPARISON")
print("=" * 75)

numerical_results = []

for column in numerical_columns:

    real_values = real[column]
    synthetic_values = synthetic[column]

    result = {
        "column": column,

        "real_mean": real_values.mean(),
        "synthetic_mean": synthetic_values.mean(),

        "real_median": real_values.median(),
        "synthetic_median": synthetic_values.median(),

        "real_std": real_values.std(),
        "synthetic_std": synthetic_values.std(),

        "real_min": real_values.min(),
        "synthetic_min": synthetic_values.min(),

        "real_max": real_values.max(),
        "synthetic_max": synthetic_values.max(),
    }

    numerical_results.append(result)

    print(f"\n{column}")

    print(
        f"  Mean   : "
        f"{real_values.mean():.2f} "
        f"vs "
        f"{synthetic_values.mean():.2f}"
    )

    print(
        f"  Median : "
        f"{real_values.median():.2f} "
        f"vs "
        f"{synthetic_values.median():.2f}"
    )

    print(
        f"  Std    : "
        f"{real_values.std():.2f} "
        f"vs "
        f"{synthetic_values.std():.2f}"
    )

    print(
        f"  Range  : "
        f"{real_values.min():.2f} - "
        f"{real_values.max():.2f} "
        f"vs "
        f"{synthetic_values.min():.2f} - "
        f"{synthetic_values.max():.2f}"
    )


numerical_df = pd.DataFrame(numerical_results)

numerical_df.to_csv(
    REPORT_DIR / "numerical_comparison.csv",
    index=False
)


# ============================================================
# CATEGORICAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("CATEGORICAL DISTRIBUTION COMPARISON")
print("=" * 75)

categorical_results = []

for column in categorical_columns:

    real_distribution = (
        real[column]
        .value_counts(normalize=True)
        .sort_index()
    )

    synthetic_distribution = (
        synthetic[column]
        .value_counts(normalize=True)
        .sort_index()
    )

    all_categories = sorted(
        set(real_distribution.index)
        | set(synthetic_distribution.index)
    )

    print(f"\n{column}")

    for category in all_categories:

        real_percentage = (
            real_distribution.get(category, 0) * 100
        )

        synthetic_percentage = (
            synthetic_distribution.get(category, 0) * 100
        )

        difference = (
            synthetic_percentage - real_percentage
        )

        categorical_results.append({
            "column": column,
            "category": category,
            "real_percentage": real_percentage,
            "synthetic_percentage": synthetic_percentage,
            "difference": difference,
        })

        print(
            f"  {category}: "
            f"Real={real_percentage:.2f}% | "
            f"Synthetic={synthetic_percentage:.2f}% | "
            f"Difference={difference:+.2f}%"
        )


categorical_df = pd.DataFrame(
    categorical_results
)

categorical_df.to_csv(
    REPORT_DIR / "categorical_comparison.csv",
    index=False
)


# ============================================================
# CORRELATION COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION ANALYSIS")
print("=" * 75)

real_correlation = real.corr(numeric_only=True)

synthetic_correlation = synthetic.corr(
    numeric_only=True
)

correlation_difference = (
    synthetic_correlation - real_correlation
)

print("\nReal correlation matrix:")
print(real_correlation.round(2))

print("\nSynthetic correlation matrix:")
print(synthetic_correlation.round(2))

print("\nAbsolute correlation difference:")

print(
    correlation_difference
    .abs()
    .round(2)
)


real_correlation.to_csv(
    REPORT_DIR / "real_correlation.csv"
)

synthetic_correlation.to_csv(
    REPORT_DIR / "synthetic_correlation.csv"
)

correlation_difference.to_csv(
    REPORT_DIR / "correlation_difference.csv"
)


# ============================================================
# EXACT DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 75)
print("EXACT RECORD DUPLICATE CHECK")
print("=" * 75)

real_records = set(
    map(tuple, real.to_numpy())
)

synthetic_records = set(
    map(tuple, synthetic.to_numpy())
)

exact_matches = (
    real_records.intersection(
        synthetic_records
    )
)

print(
    f"Unique real records     : {len(real_records)}"
)

print(
    f"Unique synthetic records: {len(synthetic_records)}"
)

print(
    f"Exact matching records  : {len(exact_matches)}"
)


if len(exact_matches) > 0:

    print(
        "\nWARNING:"
        " Synthetic records exactly matching"
        " real records were found."
    )

else:

    print(
        "\nNo exact real/synthetic record matches found."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "real_rows": len(real),
    "synthetic_rows": len(synthetic),
    "real_columns": len(real.columns),
    "synthetic_columns": len(synthetic.columns),
    "exact_record_matches": len(exact_matches),
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    REPORT_DIR / "quality_summary.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("DETAILED QUALITY ANALYSIS COMPLETE")
print("=" * 75)

print("\nReports generated:")

print("  reports/numerical_comparison.csv")
print("  reports/categorical_comparison.csv")
print("  reports/real_correlation.csv")
print("  reports/synthetic_correlation.csv")
print("  reports/correlation_difference.csv")
print("  reports/quality_summary.csv")