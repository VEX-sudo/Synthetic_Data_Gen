import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# ============================================================
# CONFIGURATION
# ============================================================

REAL_PATH = Path("data/heart_unique.csv")
SYNTHETIC_PATH = Path("synthetic_data/heart_synthetic.csv")

REPORT_DIR = Path("reports/privacy")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("PRIVACY RISK ANALYSIS")
print("=" * 75)

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print(f"\nReal records      : {len(real)}")
print(f"Synthetic records : {len(synthetic)}")


# ============================================================
# FEATURE DEFINITIONS
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
# PREPARE NUMERICAL FEATURES
# ============================================================

print("\nPreparing numerical features...")

scaler = StandardScaler()

real_numerical = scaler.fit_transform(
    real[numerical_columns]
)

synthetic_numerical = scaler.transform(
    synthetic[numerical_columns]
)


# ============================================================
# PREPARE CATEGORICAL FEATURES
# ============================================================

print("Preparing categorical features...")

# Convert categorical columns to strings first
real_categorical = real[categorical_columns].astype(str)
synthetic_categorical = synthetic[categorical_columns].astype(str)

# Create one-hot encoding using the combined category space
combined = pd.concat(
    [real_categorical, synthetic_categorical],
    axis=0
)

combined_encoded = pd.get_dummies(
    combined,
    columns=categorical_columns,
    dtype=float
)

real_encoded = combined_encoded.iloc[
    :len(real)
].to_numpy()

synthetic_encoded = combined_encoded.iloc[
    len(real):
].to_numpy()


# ============================================================
# COMBINE FEATURES
# ============================================================

real_features = np.hstack(
    [real_numerical, real_encoded]
)

synthetic_features = np.hstack(
    [synthetic_numerical, synthetic_encoded]
)

print(
    f"Feature dimensions: "
    f"{real_features.shape[1]}"
)


# ============================================================
# CALCULATE DISTANCES
# ============================================================

print("\nCalculating synthetic → real distances...")

distance_matrix = pairwise_distances(
    synthetic_features,
    real_features,
    metric="euclidean"
)

nearest_distances = distance_matrix.min(
    axis=1
)

nearest_indices = distance_matrix.argmin(
    axis=1
)


# ============================================================
# DISTANCE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("NEAREST-NEIGHBOR DISTANCE STATISTICS")
print("=" * 75)

print(
    f"\nMinimum distance : "
    f"{nearest_distances.min():.4f}"
)

print(
    f"Mean distance    : "
    f"{nearest_distances.mean():.4f}"
)

print(
    f"Median distance  : "
    f"{np.median(nearest_distances):.4f}"
)

print(
    f"Maximum distance : "
    f"{nearest_distances.max():.4f}"
)

print(
    f"25th percentile  : "
    f"{np.percentile(nearest_distances, 25):.4f}"
)

print(
    f"75th percentile  : "
    f"{np.percentile(nearest_distances, 75):.4f}"
)


# ============================================================
# LOW-DISTANCE RECORDS
# ============================================================

print("\n" + "=" * 75)
print("CLOSEST SYNTHETIC → REAL RECORDS")
print("=" * 75)

closest_indices = np.argsort(
    nearest_distances
)[:10]

close_records = []

for synthetic_index in closest_indices:

    real_index = nearest_indices[
        synthetic_index
    ]

    distance = nearest_distances[
        synthetic_index
    ]

    print(
        f"\nSynthetic #{synthetic_index}"
        f" → Real #{real_index}"
    )

    print(
        f"Distance: {distance:.4f}"
    )

    close_records.append({
        "synthetic_index": synthetic_index,
        "real_index": real_index,
        "distance": distance,
    })


close_df = pd.DataFrame(
    close_records
)

close_df.to_csv(
    REPORT_DIR / "closest_records.csv",
    index=False
)


# ============================================================
# DISTANCE DISTRIBUTION
# ============================================================

distance_df = pd.DataFrame({
    "synthetic_index": range(
        len(nearest_distances)
    ),
    "nearest_real_distance": nearest_distances,
    "nearest_real_index": nearest_indices,
})

distance_df.to_csv(
    REPORT_DIR / "nearest_neighbor_distances.csv",
    index=False
)


# ============================================================
# RISK THRESHOLDS
# ============================================================

print("\n" + "=" * 75)
print("RELATIVE PRIVACY RISK ANALYSIS")
print("=" * 75)

# Use percentiles instead of arbitrary universal thresholds.
# This is a relative screening tool, NOT a formal privacy guarantee.

threshold_5 = np.percentile(
    nearest_distances,
    5
)

threshold_10 = np.percentile(
    nearest_distances,
    10
)

threshold_25 = np.percentile(
    nearest_distances,
    25
)

print(
    f"\n5th percentile distance  : "
    f"{threshold_5:.4f}"
)

print(
    f"10th percentile distance : "
    f"{threshold_10:.4f}"
)

print(
    f"25th percentile distance : "
    f"{threshold_25:.4f}"
)

print(
    "\nRecords in lowest 5% distance : "
    f"{(nearest_distances <= threshold_5).sum()}"
)

print(
    "Records in lowest 10% distance: "
    f"{(nearest_distances <= threshold_10).sum()}"
)

print(
    "Records in lowest 25% distance: "
    f"{(nearest_distances <= threshold_25).sum()}"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "real_records": len(real),
    "synthetic_records": len(synthetic),
    "minimum_distance": nearest_distances.min(),
    "mean_distance": nearest_distances.mean(),
    "median_distance": np.median(nearest_distances),
    "maximum_distance": nearest_distances.max(),
    "p05_distance": threshold_5,
    "p10_distance": threshold_10,
    "p25_distance": threshold_25,
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    REPORT_DIR / "privacy_summary.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("PRIVACY ANALYSIS COMPLETE")
print("=" * 75)

print("\nReports generated:")

print(
    "  reports/privacy/privacy_summary.csv"
)

print(
    "  reports/privacy/nearest_neighbor_distances.csv"
)

print(
    "  reports/privacy/closest_records.csv"
)