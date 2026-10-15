import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

REAL_PATH = Path("data/heart_unique.csv")
SYNTHETIC_PATH = Path("synthetic_data/heart_synthetic.csv")
DISTANCE_PATH = Path(
    "reports/privacy/nearest_neighbor_distances.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)
distances = pd.read_csv(DISTANCE_PATH)


# ============================================================
# GET CLOSEST RECORDS
# ============================================================

closest = distances.sort_values(
    "nearest_real_distance"
).head(10)


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("=" * 80)
print("CLOSEST SYNTHETIC ↔ REAL RECORD COMPARISON")
print("=" * 80)

for _, row in closest.iterrows():

    synthetic_index = int(
        row["synthetic_index"]
    )

    real_index = int(
        row["nearest_real_index"]
    )

    distance = row[
        "nearest_real_distance"
    ]

    print("\n" + "-" * 80)

    print(
        f"Distance: {distance:.4f}"
    )

    print(
        f"Synthetic record #{synthetic_index}"
    )

    print(
        synthetic.iloc[synthetic_index]
        .to_string()
    )

    print(
        f"\nReal record #{real_index}"
    )

    print(
        real.iloc[real_index]
        .to_string()
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_rows = []

for _, row in closest.iterrows():

    synthetic_index = int(
        row["synthetic_index"]
    )

    real_index = int(
        row["nearest_real_index"]
    )

    comparison = {
        "synthetic_index": synthetic_index,
        "real_index": real_index,
        "distance": row[
            "nearest_real_distance"
        ],
    }

    for column in real.columns:

        comparison[
            f"real_{column}"
        ] = real.iloc[
            real_index
        ][column]

        comparison[
            f"synthetic_{column}"
        ] = synthetic.iloc[
            synthetic_index
        ][column]

    comparison_rows.append(
        comparison
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

output_path = Path(
    "reports/privacy/closest_record_comparison.csv"
)

comparison_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 80)
print("PRIVACY RECORD INSPECTION COMPLETE")
print("=" * 80)

print(
    f"\nSaved to: {output_path}"
)