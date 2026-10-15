import pandas as pd
from pathlib import Path

from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality


# ============================================================
# CONFIGURATION
# ============================================================

REAL_DATA_PATH = Path("data/heart_unique.csv")
SYNTHETIC_DATA_PATH = Path(
    "synthetic_data/heart_synthetic.csv"
)
METADATA_PATH = Path(
    "models/heart_metadata.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SYNTHETIC DATA QUALITY EVALUATION")
print("=" * 70)

print("\nLoading real dataset...")

real_data = pd.read_csv(
    REAL_DATA_PATH
)

print(
    f"Real data: "
    f"{real_data.shape[0]} rows × "
    f"{real_data.shape[1]} columns"
)


print("\nLoading synthetic dataset...")

synthetic_data = pd.read_csv(
    SYNTHETIC_DATA_PATH
)

print(
    f"Synthetic data: "
    f"{synthetic_data.shape[0]} rows × "
    f"{synthetic_data.shape[1]} columns"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

metadata = Metadata.load_from_json(
    filepath=METADATA_PATH
)

metadata.validate()

print("Metadata validation: PASSED")


# ============================================================
# EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING SDV QUALITY EVALUATION")
print("=" * 70)

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)


# ============================================================
# DISPLAY SCORE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL QUALITY SCORE")
print("=" * 70)

score = quality_report.get_score()

print(f"Score: {score:.4f}")
print(f"Percentage: {score * 100:.2f}%")


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_PATH = Path(
    "reports/quality_report.pkl"
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

quality_report.save(
    filepath=REPORT_PATH
)

print("\nQuality report saved to:")
print(REPORT_PATH)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("QUALITY EVALUATION COMPLETE")
print("=" * 70)