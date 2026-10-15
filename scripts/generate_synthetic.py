import pandas as pd
from pathlib import Path

from sdv.single_table import CTGANSynthesizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/heart_ctgan.pkl")
OUTPUT_PATH = Path("synthetic_data/heart_synthetic.csv")

NUM_ROWS = 302


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SYNTHETIC DATA GENERATION")
print("=" * 70)

print("\nLoading trained CTGAN model...")

synthesizer = CTGANSynthesizer.load(
    filepath=MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GENERATE DATA
# ============================================================

print("\nGenerating synthetic records...")
print(f"Requested rows: {NUM_ROWS}")

synthetic_data = synthesizer.sample(
    num_rows=NUM_ROWS
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE SYNTHETIC DATA
# ============================================================

synthetic_data.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SYNTHETIC DATA GENERATED")
print("=" * 70)

print(f"Rows    : {synthetic_data.shape[0]}")
print(f"Columns : {synthetic_data.shape[1]}")

print(f"\nSaved to:")
print(OUTPUT_PATH)


# ============================================================
# DISPLAY FIRST RECORDS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 SYNTHETIC RECORDS")
print("=" * 70)

print(synthetic_data.head(10).to_string(index=False))


# ============================================================
# DISPLAY DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("SYNTHETIC DATA TYPES")
print("=" * 70)

print(synthetic_data.dtypes)


# ============================================================
# DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SYNTHETIC TARGET DISTRIBUTION")
print("=" * 70)

print(
    synthetic_data["target"].value_counts()
)

print("\nPercentage:")

print(
    (synthetic_data["target"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)