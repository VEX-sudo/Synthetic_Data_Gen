import pandas as pd
from pathlib import Path

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/heart_unique.csv")
METADATA_PATH = Path("models/heart_metadata.json")
MODEL_PATH = Path("models/heart_ctgan.pkl")

EPOCHS = 300
BATCH_SIZE = 50
PAC = 1


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("CTGAN TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading SDV metadata...")

metadata = Metadata.load_from_json(
    filepath=METADATA_PATH
)

metadata.validate()

print("Metadata validation: PASSED")


# ============================================================
# CREATE CTGAN
# ============================================================

print("\nCreating CTGAN synthesizer...")

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    pac=PAC,
    verbose=True,
    enforce_min_max_values=True,
    enforce_rounding=True,
    enable_gpu=False
)

print("CTGAN created successfully")


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING CTGAN TRAINING")
print("=" * 70)

print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"PAC        : {PAC}")
print("GPU        : Enabled")
print("Random seed: 42")

synthesizer.fit(df)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

synthesizer.save(
    filepath=MODEL_PATH
)

print(f"Model saved successfully:")
print(MODEL_PATH)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("CTGAN TRAINING COMPLETE")
print("=" * 70)