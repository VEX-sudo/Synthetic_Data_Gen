import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

REAL_PATH = Path("data/heart_unique.csv")
SYNTHETIC_PATH = Path("synthetic_data/heart_synthetic.csv")

REPORT_DIR = Path("reports/ml_utility")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("ML UTILITY EVALUATION")
print("=" * 75)

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print(f"\nReal dataset      : {real.shape}")
print(f"Synthetic dataset : {synthetic.shape}")


# ============================================================
# FEATURES / TARGET
# ============================================================

TARGET = "target"

feature_columns = [
    column
    for column in real.columns
    if column != TARGET
]

X_real = real[feature_columns]
y_real = real[TARGET]

X_synthetic = synthetic[feature_columns]
y_synthetic = synthetic[TARGET]


# ============================================================
# SPLIT REAL DATA
# ============================================================

print("\nSplitting real dataset...")

X_real_train, X_real_test, y_real_train, y_real_test = (
    train_test_split(
        X_real,
        y_real,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_real,
    )
)

print(
    f"Real training records : {len(X_real_train)}"
)

print(
    f"Real testing records  : {len(X_real_test)}"
)


# ============================================================
# STANDARDIZATION
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
]


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def prepare_features(
    train_data,
    test_data,
    numerical_columns,
    categorical_columns,
):
    """
    Prepare numerical and categorical features.

    Numerical features are standardized.
    Categorical features are one-hot encoded.
    """

    train_num = train_data[
        numerical_columns
    ].copy()

    test_num = test_data[
        numerical_columns
    ].copy()

    scaler = StandardScaler()

    train_num_scaled = scaler.fit_transform(
        train_num
    )

    test_num_scaled = scaler.transform(
        test_num
    )

    train_cat = train_data[
        categorical_columns
    ].astype(str)

    test_cat = test_data[
        categorical_columns
    ].astype(str)

    combined = pd.concat(
        [train_cat, test_cat],
        axis=0
    )

    combined_encoded = pd.get_dummies(
        combined,
        columns=categorical_columns,
        dtype=float
    )

    train_cat_encoded = combined_encoded.iloc[
        :len(train_cat)
    ].to_numpy()

    test_cat_encoded = combined_encoded.iloc[
        len(train_cat):
    ].to_numpy()

    X_train = np.hstack(
        [
            train_num_scaled,
            train_cat_encoded,
        ]
    )

    X_test = np.hstack(
        [
            test_num_scaled,
            test_cat_encoded,
        ]
    )

    return X_train, X_test


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# EXPERIMENT A
# REAL TRAINING → REAL TEST
# ============================================================

print(
    "\nPreparing REAL → REAL experiment..."
)

X_train_real, X_test_real = prepare_features(
    X_real_train,
    X_real_test,
    numerical_columns,
    categorical_columns,
)

real_result = evaluate_model(
    name="REAL TRAINING → REAL TEST",
    X_train=X_train_real,
    y_train=y_real_train,
    X_test=X_test_real,
    y_test=y_real_test,
)


# ============================================================
# EXPERIMENT B
# SYNTHETIC TRAINING → REAL TEST
# ============================================================

print(
    "\nPreparing SYNTHETIC → REAL experiment..."
)

X_train_synthetic, X_test_synthetic = (
    prepare_features(
        X_synthetic,
        X_real_test,
        numerical_columns,
        categorical_columns,
    )
)

synthetic_result = evaluate_model(
    name="SYNTHETIC TRAINING → REAL TEST",
    X_train=X_train_synthetic,
    y_train=y_synthetic,
    X_test=X_test_synthetic,
    y_test=y_real_test,
)


# ============================================================
# COMPARISON
# ============================================================

results = pd.DataFrame(
    [
        real_result,
        synthetic_result,
    ]
)

results["utility_gap"] = (
    results.iloc[0]["roc_auc"]
    - results["roc_auc"]
)

print("\n" + "=" * 75)
print("MODEL UTILITY COMPARISON")
print("=" * 75)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    REPORT_DIR / "ml_utility_results.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("ML UTILITY EVALUATION COMPLETE")
print("=" * 75)

print(
    f"\nResults saved to:"
    f" {REPORT_DIR / 'ml_utility_results.csv'}"
)