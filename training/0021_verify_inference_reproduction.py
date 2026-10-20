from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ============================================================
# GLUCOTWIN - HISTORICAL INFERENCE REPRODUCTION TEST
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    TRAINING_DIR.parent
    / "DATASET"
    / "processed"
    / "cgm_t1dm_t2dm_model_ready.csv"
)

RESULTS_DIR = TRAINING_DIR / "results_causal"
MODELS_DIR = RESULTS_DIR / "models"

SAVED_PREDICTIONS_FILE = (
    RESULTS_DIR / "causal_test_predictions.csv"
)

HORIZONS = [30, 60, 90, 120]

print("=" * 72)
print("GLUCOTWIN - HISTORICAL INFERENCE REPRODUCTION")
print("=" * 72)

for path in [DATA_FILE, SAVED_PREDICTIONS_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

df = pd.read_csv(DATA_FILE)

df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

df = (
    df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    )
    .reset_index(drop=True)
)




saved_predictions = pd.read_csv(SAVED_PREDICTIONS_FILE)

print("Dataset rows:", len(df))
print("Saved prediction rows:", len(saved_predictions))
print("Dataset path:", DATA_FILE)

required_metadata = {"split", "patient_uid", "segment_id", "dt", "cgm"}

missing_metadata = required_metadata - set(df.columns)

if missing_metadata:
    raise ValueError(
        f"Dataset is missing required columns: {sorted(missing_metadata)}"
    )

if "_row_index" not in saved_predictions.columns:
    raise ValueError("Saved predictions lack the _row_index column.")

if "horizon_minutes" not in saved_predictions.columns:
    raise ValueError("Saved predictions lack horizon_minutes.")

df["split"] = df["split"].astype(str).str.lower()
saved_predictions["split"] = (
    saved_predictions["split"].astype(str).str.lower()
)

# The original training script used these row indices to identify
# the source rows for each saved prediction.

all_passed = True

for horizon in HORIZONS:
    print("\n" + "-" * 72)
    print(f"HORIZON +{horizon} MINUTES")
    print("-" * 72)

    artifact_path = (
        MODELS_DIR / f"lightgbm_causal_t{horizon}.joblib"
    )

    if not artifact_path.exists():
        print("FAIL: Model artifact missing:", artifact_path)
        all_passed = False
        continue

    artifact = joblib.load(artifact_path)

    features = artifact["features"]
    model = artifact["model"]
    imputer = artifact["imputer"]

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        print("FAIL: Dataset lacks required model features:")
        print(missing_features)
        all_passed = False
        continue

    expected = saved_predictions[
        (saved_predictions["horizon_minutes"] == horizon)
        & (saved_predictions["split"] == "test")
    ].copy()

    if expected.empty:
        print("FAIL: No saved test predictions for this horizon.")
        all_passed = False
        continue

    expected["_row_index"] = pd.to_numeric(
        expected["_row_index"], errors="coerce"
    )

    expected = expected.dropna(subset=["_row_index"])
    expected["_row_index"] = expected["_row_index"].astype(int)

    if (
        (expected["_row_index"] < 0).any()
        or (expected["_row_index"] >= len(df)).any()
    ):
        print("FAIL: Saved row indices are outside dataset bounds.")
        all_passed = False
        continue

    source_rows = df.loc[
        expected["_row_index"].to_numpy()
    ].copy()

    # Confirm these are the same split and patient/segment rows.
    split_matches = (
        source_rows["split"].to_numpy()
        == expected["split"].to_numpy()
    )

    if not split_matches.all():
        print("FAIL: Saved indices do not match the expected split.")
        all_passed = False
        continue

    # Reuse the artifact's fitted imputer and model.
    X = source_rows[features]
    X_imputed = imputer.transform(X)

    # The saved artifact identifies its prediction target as mg/dL.
    # Use model predictions directly; do not inverse-scale them.
    predictions_mgdl = model.predict(X_imputed)


    expected_mgdl = expected["prediction_mgdl"].to_numpy()

    difference = np.abs(predictions_mgdl - expected_mgdl)

    mae_difference = mean_absolute_error(
        expected_mgdl, predictions_mgdl
    )

    max_difference = float(difference.max())

    # Small numerical differences can occur across library versions.
    passed = np.allclose(
        predictions_mgdl,
        expected_mgdl,
        rtol=1e-4,
        atol=0.05,
    )

    print("Test rows checked:", len(expected))
    print(f"Mean absolute prediction difference: {mae_difference:.6f} mg/dL")
    print(f"Maximum absolute difference: {max_difference:.6f} mg/dL")
    print("REPRODUCTION:", "PASS" if passed else "CHECK")

    if not passed:
        all_passed = False

print("\n" + "=" * 72)
print("FINAL RESULT:", "ALL HORIZONS REPRODUCED" if all_passed else "REVIEW REQUIRED")
print("=" * 72)

print(
    "This verifies historical prediction reproduction. "
    "It does not prove causal correctness or clinical validity."
)
