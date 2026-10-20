from pathlib import Path
import pandas as pd
import joblib

# ============================================================
# GLUCOTWIN: VERIFY CAUSAL MULTI-HORIZON RESULTS
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TRAINING_DIR / "results_causal"
MODELS_DIR = RESULTS_DIR / "models"

METRICS_FILE = RESULTS_DIR / "causal_multihorizon_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "causal_test_predictions.csv"
COUNTS_FILE = RESULTS_DIR / "causal_row_counts.csv"

HORIZONS = [30, 60, 90, 120]

print("=" * 65)
print("GLUCOTWIN - CAUSAL RESULTS VERIFICATION")
print("=" * 65)

errors = []
warnings = []

# ------------------------------------------------------------
# 1. Verify directories
# ------------------------------------------------------------

print("\n[1/5] Checking result directories...")

for folder in [RESULTS_DIR, MODELS_DIR]:
    if folder.exists():
        print(f"[OK] {folder}")
    else:
        print(f"[MISSING] {folder}")
        errors.append(f"Missing directory: {folder}")

if errors:
    print("\nCannot continue: required directories are missing.")
    print("Check that this script is inside the training folder.")
    raise SystemExit(1)

# ------------------------------------------------------------
# 2. Verify model files
# ------------------------------------------------------------

print("\n[2/5] Checking saved models...")

for horizon in HORIZONS:
    model_path = MODELS_DIR / f"lightgbm_causal_t{horizon}.joblib"

    if not model_path.exists():
        print(f"[MISSING] {model_path.name}")
        errors.append(f"Missing model: {model_path.name}")
        continue

    try:
        model = joblib.load(model_path)

        if hasattr(model, "n_features_in_"):
            print(
                f"[OK] {model_path.name} | "
                f"Features: {model.n_features_in_}"
            )
        else:
            print(f"[OK] {model_path.name} | Loaded successfully")

    except Exception as exc:
        print(f"[ERROR] {model_path.name}: {exc}")
        errors.append(f"Could not load {model_path.name}")

# ------------------------------------------------------------
# 3. Verify metrics
# ------------------------------------------------------------

print("\n[3/5] Checking evaluation metrics...")

if not METRICS_FILE.exists():
    print(f"[MISSING] {METRICS_FILE.name}")
    errors.append("Metrics CSV is missing.")
else:
    metrics = pd.read_csv(METRICS_FILE)

    print(f"Rows: {len(metrics)}")
    print("Columns:", list(metrics.columns))
    print("\nMetrics preview:")
    print(metrics.to_string(index=False))

    if metrics.empty:
        errors.append("Metrics CSV is empty.")

    required = {"horizon_minutes", "split", "MAE", "RMSE", "R2"}
    missing = required - set(metrics.columns)

    if missing:
        warnings.append(
            "Metrics use different column names or lack: "
            + ", ".join(sorted(missing))
        )
    else:
        if metrics.duplicated(["horizon_minutes", "split"]).any():
            warnings.append("Duplicate horizon/split metric rows found.")

        if metrics[["MAE", "RMSE", "R2"]].isna().any().any():
            warnings.append("Some metrics contain missing values.")

        if (metrics[["MAE", "RMSE"]] < 0).any().any():
            warnings.append("Negative MAE or RMSE detected.")

        print("\nTest-set summary:")
        test_metrics = metrics[
            metrics["split"].astype(str).str.lower().eq("test")
        ]

        if not test_metrics.empty:
            print(
                test_metrics[
                    ["horizon_minutes", "MAE", "RMSE", "R2"]
                ].sort_values("horizon_minutes").to_string(index=False)
            )
        else:
            warnings.append("No test-set metrics found.")

# ------------------------------------------------------------
# 4. Verify test predictions
# ------------------------------------------------------------

print("\n[4/5] Checking test predictions...")

if not PREDICTIONS_FILE.exists():
    print(f"[MISSING] {PREDICTIONS_FILE.name}")
    errors.append("Prediction CSV is missing.")
else:
    predictions = pd.read_csv(PREDICTIONS_FILE)

    print(f"Prediction rows: {len(predictions)}")
    print("Columns:", list(predictions.columns))
    print("\nFirst five rows:")
    print(predictions.head().to_string(index=False))

    if predictions.empty:
        errors.append("Prediction CSV is empty.")

    # Identify likely target and prediction columns.
    target_candidates = [
        c for c in predictions.columns
        if c.lower() in {
            "actual", "y_true", "target", "actual_glucose",
            "actual_cgm", "y_actual"
        }
    ]

    prediction_candidates = [
        c for c in predictions.columns
        if c.lower() in {
            "prediction", "predicted", "y_pred",
            "predicted_glucose", "predicted_cgm"
        }
    ]

    if target_candidates and prediction_candidates:
        actual_col = target_candidates[0]
        predicted_col = prediction_candidates[0]

        actual = pd.to_numeric(
            predictions[actual_col], errors="coerce"
        )
        predicted = pd.to_numeric(
            predictions[predicted_col], errors="coerce"
        )

        valid = actual.notna() & predicted.notna()

        print(f"\nValid actual/prediction pairs: {valid.sum()}")
        print(f"Missing/invalid pairs: {(~valid).sum()}")

        if valid.any():
            print(
                "Prediction range:",
                float(predicted[valid].min()),
                "to",
                float(predicted[valid].max())
            )
            print(
                "Actual range:",
                float(actual[valid].min()),
                "to",
                float(actual[valid].max())
            )
    else:
        print(
            "\nColumn names differ from common names; "
            "manual inspection may be needed."
        )

# ------------------------------------------------------------
# 5. Verify row counts and summarize
# ------------------------------------------------------------

print("\n[5/5] Checking row counts...")

if not COUNTS_FILE.exists():
    print(f"[MISSING] {COUNTS_FILE.name}")
    warnings.append("Row-count CSV is missing.")
else:
    counts = pd.read_csv(COUNTS_FILE)
    print(f"Row-count records: {len(counts)}")
    print(counts.to_string(index=False))

print("\n" + "=" * 65)
print("VERIFICATION SUMMARY")
print("=" * 65)

if not errors and not warnings:
    print("All structural checks passed.")
elif not errors:
    print("Core files are present, but review the warnings.")
else:
    print(f"Errors found: {len(errors)}")

for item in errors:
    print("[ERROR]", item)

for item in warnings:
    print("[WARNING]", item)

print("\nImportant:")
print("- This script checks files and basic metric consistency.")
print("- It does not prove that features are leakage-free.")
print("- It does not establish clinical validity or safety.")
print("\nVerification finished.")
