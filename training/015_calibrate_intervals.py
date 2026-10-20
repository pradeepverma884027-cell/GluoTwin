from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# GLUCOTWIN - SPLIT CONFORMAL PREDICTION INTERVALS
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TRAINING_DIR / "results_causal"

PREDICTIONS_FILE = (
    RESULTS_DIR / "causal_test_predictions.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "conformal_interval_results.csv"
)

ALPHA_LEVELS = {
    "80%": 0.20,
    "90%": 0.10,
}

print("=" * 65)
print("GLUCOTWIN - PREDICTION INTERVAL CALIBRATION")
print("=" * 65)

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Predictions file not found:\n{PREDICTIONS_FILE}"
    )

df = pd.read_csv(PREDICTIONS_FILE)

required = {
    "horizon_minutes",
    "split",
    "prediction_mgdl",
    "actual_mgdl",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

df["split"] = df["split"].astype(str).str.lower()

for col in [
    "horizon_minutes",
    "prediction_mgdl",
    "actual_mgdl",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "horizon_minutes",
        "prediction_mgdl",
        "actual_mgdl",
    ]
).copy()

df["absolute_error"] = (
    df["actual_mgdl"] - df["prediction_mgdl"]
).abs()

results = []

for horizon in sorted(df["horizon_minutes"].unique()):
    horizon_df = df[
        df["horizon_minutes"] == horizon
    ]

    validation = horizon_df[
        horizon_df["split"] == "validation"
    ]

    test = horizon_df[
        horizon_df["split"] == "test"
    ]

    print(f"\n{'-' * 55}")
    print(f"HORIZON: +{int(horizon)} MINUTES")
    print(f"{'-' * 55}")

    if validation.empty or test.empty:
        print("Skipping: validation or test rows are missing.")
        continue

    validation_errors = validation[
        "absolute_error"
    ].to_numpy()

    test_actual = test["actual_mgdl"].to_numpy()
    test_prediction = test["prediction_mgdl"].to_numpy()

    print(f"Validation examples: {len(validation)}")
    print(f"Test examples: {len(test)}")

    for label, alpha in ALPHA_LEVELS.items():
        # Finite-sample split conformal quantile.
        n = len(validation_errors)
        rank = int(np.ceil((n + 1) * (1 - alpha)))

        if rank > n:
            q = np.inf
        else:
            q = np.sort(validation_errors)[rank - 1]

        lower = test_prediction - q
        upper = test_prediction + q

        covered = (
            (test_actual >= lower)
            & (test_actual <= upper)
        )

        coverage = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))

        result = {
            "horizon_minutes": int(horizon),
            "interval": label,
            "validation_rows": len(validation),
            "test_rows": len(test),
            "calibration_quantile_mgdl": float(q),
            "test_coverage": coverage,
            "target_coverage": 1 - alpha,
            "mean_interval_width_mgdl": mean_width,
        }

        results.append(result)

        print(f"\nTarget interval: {label}")
        print(f"Calibration radius: {q:.2f} mg/dL")
        print(f"Test coverage: {coverage:.1%}")
        print(f"Mean interval width: {mean_width:.2f} mg/dL")

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No interval results were generated.")

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("CALIBRATION COMPLETE")
print("=" * 65)
print(f"Saved results to:\n{OUTPUT_FILE}")
print("\nImportant:")
print("- Calibration uses validation residuals only.")
print("- Test coverage evaluates the resulting intervals.")
print("- Coverage is empirical, not a guarantee for every patient.")
print("- These intervals are not clinical safety guarantees.")
