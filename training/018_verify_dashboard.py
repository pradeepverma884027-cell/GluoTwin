from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# GLUCOTWIN - DASHBOARD METRIC VERIFICATION
# ============================================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results_causal"

metrics_path = RESULTS / "causal_multihorizon_metrics.csv"
predictions_path = RESULTS / "causal_test_predictions.csv"
intervals_path = RESULTS / "conformal_interval_results.csv"

for path in [metrics_path, predictions_path, intervals_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

metrics = pd.read_csv(metrics_path)
predictions = pd.read_csv(predictions_path)
intervals = pd.read_csv(intervals_path)

print("=" * 70)
print("GLUCOTWIN DASHBOARD VERIFICATION")
print("=" * 70)

for horizon in [30, 60, 90, 120]:
    print(f"\n{'-' * 60}")
    print(f"HORIZON +{horizon} MINUTES")
    print(f"{'-' * 60}")

    for split in ["validation", "test"]:
        subset = predictions[
            (predictions["horizon_minutes"] == horizon)
            & (predictions["split"].str.lower() == split)
        ].dropna(subset=["actual_mgdl", "prediction_mgdl"])

        saved = metrics[
            (metrics["horizon_minutes"] == horizon)
            & (metrics["split"].str.lower() == split)
            & metrics["model"].str.contains("LightGBM", case=False)
        ]

        if subset.empty or saved.empty:
            print(f"{split}: MISSING DATA")
            continue

        y_true = subset["actual_mgdl"].to_numpy()
        y_pred = subset["prediction_mgdl"].to_numpy()

        calculated = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        }

        expected = saved.iloc[0]

        print(
            f"\n{split.upper()} | rows={len(subset):,}"
        )

        all_match = True

        for name, value in calculated.items():
            expected_value = float(expected[name])
            difference = abs(value - expected_value)
            matches = difference < 0.02
            all_match = all_match and matches

            print(
                f"{name}: calculated={value:.5f} | "
                f"saved={expected_value:.5f} | "
                f"{'PASS' if matches else 'CHECK'}"
            )

        print("METRIC CHECK:", "PASS" if all_match else "REVIEW")

    # Recalculate interval coverage on test predictions.
    test = predictions[
        (predictions["horizon_minutes"] == horizon)
        & (predictions["split"].str.lower() == "test")
    ].dropna(subset=["actual_mgdl", "prediction_mgdl"])

    for label in ["80%", "90%"]:
        row = intervals[
            (intervals["horizon_minutes"] == horizon)
            & (intervals["interval"] == label)
        ]

        if test.empty or row.empty:
            print(f"{label} interval: MISSING DATA")
            continue

        radius = float(row.iloc[0]["calibration_quantile_mgdl"])

        actual = test["actual_mgdl"].to_numpy()
        predicted = test["prediction_mgdl"].to_numpy()

        coverage = np.mean(
            (actual >= predicted - radius)
            & (actual <= predicted + radius)
        )

        saved_coverage = float(row.iloc[0]["test_coverage"])

        print(
            f"\n{label} interval coverage: "
            f"recalculated={coverage:.4%} | "
            f"saved={saved_coverage:.4%} | "
            f"{'PASS' if abs(coverage - saved_coverage) < 0.005 else 'CHECK'}"
        )

print("\n" + "=" * 70)
print("VERIFICATION FINISHED")
print("=" * 70)
print("PASS means the recalculated values agree within the stated tolerance.")
print("This checks consistency, not clinical validity or absence of leakage.")
