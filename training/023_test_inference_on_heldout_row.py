"""
GlucoTwin inference engine test on a saved eligible held-out row.

This checks that the reusable engine reproduces saved test predictions for
one row that was already retained by the causal training pipeline. It does
not independently prove that the feature pipeline is causal.
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

TRAINING_DIR = Path(__file__).resolve().parent
DATA_FILE = (
    TRAINING_DIR.parent
    / "DATASET"
    / "processed"
    / "cgm_t1dm_t2dm_model_ready.csv"
)
PREDICTIONS_FILE = (
    TRAINING_DIR / "results_causal" / "causal_test_predictions.csv"
)
ENGINE_FILE = TRAINING_DIR / "022_glucotwin_inference_engine.py"
HORIZONS = [30, 60, 90, 120]


def load_engine_class():
    if not ENGINE_FILE.is_file():
        raise FileNotFoundError(f"Inference engine file not found:\n{ENGINE_FILE}")

    spec = importlib.util.spec_from_file_location(
        "glucotwin_inference_engine", ENGINE_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load engine module from {ENGINE_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GlucoTwinInferenceEngine


def main():
    print("=" * 72)
    print("GLUCOTWIN - HELD-OUT ROW INFERENCE TEST")
    print("=" * 72)

    for path in (DATA_FILE, PREDICTIONS_FILE, ENGINE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    # Match the sorted/reset row order used by the causal training pipeline.
    df = pd.read_csv(DATA_FILE)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)
    df["split"] = df["split"].astype(str).str.lower()

    saved = pd.read_csv(PREDICTIONS_FILE)
    saved["split"] = saved["split"].astype(str).str.lower()
    saved["_row_index"] = pd.to_numeric(saved["_row_index"], errors="coerce")
    saved["horizon_minutes"] = pd.to_numeric(
        saved["horizon_minutes"], errors="coerce"
    )

    needed = {"_row_index", "split", "horizon_minutes", "prediction_mgdl"}
    missing = needed - set(saved.columns)
    if missing:
        raise ValueError(
            f"Saved predictions CSV lacks required columns: {sorted(missing)}"
        )

    saved = saved.dropna(subset=["_row_index", "horizon_minutes"]).copy()
    saved["_row_index"] = saved["_row_index"].astype(int)
    saved["horizon_minutes"] = saved["horizon_minutes"].astype(int)

    per_horizon = {}
    for horizon in HORIZONS:
        part = saved[
            (saved["split"] == "test")
            & (saved["horizon_minutes"] == horizon)
        ]
        per_horizon[horizon] = set(part["_row_index"].tolist())

    common_indices = set.intersection(*(per_horizon[h] for h in HORIZONS))
    common_indices = {
        idx for idx in common_indices if 0 <= idx < len(df)
    }

    if not common_indices:
        raise RuntimeError(
            "No common held-out row index was found across all four horizons."
        )

    row_index = min(common_indices)
    source_row = df.iloc[row_index]

    if str(source_row["split"]).lower() != "test":
        raise RuntimeError(
            f"Selected row {row_index} is not in the test split."
        )

    Engine = load_engine_class()
    engine = Engine()

    # This row was included by the causal training pipeline for every horizon.
    # That is inherited eligibility, not an independent causal audit.
    feature_values = {
        name: source_row[name] for name in engine.features
    }

    result = engine.predict(feature_values, history_verified=True)

    print("Dataset rows:", len(df))
    print("Selected held-out row index:", row_index)
    print("Split:", source_row["split"])
    print("Patient UID:", source_row["patient_uid"])
    print("Segment ID:", source_row["segment_id"])
    print("Timestamp:", source_row["dt"])
    print("\nPrediction comparison")
    print("-" * 72)

    all_passed = True
    for item in result["predictions"]:
        horizon = item["horizon_minutes"]
        expected_rows = saved[
            (saved["split"] == "test")
            & (saved["horizon_minutes"] == horizon)
            & (saved["_row_index"] == row_index)
        ]

        if expected_rows.empty:
            print(f"+{horizon} min: FAIL - saved row not found")
            all_passed = False
            continue

        expected = float(expected_rows.iloc[0]["prediction_mgdl"])
        actual = float(item["prediction_mgdl"])
        difference = abs(actual - expected)
        passed = bool(np.isclose(actual, expected, rtol=1e-4, atol=0.05))
        all_passed = all_passed and passed

        print(f"+{horizon} min")
        print(f"  Engine prediction: {actual:.6f} mg/dL")
        print(f"  Saved prediction:  {expected:.6f} mg/dL")
        print(f"  Absolute diff:     {difference:.8f} mg/dL")
        print(f"  Reproduction:      {'PASS' if passed else 'CHECK'}")
        print(
            f"  80% interval:      "
            f"{item['prediction_intervals']['80%']['lower_mgdl']:.2f} to "
            f"{item['prediction_intervals']['80%']['upper_mgdl']:.2f} mg/dL"
        )
        print(
            f"  90% interval:      "
            f"{item['prediction_intervals']['90%']['lower_mgdl']:.2f} to "
            f"{item['prediction_intervals']['90%']['upper_mgdl']:.2f} mg/dL"
        )

    print("\n" + "=" * 72)
    print("FINAL RESULT:", "PASS" if all_passed else "REVIEW REQUIRED")
    print("=" * 72)
    print(
        "This is a reproduction test against a previously eligible held-out "
        "row. It does not independently validate causal feature timing, "
        "interval coverage on new data, or clinical safety."
    )


if __name__ == "__main__":
    main()
