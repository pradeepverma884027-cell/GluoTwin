from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(r"C:\Users\pradeep\Desktop\gluotwinv2")
BASE = ROOT / "dataset_from_beginning" / "DATASET"
TRAINING = BASE / "training"
RESULTS = TRAINING / "results"
MODELS = RESULTS / "models"

DATA_PATH = (
    BASE / "DATASET" / "processed" /
    "cgm_t1dm_t2dm_model_ready.csv"
)

TARGETS = [
    "target_cgm_t30",
    "target_cgm_t60",
    "target_cgm_t90",
    "target_cgm_t120",
]

def main():
    print("=" * 65)
    print("GLUCOTWIN — MULTIHORIZON METRIC VERIFICATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset shape: {df.shape}")

    # Reconstruct future targets from CGM observations.
    # This verifies whether the generated labels align with timestamps.
    if "cgm" not in df.columns:
        print("\nCould not find a column named 'cgm'.")
        print("Available glucose columns:")
        print([c for c in df.columns if "cgm" in c.lower()])
        return

    if not {"patient_uid", "segment_id", "dt"}.issubset(df.columns):
        print("\nRequired patient/segment/timestamp columns are missing.")
        return

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)

    groups = df.groupby(
        ["patient_uid", "segment_id"], sort=False
    )

    print("\n--- DIRECT TARGET ALIGNMENT CHECK ---")
    for horizon, steps in [(30, 2), (60, 4), (90, 6), (120, 8)]:
        future_cgm = groups["cgm"].shift(-steps)
        future_dt = groups["dt"].shift(-steps)
        expected_dt = df["dt"] + pd.Timedelta(minutes=horizon)

        valid = (
            future_dt.eq(expected_dt)
            & future_cgm.notna()
        )

        target_name = f"target_cgm_t{horizon}"
        if target_name in df.columns:
            stored = df[target_name]
            both = valid & stored.notna()
            if both.any():
                difference = (
                    stored.loc[both] - future_cgm.loc[both]
                ).abs()
                print(
                    f"+{horizon} min: "
                    f"matched={int(both.sum()):,}, "
                    f"max target difference={difference.max():.6f}, "
                    f"mean difference={difference.mean():.6f}"
                )
            else:
                print(f"+{horizon} min: no comparable target rows")
        else:
            print(f"+{horizon} min: stored target column absent")

    print("\n--- MODEL ARTIFACT CHECK ---")
    for target in TARGETS:
        model_path = MODELS / f"lightgbm_{target}.joblib"

        if not model_path.exists():
            print(f"\nMissing model: {model_path.name}")
            continue

        artifact = joblib.load(model_path)
        print(f"\n{target}")
        print("Artifact type:", type(artifact).__name__)

        if isinstance(artifact, dict):
            print("Artifact keys:", list(artifact.keys()))
            print("Task:", artifact.get("task"))
            print("Target:", artifact.get("target"))

            model = artifact.get("model")
            features = artifact.get("features", [])
            medians = artifact.get("medians")

            print("Feature count:", len(features))
            print("Contains target as feature:", target in features)

            suspicious = [
                c for c in features
                if c.startswith("target_")
                or c in {
                    "split", "patient_uid", "segment_id",
                    "dt", "future_cgm"
                }
            ]
            print("Potentially suspicious features:", suspicious)

            if model is not None and features:
                print("Model feature count:", getattr(
                    model, "n_features_in_", "unknown"
                ))
        else:
            print("This artifact is not a dictionary.")
            print("Inspect its training script before evaluating it.")

    print("\n--- EXISTING METRICS ---")
    metrics_path = RESULTS / "multihorizon_metrics.csv"
    if metrics_path.exists():
        print(pd.read_csv(metrics_path).to_string(index=False))
    else:
        print("multihorizon_metrics.csv not found.")

    print("\nCheck complete. No files modified and no models retrained.")

if __name__ == "__main__":
    main()