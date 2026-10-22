
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path(r"C:\Users\pradeep\Desktop\gluotwinv2")
BASE = ROOT / "dataset_from_beginning" / "DATASET"
DATA = BASE / "DATASET" / "processed" / "cgm_t1dm_t2dm_model_ready.csv"
RESULTS = BASE / "training" / "results"
MODELS = RESULTS / "models"

def main():
    print("=" * 65)
    print("GLUCOTWIN — SCALE AND MODEL FILE DIAGNOSTICS")
    print("=" * 65)

    df = pd.read_csv(DATA)
    print("\nDataset shape:", df.shape)

    print("\n--- GLUCOSE-RELATED COLUMNS ---")
    glucose_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["cgm", "glucose", "target"])
    ]

    for col in glucose_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values):
            print(
                f"{col:30s} "
                f"min={values.min():10.4f} "
                f"median={values.median():10.4f} "
                f"mean={values.mean():10.4f} "
                f"max={values.max():10.4f}"
            )

    print("\n--- FUTURE TARGET COMPARISON ---")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)

    groups = df.groupby(["patient_uid", "segment_id"], sort=False)

    for horizon, steps in [(30, 2), (60, 4)]:
        future = groups["cgm"].shift(-steps)
        future_dt = groups["dt"].shift(-steps)
        expected_dt = df["dt"] + pd.Timedelta(minutes=horizon)
        target = f"target_cgm_t{horizon}"

        valid = future_dt.eq(expected_dt) & future.notna()
        if target not in df.columns:
            print(f"{target}: not present in source dataset")
            continue

        valid &= df[target].notna()
        actual = pd.to_numeric(df.loc[valid, target], errors="coerce")
        future_values = pd.to_numeric(future.loc[valid], errors="coerce")
        valid_pairs = actual.notna() & future_values.notna()

        actual = actual[valid_pairs]
        future_values = future_values[valid_pairs]

        if len(actual) == 0:
            print(f"{target}: no comparable rows")
            continue

        diff = actual.to_numpy() - future_values.to_numpy()
        print(f"\n{target} vs future cgm (+{horizon} min)")
        print("Comparable rows:", len(actual))
        print("Target mean:", round(actual.mean(), 4))
        print("Future cgm mean:", round(future_values.mean(), 4))
        print("Target-future mean difference:", round(diff.mean(), 4))
        print("Mean absolute difference:", round(np.abs(diff).mean(), 4))
        print("Correlation:", round(actual.corr(future_values), 6))

    print("\n--- MODEL FILES FOUND ---")
    files = list(MODELS.glob("*")) if MODELS.exists() else []

    if not files:
        print("No files found in:", MODELS)
    else:
        for path in files:
            print(path.name)

            if path.suffix.lower() not in {".joblib", ".pkl"}:
                continue

            try:
                artifact = joblib.load(path)
                if isinstance(artifact, dict):
                    print("  keys:", list(artifact.keys()))
                    print("  target:", artifact.get("target"))
                    print("  feature count:", len(artifact.get("features", [])))
                    features = artifact.get("features", [])
                    suspicious = [
                        f for f in features
                        if f.startswith("target_")
                        or f in {"patient_uid", "segment_id", "dt", "split"}
                    ]
                    print("  suspicious feature names:", suspicious)
            except Exception as e:
                print("  Could not inspect:", e)

    print("\nDiagnostic finished. No files modified.")

if __name__ == "__main__":
    main()
