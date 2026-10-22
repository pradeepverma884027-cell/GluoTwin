
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\pradeep\Desktop\gluotwinv2")
BASE = ROOT / "dataset_from_beginning" / "DATASET"

DATA = (
    BASE / "DATASET" / "processed" /
    "cgm_t1dm_t2dm_model_ready.csv"
)
SCHEMA = (
    BASE / "DATASET" / "processed" /
    "schema_cgm_t1dm_t2dm.json"
)

def main():
    df = pd.read_csv(DATA)

    with open(SCHEMA, "r", encoding="utf-8") as f:
        schema = json.load(f)

    mean, std = schema["scaled_cols_mean_std"]["cgm"]

    print("=" * 65)
    print("GLUCOTWIN — CORRECTED TARGET SCALE VERIFICATION")
    print("=" * 65)
    print(f"CGM inverse transform: value * {std:.8f} + {mean:.8f}")

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)

    groups = df.groupby(["patient_uid", "segment_id"], sort=False)

    for horizon, steps in [(30, 2), (60, 4), (90, 6), (120, 8)]:
        future_scaled = groups["cgm"].shift(-steps)
        future_dt = groups["dt"].shift(-steps)
        future_interpolated = groups["cgm_interpolated"].shift(-steps)

        expected_dt = df["dt"] + pd.Timedelta(minutes=horizon)
        future_raw = future_scaled * std + mean

        valid = (
            future_dt.eq(expected_dt)
            & future_scaled.notna()
            & future_interpolated.eq(0)
        )

        target = f"target_cgm_t{horizon}"

        if target in df.columns:
            valid &= df[target].notna()
            stored = pd.to_numeric(df[target], errors="coerce")
            valid &= stored.notna()
            actual = stored[valid]
            predicted_from_cgm = future_raw[valid]
            comparison = "stored target vs inverse-transformed future CGM"
        else:
            actual = future_raw[valid]
            predicted_from_cgm = None
            comparison = "inverse-transformed future CGM only"

        if len(actual) == 0:
            print(f"\n+{horizon} min: no valid rows")
            continue

        print(f"\n+{horizon} minutes — {comparison}")
        print("Valid rows:", len(actual))
        print("Mean:", round(actual.mean(), 3))
        print("Std:", round(actual.std(), 3))
        print("Min:", round(actual.min(), 3))
        print("Max:", round(actual.max(), 3))

        if predicted_from_cgm is not None:
            diff = actual.to_numpy() - predicted_from_cgm.to_numpy()
            corr = actual.corr(predicted_from_cgm)
            print("Mean absolute difference:", round(np.abs(diff).mean(), 6))
            print("Maximum absolute difference:", round(np.abs(diff).max(), 6))
            print("Correlation:", round(corr, 6))

    print("\nVerification complete. No files modified.")

if __name__ == "__main__":
    main()
