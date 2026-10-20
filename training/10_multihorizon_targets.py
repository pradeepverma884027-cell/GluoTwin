
from pathlib import Path
import pandas as pd

TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR.parent

# Find the source dataset automatically
matches = list(
    DATASET_DIR.rglob("cgm_t1dm_t2dm_model_ready.csv")
)

if not matches:
    raise FileNotFoundError(
        "Could not find cgm_t1dm_t2dm_model_ready.csv"
    )

source_path = next(
    (p for p in matches if "processed" in p.parts),
    matches[0]
)

print("Source dataset:", source_path)

df = pd.read_csv(source_path, low_memory=False)

required = {"patient_uid", "segment_id", "dt", "cgm"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Required columns missing: {sorted(missing)}")

df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

if df["dt"].isna().any():
    raise ValueError("Some timestamps could not be parsed.")

# Sort before shifting to future observations
df = df.sort_values(
    ["patient_uid", "segment_id", "dt"]
).reset_index(drop=True)

group_cols = ["patient_uid", "segment_id"]
grouped = df.groupby(group_cols, sort=False)

# Derive future labels only within each patient segment.
# A target is kept only when its timestamp is exactly the
# requested number of minutes after the current timestamp.
for horizon in [90, 120]:
    steps = horizon // 15

    future_glucose = grouped["cgm"].shift(-steps)
    future_time = grouped["dt"].shift(-steps)

    actual_minutes = (
        future_time - df["dt"]
    ).dt.total_seconds() / 60

    valid_time = actual_minutes.eq(horizon)

    target_name = f"target_cgm_t{horizon}"

    df[target_name] = future_glucose.where(valid_time)

    print(f"\n{target_name}")
    print("Valid labels:", int(df[target_name].notna().sum()))
    print("Missing labels:", int(df[target_name].isna().sum()))

# Save separately; do not overwrite the original dataset
output_dir = TRAINING_DIR / "results"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "cgm_multihorizon_targets.csv"
df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("MULTI-HORIZON TARGET CREATION COMPLETE")
print("Saved to:", output_path)
print("Original dataset was not modified.")
print("=" * 60)
