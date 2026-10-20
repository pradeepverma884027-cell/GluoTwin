
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

CANDIDATES = [
    PROJECT / "DATASET" / "processed",
    PROJECT / "processed",
    PROJECT / "DATASET" / "DATASET" / "processed",
]

DATASET_NAME = "cgm_t1dm_t2dm"

folder = next(
    (
        p for p in CANDIDATES
        if (p / f"{DATASET_NAME}_model_ready.csv").exists()
        and (p / f"schema_{DATASET_NAME}.json").exists()
    ),
    None,
)

if folder is None:
    raise FileNotFoundError("Dataset and schema files were not found.")

df = pd.read_csv(
    folder / f"{DATASET_NAME}_model_ready.csv",
    low_memory=False
)

with open(
    folder / f"schema_{DATASET_NAME}.json",
    encoding="utf-8"
) as f:
    schema = json.load(f)

print("=" * 65)
print("GLUCOTWIN — TARGET SCALE CHECK")
print("=" * 65)

print("\n--- TARGET SUMMARY ---")

targets = [
    "target_cgm_t30",
    "target_cgm_t60",
]

for target in targets:
    if target not in df.columns:
        continue

    values = pd.to_numeric(df[target], errors="coerce")

    print(f"\n{target}")
    print(f"Min:    {values.min():.6f}")
    print(f"Max:    {values.max():.6f}")
    print(f"Mean:   {values.mean():.6f}")
    print(f"Std:    {values.std():.6f}")
    print(f"Median: {values.median():.6f}")

print("\n--- SCHEMA SCALING INFORMATION ---")
scaling = schema.get("scaled_cols_mean_std", {})

for target in [
    "target_cgm_t30",
    "target_cgm_t60",
    "target_cgm_t90",
    "target_cgm_t120",
]:
    print(f"\n{target}:")
    if isinstance(scaling, dict) and target in scaling:
        print(scaling[target])
    elif isinstance(scaling, dict):
        print("No scaling entry found.")
    else:
        print("Scaling metadata format:", type(scaling).__name__)
        print(scaling)

print("\n--- SCHEMA UNSCALED COLUMNS ---")
print(schema.get("unscaled_cols", []))

print("\n--- COMPLETE")
print("No files modified; no models retrained.")
