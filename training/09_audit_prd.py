
from pathlib import Path
import json
import pandas as pd

# Automatically locate the project folders
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR.parent

print("=" * 65)
print("GLUCOTWIN — AUTOMATIC PRD DATA AUDIT")
print("=" * 65)

# Search the dataset folder recursively
csv_files = list(DATASET_DIR.rglob("cgm_t1dm_t2dm_model_ready.csv"))
schema_files = list(DATASET_DIR.rglob("cgm_t1dm_t2dm_model_ready_schema.json"))

# If exact filenames differ, discover possible alternatives
if not csv_files:
    csv_files = [
        p for p in DATASET_DIR.rglob("*.csv")
        if "cgm" in p.name.lower()
        and "t1dm" in p.name.lower()
        and "t2dm" in p.name.lower()
    ]

if not schema_files:
    schema_files = [
        p for p in DATASET_DIR.rglob("*.json")
        if "schema" in p.name.lower()
        and "cgm" in p.name.lower()
    ]

print("\n--- DISCOVERED DATA FILES ---")
for path in csv_files:
    print(f"CSV: {path}")

for path in schema_files:
    print(f"SCHEMA: {path}")

if not csv_files:
    print("\nNo matching CGM CSV was found.")
    print("Available CSV files:")
    for path in DATASET_DIR.rglob("*.csv"):
        print(f"  {path.relative_to(DATASET_DIR)}")
    raise FileNotFoundError(
        "Please check the discovered filenames printed above."
    )

# Prefer the processed model-ready file when several matches exist
csv_path = next(
    (p for p in csv_files if "processed" in p.parts),
    csv_files[0]
)

print(f"\nUsing dataset: {csv_path}")

# Load dataset
df = pd.read_csv(csv_path, low_memory=False)

print("\n--- DATASET OVERVIEW ---")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Duplicate rows: {df.duplicated().sum():,}")

# Read schema if found
if schema_files:
    schema_path = schema_files[0]
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    print(f"\nSchema loaded: {schema_path.name}")
    if isinstance(schema, dict):
        print("Schema keys:", list(schema.keys()))
else:
    print("\nNo matching schema JSON found.")

# Feature availability
groups = {
    "CGM / glucose": [
        "cgm", "glucose", "bg", "sensor"
    ],
    "Meal / nutrition": [
        "meal", "carb", "food", "protein", "fat", "fiber"
    ],
    "EHR / clinical": [
        "age", "sex", "bmi", "hba1c", "diabetes_duration",
        "fasting_glucose", "medication", "comorbid"
    ],
    "Wearables / activity": [
        "heart_rate", "hr_", "steps", "activity", "wearable"
    ],
    "Sleep": ["sleep"],
    "Data quality": [
        "missing", "interpolated", "coverage", "quality"
    ],
    "Patient / time": [
        "patient", "timestamp", "datetime", "segment"
    ],
}

print("\n--- FEATURE AVAILABILITY ---")
for group, keywords in groups.items():
    matched = [
        col for col in df.columns
        if any(k in str(col).lower() for k in keywords)
    ]

    print(f"\n{group}: {len(matched)} columns")
    for col in matched:
        print(f"  - {col}")

# Target availability
targets = [
    "target_cgm_t30",
    "target_cgm_t60",
    "target_cgm_t90",
    "target_cgm_t120",
    "target_hypo_60",
]

print("\n--- TARGET AVAILABILITY ---")
for target in targets:
    if target in df.columns:
        print(
            f"FOUND {target}: "
            f"{df[target].notna().sum():,} usable values"
        )
    else:
        print(f"MISSING {target}")

# Missingness report
print("\n--- TOP 15 MISSING COLUMNS ---")
missing = (df.isna().mean() * 100).sort_values(ascending=False)
print(missing.head(15).round(2).to_string())

# Save complete audit report
output_dir = TRAINING_DIR / "results"
output_dir.mkdir(parents=True, exist_ok=True)

report = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_percent": [
        round(df[c].isna().mean() * 100, 2)
        for c in df.columns
    ],
})

report.to_csv(output_dir / "prd_data_audit.csv", index=False)

print("\n" + "=" * 65)
print("AUDIT COMPLETE")
print(f"Report saved to: {output_dir / 'prd_data_audit.csv'}")
print("No model was retrained or dataset modified.")
print("=" * 65)
