
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# --------------------------------------------------
# PATHS
# --------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "results"
MODELS = RESULTS / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    PROJECT / "DATASET" / "processed",
    PROJECT / "processed",
    PROJECT / "DATASET" / "DATASET" / "processed",
]

DATASET_NAME = "cgm_t1dm_t2dm"
TARGETS = {
    30: "target_cgm_t30",
    60: "target_cgm_t60",
    90: "target_cgm_t90",
    120: "target_cgm_t120",
}

print("=" * 70)
print("GLUCOTWIN — MULTI-HORIZON MODEL TRAINING")
print("=" * 70)

# --------------------------------------------------
# FIND DATA AND SCHEMA
# --------------------------------------------------
data_dir = None

for folder in DATA_CANDIDATES:
    csv_path = folder / f"{DATASET_NAME}_model_ready.csv"
    schema_path = folder / f"schema_{DATASET_NAME}.json"

    if csv_path.is_file() and schema_path.is_file():
        data_dir = folder
        break

if data_dir is None:
    raise FileNotFoundError(
        "Could not locate the model-ready CSV and schema."
    )

source_path = data_dir / f"{DATASET_NAME}_model_ready.csv"
schema_path = data_dir / f"schema_{DATASET_NAME}.json"

print("Dataset:", source_path)

df = pd.read_csv(source_path, low_memory=False)

with open(schema_path, "r", encoding="utf-8") as f:
    schema = json.load(f)

# Use only features declared in the original schema.
features = list(schema["features"])

# Never allow target columns to become input features.
target_names = set(TARGETS.values())
features = [c for c in features if c not in target_names]

required = {
    "patient_uid", "segment_id", "dt", "split", "cgm"
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Verify split labels exist.
df["split"] = df["split"].astype(str).str.lower()
valid_splits = {"train", "val", "validation", "valid", "test"}

if not set(df["split"].unique()).issubset(valid_splits):
    raise ValueError(
        f"Unexpected split values: {df['split'].unique().tolist()}"
    )

# --------------------------------------------------
# BUILD CONSISTENT FUTURE LABELS
# --------------------------------------------------
df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

if df["dt"].isna().any():
    raise ValueError("Invalid timestamps found in dataset.")

df = df.sort_values(
    ["patient_uid", "segment_id", "dt"]
).reset_index(drop=True)

group_cols = ["patient_uid", "segment_id"]
grouped = df.groupby(group_cols, sort=False)

# Do not train on labels whose future glucose is marked
# as interpolated, when that flag is available.
has_interpolation_flag = "cgm_interpolated" in df.columns

label_counts = {}

for minutes, target in TARGETS.items():
    steps = minutes // 15

    future_cgm = grouped["cgm"].shift(-steps)
    future_dt = grouped["dt"].shift(-steps)

    elapsed_minutes = (
        future_dt - df["dt"]
    ).dt.total_seconds() / 60

    valid = elapsed_minutes.eq(minutes) & future_cgm.notna()

    if has_interpolation_flag:
        future_interpolated = grouped[
            "cgm_interpolated"
        ].shift(-steps)

        valid &= pd.to_numeric(
            future_interpolated, errors="coerce"
        ).eq(0)

    df[target] = future_cgm.where(valid)
    label_counts[target] = int(df[target].notna().sum())

print("\nValid labels:")
for target, count in label_counts.items():
    print(f"  {target}: {count:,}")

# --------------------------------------------------
# CHECK PATIENT SPLIT INTEGRITY
# --------------------------------------------------
patient_split_counts = df.groupby("patient_uid")["split"].nunique()

if (patient_split_counts > 1).any():
    raise ValueError(
        "Some patients appear in multiple splits. "
        "Stop training and audit the split first."
    )

train_patients = set(
    df.loc[df["split"] == "train", "patient_uid"].unique()
)
val_patients = set(
    df.loc[df["split"].isin(["val", "validation", "valid"]),
           "patient_uid"].unique()
)
test_patients = set(
    df.loc[df["split"] == "test", "patient_uid"].unique()
)

if (
    train_patients & val_patients
    or train_patients & test_patients
    or val_patients & test_patients
):
    raise ValueError("Patient overlap detected between splits.")

print("\nPatient counts:")
print("  Train:", len(train_patients))
print("  Validation:", len(val_patients))
print("  Test:", len(test_patients))

# --------------------------------------------------
# FEATURE PREPARATION
# --------------------------------------------------
absent_features = [c for c in features if c not in df.columns]

if absent_features:
    raise ValueError(
        f"Schema features missing from data: {absent_features}"
    )

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only numeric model features.
features = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c])
]

if not features:
    raise ValueError("No numeric features remain.")

# --------------------------------------------------
# TRAIN ONE MODEL PER HORIZON
# --------------------------------------------------
all_metrics = []

for minutes, target in TARGETS.items():
    print(f"\n{'-' * 60}")
    print(f"Training horizon: +{minutes} minutes")
    print(f"Target: {target}")

    train = df[df["split"] == "train"].dropna(
        subset=[target]
    ).copy()

    val = df[df["split"].isin(
        ["val", "validation", "valid"]
    )].dropna(subset=[target]).copy()

    test = df[df["split"] == "test"].dropna(
        subset=[target]
    ).copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split for {target}. "
            "Check labels and split assignments."
        )

    # Fit imputation values on training data only.
    medians = train[features].median().fillna(0)

    X_train = train[features].fillna(medians)
    X_val = val[features].fillna(medians)
    X_test = test[features].fillna(medians)

    y_train = train[target].astype(float)
    y_val = val[target].astype(float)
    y_test = test[target].astype(float)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    elapsed = time.time() - start

    for split_name, X, y in [
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        predictions = model.predict(X)

        metrics = {
            "horizon_minutes": minutes,
            "target": target,
            "split": split_name,
            "rows": len(y),
            "MAE": mean_absolute_error(y, predictions),
            "RMSE": np.sqrt(
                mean_squared_error(y, predictions)
            ),
            "R2": r2_score(y, predictions),
        }

        all_metrics.append(metrics)

        print(
            f"{split_name}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"R2={metrics['R2']:.5f}"
        )

    # Save model and metadata in a consistent artifact.
    model_path = MODELS / f"lightgbm_cgm_t{minutes}.joblib"

    import joblib

    artifact = {
        "model": model,
        "features": features,
        "medians": medians.to_dict(),
        "target": target,
        "horizon_minutes": minutes,
        "dataset_name": DATASET_NAME,
        "schema": schema,
        "task": "regression",
    }

    joblib.dump(artifact, model_path)

    print(f"Saved model: {model_path}")
    print(f"Training time: {elapsed:.1f} seconds")

# --------------------------------------------------
# SAVE COMPARISON RESULTS
# --------------------------------------------------
metrics_path = RESULTS / "multihorizon_metrics.csv"

pd.DataFrame(all_metrics).to_csv(
    metrics_path, index=False
)

print("\n" + "=" * 70)
print("MULTI-HORIZON TRAINING COMPLETE")
print("Metrics:", metrics_path)
print("Models:", MODELS)
print("=" * 70)


