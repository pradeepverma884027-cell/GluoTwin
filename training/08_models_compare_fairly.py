
"""Fair comparison of five ML models and GRU on identical test examples."""

import gc

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import load_data, RESULTS, MODELS


# ---------------- SETTINGS ----------------

DATASET_NAME = "cgm_t1dm_t2dm"
TARGET = "target_cgm_t30"
SEQ_LEN = 8

INTERVAL = pd.Timedelta(minutes=15)
HORIZON = pd.Timedelta(minutes=30)

MODEL_FILES = {
    "Linear Regression": "linear__cgm_t1dm_t2dm__target_cgm_t30.joblib",
    "Random Forest": "random_forest__cgm_t1dm_t2dm__target_cgm_t30.joblib",
    "Extra Trees": "extra_trees__cgm_t1dm_t2dm__target_cgm_t30.joblib",
    "LightGBM": "lightgbm__cgm_t1dm_t2dm__target_cgm_t30.joblib",
    "XGBoost": "xgboost__cgm_t1dm_t2dm__target_cgm_t30.joblib",
}

GRU_FILE = MODELS / "gru_cgm_t30.keras"


def find_column(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def calculate_metrics(actual, predicted):
    predicted = np.asarray(predicted).reshape(-1)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


# ---------------- LOAD DATA ----------------

df, features, targets, schema = load_data(DATASET_NAME)

if TARGET not in targets:
    raise ValueError(f"Target {TARGET!r} is missing from the schema.")

time_col = find_column(
    df.columns, ["dt", "datetime", "timestamp", "time"]
)
segment_col = find_column(
    df.columns, ["segment_id", "seg"]
)

required = [
    "split",
    "patient_uid",
    TARGET,
    "cgm_interpolated",
]

if time_col is None or segment_col is None:
    raise ValueError(
        "Timestamp or segment column not found. "
        "Check the processed CSV column names."
    )

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

for col in features + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=features + [
        TARGET, time_col, "patient_uid", segment_col, "split"
    ]
).copy()

df = df.sort_values(
    ["patient_uid", segment_col, time_col]
).reset_index(drop=True)


# ---------------- COMMON TEST EXAMPLES ----------------

test = df[df["split"] == "test"]

X_rows = []
X_sequences = []
y_true = []
metadata = []

for (patient, segment), group in test.groupby(
    ["patient_uid", segment_col],
    sort=False,
    dropna=False,
):
    group = group.sort_values(time_col).reset_index(drop=True)

    values = group[features].to_numpy(dtype=np.float32)
    labels = group[TARGET].to_numpy(dtype=np.float32)
    times = group[time_col]

    interpolated = (
        group["cgm_interpolated"].fillna(1).to_numpy() > 0
    )

    for end in range(SEQ_LEN - 1, len(group) - 2):
        start = end - SEQ_LEN + 1
        future = end + 2

        history_times = times.iloc[start:end + 1]

        # Require eight observations at regular 15-minute intervals.
        if not history_times.diff().dropna().eq(INTERVAL).all():
            continue

        # Confirm the future reading is exactly 30 minutes ahead.
        if times.iloc[future] - times.iloc[end] != HORIZON:
            continue

        # Exclude interpolated input readings and interpolated label.
        if interpolated[start:end + 1].any():
            continue

        if interpolated[future]:
            continue

        X_rows.append(values[end])
        X_sequences.append(values[start:end + 1])
        y_true.append(labels[end])

        metadata.append({
            "patient_uid": patient,
            "segment": segment,
            "timestamp": times.iloc[end],
        })

if not X_rows:
    raise ValueError("No eligible test examples were found.")

X_rows = np.asarray(X_rows, dtype=np.float32)
X_sequences = np.asarray(X_sequences, dtype=np.float32)
y_true = np.asarray(y_true, dtype=np.float32)

print("Common eligible test examples:", len(y_true))
print("Tabular feature matrix:", X_rows.shape)
print("GRU sequence matrix:", X_sequences.shape)


# ---------------- EVALUATE TABULAR MODELS ----------------

prediction_columns = {"actual": y_true.copy()}
results = []

for name, filename in MODEL_FILES.items():
    path = MODELS / filename
    print(f"\nEvaluating {name}...")

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    artifact = joblib.load(path)

    try:
        # Your saved .joblib files contain dictionaries.
        if isinstance(artifact, dict):
            estimator = artifact["model"]
            saved_features = artifact["features"]
            saved_medians = artifact["medians"]

            # Reproduce the saved feature order and imputation.
            X_model = pd.DataFrame(X_rows, columns=features)
            X_model = X_model.reindex(columns=saved_features)

            for col in saved_features:
                X_model[col] = pd.to_numeric(
                    X_model[col], errors="coerce"
                ).fillna(saved_medians[col])

            predictions = estimator.predict(X_model)

        else:
            # Fallback for files storing an estimator directly.
            predictions = artifact.predict(X_rows)

        predictions = np.asarray(predictions).reshape(-1)

        if len(predictions) != len(y_true):
            raise ValueError(
                f"{name} returned {len(predictions)} predictions; "
                f"expected {len(y_true)}."
            )

        prediction_columns[name] = predictions
        row = {
            "model": name,
            **calculate_metrics(y_true, predictions),
        }
        results.append(row)

        print(row)

    finally:
        del artifact
        gc.collect()


# ---------------- EVALUATE GRU ----------------

print("\nEvaluating GRU...")

if not GRU_FILE.is_file():
    raise FileNotFoundError(f"GRU model not found: {GRU_FILE}")

gru = tf.keras.models.load_model(GRU_FILE, compile=False)

try:
    gru_predictions = gru.predict(
        X_sequences,
        batch_size=256,
        verbose=1,
    ).reshape(-1)

    if len(gru_predictions) != len(y_true):
        raise ValueError("GRU prediction count does not match test rows.")

    prediction_columns["GRU"] = gru_predictions

    row = {
        "model": "GRU",
        **calculate_metrics(y_true, gru_predictions),
    }
    results.append(row)
    print(row)

finally:
    del gru
    gc.collect()


# ---------------- SAVE COMPARISON ----------------

ranking = (
    pd.DataFrame(results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

ranking.insert(0, "rank_by_RMSE", np.arange(1, len(ranking) + 1))

ranking_path = RESULTS / "fair_model_comparison.csv"
predictions_path = RESULTS / "fair_test_predictions.csv"

ranking.to_csv(ranking_path, index=False)

prediction_df = pd.DataFrame(prediction_columns)
prediction_df.insert(
    0, "timestamp", [item["timestamp"] for item in metadata]
)
prediction_df.insert(
    0, "patient_uid", [item["patient_uid"] for item in metadata]
)
prediction_df.to_csv(predictions_path, index=False)

print("\n===== FAIR MODEL COMPARISON =====")
print(ranking.to_string(index=False))

print(f"\nModels evaluated: {len(results)} / 6")
print(f"Common test examples: {len(y_true)}")
print(f"Ranking saved to: {ranking_path}")
print(f"Predictions saved to: {predictions_path}")
