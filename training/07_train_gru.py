
"""Train a GRU model for 30-minute-ahead glucose prediction."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the existing project utilities.
from common import load_data, RESULTS, MODELS, RANDOM_STATE


# -------------------- SETTINGS --------------------

DATASET_NAME = "cgm_t1dm_t2dm"
TARGET = "target_cgm_t30"

SEQUENCE_LENGTH = 8       # 8 observations = 2 hours of history
EPOCHS = 100              # Early stopping may finish earlier
BATCH_SIZE = 256
GRU_UNITS = 64
PATIENCE = 10

MODEL_PATH = MODELS / "gru_cgm_t30.keras"
METRICS_PATH = RESULTS / "gru_metrics.csv"
HISTORY_PATH = RESULTS / "gru_training_history.csv"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# -------------------- LOAD DATA --------------------

df, features, targets, schema = load_data(DATASET_NAME)

if TARGET not in targets:
    raise ValueError(f"Target {TARGET!r} is not in the dataset schema.")

required = ["split", "patient_uid", TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Find the time and segment columns without hardcoding the segment name.
time_col = next(
    (c for c in ["dt", "datetime", "timestamp", "time"] if c in df.columns),
    None,
)
segment_col = next(
    (c for c in ["segment_id", "seg"] if c in df.columns),
    None,
)

if time_col is None:
    raise ValueError("Could not find a timestamp column in the CSV.")
if segment_col is None:
    raise ValueError("Could not find a segment column in the CSV.")

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

# Convert model features and target to numeric values.
for col in features + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=features + [TARGET, time_col, "patient_uid", segment_col, "split"]
).copy()

# The preprocessing pipeline may interpolate missing CGM values.
# We exclude sequences containing interpolated input readings and labels
# whose actual future CGM reading was interpolated.
if "cgm_interpolated" not in df.columns:
    raise ValueError(
        "cgm_interpolated flag not found; inspect the preprocessing "
        "before training to avoid possible temporal leakage."
    )

df = df.sort_values(
    ["patient_uid", segment_col, time_col]
).reset_index(drop=True)

print("Dataset:", DATASET_NAME)
print("Rows available:", len(df))
print("Features:", len(features))
print("Split counts:")
print(df["split"].value_counts().to_string())


# -------------------- BUILD SEQUENCES --------------------

def build_sequences(split_name):
    """Build fixed-length histories without crossing patient/segment boundaries."""

    X_list = []
    y_list = []

    subset = df[df["split"] == split_name]

    grouped = subset.groupby(
        ["patient_uid", segment_col],
        sort=False,
        dropna=False,
    )

    for _, group in grouped:
        group = group.sort_values(time_col).reset_index(drop=True)

        values = group[features].to_numpy(dtype=np.float32)
        labels = group[TARGET].to_numpy(dtype=np.float32)
        times = group[time_col]
        interpolated = (
            group["cgm_interpolated"].fillna(1).to_numpy() > 0
        )

        # The label at row i refers to the CGM reading two 15-minute
        # steps later. That future reading must be real, not interpolated.
        for end in range(SEQUENCE_LENGTH - 1, len(group) - 2):
            start = end - SEQUENCE_LENGTH + 1
            future = end + 2

            # Require regular 15-minute history and a 30-minute horizon.
            history_times = times.iloc[start:end + 1]
            if not history_times.diff().dropna().eq(
                pd.Timedelta(minutes=15)
            ).all():
                continue

            if times.iloc[future] - times.iloc[end] != pd.Timedelta(minutes=30):
                continue

            # Avoid interpolated CGM values in the input history or label.
            if interpolated[start:end + 1].any():
                continue
            if interpolated[future]:
                continue

            X_list.append(values[start:end + 1])
            y_list.append(labels[end])

    if not X_list:
        raise ValueError(
            f"No valid sequences found for split {split_name!r}. "
            "Check the timestamp, segment, and interpolation columns."
        )

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


X_train, y_train = build_sequences("train")
X_val, y_val = build_sequences("val")
X_test, y_test = build_sequences("test")

print("\nSequence shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# -------------------- BUILD GRU MODEL --------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, len(features))
    ),
    tf.keras.layers.GRU(GRU_UNITS, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ],
)

model.summary()


# -------------------- TRAIN --------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)

pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False)


# -------------------- FINAL EVALUATION --------------------

def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


val_predictions = model.predict(X_val, verbose=0).reshape(-1)
test_predictions = model.predict(X_test, verbose=0).reshape(-1)

val_metrics = calculate_metrics(y_val, val_predictions)
test_metrics = calculate_metrics(y_test, test_predictions)

results = pd.DataFrame([
    {
        "model": "GRU",
        "dataset": DATASET_NAME,
        "target": TARGET,
        "split": "validation",
        "sequences": len(y_val),
        **val_metrics,
    },
    {
        "model": "GRU",
        "dataset": DATASET_NAME,
        "target": TARGET,
        "split": "test",
        "sequences": len(y_test),
        **test_metrics,
    },
])

results.to_csv(METRICS_PATH, index=False)

print("\n===== GRU RESULTS =====")
print(results.to_string(index=False))
print(f"\nBest model saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")
print(f"Training history saved to: {HISTORY_PATH}")
print(f"Epochs completed: {len(history.history['loss'])}")
