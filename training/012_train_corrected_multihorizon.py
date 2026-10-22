
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\pradeep\Desktop\gluotwinv2")
BASE = ROOT / "dataset_from_beginning" / "DATASET"
TRAINING = BASE / "training"
PROCESSED = BASE / "DATASET" / "processed"

DATA_PATH = PROCESSED / "cgm_t1dm_t2dm_model_ready.csv"
SCHEMA_PATH = PROCESSED / "schema_cgm_t1dm_t2dm.json"

OUTPUT = TRAINING / "results_corrected"
MODEL_DIR = OUTPUT / "models"

HORIZONS = {
    30: 2,
    60: 4,
    90: 6,
    120: 8,
}

RANDOM_STATE = 42


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GLUCOTWIN - CORRECTED MULTIHORIZON TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CHECK FILES
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found:\n{SCHEMA_PATH}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2. LOAD DATA AND SCHEMA
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)

    print("\nDataset shape:", df.shape)

    required = {
        "dt",
        "patient_uid",
        "segment_id",
        "split",
        "cgm",
        "cgm_interpolated",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --------------------------------------------------------
    # 3. FIX SPLIT LABELS: val -> validation
    # --------------------------------------------------------

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"val": "validation"})
    )

    split_names = set(df["split"].unique())

    if not {"train", "validation", "test"}.issubset(split_names):
        raise ValueError(
            f"Expected train/val/test splits; found {sorted(split_names)}"
        )

    print("\nDetected splits:")
    print(df["split"].value_counts().to_string())

    # --------------------------------------------------------
    # 4. CHECK PATIENT-LEVEL SPLIT INTEGRITY
    # --------------------------------------------------------

    patient_split_counts = df.groupby("patient_uid")["split"].nunique()

    if (patient_split_counts > 1).any():
        raise ValueError(
            "Patient leakage detected: some patients occur in multiple splits."
        )

    print("\nPatient-level split check: PASSED")

    # --------------------------------------------------------
    # 5. FEATURES AND ORIGINAL CGM SCALE
    # --------------------------------------------------------

    features = [
        feature
        for feature in schema["features"]
        if feature in df.columns and not feature.startswith("target_")
    ]

    if not features:
        raise ValueError("No valid features found in the schema.")

    if "cgm" not in features:
        raise ValueError("'cgm' must be included in the feature list.")

    cgm_mean, cgm_std = schema["scaled_cols_mean_std"]["cgm"]

    if not np.isfinite(cgm_std) or cgm_std <= 0:
        raise ValueError("Invalid CGM standard deviation in schema.")

    print("\nFeature count:", len(features))
    print("CGM inverse-transform mean:", cgm_mean)
    print("CGM inverse-transform std:", cgm_std)

    # --------------------------------------------------------
    # 6. SORT ROWS AND CREATE SEQUENCE GROUPS
    # --------------------------------------------------------

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)

    groups = df.groupby(
        ["patient_uid", "segment_id"],
        sort=False,
    )

    # Numeric feature conversion
    X_all = df[features].apply(pd.to_numeric, errors="coerce")

    results = []
    test_predictions = []

    # --------------------------------------------------------
    # 7. TRAIN FOUR HORIZON MODELS
    # --------------------------------------------------------

    for horizon, steps in HORIZONS.items():

        print("\n" + "-" * 70)
        print(f"TRAINING HORIZON: +{horizon} MINUTES")
        print("-" * 70)

        # Future CGM is standardized in the dataset.
        future_scaled = groups["cgm"].shift(-steps)
        future_dt = groups["dt"].shift(-steps)
        future_interpolated = groups["cgm_interpolated"].shift(-steps)

        expected_dt = df["dt"] + pd.Timedelta(minutes=horizon)

        # Convert future CGM to the original scale.
        target = future_scaled * cgm_std + cgm_mean

        # Only use observations exactly at the requested future timestamp.
        valid_target = (
            df["dt"].notna()
            & future_dt.eq(expected_dt)
            & future_scaled.notna()
            & future_interpolated.eq(0)
            & pd.to_numeric(
                df["cgm_interpolated"], errors="coerce"
            ).eq(0)
            & target.notna()
            & np.isfinite(target)
        )

        # Use the dataset's existing patient-level assignments.
        train_mask = valid_target & df["split"].eq("train")
        val_mask = valid_target & df["split"].eq("validation")
        test_mask = valid_target & df["split"].eq("test")

        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            raise ValueError(
                f"Empty train, validation, or test set for +{horizon} minutes."
            )

        X_train = X_all.loc[train_mask]
        X_val = X_all.loc[val_mask]
        X_test = X_all.loc[test_mask]

        y_train = target.loc[train_mask]
        y_val = target.loc[val_mask]
        y_test = target.loc[test_mask]

        print("Training rows:", len(y_train))
        print("Validation rows:", len(y_val))
        print("Test rows:", len(y_test))

        # ----------------------------------------------------
        # 8. IMPUTE MISSING FEATURES USING TRAINING DATA ONLY
        # ----------------------------------------------------

        imputer = SimpleImputer(strategy="median")

        X_train_i = imputer.fit_transform(X_train)
        X_val_i = imputer.transform(X_val)
        X_test_i = imputer.transform(X_test)

        # ----------------------------------------------------
        # 9. TRAIN LIGHTGBM
        # ----------------------------------------------------

        model = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(X_train_i, y_train)

        # ----------------------------------------------------
        # 10. EVALUATE MODEL AND LAST-CGM BASELINE
        # ----------------------------------------------------

        for split_name, X_i, y_true, mask in [
            ("validation", X_val_i, y_val, val_mask),
            ("test", X_test_i, y_test, test_mask),
        ]:

            predictions = model.predict(X_i)

            model_metrics = calculate_metrics(y_true, predictions)

            results.append({
                "horizon_minutes": horizon,
                "model": "LightGBM",
                "split": split_name,
                "rows": len(y_true),
                **model_metrics,
            })

            print(
                f"\nLightGBM - {split_name.upper()}\n"
                f"MAE:  {model_metrics['MAE']:.4f}\n"
                f"RMSE: {model_metrics['RMSE']:.4f}\n"
                f"R2:   {model_metrics['R2']:.5f}"
            )

            # Baseline: predict that future glucose equals current glucose.
            baseline = (
                pd.to_numeric(df.loc[mask, "cgm"], errors="coerce")
                * cgm_std
                + cgm_mean
            ).to_numpy()

            baseline_metrics = calculate_metrics(y_true, baseline)

            results.append({
                "horizon_minutes": horizon,
                "model": "Last_CGM_Baseline",
                "split": split_name,
                "rows": len(y_true),
                **baseline_metrics,
            })

            print(
                f"\nLAST-CGM BASELINE - {split_name.upper()}\n"
                f"MAE:  {baseline_metrics['MAE']:.4f}\n"
                f"RMSE: {baseline_metrics['RMSE']:.4f}\n"
                f"R2:   {baseline_metrics['R2']:.5f}"
            )

            # Save test predictions for later analysis.
            if split_name == "test":
                prediction_frame = df.loc[
                    mask,
                    ["patient_uid", "segment_id", "dt"],
                ].copy()

                prediction_frame["horizon_minutes"] = horizon
                prediction_frame["actual"] = y_true.to_numpy()
                prediction_frame["prediction"] = predictions
                prediction_frame["baseline_prediction"] = baseline

                test_predictions.append(prediction_frame)

        # ----------------------------------------------------
        # 11. SAVE MODEL AND IMPUTER TOGETHER
        # ----------------------------------------------------

        artifact = {
            "model": model,
            "imputer": imputer,
            "features": features,
            "target": f"target_cgm_t{horizon}",
            "horizon_minutes": horizon,
            "cgm_mean": float(cgm_mean),
            "cgm_std": float(cgm_std),
            "schema": schema,
            "task": "regression",
            "scale": "original_cgm_scale",
        }

        model_path = MODEL_DIR / f"lightgbm_corrected_t{horizon}.joblib"
        joblib.dump(artifact, model_path)

        print("\nSaved model:", model_path)

    # --------------------------------------------------------
    # 12. SAVE METRICS AND TEST PREDICTIONS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(results)

    metrics_path = OUTPUT / "corrected_multihorizon_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = OUTPUT / "corrected_test_predictions.csv"

    if test_predictions:
        pd.concat(
            test_predictions,
            ignore_index=True,
        ).to_csv(predictions_path, index=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("\nMetrics saved to:")
    print(metrics_path)

    print("\nTest predictions saved to:")
    print(predictions_path)

    print("\nModels saved in:")
    print(MODEL_DIR)

    print("\nMETRICS SUMMARY")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nOriginal dataset and existing models were not overwritten.")


if __name__ == "__main__":
    main()
