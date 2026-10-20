
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

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR.parent

DATA_PATH = (
    DATASET_DIR / "DATASET" / "processed"
    / "cgm_t1dm_t2dm_model_ready.csv"
)
SCHEMA_PATH = (
    DATASET_DIR / "DATASET" / "processed"
    / "schema_cgm_t1dm_t2dm.json"
)

OUTPUT_DIR = TRAINING_DIR / "results_causal"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 90, 120]
HISTORY_STEPS = 96  # 24 hours at 15-minute intervals

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    print("=" * 68)
    print("GLUCOTWIN CAUSAL MULTI-HORIZON TRAINING")
    print("=" * 68)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Update DATA_PATH to your actual processed CSV."
        )

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema not found:\n{SCHEMA_PATH}\n"
            "Update SCHEMA_PATH to your actual schema JSON."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    required = {
        "patient_uid", "segment_id", "dt", "split",
        "cgm", "cgm_interpolated"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")

    features = schema.get("features", [])
    if not features:
        raise ValueError("No feature list found in the schema JSON.")

    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise ValueError(
            f"Schema features missing from CSV: {missing_features}"
        )

    scale_params = schema.get("scaled_cols_mean_std", {})
    if "cgm" not in scale_params:
        raise ValueError(
            "Schema is missing scaled_cols_mean_std['cgm']; "
            "cannot convert CGM values back to mg/dL."
        )

    cgm_mean, cgm_std = map(float, scale_params["cgm"])
    if not np.isfinite(cgm_std) or cgm_std <= 0:
        raise ValueError(f"Invalid CGM standard deviation: {cgm_std}")

    # Normalize split naming without changing patient assignments.
    df["split"] = df["split"].astype(str).str.lower().replace({
        "val": "validation"
    })

    allowed_splits = {"train", "validation", "test"}
    if set(df["split"].dropna().unique()) - allowed_splits:
        raise ValueError(
            f"Unexpected split values: {df['split'].unique().tolist()}"
        )

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["cgm_interpolated"] = pd.to_numeric(
        df["cgm_interpolated"], errors="coerce"
    ).fillna(1).astype(int)

    # Sort before any shift or rolling operation.
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)

    groups = ["patient_uid", "segment_id"]
    grouped = df.groupby(groups, sort=False)

    # -----------------------------------------------------
    # 1. Conservative interpolation-history filter
    # -----------------------------------------------------
    # A row is eligible only if none of the current CGM
    # reading and the previous 95 readings was interpolated.
    # This is a 96-row window, not a guarantee of 24 hours
    # of uninterrupted history at segment boundaries.
    df["_interpolated_in_history"] = (
        grouped["cgm_interpolated"]
        .transform(
            lambda s: s.rolling(
                window=HISTORY_STEPS,
                min_periods=1
            ).max()
        )
        .fillna(1)
        .astype(int)
    )

    df["_history_is_clean"] = (
        df["_interpolated_in_history"] == 0
    )

    before = len(df)
    clean_history_rows = int(df["_history_is_clean"].sum())

    print(f"Dataset rows: {before:,}")
    print(f"Rows with clean interpolation history: {clean_history_rows:,}")
    print(
        f"Rows excluded by history filter: "
        f"{before - clean_history_rows:,}"
    )

    # Ensure the feature matrix cannot include targets or helper columns.
    forbidden = {
        "target_cgm_t30", "target_cgm_t60",
        "target_cgm_t90", "target_cgm_t120",
        "_interpolated_in_history", "_history_is_clean",
    }
    features = [c for c in features if c not in forbidden]

    if "cgm_interpolated" not in features:
        print(
            "Note: cgm_interpolated is not in the schema feature list."
        )

    # -----------------------------------------------------
    # 2. Create future targets from the correct future row
    # -----------------------------------------------------
    # The standardized CGM values are converted back to mg/dL.
    # Exact timestamp checks prevent crossing gaps or segments.
    all_metrics = []
    all_predictions = []
    audit_rows = []

    for horizon in HORIZONS:
        steps = horizon // 15
        if horizon % 15 != 0:
            raise ValueError("Horizons must be multiples of 15 minutes.")

        future_cgm = grouped["cgm"].shift(-steps)
        future_interpolated = grouped["cgm_interpolated"].shift(-steps)
        future_dt = grouped["dt"].shift(-steps)
        future_segment = grouped["segment_id"].shift(-steps)

        expected_dt = df["dt"] + pd.to_timedelta(horizon, unit="min")

        target_mgdl = future_cgm * cgm_std + cgm_mean

        valid = (
            df["_history_is_clean"]
            & df["dt"].notna()
            & future_dt.eq(expected_dt)
            & future_segment.eq(df["segment_id"])
            & future_interpolated.eq(0)
            & target_mgdl.notna()
            & np.isfinite(target_mgdl)
            & df["split"].isin(allowed_splits)
        )

        work = df.loc[valid, ["split"] + features].copy()
        work["_target"] = target_mgdl.loc[valid].astype(float).to_numpy()
        work["_row_index"] = df.index[valid]

        print(f"\n--- Horizon +{horizon} minutes ---")
        print(f"Eligible examples: {len(work):,}")

        split_counts = work["split"].value_counts().to_dict()
        print(f"Split counts: {split_counts}")

        train = work["split"].eq("train")
        validation = work["split"].eq("validation")
        test = work["split"].eq("test")

        if train.sum() == 0 or validation.sum() == 0 or test.sum() == 0:
            raise ValueError(
                f"Horizon {horizon}: a split has no eligible rows."
            )

        X_train = work.loc[train, features].replace(
            [np.inf, -np.inf], np.nan
        )
        X_val = work.loc[validation, features].replace(
            [np.inf, -np.inf], np.nan
        )
        X_test = work.loc[test, features].replace(
            [np.inf, -np.inf], np.nan
        )

        y_train = work.loc[train, "_target"]
        y_val = work.loc[validation, "_target"]
        y_test = work.loc[test, "_target"]

        # Fit imputation using training data only.
        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_val_i = imputer.transform(X_val)
        X_test_i = imputer.transform(X_test)

        model = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(X_train_i, y_train)

        predictions = {
            "validation": model.predict(X_val_i),
            "test": model.predict(X_test_i),
        }
        actuals = {
            "validation": y_val.to_numpy(),
            "test": y_test.to_numpy(),
        }

        for split_name in ["validation", "test"]:
            pred = predictions[split_name]
            actual = actuals[split_name]
            metrics = calculate_metrics(actual, pred)

            print(
                f"{split_name:10s} | "
                f"MAE={metrics['MAE']:.3f} | "
                f"RMSE={metrics['RMSE']:.3f} | "
                f"R2={metrics['R2']:.4f}"
            )

            all_metrics.append({
                "horizon_minutes": horizon,
                "model": "LightGBM_CausalFiltered",
                "split": split_name,
                "rows": len(actual),
                **metrics,
            })

            # Last-CGM baseline uses the current CGM converted to mg/dL.
            current_cgm = (
                df.loc[work.loc[
                    work["split"].eq(split_name), "_row_index"
                ], "cgm"].to_numpy() * cgm_std + cgm_mean
            )
            baseline = calculate_metrics(actual, current_cgm)

            all_metrics.append({
                "horizon_minutes": horizon,
                "model": "Last_CGM_Baseline",
                "split": split_name,
                "rows": len(actual),
                **baseline,
            })

            prediction_rows = work.loc[
                work["split"].eq(split_name), ["_row_index", "_target"]
            ].copy()
            prediction_rows["horizon_minutes"] = horizon
            prediction_rows["split"] = split_name
            prediction_rows["prediction_mgdl"] = pred
            prediction_rows["baseline_mgdl"] = current_cgm
            prediction_rows["actual_mgdl"] = actual
            all_predictions.append(prediction_rows)

        artifact = {
            "model": model,
            "imputer": imputer,
            "features": features,
            "horizon_minutes": horizon,
            "cgm_mean": cgm_mean,
            "cgm_std": cgm_std,
            "schema": schema,
            "task": "causal_filtered_multihorizon_regression",
            "history_filter_steps": HISTORY_STEPS,
            "target_scale": "mg/dL",
        }

        model_path = MODEL_DIR / f"lightgbm_causal_t{horizon}.joblib"
        joblib.dump(artifact, model_path)
        print(f"Saved model: {model_path}")

        audit_rows.append({
            "horizon_minutes": horizon,
            "history_filter_steps": HISTORY_STEPS,
            "eligible_rows": len(work),
            "train_rows": int(train.sum()),
            "validation_rows": int(validation.sum()),
            "test_rows": int(test.sum()),
        })

    # -----------------------------------------------------
    # 3. Save results separately
    # -----------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUTPUT_DIR / "causal_multihorizon_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = OUTPUT_DIR / "causal_test_predictions.csv"
    pd.concat(all_predictions, ignore_index=True).to_csv(
        predictions_path, index=False
    )

    audit_path = OUTPUT_DIR / "causal_row_counts.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    print("\n" + "=" * 68)
    print("CAUSAL TRAINING COMPLETE")
    print("=" * 68)
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Row counts: {audit_path}")
    print(f"Models: {MODEL_DIR}")
    print(
        "\nReminder: this filter addresses interpolation history only. "
        "Review feature timing and preprocessing causality before "
        "claiming the pipeline is fully leakage-free."
    )


if __name__ == "__main__":
    main()
