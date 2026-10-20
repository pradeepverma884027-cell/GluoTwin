from __future__ import annotations

"""Train isolated v3 candidate models; never overwrites existing GlucoTwin models/results."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score
import joblib

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
V3_DIR = HERE / "results_causal" / "preprocessing_v3"
OUT_DIR = HERE / "results_causal" / "v3_model_comparison"
MODEL_PATH = V3_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
SCHEMA_PATH = V3_DIR / "schema_cgm_t1dm_t2dm_causal.json"

SEED = 42


def regression_metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def binary_metrics(y, prob):
    y = np.asarray(y).astype(int)
    out = {
        "prevalence": float(y.mean()),
        "PR_AUC": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else None,
        "ROC_AUC": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
    }
    return out


def main():
    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            "V3 candidate files not found. Expected:\n"
            f"  {MODEL_PATH}\n  {SCHEMA_PATH}"
        )

    df = pd.read_csv(MODEL_PATH, parse_dates=["dt"], low_memory=False)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]
    targets = schema["targets"]

    required = ["patient_uid", "segment_id", "dt", "split"] + features + targets
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[features].isna().any().any():
        raise ValueError("Feature matrix has missing values; refusing to train.")
    if df[targets].isna().any().any():
        raise ValueError("Targets have missing values; refusing to train.")
    if df.duplicated(["patient_uid", "segment_id", "dt"]).any():
        raise ValueError("Duplicate patient/segment/timestamp keys; refusing to train.")
    if df.groupby("patient_uid")["split"].nunique().max() != 1:
        raise ValueError("Patient appears in multiple splits; refusing to train.")
    if set(df["split"].unique()) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test splits, found {df['split'].unique().tolist()}")

    # Ensure candidate features and targets do not overlap.
    if set(features) & set(targets):
        raise ValueError("Feature/target name overlap detected.")

    # Keep all output separate from prior model artifacts.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    split_summary = {
        "rows": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "patients": {str(k): int(v) for k, v in df.groupby("split")["patient_uid"].nunique().items()},
        "feature_count": len(features),
        "targets": targets,
        "patient_split_overlap": False,
        "source_model_ready_csv": str(MODEL_PATH),
        "output_directory": str(OUT_DIR),
        "note": "New isolated candidate run. Existing models/results are not overwritten.",
    }
    (OUT_DIR / "v3_run_metadata.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print("V3 candidate validation passed.")
    print(json.dumps(split_summary, indent=2))

    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None

    if lgb is None and xgb is None:
        raise ImportError("Neither lightgbm nor xgboost is installed. Install one or both to train candidate models.")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    rows = []

    for target in targets:
        task = "binary" if target == "target_hypo_60" else "regression"
        Xtr, ytr = train[features], train[target]
        Xva, yva = val[features], val[target]
        Xte, yte = test[features], test[target]

        candidates = []
        if lgb is not None:
            if task == "regression":
                model = lgb.LGBMRegressor(
                    n_estimators=3000, learning_rate=0.03, num_leaves=31,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1
                )
                fit_kw = {"eval_set": [(Xva, yva)], "callbacks": [lgb.early_stopping(100, verbose=False)]}
            else:
                pos = max(float(ytr.mean()), 1e-6)
                model = lgb.LGBMClassifier(
                    n_estimators=3000, learning_rate=0.03, num_leaves=31,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, scale_pos_weight=(1 - pos) / pos,
                    random_state=SEED, n_jobs=-1, verbose=-1
                )
                fit_kw = {"eval_set": [(Xva, yva)], "callbacks": [lgb.early_stopping(100, verbose=False)]}
            candidates.append(("lightgbm", model, fit_kw))

        if xgb is not None:
            common = dict(
                n_estimators=3000, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                device="cpu", early_stopping_rounds=100, n_jobs=-1,
                random_state=SEED
            )
            if task == "regression":
                model = xgb.XGBRegressor(**common)
            else:
                pos = max(float(ytr.mean()), 1e-6)
                model = xgb.XGBClassifier(**common, scale_pos_weight=(1 - pos) / pos, eval_metric="logloss")
            candidates.append(("xgboost", model, {"eval_set": [(Xva, yva)], "verbose": False}))

        for model_name, model, fit_kw in candidates:
            print(f"\nTraining {model_name} | {target} | {task}", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, ytr, **fit_kw)

            if task == "regression":
                val_pred = model.predict(Xva)
                test_pred = model.predict(Xte)
                metrics_val = regression_metrics(yva, val_pred)
                metrics_test = regression_metrics(yte, test_pred)

                # Same-as-current-glucose baseline in original mg/dL units.
                # Current cgm feature was z-scored, so invert via schema's train-only parameters.
                scale = schema.get("scaled_cols_mean_std_train_only", {}).get("cgm")
                if scale:
                    cgm_now = test["cgm"].to_numpy() * float(scale["std"]) + float(scale["mean"])
                    base = regression_metrics(yte, cgm_now)
                    for k, v in base.items():
                        metrics_test[f"baseline_{k}"] = v
                for split_name, metrics in [("val", metrics_val), ("test", metrics_test)]:
                    rows.append({"model": model_name, "target": target, "task": task, "split": split_name, **metrics})
            else:
                val_prob = model.predict_proba(Xva)[:, 1]
                test_prob = model.predict_proba(Xte)[:, 1]
                metrics_val = binary_metrics(yva, val_prob)
                metrics_test = binary_metrics(yte, test_prob)
                for split_name, metrics in [("val", metrics_val), ("test", metrics_test)]:
                    rows.append({"model": model_name, "target": target, "task": task, "split": split_name, **metrics})

            model_file = OUT_DIR / "models" / f"{model_name}__{target}.joblib"
            joblib.dump(model, model_file)
            print(f"Saved isolated candidate model: {model_file}", flush=True)

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "v3_candidate_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nV3 CANDIDATE METRICS")
    print(metrics_df.round(5).to_string(index=False))
    print(f"\nMetrics saved: {metrics_path}")
    print("Finished. Existing production/current models were not modified.")
    print("Note: metrics are research evaluation only; not clinical validation.")


if __name__ == "__main__":
    main()
