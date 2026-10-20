"""Shared utilities for training one model at a time. Dataset paths are set here."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_CANDIDATES = [
    PROJECT / "DATASET" / "processed",   # extracted ZIP's actual layout
    PROJECT / "processed",
    PROJECT / "DATASET" / "DATASET" / "processed",
]
RESULTS = HERE / "results"
MODELS = RESULTS / "models"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Change these constants instead of typing paths in the terminal.
DATASET_NAME = "cgm_t1dm_t2dm"
TARGET_COLUMN = "target_cgm_t30"
RANDOM_STATE = 42

DATASET_OPTIONS = ["cgm_t1dm_t2dm", "glucotwin", "ohhim", "smart_wearable"]


def locate_data_dir(dataset_name=DATASET_NAME):
    for folder in DATA_CANDIDATES:
        if (folder / f"{dataset_name}_model_ready.csv").is_file() and (folder / f"schema_{dataset_name}.json").is_file():
            return folder
    raise FileNotFoundError(
        f"Could not locate {dataset_name}_model_ready.csv and schema_{dataset_name}.json. "
        "Extract the ZIP and keep the folder structure unchanged. Expected location: "
        "DATASET/DATASET/processed/"
    )


def load_data(dataset_name=DATASET_NAME):
    folder = locate_data_dir(dataset_name)
    df = pd.read_csv(folder / f"{dataset_name}_model_ready.csv", low_memory=False)
    with open(folder / f"schema_{dataset_name}.json", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema["features"]
    targets = schema["targets"]
    if "split" not in df.columns:
        raise ValueError("Expected a 'split' column containing train/val/test.")
    return df, features, targets, schema


def task_type(train_y):
    vals = pd.Series(train_y).dropna().unique()
    return "classification" if len(vals) <= 2 else "regression"


def score(task, y, pred, probabilities=None, threshold=0.5):
    y = np.asarray(y)
    pred = np.asarray(pred)
    if task == "regression":
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "MAE": float(mean_absolute_error(y, pred)),
            "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        }
    p = np.asarray(probabilities if probabilities is not None else pred, dtype=float)
    labels = (p >= threshold).astype(int)
    result = {
        "Accuracy": float(accuracy_score(y, labels)),
        "Precision": float(precision_score(y, labels, zero_division=0)),
        "Recall": float(recall_score(y, labels, zero_division=0)),
        "F1": float(f1_score(y, labels, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        result["ROC_AUC"] = float(roc_auc_score(y, p))
        result["PR_AUC"] = float(average_precision_score(y, p))
    return result


def run_model(model_name, estimator_factory, dataset_name=DATASET_NAME, target=TARGET_COLUMN,
              early_stopping_fit=None):
    df, features, targets, schema = load_data(dataset_name)
    if target not in targets or target not in df.columns:
        raise ValueError(f"Target '{target}' not in schema. Available targets: {targets}")
    train = df[df.split.astype(str).str.lower() == "train"].copy()
    val = df[df.split.astype(str).str.lower().isin(["val", "validation", "valid"])].copy()
    test = df[df.split.astype(str).str.lower() == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Expected non-empty train, val, and test splits.")
    # Data preparation was performed by the dataset pipeline; guard against any bad values.
    for part in (train, val, test):
        part[features] = part[features].apply(pd.to_numeric, errors="coerce")
        part[target] = pd.to_numeric(part[target], errors="coerce")
    train = train.dropna(subset=[target]); val = val.dropna(subset=[target]); test = test.dropna(subset=[target])
    # Features with missing values are filled using train-only medians.
    medians = train[features].median(numeric_only=True).reindex(features).fillna(0)
    Xtr = train[features].fillna(medians)
    Xv = val[features].fillna(medians)
    Xte = test[features].fillna(medians)
    ytr, yv, yte = train[target], val[target], test[target]
    task = task_type(ytr)
    estimator = estimator_factory(task, ytr)
    if early_stopping_fit:
        early_stopping_fit(estimator, Xtr, ytr, Xv, yv, task)
    else:
        estimator.fit(Xtr, ytr)
    # Choose classification threshold on validation data only (max F1).
    threshold = 0.5
    def get_predictions(X):
        if task == "classification":
            return estimator.predict_proba(X)[:, 1]
        return estimator.predict(X)
    if task == "classification":
        pv = get_predictions(Xv)
        thresholds = np.linspace(0.05, 0.95, 181)
        from sklearn.metrics import f1_score
        threshold = max(thresholds, key=lambda t: f1_score(yv, (pv >= t).astype(int), zero_division=0))
    rows = []
    for split_name, X, y in [("validation", Xv, yv), ("test", Xte, yte)]:
        p = get_predictions(X)
        rows.append({"dataset": dataset_name, "model": model_name, "target": target,
                     "task": task, "split": split_name, "threshold": threshold if task == "classification" else None,
                     **score(task, y, p, probabilities=p if task == "classification" else None, threshold=threshold)})
    metrics_path = RESULTS / f"{model_name}__{dataset_name}__{target}__metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    artifact = {"model": estimator, "features": features, "medians": medians.to_dict(),
                "dataset_name": dataset_name, "target": target, "task": task,
                "threshold": float(threshold), "schema": schema}
    model_path = MODELS / f"{model_name}__{dataset_name}__{target}.joblib"
    joblib.dump(artifact, model_path)
    if hasattr(estimator, "feature_importances_"):
        imp = pd.DataFrame({"feature": features, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(RESULTS / f"{model_name}__{dataset_name}__{target}__feature_importance.csv", index=False)
    print("\nTraining finished")
    print(f"Dataset: {dataset_name} | Target: {target} | Task: {task}")
    print(f"Train/validation/test rows: {len(train):,}/{len(val):,}/{len(test):,}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    return estimator
