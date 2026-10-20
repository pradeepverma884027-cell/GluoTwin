from __future__ import annotations

"""Read-only evaluation of saved GlucoTwin v3 candidates; does not retrain or alter models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "results_causal"
V3_DIR = ROOT / "preprocessing_v3"
MODEL_DIR = ROOT / "v3_model_comparison" / "models"
OUT_DIR = ROOT / "v3_model_comparison"
DATA_PATH = V3_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
SCHEMA_PATH = V3_DIR / "schema_cgm_t1dm_t2dm_causal.json"
TARGET = "target_hypo_60"
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]


def safe_auc(y, p, metric):
    return float(metric(y, p)) if len(np.unique(y)) > 1 else None


def main():
    for path in [DATA_PATH, SCHEMA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]
    df = pd.read_csv(DATA_PATH, parse_dates=["dt"], low_memory=False)
    required = ["patient_uid", "split", TARGET] + features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = []
    patient_rows = []
    threshold_rows = []
    for split in ["train", "val", "test"]:
        part = df[df["split"].eq(split)].copy()
        y = part[TARGET].astype(int).to_numpy()
        patient_rates = part.groupby("patient_uid")[TARGET].agg(["count", "mean", "sum"])
        patient_rows.append({
            "split": split,
            "rows": int(len(part)),
            "patients": int(part["patient_uid"].nunique()),
            "positive_rows": int(y.sum()),
            "prevalence": float(y.mean()),
            "patient_prevalence_min": float(patient_rates["mean"].min()),
            "patient_prevalence_median": float(patient_rates["mean"].median()),
            "patient_prevalence_max": float(patient_rates["mean"].max()),
            "patients_with_zero_positive_rows": int((patient_rates["sum"] == 0).sum()),
            "patients_with_positive_rows": int((patient_rates["sum"] > 0).sum()),
        })

        if split not in ["val", "test"]:
            continue
        for model_name in ["lightgbm", "xgboost"]:
            model_path = MODEL_DIR / f"{model_name}__{TARGET}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Saved candidate model missing: {model_path}")
            model = joblib.load(model_path)
            prob = model.predict_proba(part[features])[:, 1]
            rows.append({
                "model": model_name,
                "split": split,
                "rows": int(len(part)),
                "prevalence": float(y.mean()),
                "PR_AUC": safe_auc(y, prob, average_precision_score),
                "ROC_AUC": safe_auc(y, prob, roc_auc_score),
                "Brier_score": float(brier_score_loss(y, prob)),
                "mean_predicted_probability": float(np.mean(prob)),
            })
            for threshold in THRESHOLDS:
                pred = (prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
                threshold_rows.append({
                    "model": model_name,
                    "split": split,
                    "threshold": threshold,
                    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                    "precision": float(precision_score(y, pred, zero_division=0)),
                    "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
                    "specificity": float(tn / (tn + fp)) if tn + fp else None,
                    "F1": float(f1_score(y, pred, zero_division=0)),
                })

    patient_path = OUT_DIR / "052_hypo_patient_prevalence.csv"
    metrics_path = OUT_DIR / "052_hypo_probability_metrics.csv"
    threshold_path = OUT_DIR / "052_hypo_threshold_tradeoffs.csv"
    pd.DataFrame(patient_rows).to_csv(patient_path, index=False)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)

    print("=" * 70)
    print("GLUCOTWIN V3 - HYPOGLYCEMIA EVALUATION (READ ONLY)")
    print("=" * 70)
    print("\nPrevalence and patient shift:")
    print(pd.DataFrame(patient_rows).round(5).to_string(index=False))
    print("\nProbability metrics:")
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    print("\nThreshold tradeoffs (exploratory only; thresholds are not clinical recommendations):")
    print(pd.DataFrame(threshold_rows).round(5).to_string(index=False))
    print("\nReports saved:")
    print(patient_path)
    print(metrics_path)
    print(threshold_path)
    print("\nRead-only: no training, data, or model files were modified.")
    print("These are retrospective research metrics, not clinical validation or medical advice.")


if __name__ == "__main__":
    main()
