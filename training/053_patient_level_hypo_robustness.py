from __future__ import annotations

"""Read-only per-patient robustness analysis for GlucoTwin v3 hypoglycemia candidates."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score, brier_score_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results_causal"
V3_DIR = ROOT / "preprocessing_v3"
RUN_DIR = ROOT / "v3_model_comparison"
MODEL_DIR = RUN_DIR / "models"
DATA_PATH = V3_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
SCHEMA_PATH = V3_DIR / "schema_cgm_t1dm_t2dm_causal.json"
TARGET = "target_hypo_60"
THRESHOLD = 0.20  # exploratory only; not a clinical threshold


def auc_or_none(y, p, fn):
    return float(fn(y, p)) if len(np.unique(y)) > 1 else np.nan


def main():
    for p in [DATA_PATH, SCHEMA_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]
    df = pd.read_csv(DATA_PATH, low_memory=False)
    test = df[df["split"].eq("test")].copy()
    y = test[TARGET].astype(int).to_numpy()

    all_patient_summaries = []
    aggregate = []
    for model_name in ["lightgbm", "xgboost"]:
        path = MODEL_DIR / f"{model_name}__{TARGET}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        model = joblib.load(path)
        test[f"prob_{model_name}"] = model.predict_proba(test[features])[:, 1]
        p = test[f"prob_{model_name}"].to_numpy()
        pred = (p >= THRESHOLD).astype(int)
        aggregate.append({
            "model": model_name,
            "test_rows": int(len(test)),
            "test_patients": int(test["patient_uid"].nunique()),
            "threshold_exploratory": THRESHOLD,
            "PR_AUC_row_weighted": auc_or_none(y, p, average_precision_score),
            "ROC_AUC_row_weighted": auc_or_none(y, p, roc_auc_score),
            "Brier_row_weighted": float(brier_score_loss(y, p)),
            "precision_at_threshold": float(precision_score(y, pred, zero_division=0)),
            "recall_at_threshold": float(recall_score(y, pred, zero_division=0)),
        })

        for patient, g in test.groupby("patient_uid", sort=True):
            gy = g[TARGET].astype(int).to_numpy()
            gp = g[f"prob_{model_name}"].to_numpy()
            gpred = (gp >= THRESHOLD).astype(int)
            positives = int(gy.sum())
            negatives = int(len(gy) - positives)
            all_patient_summaries.append({
                "model": model_name,
                "patient_uid": patient,
                "rows": int(len(g)),
                "positive_rows": positives,
                "prevalence": float(gy.mean()),
                "PR_AUC": auc_or_none(gy, gp, average_precision_score),
                "ROC_AUC": auc_or_none(gy, gp, roc_auc_score),
                "Brier": float(brier_score_loss(gy, gp)),
                "TP_at_exploratory_threshold": int(((gpred == 1) & (gy == 1)).sum()),
                "FP_at_exploratory_threshold": int(((gpred == 1) & (gy == 0)).sum()),
                "FN_at_exploratory_threshold": int(((gpred == 0) & (gy == 1)).sum()),
                "sensitivity_at_exploratory_threshold": float(recall_score(gy, gpred, zero_division=0)),
                "specificity_at_exploratory_threshold": (
                    float(((gpred == 0) & (gy == 0)).sum() / negatives) if negatives else np.nan
                ),
                "has_both_classes": bool(len(np.unique(gy)) == 2),
            })

    patient_df = pd.DataFrame(all_patient_summaries)
    aggregate_df = pd.DataFrame(aggregate)
    OUT_PATIENT = RUN_DIR / "053_hypo_per_patient_metrics.csv"
    OUT_AGG = RUN_DIR / "053_hypo_aggregate_summary.csv"
    patient_df.to_csv(OUT_PATIENT, index=False)
    aggregate_df.to_csv(OUT_AGG, index=False)

    print("=" * 72)
    print("GLUCOTWIN V3 - PATIENT-LEVEL HYPOGLYCEMIA ROBUSTNESS (READ ONLY)")
    print("=" * 72)
    print("\nAggregate test summary:")
    print(aggregate_df.round(5).to_string(index=False))
    print("\nPer-patient metrics:")
    print(patient_df.round(5).to_string(index=False))
    print("\nAcross-patient summary by model (NaN AUCs excluded where a patient has one class):")
    for model_name, g in patient_df.groupby("model"):
        valid_pr = g["PR_AUC"].dropna()
        valid_roc = g["ROC_AUC"].dropna()
        print(f"\n{model_name}:")
        print(f"  patients: {g['patient_uid'].nunique()}")
        print(f"  patients with both target classes: {int(g['has_both_classes'].sum())}")
        print(f"  median patient prevalence: {g['prevalence'].median():.5f}")
        print(f"  patient prevalence range: {g['prevalence'].min():.5f} to {g['prevalence'].max():.5f}")
        print(f"  median per-patient PR-AUC (eligible patients): {valid_pr.median():.5f}" if len(valid_pr) else "  median per-patient PR-AUC: unavailable")
        print(f"  median per-patient ROC-AUC (eligible patients): {valid_roc.median():.5f}" if len(valid_roc) else "  median per-patient ROC-AUC: unavailable")
        print(f"  median per-patient Brier: {g['Brier'].median():.5f}")
        print(f"  patients with zero positive labels: {int((g['positive_rows'] == 0).sum())}")
    print("\nSaved reports:")
    print(OUT_PATIENT)
    print(OUT_AGG)
    print("\nRead-only: no datasets or models were modified.")
    print("Retrospective research analysis only; not clinical validation or medical advice.")


if __name__ == "__main__":
    main()
