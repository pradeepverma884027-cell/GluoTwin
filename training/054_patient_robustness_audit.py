from __future__ import annotations

"""Read-only patient-robustness audit for GlucoTwin v3 hypoglycemia candidates.

Loads saved models and the existing v3 model-ready dataset.
Does not retrain, modify, or overwrite any model or dataset.
Writes new Audit 054 CSV/JSON reports only.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "results_causal"
V3_DIR = ROOT / "preprocessing_v3"
MODEL_DIR = ROOT / "v3_model_comparison" / "models"
OUT_DIR = ROOT / "v3_model_comparison"
DATA_PATH = V3_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
SCHEMA_PATH = V3_DIR / "schema_cgm_t1dm_t2dm_causal.json"
TARGET = "target_hypo_60"
THRESHOLD = 0.20
SENSITIVITY_EXCLUDE_PATIENT = "T2DM_2077"


def safe_auc(y: np.ndarray, p: np.ndarray, metric) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(metric(y, p))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "PR_AUC": safe_auc(y, p, average_precision_score),
        "ROC_AUC": safe_auc(y, p, roc_auc_score),
        "Brier": float(brier_score_loss(y, p)) if len(y) else None,
        "mean_predicted_probability": float(np.mean(p)) if len(p) else None,
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
    }


def main() -> None:
    for path in (DATA_PATH, SCHEMA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = ["patient_uid", "split", TARGET, *features]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    test = df[df["split"].eq("test")].copy()
    if test.empty:
        raise ValueError("No rows found with split='test'.")

    model_probs: dict[str, np.ndarray] = {}
    patient_records = []
    for model_name in ("lightgbm", "xgboost"):
        model_path = MODEL_DIR / f"{model_name}__{TARGET}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved candidate model missing: {model_path}")
        model = joblib.load(model_path)
        probabilities = model.predict_proba(test[features])[:, 1]
        model_probs[model_name] = probabilities
        y_all = test[TARGET].astype(int).to_numpy()

        for patient_uid, idx in test.groupby("patient_uid", sort=True).groups.items():
            positions = test.index.get_indexer(idx)
            y = test.loc[idx, TARGET].astype(int).to_numpy()
            p = probabilities[positions]
            metrics = binary_metrics(y, p, THRESHOLD)
            patient_records.append({
                "model": model_name,
                "patient_uid": patient_uid,
                "has_both_classes": bool(len(np.unique(y)) == 2),
                **metrics,
            })

    patient_df = pd.DataFrame(patient_records)
    # Macro metrics average eligible patients equally. PR-AUC/ROC-AUC are only
    # averaged across patients whose test rows contain both classes.
    macro_records = []
    for model_name in ("lightgbm", "xgboost"):
        part = patient_df[patient_df["model"].eq(model_name)]
        eligible = part[part["has_both_classes"]]
        macro_records.append({
            "model": model_name,
            "test_patients": int(part["patient_uid"].nunique()),
            "patients_with_both_classes": int(len(eligible)),
            "patient_macro_PR_AUC": float(eligible["PR_AUC"].dropna().mean()),
            "patient_macro_ROC_AUC": float(eligible["ROC_AUC"].dropna().mean()),
            "patient_macro_Brier_all_patients": float(part["Brier"].mean()),
            "patient_macro_precision_at_threshold": float(part["precision"].mean()),
            "patient_macro_sensitivity_positive_patients": float(
                eligible["sensitivity"].dropna().mean()
            ),
            "patient_macro_specificity_all_patients": float(part["specificity"].dropna().mean()),
        })

    y = test[TARGET].astype(int).to_numpy()
    aggregate_records = []
    for model_name, p in model_probs.items():
        aggregate_records.append({
            "analysis": "full_test",
            "excluded_patient": "",
            "model": model_name,
            **binary_metrics(y, p, THRESHOLD),
        })
        keep = test["patient_uid"].astype(str).to_numpy() != SENSITIVITY_EXCLUDE_PATIENT
        if keep.any():
            aggregate_records.append({
                "analysis": "sensitivity_excluding_patient",
                "excluded_patient": SENSITIVITY_EXCLUDE_PATIENT,
                "model": model_name,
                **binary_metrics(y[keep], p[keep], THRESHOLD),
            })

    patient_path = OUT_DIR / "054_hypo_per_patient_metrics.csv"
    macro_path = OUT_DIR / "054_hypo_patient_macro_summary.csv"
    aggregate_path = OUT_DIR / "054_hypo_prevalence_sensitivity.csv"
    json_path = OUT_DIR / "054_hypo_robustness_audit.json"

    patient_df.to_csv(patient_path, index=False)
    macro_df = pd.DataFrame(macro_records)
    macro_df.to_csv(macro_path, index=False)
    aggregate_df = pd.DataFrame(aggregate_records)
    aggregate_df.to_csv(aggregate_path, index=False)

    report = {
        "audit": "054_patient_robustness_and_prevalence_sensitivity",
        "read_only": True,
        "target": TARGET,
        "threshold_exploratory_not_clinical": THRESHOLD,
        "excluded_patient_for_sensitivity_only": SENSITIVITY_EXCLUDE_PATIENT,
        "test_rows": int(len(test)),
        "test_patients": int(test["patient_uid"].nunique()),
        "features_count": int(len(features)),
        "outputs": [str(patient_path), str(macro_path), str(aggregate_path)],
        "limitations": [
            "Retrospective dataset evaluation only; not clinical validation.",
            "The 0.20 threshold is exploratory and not a clinical recommendation.",
            "Patient-macro metrics are descriptive; uncertainty is substantial with few test patients.",
            "Excluding one patient is sensitivity analysis only and does not replace the full test result.",
        ],
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 76)
    print("GLUCOTWIN V3 — AUDIT 054: PATIENT ROBUSTNESS (READ ONLY)")
    print("=" * 76)
    print(f"Test rows: {len(test)} | test patients: {test['patient_uid'].nunique()}")
    print(f"Exploratory threshold: {THRESHOLD:.2f} (not a clinical recommendation)")
    print("\nPatient-macro summary:")
    print(macro_df.round(5).to_string(index=False))
    print("\nFull-test vs sensitivity analysis excluding T2DM_2077:")
    print(aggregate_df.round(5).to_string(index=False))
    print("\nPer-patient metrics:")
    display_cols = [
        "model", "patient_uid", "rows", "positive_rows", "prevalence",
        "PR_AUC", "ROC_AUC", "Brier", "TP", "FP", "FN",
        "sensitivity", "specificity", "has_both_classes",
    ]
    print(patient_df[display_cols].round(5).to_string(index=False))
    print("\nReports saved:")
    for path in (patient_path, macro_path, aggregate_path, json_path):
        print(path)
    print("\nRead-only: models and dataset were not modified or retrained.")
    print("These retrospective results are not clinical validation or medical advice.")


if __name__ == "__main__":
    main()
