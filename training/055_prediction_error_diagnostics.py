from __future__ import annotations

"""GlucoTwin Audit 055: read-only prediction error and prevalence diagnostics.

Loads the existing v3 model-ready data and saved LightGBM/XGBoost models.
Does not train, tune, overwrite, or modify any model or dataset.
Creates new CSV/JSON reports only.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "results_causal"
V3_DIR = ROOT / "preprocessing_v3"
MODEL_DIR = ROOT / "v3_model_comparison" / "models"
OUT_DIR = ROOT / "v3_model_comparison"
DATA_PATH = V3_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
SCHEMA_PATH = V3_DIR / "schema_cgm_t1dm_t2dm_causal.json"
TARGET = "target_hypo_60"
THRESHOLD = 0.20
FOCUS_PATIENTS = ["T2DM_2072", "T2DM_2095", "T2DM_2078", "T2DM_2077"]


def brier_by_group(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(brier_score_loss(y, p))


def main() -> None:
    for path in (DATA_PATH, SCHEMA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = ["patient_uid", "dt", "split", TARGET, *features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    test = df[df["split"].eq("test")].copy()
    if test.empty:
        raise ValueError("No rows found with split='test'.")

    all_patient_rows = []
    focus_error_rows = []
    segment_rows = []
    prevalence_rows = []
    source_feature_rows = []
    probs_by_model = {}

    for model_name in ("lightgbm", "xgboost"):
        model_path = MODEL_DIR / f"{model_name}__{TARGET}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model missing: {model_path}")
        model = joblib.load(model_path)
        prob = model.predict_proba(test[features])[:, 1]
        probs_by_model[model_name] = prob

        pred = (prob >= THRESHOLD).astype(int)
        work = test[["patient_uid", "dt", TARGET]].copy()
        work["probability"] = prob
        work["prediction"] = pred
        work["error_type"] = np.select(
            [(work[TARGET].astype(int).to_numpy() == 1) & (pred == 0),
             (work[TARGET].astype(int).to_numpy() == 0) & (pred == 1)],
            ["false_negative", "false_positive"],
            default="correct",
        )
        work["model"] = model_name

        for patient_uid, group in work.groupby("patient_uid", sort=True):
            y = group[TARGET].astype(int).to_numpy()
            p = group["probability"].to_numpy()
            pr = group["prediction"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
            all_patient_rows.append({
                "model": model_name,
                "patient_uid": patient_uid,
                "rows": int(len(group)),
                "positive_rows": int(y.sum()),
                "prevalence": float(y.mean()),
                "mean_predicted_probability": float(p.mean()),
                "mean_probability_minus_prevalence": float(p.mean() - y.mean()),
                "Brier": brier_by_group(y, p),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
                "specificity": float(tn / (tn + fp)) if tn + fp else None,
                "has_both_classes": bool(len(np.unique(y)) == 2),
            })

            if patient_uid in FOCUS_PATIENTS:
                errors = group[group["error_type"] != "correct"]
                focus_error_rows.append({
                    "model": model_name,
                    "patient_uid": patient_uid,
                    "rows": int(len(group)),
                    "positive_rows": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "false_negatives": int(fn),
                    "false_positives": int(fp),
                    "correct_rows": int((group["error_type"] == "correct").sum()),
                    "error_rows": int(len(errors)),
                    "mean_probability": float(p.mean()),
                    "median_probability": float(np.median(p)),
                    "p90_probability": float(np.quantile(p, 0.90)),
                    "p95_probability": float(np.quantile(p, 0.95)),
                })

        # Descriptive segment diagnostics by calendar month; no inference that
        # temporal patterns are causal. Missing dates are grouped explicitly.
        work["month"] = work["dt"].dt.to_period("M").astype(str)
        work.loc[work["dt"].isna(), "month"] = "UNKNOWN_DATE"
        for month, group in work.groupby("month", sort=True):
            y = group[TARGET].astype(int).to_numpy()
            p = group["probability"].to_numpy()
            pr = group["prediction"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
            segment_rows.append({
                "model": model_name,
                "month": month,
                "rows": int(len(group)),
                "positive_rows": int(y.sum()),
                "prevalence": float(y.mean()),
                "mean_predicted_probability": float(p.mean()),
                "Brier": brier_by_group(y, p),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
                "specificity": float(tn / (tn + fp)) if tn + fp else None,
            })

        # Feature distributions in focus patients vs all other test patients.
        # These are descriptive only; features are not assumed to explain errors.
        focus_mask = test["patient_uid"].isin(FOCUS_PATIENTS)
        for feature in features:
            vals_focus = pd.to_numeric(test.loc[focus_mask, feature], errors="coerce")
            vals_other = pd.to_numeric(test.loc[~focus_mask, feature], errors="coerce")
            source_feature_rows.append({
                "feature": feature,
                "focus_patients": ";".join(FOCUS_PATIENTS),
                "focus_rows": int(vals_focus.notna().sum()),
                "other_rows": int(vals_other.notna().sum()),
                "focus_mean": float(vals_focus.mean()) if vals_focus.notna().any() else None,
                "other_mean": float(vals_other.mean()) if vals_other.notna().any() else None,
                "focus_median": float(vals_focus.median()) if vals_focus.notna().any() else None,
                "other_median": float(vals_other.median()) if vals_other.notna().any() else None,
                "focus_missing_fraction": float(vals_focus.isna().mean()),
                "other_missing_fraction": float(vals_other.isna().mean()),
            })

    patient_df = pd.DataFrame(all_patient_rows)
    focus_df = pd.DataFrame(focus_error_rows)
    segment_df = pd.DataFrame(segment_rows)
    feature_df = pd.DataFrame(source_feature_rows).drop_duplicates(subset=["feature"])

    # Patient-level probability calibration summary, grouped into probability bins.
    calibration_rows = []
    for model_name, prob in probs_by_model.items():
        cal = test[["patient_uid", TARGET]].copy()
        cal["probability"] = prob
        cal["probability_bin"] = pd.cut(
            cal["probability"],
            bins=[-np.inf, 0.05, 0.10, 0.20, 0.30, 0.50, np.inf],
            labels=["<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50", ">=0.50"],
            right=False,
        )
        cal["model"] = model_name
        for bin_name, group in cal.groupby("probability_bin", observed=False):
            if len(group) == 0:
                continue
            calibration_rows.append({
                "model": model_name,
                "probability_bin": str(bin_name),
                "rows": int(len(group)),
                "mean_predicted_probability": float(group["probability"].mean()),
                "observed_prevalence": float(group[TARGET].mean()),
                "absolute_gap": float(abs(group["probability"].mean() - group[TARGET].mean())),
            })
    calibration_df = pd.DataFrame(calibration_rows)

    paths = {
        "patient_calibration": OUT_DIR / "055_patient_calibration_diagnostics.csv",
        "focus_errors": OUT_DIR / "055_focus_patient_error_summary.csv",
        "monthly_segments": OUT_DIR / "055_monthly_error_segments.csv",
        "feature_distributions": OUT_DIR / "055_focus_feature_distributions.csv",
        "probability_bins": OUT_DIR / "055_probability_bin_calibration.csv",
        "metadata": OUT_DIR / "055_diagnostics_audit.json",
    }
    patient_df.to_csv(paths["patient_calibration"], index=False)
    focus_df.to_csv(paths["focus_errors"], index=False)
    segment_df.to_csv(paths["monthly_segments"], index=False)
    feature_df.to_csv(paths["feature_distributions"], index=False)
    calibration_df.to_csv(paths["probability_bins"], index=False)

    report = {
        "audit": "055_prediction_error_and_calibration_diagnostics",
        "read_only": True,
        "target": TARGET,
        "threshold_exploratory_not_clinical": THRESHOLD,
        "focus_patients": FOCUS_PATIENTS,
        "test_rows": int(len(test)),
        "test_patients": int(test["patient_uid"].nunique()),
        "features_count": int(len(features)),
        "outputs": {k: str(v) for k, v in paths.items()},
        "limitations": [
            "Retrospective descriptive diagnostics only; not clinical validation.",
            "Threshold 0.20 is exploratory and not a clinical recommendation.",
            "Monthly groups may be small and may not represent independent samples.",
            "Feature distribution differences do not establish causation.",
            "Probability bins summarize calibration coarsely; they are not a full calibration study.",
        ],
    }
    paths["metadata"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 76)
    print("GLUCOTWIN V3 — AUDIT 055: PREDICTION ERROR & CALIBRATION")
    print("=" * 76)
    print(f"Test rows: {len(test)} | test patients: {test['patient_uid'].nunique()}")
    print(f"Threshold: {THRESHOLD:.2f} (exploratory only, not clinical advice)")
    print("\nFocus-patient error summary:")
    print(focus_df.round(5).to_string(index=False))
    print("\nPer-patient probability calibration and errors:")
    print(patient_df.round(5).to_string(index=False))
    print("\nProbability-bin calibration:")
    print(calibration_df.round(5).to_string(index=False))
    print("\nMonthly error segments:")
    print(segment_df.round(5).to_string(index=False))
    print("\nFocus-patient vs other-patient feature distributions (descriptive):")
    print(feature_df.round(5).to_string(index=False))
    print("\nReports saved:")
    for path in paths.values():
        print(path)
    print("\nRead-only: no models or datasets were modified or retrained.")
    print("Retrospective research diagnostics only; not clinical validation or medical advice.")


if __name__ == "__main__":
    main()
