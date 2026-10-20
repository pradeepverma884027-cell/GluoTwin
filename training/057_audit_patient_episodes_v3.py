"""
GlucoTwin v3 patient episode and false-negative audit (READ ONLY)

Place this script in:
  DATASET/training/

Run:
  python -u 057_audit_patient_episodes_v3.py

Inputs:
  results_causal/preprocessing_v3/cgm_t1dm_t2dm_causal_clean.csv
  results_causal/preprocessing_v3/cgm_t1dm_t2dm_causal_model_ready.csv
  results_causal/v3_model_comparison/models/xgboost__target_hypo_60.joblib
  results_causal/v3_model_comparison/models/lightgbm__target_hypo_60.joblib

This script loads existing models only. It does not train models, modify datasets,
or overwrite model files.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DIR = SCRIPT_DIR / "results_causal" / "preprocessing_v3"
MODEL_DIR = SCRIPT_DIR / "results_causal" / "v3_model_comparison"
CLEAN_PATH = PREP_DIR / "cgm_t1dm_t2dm_causal_clean.csv"
MODEL_PATH = PREP_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
XGB_PATH = MODEL_DIR / "models" / "xgboost__target_hypo_60.joblib"
LGB_PATH = MODEL_DIR / "models" / "lightgbm__target_hypo_60.joblib"
OUT_DIR = MODEL_DIR
FOCUS_PATIENTS = ["T2DM_2072", "T2DM_2077", "T2DM_2078"]
THRESHOLD = 0.20
LOW_CGM = 70.0
GRID_MINUTES = 15

# Features used by the v3 candidate training script.
FEATURES = [
    "cgm", "cgm_lag_15m", "cgm_lag_30m", "cgm_lag_45m", "cgm_lag_60m",
    "cgm_lag_90m", "cgm_lag_120m", "cgm_delta_15", "cgm_delta_30",
    "cgm_delta_60", "cgm_roll_mean_60m", "cgm_roll_std_60m",
    "cgm_roll_min_120m", "csii_basal", "fast_insulin_last_4h",
    "meal_grams_last_2h", "mins_since_meal", "mins_since_fast_insulin",
    "oral_agent_count", "cgm_mean_6h", "cgm_std_6h", "cgm_mean_24h",
    "cgm_tir_24h", "cgm_tbr_24h", "iob_proxy", "cob_proxy",
    "csii_suspend", "sc_long_acting", "meal_logged", "cgm_interpolated",
    "is_t2dm", "hour_sin", "hour_cos",
]


def require_files():
    paths = [CLEAN_PATH, MODEL_PATH, XGB_PATH, LGB_PATH]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required file(s) missing:\n  " + "\n  ".join(missing) +
            "\nConfirm the v3 candidate model filenames and locations."
        )


def safe_auc(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    result = {
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        "prevalence": float(np.mean(y)) if len(y) else None,
        "pr_auc": None,
        "roc_auc": None,
        "brier": None,
    }
    if len(y) and len(np.unique(y)) > 1:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["brier"] = float(brier_score_loss(y, p))
    elif len(y):
        result["brier"] = float(np.mean((p - y) ** 2))
    return result


def predict_probability(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    return np.clip(pred, 0.0, 1.0)


def find_episode_groups(frame, label_col, value_col=None):
    """Return contiguous runs of positive labels per patient and segment."""
    records = []
    for (pid, seg), g in frame.groupby(["patient_uid", "segment_id"], sort=False):
        g = g.sort_values("dt").reset_index(drop=True)
        labels = pd.to_numeric(g[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        values = pd.to_numeric(g[value_col], errors="coerce").to_numpy() if value_col else None
        dts = pd.to_datetime(g["dt"]).to_numpy()
        start = None
        for i, label in enumerate(labels):
            if label == 1 and start is None:
                start = i
            if start is not None:
                at_end = label != 1 or i == len(labels) - 1
                if at_end:
                    end = i - 1 if label != 1 else i
                    if end >= start:
                        records.append({
                            "patient_uid": str(pid),
                            "segment_id": int(seg),
                            "start_dt": pd.Timestamp(dts[start]),
                            "end_dt": pd.Timestamp(dts[end]),
                            "positive_slots": int(end - start + 1),
                            "span_minutes": int(end - start) * GRID_MINUTES,
                            "min_value_in_run": (
                                float(np.nanmin(values[start:end + 1]))
                                if values is not None and np.isfinite(values[start:end + 1]).any()
                                else np.nan
                            ),
                        })
                    start = None
    return records


def main():
    require_files()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["dt"], low_memory=False)
    data = pd.read_csv(MODEL_PATH, parse_dates=["dt"], low_memory=False)

    missing_features = [c for c in FEATURES if c not in data.columns]
    if missing_features:
        raise ValueError(f"Model-ready CSV is missing features: {missing_features}")
    required = ["patient_uid", "segment_id", "dt", "split", "target_hypo_60"]
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Model-ready CSV is missing columns: {missing_cols}")

    XGB = joblib.load(XGB_PATH)
    LGB = joblib.load(LGB_PATH)
    X = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    # Do not silently impute for a model that expects clean numeric input.
    if X.isna().any().any():
        raise ValueError(
            f"Feature matrix has {int(X.isna().sum().sum())} missing cells; "
            "inspect preprocessing before predicting."
        )

    data = data.copy()
    data["y_true"] = pd.to_numeric(data["target_hypo_60"], errors="raise").astype(int)
    data["xgb_probability"] = predict_probability(XGB, X)
    data["lgb_probability"] = predict_probability(LGB, X)
    data["xgb_pred_020"] = (data["xgb_probability"] >= THRESHOLD).astype(int)
    data["lgb_pred_020"] = (data["lgb_probability"] >= THRESHOLD).astype(int)

    # Join actual clean-grid CGM values for the prediction timestamps.
    clean_cgm = clean[["patient_uid", "segment_id", "dt", "cgm"]].copy()
    clean_cgm["dt"] = pd.to_datetime(clean_cgm["dt"])
    data = data.merge(
        clean_cgm.rename(columns={"cgm": "current_cgm"}),
        on=["patient_uid", "segment_id", "dt"], how="left", validate="one_to_one"
    )

    # Episode labels from target positives and actual current CGM < 70 are different constructs.
    clean_focus = clean_cgm[clean_cgm["patient_uid"].astype(str).isin(FOCUS_PATIENTS)].copy()
    clean_focus["actual_low_now"] = (pd.to_numeric(clean_focus["cgm"], errors="coerce") < LOW_CGM).astype(int)
    actual_episode_rows = find_episode_groups(
        clean_focus, "actual_low_now", "cgm"
    )
    for row in actual_episode_rows:
        row["episode_type"] = "actual_current_cgm_below_70"
    focus_model = data[data["patient_uid"].astype(str).isin(FOCUS_PATIENTS)].copy()
    target_episode_rows = find_episode_groups(
        focus_model, "y_true", "current_cgm"
    )
    for row in target_episode_rows:
        row["episode_type"] = "future_hypo60_target_positive_run"

    # Detailed row-level output: particularly false negatives, plus nearby actual low CGM.
    detail = focus_model.copy()
    detail["xgb_false_negative"] = (detail["y_true"] == 1) & (detail["xgb_pred_020"] == 0)
    detail["lgb_false_negative"] = (detail["y_true"] == 1) & (detail["lgb_pred_020"] == 0)
    detail["xgb_false_positive"] = (detail["y_true"] == 0) & (detail["xgb_pred_020"] == 1)
    detail["lgb_false_positive"] = (detail["y_true"] == 0) & (detail["lgb_pred_020"] == 1)
    detail["model_disagreement"] = detail["xgb_pred_020"] != detail["lgb_pred_020"]

    # Patient-level metrics and focused FN rows.
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    patient_rows = []
    for pid in FOCUS_PATIENTS:
        g = detail[detail["patient_uid"].astype(str) == pid]
        if g.empty:
            continue
        for model_name, prob_col, pred_col in [
            ("XGBoost", "xgb_probability", "xgb_pred_020"),
            ("LightGBM", "lgb_probability", "lgb_pred_020"),
        ]:
            y = g["y_true"].to_numpy()
            p = g[prob_col].to_numpy()
            pred = g[pred_col].to_numpy()
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            auc = safe_auc(y, p)
            patient_rows.append({
                "patient_uid": pid,
                "model": model_name,
                "rows": int(len(g)),
                **auc,
                "threshold": THRESHOLD,
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "sensitivity": float(recall_score(y, pred, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if tn + fp else None,
                "f1": float(f1_score(y, pred, zero_division=0)),
                "mean_predicted_probability": float(np.mean(p)),
                "mean_current_cgm_positive_labels": float(g.loc[g["y_true"] == 1, "current_cgm"].mean()),
                "mean_current_cgm_false_negatives": float(g.loc[(g["y_true"] == 1) & (g[pred_col] == 0), "current_cgm"].mean()),
            })

    # Compact detailed rows of FN and disagreements, with feature columns for investigation.
    selected = detail[
        detail["xgb_false_negative"] | detail["lgb_false_negative"] |
        detail["model_disagreement"]
    ].copy()
    keep = [
        "patient_uid", "segment_id", "dt", "split", "current_cgm", "y_true",
        "xgb_probability", "xgb_pred_020", "xgb_false_negative", "xgb_false_positive",
        "lgb_probability", "lgb_pred_020", "lgb_false_negative", "lgb_false_positive",
        "model_disagreement",
    ] + FEATURES
    selected = selected[[c for c in keep if c in selected.columns]].sort_values(["patient_uid", "dt"])

    # Summaries by month to expose temporal concentration, not causal explanations.
    detail["month"] = detail["dt"].dt.to_period("M").astype(str)
    monthly_rows = []
    for (pid, month), g in detail.groupby(["patient_uid", "month"]):
        if len(g) < 10:
            continue
        for model_name, prob_col, pred_col in [
            ("XGBoost", "xgb_probability", "xgb_pred_020"),
            ("LightGBM", "lgb_probability", "lgb_pred_020"),
        ]:
            y = g["y_true"].to_numpy()
            pred = g[pred_col].to_numpy()
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            monthly_rows.append({
                "patient_uid": pid, "month": month, "model": model_name,
                "rows": int(len(g)), "positive_labels": int(y.sum()),
                "prevalence": float(y.mean()), "tp": int(tp), "fp": int(fp),
                "tn": int(tn), "fn": int(fn),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
                "specificity": float(tn / (tn + fp)) if tn + fp else None,
                "mean_probability": float(g[prob_col].mean()),
            })

    # Run episodes can overlap one actual low event because the target looks 60 min ahead.
    episodes = pd.DataFrame(actual_episode_rows + target_episode_rows)
    patient_summary = pd.DataFrame(patient_rows)
    monthly = pd.DataFrame(monthly_rows)

    # Restrict summary rows to focus patients and clearly mark threshold as exploratory.
    outputs = {
        "patient_metrics": OUT_DIR / "057_focus_patient_metrics.csv",
        "episodes": OUT_DIR / "057_focus_patient_episodes.csv",
        "false_negatives_and_disagreements": OUT_DIR / "057_false_negatives_and_disagreements.csv",
        "monthly": OUT_DIR / "057_focus_monthly_metrics.csv",
        "json": OUT_DIR / "057_patient_episode_audit.json",
    }
    patient_summary.to_csv(outputs["patient_metrics"], index=False)
    episodes.to_csv(outputs["episodes"], index=False)
    selected.to_csv(outputs["false_negatives_and_disagreements"], index=False)
    monthly.to_csv(outputs["monthly"], index=False)

    report = {
        "audit": "GlucoTwin v3 focus-patient episode and false-negative audit",
        "read_only": True,
        "models_loaded_not_trained": [str(XGB_PATH), str(LGB_PATH)],
        "threshold_used_for_exploratory_comparison": THRESHOLD,
        "focus_patients": FOCUS_PATIENTS,
        "rows_by_focus_patient": {
            str(pid): int((detail["patient_uid"].astype(str) == pid).sum())
            for pid in FOCUS_PATIENTS
        },
        "actual_current_cgm_below_70_episode_count": int(len(actual_episode_rows)),
        "future_hypo60_positive_run_count": int(len(target_episode_rows)),
        "xgb_false_negative_rows": int(detail["xgb_false_negative"].sum()),
        "lgb_false_negative_rows": int(detail["lgb_false_negative"].sum()),
        "model_disagreement_rows": int(detail["model_disagreement"].sum()),
        "patient_metrics": patient_summary.to_dict(orient="records"),
        "important_interpretation_notes": [
            "The 0.20 probability threshold is an exploratory comparison threshold, not a clinical recommendation.",
            "A positive target means at least one of four future grid readings at +15, +30, +45, or +60 minutes is below 70 mg/dL.",
            "Consecutive positive target rows may overlap the same future low-glucose event; they are not independent events.",
            "Current CGM at a prediction timestamp is not the same as the minimum future CGM used to construct the target.",
            "This audit describes associations and errors; it does not establish clinical causes or clinical safety.",
            "No training or dataset/model modification is performed."
        ],
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 76)
    print("GLUCOTWIN - FOCUS PATIENT EPISODE / FALSE-NEGATIVE AUDIT (READ ONLY)")
    print("=" * 76)
    print(f"Exploratory threshold: {THRESHOLD:.2f}")
    print(f"Actual current CGM < {LOW_CGM:g} episodes: {len(actual_episode_rows)}")
    print(f"Future hypo60 positive-label runs: {len(target_episode_rows)}")
    print(f"XGBoost false-negative rows: {int(detail['xgb_false_negative'].sum())}")
    print(f"LightGBM false-negative rows: {int(detail['lgb_false_negative'].sum())}")
    print(f"Rows where models disagree at threshold: {int(detail['model_disagreement'].sum())}")
    print("\nPatient metrics:")
    if patient_summary.empty:
        print("No rows found for focus patients.")
    else:
        cols = ["patient_uid", "model", "rows", "positive_rows", "prevalence",
                "pr_auc", "roc_auc", "brier", "tp", "fp", "fn", "sensitivity", "specificity"]
        print(patient_summary[[c for c in cols if c in patient_summary.columns]].to_string(index=False))
    print("\nSaved reports:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("\nRead-only: no models trained and no source datasets/models modified.")


if __name__ == "__main__":
    main()
