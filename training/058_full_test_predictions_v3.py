from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE = Path(__file__).resolve().parent
PREP = BASE / "results_causal" / "preprocessing_v3"
MODEL_DIR = BASE / "results_causal" / "v3_model_comparison"

DATA_PATH = PREP / "cgm_t1dm_t2dm_causal_model_ready.csv"
XGB_PATH = MODEL_DIR / "models" / "xgboost__target_hypo_60.joblib"
LGB_PATH = MODEL_DIR / "models" / "lightgbm__target_hypo_60.joblib"
OUT_PATH = MODEL_DIR / "058_focus_test_predictions_full.csv"

FOCUS = ["T2DM_2072", "T2DM_2077", "T2DM_2078"]
THRESHOLD = 0.20

FEATURES = [
    "cgm", "cgm_lag_15m", "cgm_lag_30m", "cgm_lag_45m",
    "cgm_lag_60m", "cgm_lag_90m", "cgm_lag_120m",
    "cgm_delta_15", "cgm_delta_30", "cgm_delta_60",
    "cgm_roll_mean_60m", "cgm_roll_std_60m", "cgm_roll_min_120m",
    "csii_basal", "fast_insulin_last_4h", "meal_grams_last_2h",
    "mins_since_meal", "mins_since_fast_insulin", "oral_agent_count",
    "cgm_mean_6h", "cgm_std_6h", "cgm_mean_24h",
    "cgm_tir_24h", "cgm_tbr_24h", "iob_proxy", "cob_proxy",
    "csii_suspend", "sc_long_acting", "meal_logged",
    "cgm_interpolated", "is_t2dm", "hour_sin", "hour_cos",
]

def probability(model, X):
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    return np.clip(
        np.asarray(model.predict(X), dtype=float).reshape(-1), 0, 1
    )

def main():
    for path in [DATA_PATH, XGB_PATH, LGB_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    data = pd.read_csv(DATA_PATH, parse_dates=["dt"], low_memory=False)

    required = [
        "patient_uid", "segment_id", "dt", "split", "target_hypo_60"
    ] + FEATURES
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Restrict strictly to held-out test rows and the three focus patients.
    test = data[
        (data["split"].astype(str).str.lower() == "test")
        & data["patient_uid"].astype(str).isin(FOCUS)
    ].copy()

    if test.empty:
        raise ValueError("No focus-patient test rows found.")

    X = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Missing/non-finite feature values; aborting.")

    # Load existing models only. No training or fitting.
    xgb = joblib.load(XGB_PATH)
    lgb = joblib.load(LGB_PATH)

    test["y_true"] = pd.to_numeric(
        test["target_hypo_60"], errors="raise"
    ).astype(int)
    test["xgb_probability"] = probability(xgb, X)
    test["lgb_probability"] = probability(lgb, X)

    test["xgb_pred_020"] = (
        test["xgb_probability"] >= THRESHOLD
    ).astype(int)
    test["lgb_pred_020"] = (
        test["lgb_probability"] >= THRESHOLD
    ).astype(int)

    test["xgb_false_negative"] = (
        (test["y_true"] == 1) & (test["xgb_pred_020"] == 0)
    )
    test["lgb_false_negative"] = (
        (test["y_true"] == 1) & (test["lgb_pred_020"] == 0)
    )
    test["xgb_false_positive"] = (
        (test["y_true"] == 0) & (test["xgb_pred_020"] == 1)
    )
    test["lgb_false_positive"] = (
        (test["y_true"] == 0) & (test["lgb_pred_020"] == 1)
    )

    # Preserve every test row; do not filter to errors.
    keep = [
        "patient_uid", "segment_id", "dt", "split",
        "target_hypo_60", "y_true", "cgm",
        "xgb_probability", "xgb_pred_020", "xgb_false_negative",
        "xgb_false_positive",
        "lgb_probability", "lgb_pred_020", "lgb_false_negative",
        "lgb_false_positive",
    ] + FEATURES

    test = test.loc[:, list(dict.fromkeys(keep))]
    test = test.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)

    # New filename only; existing reports are untouched.
    test.to_csv(OUT_PATH, index=False)

    print("=" * 65)
    print("FULL FOCUS-PATIENT TEST PREDICTIONS - READ ONLY")
    print("=" * 65)
    print(f"Rows exported: {len(test)}")
    print(f"Output: {OUT_PATH}")
    print("\nRows and positive labels by patient:")
    print(
        test.groupby("patient_uid").agg(
            rows=("y_true", "size"),
            positive_labels=("y_true", "sum"),
            xgb_fn_rows=("xgb_false_negative", "sum"),
            lgb_fn_rows=("lgb_false_negative", "sum"),
        ).to_string()
    )
    print("\nModels loaded only; no training performed.")

if __name__ == "__main__":
    main()
