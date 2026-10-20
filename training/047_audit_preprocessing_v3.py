from pathlib import Path
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "results_causal" / "preprocessing_v3"
CLEAN_PATH = ROOT / "cgm_t1dm_t2dm_causal_clean.csv"
MODEL_PATH = ROOT / "cgm_t1dm_t2dm_causal_model_ready.csv"
OUT_PATH = ROOT / "047_preprocessing_v3_audit.json"
FEATURES = [
"cgm","cgm_lag_15m","cgm_lag_30m","cgm_lag_45m","cgm_lag_60m","cgm_lag_90m","cgm_lag_120m",
"cgm_delta_15","cgm_delta_30","cgm_delta_60","cgm_roll_mean_60m","cgm_roll_std_60m",
"cgm_roll_min_120m","csii_basal","fast_insulin_last_4h","meal_grams_last_2h","mins_since_meal",
"mins_since_fast_insulin","oral_agent_count","cgm_mean_6h","cgm_std_6h","cgm_mean_24h",
"cgm_tir_24h","cgm_tbr_24h","iob_proxy","cob_proxy","csii_suspend","sc_long_acting",
"meal_logged","cgm_interpolated","is_t2dm","hour_sin","hour_cos"]
TARGETS = ["target_cgm_t30","target_cgm_t60","target_hypo_60"]

def main():
    for p in (CLEAN_PATH, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required v3 file missing: {p}")
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["dt"], low_memory=False)
    model = pd.read_csv(MODEL_PATH, parse_dates=["dt"], low_memory=False)
    report = {"clean_rows": len(clean), "model_rows": len(model),
              "missing_features": [c for c in FEATURES if c not in model],
              "missing_targets": [c for c in TARGETS if c not in model]}
    keys = ["patient_uid","segment_id","dt"]
    report["clean_duplicate_keys"] = int(clean.duplicated(keys).sum())
    report["model_duplicate_keys"] = int(model.duplicated(keys).sum())
    report["missing_feature_cells"] = int(model[FEATURES].isna().sum().sum()) if not report["missing_features"] else None
    report["infinite_feature_cells"] = int(np.isinf(model[FEATURES].select_dtypes(include=[np.number])).sum().sum()) if not report["missing_features"] else None
    report["missing_target_cells"] = {c:int(model[c].isna().sum()) for c in TARGETS if c in model}
    report["patient_split_overlap_count"] = int((model.groupby("patient_uid")["split"].nunique()>1).sum()) if "split" in model else None
    report["rows_by_split"] = {str(k):int(v) for k,v in model["split"].value_counts().items()} if "split" in model else {}
    report["patients_by_split"] = {str(k):int(v) for k,v in model.groupby("split")["patient_uid"].nunique().items()} if "split" in model else {}
    cgm_map = clean.set_index(keys)["cgm"].to_dict()
    checked = {"t30":0,"t60":0,"hypo60":0}
    mismatch = {"t30":0,"t60":0,"hypo60":0}
    for pid,seg,dt,t30,t60,hypo in model[["patient_uid","segment_id","dt",*TARGETS]].itertuples(index=False,name=None):
        for name, mins, target in (("t30",30,t30),("t60",60,t60)):
            val = cgm_map.get((pid,seg,dt+pd.Timedelta(minutes=mins)),np.nan)
            if pd.notna(val):
                checked[name] += 1
                if not np.isclose(float(target),float(val),atol=1e-5): mismatch[name] += 1
        fut = [cgm_map.get((pid,seg,dt+pd.Timedelta(minutes=m)),np.nan) for m in (15,30,45,60)]
        if all(pd.notna(x) for x in fut):
            checked["hypo60"] += 1
            if not np.isclose(float(hypo),float(min(fut)<70.0)): mismatch["hypo60"] += 1
    report["target_reconstruction"] = {"checked":checked,"mismatches":mismatch,
        "note":"Exact clean-grid reconstruction; does not prove event features are causal."}
    report["clean_rows_meal_logged_and_unknown_both_one"] = int(((clean["meal_logged"]==1)&(clean["meal_unknown"]==1)).sum())
    report["clean_rows_unknown_meal_with_nonzero_grams"] = int(((clean["meal_unknown"]==1)&(clean["meal_grams"].fillna(0)!=0)).sum())
    gaps = clean.groupby(["patient_uid","segment_id"])["dt"].diff().dt.total_seconds()/60
    report["clean_non_15_minute_gaps"] = int((gaps.dropna()!=15).sum())
    OUT_PATH.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print("GLUCOTWIN - PREPROCESSING V3 READ-ONLY AUDIT")
    for k,v in report.items(): print(f"{k}: {v}")
    print(f"Report saved: {OUT_PATH}")
    print("Read-only: no dataset or model was modified.")

if __name__ == "__main__": main()
