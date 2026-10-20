"""
043_quantify_meal_flag_impact.py
Read-only audit of raw "data not available" meal entries against audit 041
and the model-ready GlucoTwin dataset.

Expected location:
DATASET/training/043_quantify_meal_flag_impact.py

This script does not modify source data, preprocessing code, or models.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
RAW_DIR = DATASET_ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
TRACE_PATH = HERE / "results_causal" / "audit" / "041_raw_to_clean_event_trace.csv"
CLEAN_CANDIDATES = [
    DATASET_ROOT / "DATASET" / "cleandataset" / "cgm_t1dm_t2dm_model_ready.csv",
    DATASET_ROOT / "DATASET" / "processed" / "cgm_t1dm_t2dm_model_ready.csv",
]
OUT_DIR = HERE / "results_causal" / "audit"
OUT_PATH = OUT_DIR / "043_unavailable_meal_impact.csv"
TOLERANCE = pd.Timedelta(minutes=7, seconds=30)


def stop(message):
    print(f"ERROR: {message}")
    sys.exit(1)


def find_col(df, names):
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def main():
    raw_paths = [("T1DM", RAW_DIR / "t1dm.csv"), ("T2DM", RAW_DIR / "t2dm.csv")]
    raw_parts = []

    for diabetes_type, path in raw_paths:
        if not path.exists():
            stop(f"Raw CSV not found: {path}")
        raw = pd.read_csv(path, low_memory=False)
        patient_col = find_col(raw, ["patient_id", "Patient ID", "Patient_ID"])
        date_col = find_col(raw, ["Date", "dt", "datetime", "timestamp"])
        diet_col = find_col(raw, ["Dietary intake", "diet_text", "dietary_intake"])
        if not all([patient_col, date_col, diet_col]):
            stop(f"Could not identify patient/date/meal columns in {path}. Columns: {list(raw.columns)}")

        part = pd.DataFrame({
            "_diabetes_type": diabetes_type,
            "_raw_patient_id": raw[patient_col].astype("string").str.strip(),
            "_raw_dt": pd.to_datetime(raw[date_col], errors="coerce"),
            "_diet_text": raw[diet_col].astype("string"),
        })
        part["_patient_uid"] = part["_diabetes_type"] + "_" + part["_raw_patient_id"].fillna("")
        part["_meal_unavailable"] = part["_diet_text"].str.contains(
            r"data\s+not\s+available", case=False, na=False, regex=True
        )
        raw_parts.append(part)

    raw_all = pd.concat(raw_parts, ignore_index=True)
    unavailable = raw_all.loc[raw_all["_meal_unavailable"]].copy()

    if not TRACE_PATH.exists():
        stop(f"Trace CSV not found: {TRACE_PATH}. Run audit 041 first.")
    trace = pd.read_csv(TRACE_PATH, low_memory=False)

    # These are the exact column names confirmed from the user's trace file.
    required_trace = ["_patient_uid", "_raw_dt", "matched_clean_dt", "match_status"]
    missing = [c for c in required_trace if c not in trace.columns]
    if missing:
        stop(f"Trace is missing expected columns {missing}. Actual columns: {list(trace.columns)}")

    trace["_raw_dt"] = pd.to_datetime(trace["_raw_dt"], errors="coerce")
    trace["matched_clean_dt"] = pd.to_datetime(trace["matched_clean_dt"], errors="coerce")

    clean_path = next((p for p in CLEAN_CANDIDATES if p.exists()), None)
    if clean_path is None:
        stop("Model-ready CSV not found in either cleandataset or processed folder.")
    clean = pd.read_csv(clean_path, low_memory=False)

    clean_patient_col = find_col(clean, ["patient_uid", "_patient_uid"])
    clean_dt_col = find_col(clean, ["dt", "Date", "timestamp", "datetime"])
    if clean_patient_col is None or clean_dt_col is None:
        stop(f"Could not identify patient/timestamp columns in clean CSV. Columns: {list(clean.columns)}")
    clean[clean_dt_col] = pd.to_datetime(clean[clean_dt_col], errors="coerce")
    clean["_audit_row_id"] = np.arange(len(clean))

    # Merge unavailable raw events to the corresponding trace rows by patient and raw time.
    # Raw data can contain duplicate patient/time events; keep the nearest trace event within tolerance.
    records = []
    trace_groups = {str(k): g for k, g in trace.groupby(trace["_patient_uid"].astype(str), sort=False)}
    clean_groups = {str(k): g for k, g in clean.groupby(clean[clean_patient_col].astype(str), sort=False)}

    for _, event in unavailable.iterrows():
        result = event.to_dict()
        result["trace_match"] = False
        result["clean_match"] = False
        result["trace_match_status"] = "no_patient_or_time_match"
        result["matched_clean_dt"] = pd.NaT
        result["match_delta_minutes"] = np.nan
        result["clean_row_id"] = np.nan

        if pd.notna(event["_raw_dt"]):
            candidates = trace_groups.get(str(event["_patient_uid"]))
            if candidates is not None and not candidates.empty:
                deltas = (candidates["_raw_dt"] - event["_raw_dt"]).abs()
                eligible = candidates.loc[deltas <= TOLERANCE].copy()
                if not eligible.empty:
                    eligible["_abs_delta"] = (eligible["_raw_dt"] - event["_raw_dt"]).abs()
                    tr = eligible.sort_values("_abs_delta").iloc[0]
                    result["trace_match"] = True
                    result["trace_match_status"] = tr["match_status"]
                    result["matched_clean_dt"] = tr["matched_clean_dt"]
                    result["match_delta_minutes"] = tr["match_delta_minutes"]
                    result["trace_meal_unavailable"] = tr.get("meal_unavailable", np.nan)

                    # A trace match only counts as a clean match when audit 041 actually
                    # found a clean timestamp and that timestamp exists in the clean CSV.
                    if pd.notna(tr["matched_clean_dt"]):
                        cg = clean_groups.get(str(event["_patient_uid"]))
                        if cg is not None and not cg.empty:
                            exact = cg.loc[cg[clean_dt_col] == tr["matched_clean_dt"]]
                            if not exact.empty:
                                cr = exact.iloc[0]
                                result["clean_match"] = True
                                result["clean_row_id"] = cr["_audit_row_id"]
                                for col in [
                                    "meal_logged", "meal_grams_last_2h", "mins_since_meal",
                                    "fast_insulin_last_4h", "mins_since_fast_insulin", "split",
                                ]:
                                    if col in clean.columns:
                                        result[f"clean_{col}"] = cr[col]
        records.append(result)

    impact = pd.DataFrame(records)

    # Reconstruct approximate original units only if stored values are z-scores,
    # using the train-only means/stds from the previous audit context.
    scaling = {
        "meal_grams_last_2h": (59.8900669312326, 140.8944464387071, "approx_meal_grams_last_2h"),
        "fast_insulin_last_4h": (1.4371261031807143, 3.163083058621087, "approx_fast_insulin_last_4h"),
        "mins_since_meal": (222.1815583723272, 126.20569833982562, "approx_mins_since_meal"),
        "mins_since_fast_insulin": (303.0154593377954, 105.33446417687357, "approx_mins_since_fast_insulin"),
    }
    for source, (mean, std, target) in scaling.items():
        col = f"clean_{source}"
        if col in impact.columns:
            impact[target] = pd.to_numeric(impact[col], errors="coerce") * std + mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    impact.to_csv(OUT_PATH, index=False)

    trace_match = impact["trace_match"].fillna(False)
    clean_match = impact["clean_match"].fillna(False)
    print("\n=== UNAVAILABLE MEAL FLAG AUDIT (READ-ONLY) ===")
    print(f"Raw unavailable-meal records: {len(unavailable):,}")
    print(f"Matched to trace within 7m30s: {int(trace_match.sum()):,}")
    print(f"Found in model-ready CSV: {int(clean_match.sum()):,}")
    if "clean_meal_logged" in impact.columns:
        logged = pd.to_numeric(impact["clean_meal_logged"], errors="coerce")
        print(f"Matched clean rows with meal_logged=1: {int((clean_match & logged.eq(1)).sum()):,}")
        print(f"Matched clean rows with meal_logged=0: {int((clean_match & logged.eq(0)).sum()):,}")
        print(f"Matched clean rows with missing meal_logged: {int((clean_match & logged.isna()).sum()):,}")
    if "clean_split" in impact.columns:
        print("\nMatched rows by split:")
        print(impact.loc[clean_match, "clean_split"].value_counts(dropna=False).to_string())
    print("\nTrace match statuses:")
    print(impact["trace_match_status"].value_counts(dropna=False).to_string())
    print(f"\nSaved: {OUT_PATH}")
    print("Read-only audit complete. No source dataset, preprocessing script, or trained model was modified.")


if __name__ == "__main__":
    main()
