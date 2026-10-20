"""
045_audit_causal_preprocessing.py
Read-only validation of the candidate dataset created by 044.

Run:
  python -u 045_audit_causal_preprocessing.py

Reads:
  results_causal/preprocessing_v2/cgm_t1dm_t2dm_causal_clean.csv
  results_causal/preprocessing_v2/cgm_t1dm_t2dm_causal_model_ready.csv
  results_causal/preprocessing_v2/schema_cgm_t1dm_t2dm_causal.json
  results_causal/preprocessing_v2/causal_preprocessing_audit.json

Writes a separate report:
  results_causal/preprocessing_v2/045_causal_dataset_audit.json

Does not modify any dataset, source script, or model.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results_causal" / "preprocessing_v2"
CLEAN_PATH = OUT_DIR / "cgm_t1dm_t2dm_causal_clean.csv"
MODEL_PATH = OUT_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
SCHEMA_PATH = OUT_DIR / "schema_cgm_t1dm_t2dm_causal.json"
BUILD_AUDIT_PATH = OUT_DIR / "causal_preprocessing_audit.json"
REPORT_PATH = OUT_DIR / "045_causal_dataset_audit.json"

def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

def main():
    for path in [CLEAN_PATH, MODEL_PATH, SCHEMA_PATH, BUILD_AUDIT_PATH]:
        if not path.exists():
            fail(f"Required file missing: {path}")

    model = pd.read_csv(MODEL_PATH, low_memory=False)
    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    build_audit = json.loads(BUILD_AUDIT_PATH.read_text(encoding="utf-8"))

    features = schema["features"]
    targets = schema["targets"]
    required_meta = ["patient_uid", "diabetes_type", "segment_id", "dt", "split"]
    missing = [c for c in required_meta + features + targets if c not in model.columns]
    if missing:
        fail(f"Missing expected model-ready columns: {missing}")

    model["dt"] = pd.to_datetime(model["dt"], errors="coerce")
    clean["dt"] = pd.to_datetime(clean["dt"], errors="coerce")

    split_counts = model["split"].value_counts(dropna=False).to_dict()
    patients_by_split = model.groupby("split")["patient_uid"].nunique().to_dict()
    patient_split_nunique = model.groupby("patient_uid")["split"].nunique()
    split_overlap = model.groupby("patient_uid")["split"].nunique().gt(1)
    dup_keys = int(model.duplicated(["patient_uid", "segment_id", "dt"]).sum())
    missing_features = int(model[features].isna().sum().sum())
    infinite_features = int(np.isinf(model[features].select_dtypes(include=[np.number])).sum().sum())
    missing_targets = {c: int(model[c].isna().sum()) for c in targets}
    bad_hypo = int((~model["target_hypo_60"].dropna().isin([0, 1])).sum())

    # Verify that the raw event flags are mutually consistent in the clean file.
    meal_logged_bad = int(((clean["meal_logged"] == 1) & (clean["meal_unknown"] == 1)).sum())
    unknown_with_grams = int(((clean["meal_unknown"] == 1) & (pd.to_numeric(clean["meal_grams"], errors="coerce").fillna(0) != 0)).sum())
    unknown_count = int((clean["meal_unknown"] == 1).sum())

    # Causal feature spot checks on the saved model-ready values are limited because
    # the continuous columns are scaled. Validate timestamp ordering and segment keys.
    segment_order_bad = 0
    for (_, _), g in clean.groupby(["patient_uid", "segment_id"], sort=False):
        times = g["dt"].dropna()
        if not times.is_monotonic_increasing:
            segment_order_bad += 1

    # Reconstruct the grid row count and check 15-minute spacing within segments.
    gap_minutes = clean.groupby(["patient_uid", "segment_id"])["dt"].diff().dt.total_seconds() / 60
    non_15_min_gaps = int((gap_minutes.notna() & ~np.isclose(gap_minutes.fillna(15), 15)).sum())

    # The target rows should correspond to future grid points within the same segment.
    # Since target columns are saved on the model file, check timestamps of source rows
    # that survived feature/target completeness. This is a structural check, not a full
    # recomputation of target values.
    model_segment_sizes = model.groupby(["patient_uid", "segment_id"]).size()
    short_segments = int((model_segment_sizes < 5).sum())

    # Split should be patient-independent.
    overlap_pass = not bool(split_overlap.any())
    report = {
        "candidate_model_rows": int(len(model)),
        "candidate_clean_rows": int(len(clean)),
        "feature_count": int(len(features)),
        "target_count": int(len(targets)),
        "rows_by_split": {str(k): int(v) for k, v in split_counts.items()},
        "patients_by_split": {str(k): int(v) for k, v in patients_by_split.items()},
        "patient_split_overlap_count": int(split_overlap.sum()),
        "patient_split_independence_pass": bool(overlap_pass),
        "duplicate_model_patient_segment_timestamp_keys": dup_keys,
        "missing_feature_cells": missing_features,
        "infinite_feature_values": infinite_features,
        "missing_targets": missing_targets,
        "invalid_hypo_target_values": bad_hypo,
        "clean_rows_meal_logged_and_unknown_both_one": meal_logged_bad,
        "clean_rows_unknown_meal_with_nonzero_grams": unknown_with_grams,
        "clean_rows_meal_unknown_one": unknown_count,
        "segments_not_time_sorted": segment_order_bad,
        "non_15_minute_gaps_in_clean_grid": non_15_min_gaps,
        "model_segments_with_fewer_than_5_rows": short_segments,
        "build_audit": build_audit,
        "limitations": [
            "This audit checks structural integrity, not clinical validity.",
            "It does not fully recompute every rolling feature and target from raw data.",
            "Causal claims still require a code-level review of event-to-grid alignment and feature calculations.",
            "Do not retrain until the audit output has been reviewed.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("GLUCOTWIN - CANDIDATE DATASET AUDIT (READ-ONLY)")
    print("=" * 72)
    print(f"Model rows: {len(model):,}")
    print(f"Clean rows: {len(clean):,}")
    print(f"Feature count: {len(features)}")
    print(f"Patient split overlap: {int(split_overlap.sum())} (expected 0)")
    print(f"Duplicate model keys: {dup_keys} (expected 0)")
    print(f"Missing feature cells: {missing_features} (expected 0)")
    print(f"Infinite feature values: {infinite_features} (expected 0)")
    print(f"meal_logged & meal_unknown both 1: {meal_logged_bad} (expected 0)")
    print(f"Unknown meals with nonzero grams: {unknown_with_grams} (expected 0)")
    print(f"Segments not sorted: {segment_order_bad} (expected 0)")
    print(f"Non-15-minute gaps in clean grid: {non_15_min_gaps} (investigate if >0)")
    print(f"\nReport saved: {REPORT_PATH}")
    print("\nNo dataset, source code, or model was modified.")
    print("Review the report before retraining.")

if __name__ == "__main__":
    main()
