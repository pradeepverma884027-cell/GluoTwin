"""
GlucoTwin feature pipeline audit
--------------------------------
Read-only audit: inspects the local preprocessing source, processed dataset,
schema, and saved model feature list. It does not modify or retrain anything.

Run this from the existing training folder.
"""

from pathlib import Path
import json
import re
import sys

import joblib
import pandas as pd

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "DATASET"
PROCESSED_DIR = DATASET_ROOT / "processed"

DATA_FILE = PROCESSED_DIR / "cgm_t1dm_t2dm_model_ready.csv"
SCHEMA_FILE = PROCESSED_DIR / "schema_cgm_t1dm_t2dm.json"
PREP_FILE = PROJECT_ROOT / "files" / "prep_pipeline.py"
MODEL_FILE = (
    TRAINING_DIR / "results_causal" / "models" / "lightgbm_causal_t30.joblib"
)

EXPECTED_FEATURES = [
    "cgm", "cgm_lag_15m", "cgm_lag_30m", "cgm_lag_45m",
    "cgm_lag_60m", "cgm_lag_90m", "cgm_lag_120m",
    "cgm_delta_15", "cgm_delta_30", "cgm_delta_60",
    "cgm_roll_mean_60m", "cgm_roll_std_60m", "cgm_roll_min_120m",
    "csii_basal", "fast_insulin_last_4h", "meal_grams_last_2h",
    "mins_since_meal", "mins_since_fast_insulin", "oral_agent_count",
    "cgm_mean_6h", "cgm_std_6h", "cgm_mean_24h", "cgm_tir_24h",
    "cgm_tbr_24h", "iob_proxy", "cob_proxy", "csii_suspend",
    "sc_long_acting", "meal_logged", "cgm_interpolated",
    "is_t2dm", "hour_sin", "hour_cos",
]

SEARCH_TERMS = [
    "groupby(", "shift(", "rolling(", "ewm(", "interpolate",
    "cgm_lag_", "cgm_delta_", "fast_insulin_last_4h",
    "meal_grams_last_2h", "mins_since_meal", "mins_since_fast_insulin",
    "cgm_mean_6h", "cgm_mean_24h", "cgm_tir_24h", "iob_proxy",
    "cob_proxy", "target_cgm_t30", "target_cgm_t60",
    "target_hypo_60", "sort_values", "segment_id", "cgm_interpolated",
]


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    section("GLUCOTWIN - CAUSAL FEATURE PIPELINE AUDIT")
    print("Training folder:", TRAINING_DIR)

    for path in [DATA_FILE, SCHEMA_FILE, PREP_FILE, MODEL_FILE]:
        print(f"{'FOUND' if path.is_file() else 'MISSING'}: {path}")

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Processed dataset not found:\n{DATA_FILE}")
    if not PREP_FILE.is_file():
        raise FileNotFoundError(f"Preprocessing source not found:\n{PREP_FILE}")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model artifact not found:\n{MODEL_FILE}")

    df = pd.read_csv(DATA_FILE, nrows=5)
    full_columns = list(pd.read_csv(DATA_FILE, nrows=0).columns)
    artifact = joblib.load(MODEL_FILE)
    model_features = list(artifact["features"])

    section("DATASET STRUCTURE")
    print("Columns:", len(full_columns))
    print("First five rows loaded for metadata inspection.")
    print("Available split values:", df["split"].dropna().unique().tolist()
          if "split" in df.columns else "split column missing")
    print("Required metadata columns:")
    for name in ["dt", "patient_uid", "segment_id", "split", "cgm"]:
        print(f"  {name}: {'OK' if name in full_columns else 'MISSING'}")

    section("MODEL FEATURE LIST")
    print("Features in artifact:", len(model_features))
    print("Features match expected 33-feature list:", model_features == EXPECTED_FEATURES)
    missing_from_dataset = [f for f in model_features if f not in full_columns]
    extra_expected = [f for f in EXPECTED_FEATURES if f not in model_features]
    print("Model features absent from dataset:", missing_from_dataset or "None")
    print("Expected features absent from artifact:", extra_expected or "None")
    for index, feature in enumerate(model_features, start=1):
        print(f"{index:02d}. {feature}")

    section("SCHEMA SUMMARY")
    if SCHEMA_FILE.is_file():
        try:
            schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
            print(json.dumps(schema, indent=2, ensure_ascii=False)[:14000])
        except Exception as exc:
            print("Could not parse schema JSON:", repr(exc))
    else:
        print("Schema file not found.")

    section("PREPROCESSING SOURCE MATCHES")
    source_lines = PREP_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    for i, line in enumerate(source_lines):
        if any(term.lower() in line.lower() for term in SEARCH_TERMS):
            hits.append(i)

    # Merge nearby hits to avoid printing the same source area repeatedly.
    ranges = []
    for hit in hits:
        start, end = max(0, hit - 3), min(len(source_lines), hit + 5)
        if ranges and start <= ranges[-1][1] + 2:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    if not ranges:
        print("No expected feature-generation terms found. Review source path/content.")
    else:
        max_chars = 24000
        printed_chars = 0
        for start, end in ranges:
            chunk = "\n".join(
                f"{i + 1:04d}: {source_lines[i]}"
                for i in range(start, end)
            )
            if printed_chars + len(chunk) > max_chars:
                print("\n[Output limit reached; remaining source excerpts omitted.]")
                break
            print("\n" + "-" * 78)
            print(chunk)
            printed_chars += len(chunk)

    section("INITIAL CAUSALITY CHECKLIST")
    checks = [
        ("Sort by patient, segment, and timestamp before shifts/rolling",
         any("sort_values" in line for line in source_lines)),
        ("Uses group/segment boundaries for temporal operations",
         any("groupby(" in line for line in source_lines)),
        ("Interpolation present in preprocessing source (requires review)",
         any("interpolate" in line.lower() for line in source_lines)),
        ("Model artifact records 96-step history filter",
         int(artifact.get("history_filter_steps", -1)) == 96),
        ("Artifact declares mg/dL target scale",
         str(artifact.get("target_scale", "")).lower() == "mg/dl"),
    ]
    for label, status in checks:
        print(f"[{'FOUND' if status else 'NOT FOUND'}] {label}")

    print("\nIMPORTANT:")
    print("- This script only finds source patterns; it does not prove causality.")
    print("- Interpolation may use future values depending on its method and timing.")
    print("- Rolling windows, event recency, target construction, and row ordering")
    print("  must be reviewed in context before enabling new/live predictions.")
    print("- No files are modified by this audit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nAUDIT FAILED:", repr(exc))
        sys.exit(1)
