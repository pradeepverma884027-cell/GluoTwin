
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# GLUCOTWIN 042 — EVENT FEATURE AUDIT
# READ-ONLY: Does not modify datasets or trained models
# ============================================================

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent

RAW_DIR = PROJECT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
AUDIT_DIR = BASE / "results_causal" / "audit"

CLEAN_CANDIDATES = [
    PROJECT / "DATASET" / "cleandataset" /
    "cgm_t1dm_t2dm_model_ready.csv",
    PROJECT / "DATASET" / "processed" /
    "cgm_t1dm_t2dm_model_ready.csv",
]

TRACE_PATH = AUDIT_DIR / "041_raw_to_clean_event_trace.csv"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Find files
# ------------------------------------------------------------

clean_path = next(
    (path for path in CLEAN_CANDIDATES if path.exists()),
    None
)

if clean_path is None:
    matches = list(
        PROJECT.rglob("cgm_t1dm_t2dm_model_ready.csv")
    )
    if not matches:
        raise FileNotFoundError(
            "Could not find cgm_t1dm_t2dm_model_ready.csv"
        )
    clean_path = matches[0]

if not TRACE_PATH.exists():
    raise FileNotFoundError(
        f"Trace file not found: {TRACE_PATH}\n"
        "Run script 041 first."
    )

print("=" * 70)
print("GLUCOTWIN EVENT FEATURE AUDIT — READ ONLY")
print("=" * 70)
print("Clean dataset:", clean_path)
print("Trace file:", TRACE_PATH)

# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

clean = pd.read_csv(clean_path, low_memory=False)
trace = pd.read_csv(TRACE_PATH, low_memory=False)

required_clean = {"patient_uid", "dt"}
missing_required = required_clean - set(clean.columns)

if missing_required:
    raise ValueError(
        f"Clean dataset missing required columns: {missing_required}"
    )

clean["patient_uid"] = clean["patient_uid"].astype("string")
clean["dt"] = pd.to_datetime(clean["dt"], errors="coerce")

# Script 041 stores this key as _patient_uid.
# Create patient_uid for a consistent merge.
if "_patient_uid" in trace.columns:
    trace["patient_uid"] = trace["_patient_uid"].astype("string")
elif "patient_uid" in trace.columns:
    trace["patient_uid"] = trace["patient_uid"].astype("string")
else:
    raise ValueError(
        "Neither '_patient_uid' nor 'patient_uid' exists in trace CSV."
    )

trace["_raw_dt"] = pd.to_datetime(
    trace.get("_raw_dt"), errors="coerce"
)

trace["matched_clean_dt"] = pd.to_datetime(
    trace.get("matched_clean_dt"), errors="coerce"
)

print("\n--- DATASET OVERVIEW ---")
print("Clean rows:", len(clean))
print("Trace rows:", len(trace))
print("Clean columns:", len(clean.columns))
print("Trace columns:", len(trace.columns))

# ------------------------------------------------------------
# 3. Inspect expected event-derived features
# ------------------------------------------------------------

expected_features = [
    "meal_logged",
    "meal_unknown",
    "meal_grams",
    "meal_grams_last_2h",
    "mins_since_meal",
    "fast_insulin",
    "fast_insulin_last_4h",
    "mins_since_fast_insulin",
    "iob_proxy",
    "cob_proxy",
]

available_features = [
    col for col in expected_features if col in clean.columns
]

missing_features = [
    col for col in expected_features if col not in clean.columns
]

print("\n--- EVENT FEATURE SCHEMA ---")
print("Available features:")
for col in available_features:
    print("  FOUND:", col)

print("\nNot stored in clean CSV:")
for col in missing_features:
    print("  NOT FOUND:", col)

# ------------------------------------------------------------
# 4. Audit raw meal descriptions in trace
# ------------------------------------------------------------

diet_col = "Dietary intake"

if diet_col in trace.columns:
    diet = (
        trace[diet_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    meal_nonempty = diet.ne("")

    meal_unavailable = diet.str.contains(
        r"data\s+not\s+available",
        case=False,
        na=False,
        regex=True,
    )

    meal_numeric_grams = diet.str.contains(
        r"\d+(?:\.\d+)?\s*g\b",
        case=False,
        na=False,
        regex=True,
    )

    print("\n--- RAW MEAL TEXT AUDIT ---")
    print("Non-empty meal descriptions:", int(meal_nonempty.sum()))
    print(
        '"data not available" records:',
        int(meal_unavailable.sum())
    )
    print(
        "Non-empty unavailable descriptions:",
        int((meal_unavailable & meal_nonempty).sum())
    )
    print(
        "Descriptions containing gram quantities:",
        int(meal_numeric_grams.sum())
    )
    print(
        "Unavailable descriptions with timestamp match:",
        int(
            (
                meal_unavailable
                & trace["matched_clean_dt"].notna()
            ).sum()
        )
    )

    print("\nExample unavailable meal descriptions:")
    examples = trace.loc[
        meal_unavailable,
        [c for c in [
            "_source_file",
            "_raw_row",
            "_patient_uid",
            "_raw_dt",
            diet_col,
            "matched_clean_dt",
            "match_status",
        ] if c in trace.columns]
    ].head(10)

    if examples.empty:
        print("No examples found.")
    else:
        print(examples.to_string(index=False))

else:
    print(
        "\nWARNING: Dietary intake column is not present "
        "in the trace CSV."
    )
    meal_unavailable = pd.Series(False, index=trace.index)

# ------------------------------------------------------------
# 5. Check timestamp quality and duplicates
# ------------------------------------------------------------

print("\n--- CLEAN TIMESTAMP AUDIT ---")

duplicate_mask = clean.duplicated(
    subset=["patient_uid", "dt"],
    keep=False,
)

print(
    "Duplicate patient/timestamp rows:",
    int(duplicate_mask.sum())
)
print(
    "Missing clean timestamps:",
    int(clean["dt"].isna().sum())
)
print(
    "Missing patient IDs:",
    int(clean["patient_uid"].isna().sum())
)

print(
    "Raw rows with invalid timestamps:",
    int(trace["_raw_dt"].isna().sum())
)
print(
    "Raw rows with no nearest clean timestamp:",
    int(trace["matched_clean_dt"].isna().sum())
)

# ------------------------------------------------------------
# 6. Join trace records to actual clean feature values
# ------------------------------------------------------------

join_features = ["patient_uid", "dt"] + available_features

clean_small = clean[join_features].copy()

clean_small = clean_small.rename(
    columns={"dt": "matched_clean_dt"}
)

# A clean timestamp should be unique per patient.
# If duplicates exist, stop rather than multiply event rows.
if clean_small.duplicated(
    ["patient_uid", "matched_clean_dt"]
).any():
    raise ValueError(
        "Duplicate patient/timestamp keys found. "
        "Resolve mapping ambiguity before feature comparisons."
    )

joined = trace.merge(
    clean_small,
    on=["patient_uid", "matched_clean_dt"],
    how="left",
    indicator=True,
    validate="many_to_one",
)

joined["_feature_joined"] = joined["_merge"].eq("both")

print("\n--- TRACE JOIN AUDIT ---")
print("Raw trace rows:", len(trace))
print(
    "Rows joined to clean features:",
    int(joined["_feature_joined"].sum())
)
print(
    "Rows without clean feature match:",
    int((~joined["_feature_joined"]).sum())
)

if "match_status" in joined.columns:
    print("\nTimestamp match status:")
    print(joined["match_status"].value_counts(dropna=False).to_string())

# ------------------------------------------------------------
# 7. Inspect unavailable meal records against clean features
# ------------------------------------------------------------

print("\n--- UNAVAILABLE MEAL FEATURE CHECK ---")

if diet_col in joined.columns:
    joined_diet = (
        joined[diet_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    unavailable_mask = joined_diet.str.contains(
        r"data\s+not\s+available",
        case=False,
        na=False,
        regex=True,
    )

    matched_unavailable = (
        unavailable_mask & joined["_feature_joined"]
    )

    print("Unavailable raw records:", int(unavailable_mask.sum()))
    print(
        "Unavailable records joined to features:",
        int(matched_unavailable.sum())
    )

    for col in [
        "meal_logged",
        "meal_unknown",
        "meal_grams",
        "meal_grams_last_2h",
        "mins_since_meal",
        "cob_proxy",
    ]:
        if col not in joined.columns:
            print(f"\n{col}: not stored in model-ready CSV")
            continue

        values = pd.to_numeric(
            joined.loc[matched_unavailable, col],
            errors="coerce",
        ).dropna()

        print(f"\n{col}:")
        print("  Valid values:", len(values))

        if not values.empty:
            print(f"  Minimum: {values.min():.6f}")
            print(f"  Median:  {values.median():.6f}")
            print(f"  Maximum: {values.max():.6f}")
            print("  Zero count:", int(values.eq(0).sum()))
            print("  Nonzero count:", int(values.ne(0).sum()))

    # Save records for manual review.
    review_columns = [
        col for col in [
            "_source_file",
            "_raw_row",
            "_patient_uid",
            "_raw_dt",
            diet_col,
            "matched_clean_dt",
            "match_status",
            *available_features,
        ]
        if col in joined.columns
    ]

    unavailable_review = joined.loc[
        unavailable_mask, review_columns
    ].copy()

    unavailable_path = (
        AUDIT_DIR / "042_unavailable_meal_review.csv"
    )
    unavailable_review.to_csv(unavailable_path, index=False)

    print("\nSaved unavailable-meal review:", unavailable_path)

else:
    print("Meal text is unavailable in trace; cannot run this check.")

# ------------------------------------------------------------
# 8. Summarize clean feature distributions
# ------------------------------------------------------------

print("\n--- CLEAN FEATURE DISTRIBUTIONS ---")

summary_rows = []

for col in available_features:
    values = pd.to_numeric(clean[col], errors="coerce")

    row = {
        "feature": col,
        "non_null": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "zero": int(values.eq(0).sum()),
        "nonzero": int(values.ne(0).sum()),
        "minimum": values.min(),
        "median": values.median(),
        "maximum": values.max(),
    }

    summary_rows.append(row)

    print(f"\n{col}")
    print("  Non-null:", row["non_null"])
    print("  Missing:", row["missing"])
    print("  Zero:", row["zero"])
    print("  Nonzero:", row["nonzero"])
    print("  Min:", row["minimum"])
    print("  Median:", row["median"])
    print("  Max:", row["maximum"])

summary_path = AUDIT_DIR / "042_event_feature_summary.csv"

pd.DataFrame(summary_rows).to_csv(
    summary_path,
    index=False,
)

# ------------------------------------------------------------
# 9. Save joined trace for deeper audit
# ------------------------------------------------------------

joined_path = AUDIT_DIR / "042_joined_event_feature_trace.csv"

joined.to_csv(joined_path, index=False)

print("\n" + "=" * 70)
print("AUDIT COMPLETE — READ ONLY")
print("=" * 70)
print("Feature summary:", summary_path)
print("Joined trace:", joined_path)

if diet_col in trace.columns:
    print(
        "Unavailable meal review:",
        AUDIT_DIR / "042_unavailable_meal_review.csv"
    )

print("\nNo raw dataset, clean dataset, model, or calibration file was changed.")
