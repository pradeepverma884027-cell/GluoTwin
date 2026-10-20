
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RAW_DIR = PROJECT_DIR / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
OUTPUT_DIR = BASE_DIR / "results_causal" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_CANDIDATES = [
    PROJECT_DIR / "DATASET" / "cleandataset" /
    "cgm_t1dm_t2dm_model_ready.csv",
    PROJECT_DIR / "DATASET" / "processed" /
    "cgm_t1dm_t2dm_model_ready.csv",
]

clean_path = next((p for p in CLEAN_CANDIDATES if p.exists()), None)

if clean_path is None:
    matches = list(PROJECT_DIR.rglob("cgm_t1dm_t2dm_model_ready.csv"))
    if not matches:
        raise FileNotFoundError("Model-ready CSV not found.")
    clean_path = matches[0]

print("Clean dataset:", clean_path)

clean = pd.read_csv(clean_path, low_memory=False)
clean["dt"] = pd.to_datetime(clean["dt"], errors="coerce")
clean["patient_uid"] = clean["patient_uid"].astype(str)

# Build a separate timestamp index for each patient.
clean = clean.dropna(subset=["dt"]).sort_values(
    ["patient_uid", "dt"]
).reset_index(drop=True)

raw_frames = []

for filename, diabetes_type in [
    ("t1dm.csv", "T1DM"),
    ("t2dm.csv", "T2DM"),
]:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path, low_memory=False)
    raw["_source_file"] = filename
    raw["_diabetes_type"] = diabetes_type
    raw["_raw_row"] = range(2, len(raw) + 2)
    raw["_raw_dt"] = pd.to_datetime(raw["Date"], errors="coerce")

    # Use the same patient-ID construction as preprocessing.
    raw["_patient_uid"] = (
        raw["_diabetes_type"] + "_" + raw["patient_id"].astype(str)
    )

    raw_frames.append(raw)

raw_all = pd.concat(raw_frames, ignore_index=True)
raw_all["_sc_text"] = (
    raw_all["Insulin dose - s.c."].fillna("").astype(str)
    if "Insulin dose - s.c." in raw_all.columns
    else ""
)
raw_all["_diet_text"] = (
    raw_all["Dietary intake"].fillna("").astype(str)
    if "Dietary intake" in raw_all.columns
    else ""
)

raw_all["has_sc_insulin"] = raw_all["_sc_text"].str.strip().ne("")
raw_all["has_meal_text"] = raw_all["_diet_text"].str.strip().ne("")
raw_all["meal_unavailable"] = raw_all["_diet_text"].str.contains(
    r"data not available", case=False, na=False
)
raw_all["has_time"] = raw_all["_raw_dt"].notna()

# Conservative text-based screening, not proof of a wrong feature.
raw_all["insulin_review_flag"] = raw_all["_sc_text"].str.contains(
    r"\b50R\b|\b40R\b|\b30R\b|"
    r"premix|mixed|glargine|detemir|degludec",
    case=False,
    na=False,
)

# Match each raw event to its nearest cleaned row for the same patient.
clean_groups = {
    patient: group.sort_values("dt").reset_index(drop=True)
    for patient, group in clean.groupby("patient_uid", sort=False)
}

matched_parts = []

for patient_uid, raw_group in raw_all.groupby("_patient_uid", sort=False):
    raw_group = raw_group.copy()
    target = clean_groups.get(patient_uid)

    if target is None or target.empty:
        raw_group["matched_clean_dt"] = pd.NaT
        raw_group["match_delta_minutes"] = float("nan")
        raw_group["match_status"] = "patient_not_found"
        matched_parts.append(raw_group)
        continue

    left = raw_group.loc[raw_group["_raw_dt"].notna()].sort_values("_raw_dt")
    invalid = raw_group.loc[raw_group["_raw_dt"].isna()].copy()

    if not left.empty:
        right = target[["dt"]].rename(
            columns={"dt": "matched_clean_dt"}
        ).sort_values("matched_clean_dt")

        result = pd.merge_asof(
            left,
            right,
            left_on="_raw_dt",
            right_on="matched_clean_dt",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=7, seconds=30),
        )

        result["match_delta_minutes"] = (
            result["matched_clean_dt"] - result["_raw_dt"]
        ).dt.total_seconds() / 60

        result["match_status"] = result["matched_clean_dt"].notna().map(
            {True: "matched_within_7m30s", False: "no_row_within_tolerance"}
        )
        matched_parts.append(result)

    if not invalid.empty:
        invalid["matched_clean_dt"] = pd.NaT
        invalid["match_delta_minutes"] = float("nan")
        invalid["match_status"] = "invalid_raw_timestamp"
        matched_parts.append(invalid)

trace = pd.concat(matched_parts, ignore_index=True)

detail_path = OUTPUT_DIR / "041_raw_to_clean_event_trace.csv"
trace.to_csv(detail_path, index=False)

summary = (
    trace.groupby(
        ["_source_file", "match_status"], dropna=False
    )
    .agg(
        records=("_raw_row", "size"),
        SC_insulin_records=("has_sc_insulin", "sum"),
        meal_text_records=("has_meal_text", "sum"),
        unavailable_meal_records=("meal_unavailable", "sum"),
        insulin_review_records=("insulin_review_flag", "sum"),
    )
    .reset_index()
)

summary_path = OUTPUT_DIR / "041_raw_to_clean_event_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nMATCH SUMMARY")
print(summary.to_string(index=False))

print("\nOVERALL")
print("Raw records:", len(trace))
print("Matched:", int(trace["match_status"].eq("matched_within_7m30s").sum()))
print("Unmatched:", int((~trace["match_status"].eq("matched_within_7m30s")).sum()))

print("\nSaved detail:", detail_path)
print("Saved summary:", summary_path)
print("\nRead-only trace complete.")
