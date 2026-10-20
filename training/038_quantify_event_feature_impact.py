
from pathlib import Path
import re
import json
import pandas as pd

# ============================================================
# GlucoTwin: Read-only event feature impact audit
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

def find_file(filename):
    matches = list(PROJECT_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {PROJECT_DIR}"
        )
    return matches[0]

RAW_DIR = find_file("t1dm.csv").parent
T1_FILE = RAW_DIR / "t1dm.csv"
T2_FILE = RAW_DIR / "t2dm.csv"

SCHEMA_FILE = find_file("schema_cgm_t1dm_t2dm.json")
SCHEMA = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
SCALING = SCHEMA.get("scaled_cols_mean_std", {})

OUT_DIR = BASE_DIR / "results_causal" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = OUT_DIR / "038_event_record_flags.csv"
SUMMARY_FILE = OUT_DIR / "038_event_impact_summary.csv"

# Screening patterns only; these do not establish clinical classification.
MIXED_PATTERN = re.compile(
    r"70\s*/\s*30|30\s*/\s*70|50\s*/\s*50|"
    r"40\s*/\s*60|premix|biphasic|"
    r"novolin\s*30r|novolin\s*50r|"
    r"humulin\s*70\s*/?\s*30|"
    r"gansulin\s*40r|scilin\s*m30",
    re.I,
)
LONG_PATTERN = re.compile(r"glargine|detemir|degludec", re.I)
REGULAR_PATTERN = re.compile(
    r"novolin\s*r|humulin\s*r|gansulin\s*r|"
    r"regular\s+insulin|insulin\s+r\b",
    re.I,
)
RAPID_PATTERN = re.compile(r"aspart|lispro|glulisine", re.I)
UNAVAILABLE_MEAL_PATTERN = re.compile(
    r"data\s+not\s+available|not\s+available|unknown",
    re.I,
)
DOSE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*IU\b", re.I)


def text_or_empty(value):
    return "" if pd.isna(value) else str(value).strip()


def screen_insulin(value):
    text = text_or_empty(value)
    if not text:
        return {
            "category": "missing",
            "mixed_flag": False,
            "long_flag": False,
            "regular_flag": False,
            "rapid_flag": False,
            "multiple_dose_numbers": 0,
            "parsed_iu_sum_audit_only": 0.0,
        }

    mixed = bool(MIXED_PATTERN.search(text))
    long_acting = bool(LONG_PATTERN.search(text))
    regular = bool(REGULAR_PATTERN.search(text))
    rapid = bool(RAPID_PATTERN.search(text))

    types_found = sum([long_acting, regular, rapid])

    if mixed:
        category = "mixed_or_premixed_review"
    elif types_found > 1:
        category = "multiple_types_review"
    elif long_acting:
        category = "long_acting_name"
    elif regular:
        category = "regular_name"
    elif rapid:
        category = "rapid_acting_name"
    else:
        category = "unknown_review"

    doses = [float(x) for x in DOSE_PATTERN.findall(text)]

    return {
        "category": category,
        "mixed_flag": mixed,
        "long_flag": long_acting,
        "regular_flag": regular,
        "rapid_flag": rapid,
        "multiple_dose_numbers": len(doses),
        "parsed_iu_sum_audit_only": sum(doses),
    }


def main():
    reports = []

    for source_file in [T1_FILE, T2_FILE]:
        if not source_file.exists():
            print("WARNING: Missing raw file:", source_file)
            continue

        df = pd.read_csv(source_file, low_memory=False)

        required = ["Insulin dose - s.c.", "Dietary intake"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(
                f"WARNING: {source_file.name} missing columns: {missing}"
            )
            continue

        insulin = df["Insulin dose - s.c."].map(screen_insulin)
        meal_text = df["Dietary intake"].map(text_or_empty)

        report = pd.DataFrame({
            "source_file": source_file.name,
            "source_row_index": df.index,
            "sc_insulin_text": df["Insulin dose - s.c."].astype("string"),
            "meal_text": df["Dietary intake"].astype("string"),
        })

        # Keep source identifiers where present.
        for col in [
            "Patient ID", "Patient", "patient_uid",
            "Date", "Time", "Timestamp", "dt",
        ]:
            if col in df.columns:
                report[col] = df[col].values

        parsed = pd.DataFrame(insulin.tolist(), index=df.index)
        for col in parsed.columns:
            report[col] = parsed[col].values

        report["meal_nonempty"] = meal_text.ne("")
        report["meal_unavailable_text"] = meal_text.map(
            lambda x: bool(UNAVAILABLE_MEAL_PATTERN.search(x))
        )

        # Mirrors the source pipeline's non-null meal_logged rule.
        # It flags possible issues; it does not assert the correct
        # replacement value for each record.
        report["current_meal_logged_rule"] = (
            df["Dietary intake"].notna().astype(int).values
        )
        report["possible_meal_flag_issue"] = (
            report["current_meal_logged_rule"].eq(1)
            & report["meal_unavailable_text"]
        )

        # Conservative audit flag only. Not used to create new doses.
        report["insulin_needs_review"] = report["category"].isin([
            "mixed_or_premixed_review",
            "multiple_types_review",
            "unknown_review",
        ])

        reports.append(report)

    if not reports:
        raise RuntimeError(
            "No reports created. Check raw paths and column names."
        )

    detail = pd.concat(reports, ignore_index=True)
    detail.to_csv(DETAIL_FILE, index=False)

    summary_rows = []

    def add_summary(metric, subset, value):
        summary_rows.append({
            "metric": metric,
            "count": int(value),
            "denominator": int(len(subset)),
            "interpretation": (
                "Audit flag only; not a confirmed feature error."
            ),
        })

    nonempty_insulin = detail[
        detail["sc_insulin_text"].notna()
        & detail["sc_insulin_text"].str.strip().ne("")
    ]

    add_summary(
        "nonempty_sc_insulin_records",
        detail,
        len(nonempty_insulin),
    )

    for category, count in (
        nonempty_insulin["category"].value_counts().items()
    ):
        add_summary(
            f"sc_insulin_category:{category}",
            nonempty_insulin,
            count,
        )

    add_summary(
        "mixed_or_premixed_records",
        nonempty_insulin,
        nonempty_insulin["mixed_flag"].sum(),
    )
    add_summary(
        "multiple_dose_number_records",
        nonempty_insulin,
        nonempty_insulin["multiple_dose_numbers"].gt(1).sum(),
    )
    add_summary(
        "insulin_records_needing_review",
        nonempty_insulin,
        nonempty_insulin["insulin_needs_review"].sum(),
    )
    add_summary(
        "meal_unavailable_text_records",
        detail,
        detail["meal_unavailable_text"].sum(),
    )
    add_summary(
        "possible_meal_flag_issue_records",
        detail,
        detail["possible_meal_flag_issue"].sum(),
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n========== EVENT IMPACT AUDIT ==========")
    print("Raw directory:", RAW_DIR)
    print("Schema:", SCHEMA_FILE)
    print("Records inspected:", len(detail))
    print("\nInsulin screening categories:")
    print(
        nonempty_insulin["category"]
        .value_counts(dropna=False)
        .to_string()
    )
    print("\nMeal flags:")
    print(
        "Unavailable meal text:",
        int(detail["meal_unavailable_text"].sum()),
    )
    print(
        "Potential meal flag issues:",
        int(detail["possible_meal_flag_issue"].sum()),
    )

    print("\nSaved record-level report:", DETAIL_FILE)
    print("Saved summary report:", SUMMARY_FILE)
    print("\nNo source data, features, models, or calibrations modified.")

    print("\nScaling parameters available for later comparison:")
    for feature in [
        "fast_insulin_last_4h",
        "meal_grams_last_2h",
        "mins_since_meal",
        "mins_since_fast_insulin",
        "iob_proxy",
        "cob_proxy",
    ]:
        print(f"  {feature}: {SCALING.get(feature, 'NOT FOUND')}")


if __name__ == "__main__":
    main()
