
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = (
    BASE_DIR.parent
    / "DATASET"
    / "raw"
    / "SHAHNGHAI-DATASET"
)
OUT_DIR = BASE_DIR / "results_causal" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "033_sc_insulin_focused_audit.csv"

SC_COLUMN = "Insulin dose - s.c."

LONG_ACTING = re.compile(
    r"glargine|detemir|degludec", re.I
)
MIXED = re.compile(
    r"70\s*/\s*30|30\s*/\s*70|50\s*/\s*50|"
    r"40\s*/\s*60|premix|biphasic|"
    r"novolin\s*30r|novolin\s*50r|"
    r"humulin\s*70\s*/?\s*30|"
    r"gansulin\s*40r|scilin\s*m30", re.I
)
RAPID = re.compile(r"aspart|lispro|glulisine", re.I)
REGULAR = re.compile(
    r"novolin\s*r|humulin\s*r|gansulin\s*r|"
    r"regular\s+insulin|insulin\s+r\b", re.I
)
DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*IU\b", re.I)


def classify(value):
    text = "" if pd.isna(value) else str(value).strip()

    if not text:
        return "missing", False, False, 0.0

    long_acting = bool(LONG_ACTING.search(text))
    mixed = bool(MIXED.search(text))
    rapid = bool(RAPID.search(text))
    regular = bool(REGULAR.search(text))

    type_count = sum([long_acting, rapid, regular])

    if mixed:
        category = "mixed_or_premixed_review"
    elif type_count > 1:
        category = "multiple_types_review"
    elif long_acting:
        category = "long_acting_name"
    elif rapid:
        category = "rapid_acting_name"
    elif regular:
        category = "regular_name"
    else:
        category = "unknown_review"

    # Mirrors the existing name-based exclusion rule.
    current_excludes = long_acting

    # Conservative screening for audit only; not a clinical decision.
    conservative_excludes = (
        long_acting or mixed or type_count > 1
        or category == "unknown_review"
    )

    dose_sum = sum(float(x) for x in DOSE.findall(text))

    return category, current_excludes, conservative_excludes, dose_sum


def main():
    reports = []

    for file_path in [
        RAW_DIR / "t1dm.csv",
        RAW_DIR / "t2dm.csv",
    ]:
        if not file_path.exists():
            print(f"FILE NOT FOUND: {file_path}")
            continue

        df = pd.read_csv(file_path, low_memory=False)

        if SC_COLUMN not in df.columns:
            print(f"Column missing in {file_path.name}: {SC_COLUMN}")
            continue

        report = pd.DataFrame({
            "source_file": file_path.name,
            "source_row_index": df.index,
            "raw_insulin_text": df[SC_COLUMN].astype("string"),
        })

        # Preserve useful identifiers if present in the raw dataset.
        for col in [
            "Patient ID", "Patient", "patient_uid",
            "Date", "Time", "Timestamp", "dt",
        ]:
            if col in df.columns:
                report[col] = df[col].values

        parsed = report["raw_insulin_text"].apply(classify)
        report[
            [
                "screening_category",
                "current_excludes_from_fast",
                "conservative_excludes_from_fast",
                "parsed_dose_sum_iu",
            ]
        ] = pd.DataFrame(parsed.tolist(), index=report.index)

        report["classification_differs"] = (
            report["current_excludes_from_fast"]
            != report["conservative_excludes_from_fast"]
        )

        report["needs_review"] = report["screening_category"].isin([
            "mixed_or_premixed_review",
            "multiple_types_review",
            "unknown_review",
        ])

        reports.append(report)

    if not reports:
        print("No audit created. Check the raw directory and column name.")
        return

    result = pd.concat(reports, ignore_index=True)
    result.to_csv(OUT_FILE, index=False)

    nonempty = result[
        result["raw_insulin_text"].notna()
        & result["raw_insulin_text"].str.strip().ne("")
    ]

    print("\n=== SUBCUTANEOUS INSULIN AUDIT ===")
    print("All rows:", len(result))
    print("Non-empty SC insulin records:", len(nonempty))
    print("\nCategories:")
    print(nonempty["screening_category"].value_counts().to_string())
    print(
        "\nClassification differences:",
        int(nonempty["classification_differs"].sum()),
    )
    print(
        "Records requiring review:",
        int(nonempty["needs_review"].sum()),
    )

    print("\nReview examples:")
    cols = [
        "source_file", "source_row_index", "raw_insulin_text",
        "screening_category", "classification_differs",
        "parsed_dose_sum_iu",
    ]
    print(
        nonempty[nonempty["needs_review"]][cols]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved: {OUT_FILE}")
    print("Audit only: no data, features, or models were modified.")


if __name__ == "__main__":
    main()
