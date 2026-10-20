
from pathlib import Path
import re
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = (
    BASE_DIR.parent
    / "DATASET"
    / "raw"
    / "SHAHNGHAI-DATASET"
)

OUTPUT_DIR = BASE_DIR / "results_causal" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "032_insulin_feature_impact.csv"

FILES = [
    RAW_DIR / "t1dm.csv",
    RAW_DIR / "t2dm.csv",
]

# --------------------------------------------------
# Conservative text screening
# This is an audit, NOT clinical classification.
# --------------------------------------------------
LONG_ACTING = re.compile(
    r"glargine|detemir|degludec", re.IGNORECASE
)

RAPID_ACTING = re.compile(
    r"aspart|lispro|glulisine", re.IGNORECASE
)

REGULAR_INSULIN = re.compile(
    r"novolin\s*r|humulin\s*r|gansulin\s*r|"
    r"regular\s+insulin|insulin\s+r\b",
    re.IGNORECASE,
)

MIXED_FORMULATION = re.compile(
    r"70\s*/\s*30|30\s*/\s*70|50\s*/\s*50|"
    r"40\s*/\s*60|premix|biphasic|"
    r"novolin\s*30r|novolin\s*50r|"
    r"humulin\s*70\s*/?\s*30|"
    r"gansulin\s*40r|scilin\s*m30",
    re.IGNORECASE,
)

# Match a numeric dose followed by IU.
DOSE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*IU\b",
    re.IGNORECASE,
)


def classify_text(value):
    text = "" if pd.isna(value) else str(value).strip()

    if not text:
        return "missing", False, 0, 0.0

    mixed = bool(MIXED_FORMULATION.search(text))

    has_long = bool(LONG_ACTING.search(text))
    has_rapid = bool(RAPID_ACTING.search(text))
    has_regular = bool(REGULAR_INSULIN.search(text))

    type_count = sum([has_long, has_rapid, has_regular])

    if mixed:
        category = "mixed_or_premixed_review"
    elif type_count > 1:
        category = "multiple_types_review"
    elif has_long:
        category = "long_acting_name"
    elif has_rapid:
        category = "rapid_acting_name"
    elif has_regular:
        category = "regular_name"
    else:
        category = "unknown_review"

    # Replicate the existing preprocessing's basic text rule:
    # if the text contains a long-acting keyword, it is excluded
    # from the fast-insulin sum. Otherwise, its dose may be included.
    current_excludes_from_fast = has_long

    # Conservative screening: do not automatically assign a fast
    # insulin dose if the product text is mixed, ambiguous, or unknown.
    conservative_excludes_from_fast = (
        has_long
        or mixed
        or type_count > 1
        or category == "unknown_review"
    )

    doses = [
        float(x)
        for x in DOSE_PATTERN.findall(text)
    ]

    parsed_dose_sum = sum(doses)

    needs_review = (
        mixed
        or type_count > 1
        or category == "unknown_review"
    )

    return (
        category,
        current_excludes_from_fast,
        int(conservative_excludes_from_fast),
        parsed_dose_sum,
    )


def main():
    all_reports = []

    for file_path in FILES:
        if not file_path.exists():
            print(f"WARNING: File not found: {file_path}")
            continue

        print(f"\nReading: {file_path.name}")
        df = pd.read_csv(file_path, low_memory=False)

        # Find likely insulin text columns without assuming
        # one exact source-dataset column name.
        insulin_columns = [
            col for col in df.columns
            if any(
                token in col.lower()
                for token in ["subcut", "insulin", "sc_units"]
            )
        ]

        if not insulin_columns:
            print("No likely insulin columns found. Skipping.")
            continue

        print("Insulin-related columns:", insulin_columns)

        for col in insulin_columns:
            report = pd.DataFrame({
                "source_file": file_path.name,
                "source_row_index": df.index,
                "source_column": col,
                "raw_insulin_text": df[col].astype("string"),
            })

            results = report["raw_insulin_text"].apply(classify_text)

            report[
                [
                    "screening_category",
                    "current_excludes_from_fast",
                    "conservative_excludes_from_fast",
                    "parsed_dose_sum_iu",
                ]
            ] = pd.DataFrame(
                results.tolist(),
                index=report.index,
            )

            report["classification_differs"] = (
                report["current_excludes_from_fast"]
                != report["conservative_excludes_from_fast"]
            )

            report["needs_manual_review"] = (
                report["screening_category"].isin([
                    "mixed_or_premixed_review",
                    "multiple_types_review",
                    "unknown_review",
                ])
            )

            all_reports.append(report)

    if not all_reports:
        print(
            "\nNo reports generated. Check RAW_DIR and the "
            "source column names."
        )
        return

    result = pd.concat(all_reports, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== AUDIT SUMMARY ==========")
    print("Rows audited:", len(result))
    print("\nScreening categories:")
    print(result["screening_category"].value_counts(dropna=False))

    print(
        "\nRecords where current and conservative screening differ:",
        int(result["classification_differs"].sum()),
    )

    print(
        "Records requiring manual review:",
        int(result["needs_manual_review"].sum()),
    )

    print("\nSaved report:")
    print(OUTPUT_FILE)
    print("\nNo source files, features, or models were modified.")


if __name__ == "__main__":
    main()
