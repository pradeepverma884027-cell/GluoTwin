
from pathlib import Path
import pandas as pd
import re

# ============================================================
# GlucoTwin 039: Raw-to-clean event trace
# Read-only audit. No training or model changes.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

OUT_DIR = BASE_DIR / "results_causal" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = OUT_DIR / "039_raw_to_clean_trace.csv"
SUMMARY_FILE = OUT_DIR / "039_raw_to_clean_summary.csv"

MIXED_PATTERN = re.compile(
    r"70\s*/\s*30|30\s*/\s*70|50\s*/\s*50|"
    r"40\s*/\s*60|premix|biphasic|"
    r"novolin\s*30r|novolin\s*50r|"
    r"humulin\s*70\s*/?\s*30|"
    r"gansulin\s*40r|scilin\s*m30",
    re.IGNORECASE,
)

UNAVAILABLE_PATTERN = re.compile(
    r"data\s+not\s+available|not\s+available",
    re.IGNORECASE,
)


def find_all(filename):
    """Find files beneath the project directory."""
    return list(PROJECT_DIR.rglob(filename))


def find_first_existing(filenames):
    """Return the first matching existing file."""
    for filename in filenames:
        matches = find_all(filename)
        if matches:
            return matches[0]
    return None


def find_column(df, candidates):
    """Find the first matching column name."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def find_raw_files():
    """Locate the two original raw CSV files."""
    t1_matches = find_all("t1dm.csv")
    t2_matches = find_all("t2dm.csv")

    if not t1_matches or not t2_matches:
        raise FileNotFoundError(
            "Could not locate both t1dm.csv and t2dm.csv "
            f"under {PROJECT_DIR}"
        )

    # Prefer files inside the known raw-data directory.
    def preferred(matches):
        for path in matches:
            if (
                "SHAHNGHAI-DATASET" in str(path).upper()
                and "RAW" in str(path).upper()
            ):
                return path
        return matches[0]

    return preferred(t1_matches), preferred(t2_matches)


def find_preprocessing_source():
    matches = find_all("prep_pipeline.py")
    if not matches:
        return None

    for path in matches:
        if path.name == "prep_pipeline.py":
            return path

    return matches[0]


def inspect_timestamp_logic(source_path):
    """Print source lines that define patient IDs and timestamps."""
    if source_path is None:
        print("\nPreprocessing source not found.")
        return

    lines = source_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    keywords = [
        "patient_uid",
        "file_no",
        "source_file",
        '["Date"]',
        "to_datetime",
        "pd.to_datetime",
        "dt",
        "combine",
        "groupby",
        "read_csv",
        "read_excel",
    ]

    relevant = set()

    for i, line in enumerate(lines):
        if any(word.lower() in line.lower() for word in keywords):
            start = max(0, i - 2)
            end = min(len(lines), i + 4)
            relevant.update(range(start, end))

    print("\n========== TIMESTAMP/PATIENT SOURCE LOGIC ==========")
    print("Source:", source_path)

    previous = -2
    for i in sorted(relevant):
        if i > previous + 1:
            print("\n--- ... ---")
        print(f"{i + 1:4}: {lines[i]}")
        previous = i


def flag_raw_records(raw, source_name):
    """Create record-level audit flags from raw source text."""
    patient_col = find_column(
        raw,
        [
            "patient_id",
            "Patient ID",
            "Patient",
            "patient_uid",
            "Patient_ID",
        ],
    )

    date_col = find_column(raw, ["Date", "date"])
    time_col = find_column(raw, ["Time", "time"])
    timestamp_col = find_column(
        raw,
        ["Timestamp", "timestamp", "DateTime", "dt"],
    )

    sc_col = find_column(
        raw,
        ["Insulin dose - s.c.", "sc_insulin_text"],
    )
    meal_col = find_column(
        raw,
        ["Dietary intake", "diet_text"],
    )

    result = pd.DataFrame({
        "source_file": source_name,
        "source_row_index": raw.index,
    })

    if patient_col:
        result["raw_patient_id"] = (
            raw[patient_col].astype("string").values
        )
    else:
        result["raw_patient_id"] = pd.NA

    # Construct timestamps only when the raw source supports them.
    if timestamp_col:
        result["raw_timestamp"] = pd.to_datetime(
            raw[timestamp_col], errors="coerce"
        ).values
        result["timestamp_basis"] = f"column:{timestamp_col}"

    elif date_col and time_col:
        combined = (
            raw[date_col].astype("string").str.strip()
            + " "
            + raw[time_col].astype("string").str.strip()
        )
        result["raw_timestamp"] = pd.to_datetime(
            combined, errors="coerce"
        ).values
        result["timestamp_basis"] = (
            f"combined:{date_col}+{time_col}"
        )

    elif date_col:
        # Date-only values are NOT treated as exact CGM timestamps.
        result["raw_date_only"] = pd.to_datetime(
            raw[date_col], errors="coerce"
        ).dt.date.astype("string").values
        result["raw_timestamp"] = pd.NaT
        result["timestamp_basis"] = "date_only_no_exact_time"

    else:
        result["raw_timestamp"] = pd.NaT
        result["timestamp_basis"] = "no_timestamp_column"

    if sc_col:
        insulin_text = raw[sc_col].astype("string")
        result["raw_sc_insulin_text"] = insulin_text.values
        result["flag_mixed_insulin"] = insulin_text.str.contains(
            MIXED_PATTERN, na=False
        ).values
        result["flag_multiple_dose_numbers"] = insulin_text.map(
            lambda value: (
                len(re.findall(
                    r"\d+(?:\.\d+)?\s*IU\b",
                    str(value),
                    re.IGNORECASE,
                )) > 1
                if pd.notna(value) else False
            )
        ).values
    else:
        result["raw_sc_insulin_text"] = pd.NA
        result["flag_mixed_insulin"] = False
        result["flag_multiple_dose_numbers"] = False

    if meal_col:
        meal_text = raw[meal_col].astype("string")
        result["raw_meal_text"] = meal_text.values
        result["flag_unavailable_meal"] = meal_text.str.contains(
            UNAVAILABLE_PATTERN, na=False
        ).values
        result["flag_meal_nonempty"] = (
            meal_text.notna()
            & meal_text.str.strip().ne("")
        ).values
    else:
        result["raw_meal_text"] = pd.NA
        result["flag_unavailable_meal"] = False
        result["flag_meal_nonempty"] = False

    result["exact_timestamp_match"] = False
    result["match_status"] = "not_checked"

    return result


def main():
    print("========== GLUCOTWIN RAW-TO-CLEAN TRACE ==========")
    print("Project directory:", PROJECT_DIR)

    t1_file, t2_file = find_raw_files()

    clean_file = find_first_existing([
        "cgm_t1dm_t2dm_clean.csv",
        "cgm_t1dm_t2dm_model_ready.csv",
    ])

    preprocessing_file = find_preprocessing_source()

    print("\nRaw T1DM file:", t1_file)
    print("Raw T2DM file:", t2_file)
    print("Candidate cleaned/model-ready file:", clean_file)

    inspect_timestamp_logic(preprocessing_file)

    if clean_file is None:
        raise FileNotFoundError(
            "No cleaned or model-ready dataset was found."
        )

    clean = pd.read_csv(clean_file, low_memory=False)
    print("\n========== CLEAN DATASET ==========")
    print("File:", clean_file)
    print("Shape:", clean.shape)
    print("Columns:", clean.columns.tolist())

    clean_patient_col = find_column(
        clean,
        ["patient_uid", "patient_id", "Patient ID", "Patient"],
    )
    clean_time_col = find_column(
        clean,
        ["dt", "timestamp", "Timestamp", "DateTime"],
    )

    if clean_patient_col is None or clean_time_col is None:
        print(
            "\nExact matching cannot proceed because the clean dataset "
            "does not expose both a patient identifier and timestamp."
        )
        clean_keys = None
    else:
        clean_keys = pd.DataFrame({
            "clean_patient_key": (
                clean[clean_patient_col]
                .astype("string")
                .str.strip()
            ),
            "clean_timestamp": pd.to_datetime(
                clean[clean_time_col], errors="coerce"
            ).dt.round("min"),
        }).drop_duplicates()

    all_reports = []

    for raw_file in [t1_file, t2_file]:
        raw = pd.read_csv(raw_file, low_memory=False)

        report = flag_raw_records(raw, raw_file.name)

        if clean_keys is None:
            report["match_status"] = (
                "clean_patient_or_timestamp_column_missing"
            )

        elif report["raw_timestamp"].notna().any():
            raw_keys = pd.DataFrame({
                "raw_patient_key": (
                    report["raw_patient_id"]
                    .astype("string")
                    .str.strip()
                ),
                "raw_timestamp_key": pd.to_datetime(
                    report["raw_timestamp"], errors="coerce"
                ).dt.round("min"),
            })

            # Exact patient-ID matching is intentional; no inferred
            # mapping between raw and processed patient IDs is made.
            clean_pairs = set(
                zip(
                    clean_keys["clean_patient_key"],
                    clean_keys["clean_timestamp"],
                )
            )

            report["exact_timestamp_match"] = [
                (patient, timestamp) in clean_pairs
                if pd.notna(patient) and pd.notna(timestamp)
                else False
                for patient, timestamp in zip(
                    raw_keys["raw_patient_key"],
                    raw_keys["raw_timestamp_key"],
                )
            ]

            report["match_status"] = report[
                "exact_timestamp_match"
            ].map({
                True: "exact_patient_timestamp_match",
                False: "no_exact_patient_timestamp_match",
            })

        else:
            report["match_status"] = (
                "raw_source_has_no_complete_timestamp"
            )

        all_reports.append(report)

    detail = pd.concat(all_reports, ignore_index=True)
    detail.to_csv(DETAIL_FILE, index=False)

    summary_rows = []

    for filename, group in detail.groupby("source_file"):
        summary_rows.append({
            "source_file": filename,
            "raw_records": len(group),
            "complete_raw_timestamps": int(
                group["raw_timestamp"].notna().sum()
            ),
            "exact_patient_timestamp_matches": int(
                group["exact_timestamp_match"].sum()
            ),
            "mixed_insulin_records": int(
                group["flag_mixed_insulin"].sum()
            ),
            "multiple_dose_number_records": int(
                group["flag_multiple_dose_numbers"].sum()
            ),
            "unavailable_meal_records": int(
                group["flag_unavailable_meal"].sum()
            ),
            "meal_nonempty_records": int(
                group["flag_meal_nonempty"].sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n========== TRACE SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\nDetailed report:", DETAIL_FILE)
    print("Summary report:", SUMMARY_FILE)
    print("\nRead-only audit complete.")
    print("No raw files, processed data, models, or calibrations modified.")


if __name__ == "__main__":
    main()
