
from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\pradeep\Desktop\gluotwinv2\dataset_from_beginning\DATASET"
)
RAW = ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"

def classify(text):
    s = str(text).lower().strip()

    if not s or s == "nan":
        return "missing"

    # Mixed formulations must be separated before individual names.
    if any(x in s for x in [
        "30r", "50r", "40r", "70/30", "m30"
    ]):
        return "mixed_or_premixed_review"

    long_acting = any(x in s for x in [
        "degludec", "detemir", "glargine", "glarigine"
    ])
    rapid = any(x in s for x in [
        "aspart", "glulisine", "lispro"
    ])
    regular = any(x in s for x in [
        "novolin r", "humulin r", "gansulin r"
    ])

    found = sum([long_acting, rapid, regular])

    if found > 1:
        return "multiple_types_review"
    if long_acting:
        return "long_acting_name"
    if rapid:
        return "rapid_acting_name"
    if regular:
        return "regular_name"
    return "unclassified_review"

for filename in ["t1dm.csv", "t2dm.csv"]:
    path = RAW / filename
    df = pd.read_csv(
        path,
        usecols=["Insulin dose - s.c."],
        low_memory=False
    )

    col = "Insulin dose - s.c."
    values = df[col].dropna().astype(str).str.strip()
    values = values[values.ne("")]

    result = pd.DataFrame({"raw_entry": values})
    result["audit_group"] = result["raw_entry"].map(classify)

    print("\n" + "=" * 70)
    print(filename)
    print("Nonempty records:", len(result))
    print("\nRecord counts by review group:")
    print(result["audit_group"].value_counts().to_string())

    print("\nDistinct entries requiring review:")
    review = result[
        result["audit_group"].isin([
            "mixed_or_premixed_review",
            "multiple_types_review",
            "unclassified_review"
        ])
    ]
    print(
        review["raw_entry"]
        .value_counts()
        .head(50)
        .to_string()
    )
