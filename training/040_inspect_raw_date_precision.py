
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

def find_raw_file(name):
    matches = list(PROJECT_DIR.rglob(name))
    matches = [
        p for p in matches
        if "SHAHNGHAI-DATASET" in str(p).upper()
        and "RAW" in str(p).upper()
    ]
    if not matches:
        raise FileNotFoundError(f"Cannot find raw file: {name}")
    return matches[0]

for name in ["t1dm.csv", "t2dm.csv"]:
    path = find_raw_file(name)
    df = pd.read_csv(path, low_memory=False)

    print(f"\n========== {name} ==========")
    print("Rows:", len(df))
    print("Date dtype:", df["Date"].dtype)
    print("First 10 raw Date values:")
    print(df["Date"].head(10).to_string(index=False))

    parsed = pd.to_datetime(df["Date"], errors="coerce")
    valid = parsed.dropna()

    print("Unparseable dates:", int(parsed.isna().sum()))

    if len(valid):
        print("Parsed minimum:", valid.min())
        print("Parsed maximum:", valid.max())
        print("Unique parsed timestamps:", int(valid.nunique()))
        print(
            "Values containing a time component:",
            int(
                df["Date"].astype("string")
                .str.contains(r"\d{1,2}:\d{2}", na=False)
                .sum()
            ),
        )

        print("Most common parsed time differences:")
        print(
            valid.sort_values()
            .diff()
            .dropna()
            .value_counts()
            .head(10)
            .to_string()
        )

print("\nRead-only inspection complete.")
