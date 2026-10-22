
from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\pradeep\Desktop\gluotwinv2\dataset_from_beginning\DATASET"
)
RAW = ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"

for filename in ("t1dm.csv", "t2dm.csv"):
    path = RAW / filename
    print("\n" + "=" * 70)
    print(filename)

    df = pd.read_csv(
        path,
        usecols=["Insulin dose - s.c."],
        low_memory=False
    )

    values = (
        df["Insulin dose - s.c."]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values.ne("")]

    print("Rows with subcutaneous insulin text:", len(values))
    print("\nMost common exact entries:")
    print(values.value_counts().head(40).to_string())

    print("\nDistinct insulin names (up to 100):")
    names = sorted(values.unique(), key=str.casefold)
    for name in names[:100]:
        print("-", name)
