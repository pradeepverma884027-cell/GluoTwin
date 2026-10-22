
from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\pradeep\Desktop\gluotwinv2\dataset_from_beginning\DATASET"
)
RAW = ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"

for name in ("t1dm.csv", "t2dm.csv"):
    path = RAW / name
    print("\n" + "=" * 70)
    print("FILE:", path)
    if not path.exists():
        print("Not found; check the exact folder spelling.")
        continue

    df = pd.read_csv(path, low_memory=False)
    print("Rows:", len(df))

    diet_cols = [
        c for c in ("Dietary intake", "饮食") if c in df.columns
    ]
    event_cols = [
        c for c in (
            "Insulin dose - s.c.",
            "Non-insulin hypoglycemic agents",
            "CSII - bolus insulin (Novolin R, IU)",
            "CSII - basal insulin (Novolin R, IU / H)",
            "Insulin dose - i.v.",
        ) if c in df.columns
    ]

    if diet_cols:
        diet = df[diet_cols].fillna("").astype(str)
        diet_text = diet.agg(" ".join, axis=1).str.strip()
        nonempty = diet_text.ne("")
        unknown = diet_text.str.contains(
            r"data not available|未记录", case=False, regex=True
        )

        print("Diet text nonempty:", int(nonempty.sum()))
        print("Diet text marked unavailable:", int(unknown.sum()))
        print("Diet text nonempty but unavailable:", int(
            (nonempty & unknown).sum()
        ))

    if event_cols:
        events = df[event_cols].notna().any(axis=1)
        print("Rows with insulin/medication fields populated:",
              int(events.sum()))

    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        print("Unparseable timestamps:", int(dt.isna().sum()))
        print("Duplicate timestamps within patient/file:",
              int(df.assign(_dt=dt).duplicated(
                  ["patient_id", "file_no", "_dt"]
              ).sum()))

    basal = "CSII - basal insulin (Novolin R, IU / H)"
    if basal in df.columns:
        print("Basal field non-null:", int(df[basal].notna().sum()))
        print("Basal values (top 15):")
        print(df[basal].value_counts(dropna=False).head(15).to_string())

    print("\nFirst 10 rows with diet or insulin data:")
    cols = [c for c in ["patient_id", "file_no", "Date"]
            + diet_cols + event_cols if c in df.columns]
    if diet_cols or event_cols:
        populated = df[diet_cols + event_cols].notna().any(axis=1)
        print(df.loc[populated, cols].head(10).to_string(index=False))
