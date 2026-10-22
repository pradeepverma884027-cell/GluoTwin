
from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\pradeep\Desktop\gluotwinv2\dataset_from_beginning\DATASET"
)

# Inspect the preprocessing source to identify the original input files.
source = ROOT / "files" / "prep_pipeline.py"
print("Preprocessing source exists:", source.exists())

# Show the dataset folders and candidate raw files.
for folder in [ROOT, ROOT / "files", ROOT / "DATASET", ROOT / "DATASET" / "processed"]:
    print(f"\n--- {folder} ---")
    if folder.exists():
        for p in sorted(folder.iterdir()):
            if p.is_file():
                print(p.name, f"({p.stat().st_size:,} bytes)")
            elif p.is_dir():
                print(p.name + "/")

# Inspect the model-ready data for event timestamps and values.
csv_path = ROOT / "DATASET" / "processed" / "cgm_t1dm_t2dm_model_ready.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path, nrows=10)
    print("\nModel-ready columns:")
    print(df.columns.tolist())
    print("\nSample rows:")
    print(df.to_string(index=False))
else:
    print("\nModel-ready CSV not found at:", csv_path)
    print("Check the folder listing above for its actual location.")





