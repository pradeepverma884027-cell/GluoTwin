
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

paths = [
    PROJECT_DIR / "DATASET" / "processed" / "cgm_t1dm_t2dm_model_ready.csv",
    PROJECT_DIR / "DATASET" / "DATASET" / "processed" / "cgm_t1dm_t2dm_model_ready.csv",
]

for path in paths:
    if path.exists():
        print("\nFOUND:", path)
        df = pd.read_csv(path, nrows=5)
        print("Columns:")
        print(df.columns.tolist())
        print("\nSample:")
        print(df.head(2).to_string(index=False))
        break
else:
    print("Processed model-ready CSV not found in expected paths.")
    print("Project directory:", PROJECT_DIR)
