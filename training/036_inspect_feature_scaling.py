
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SOURCE = PROJECT_DIR / "files" / "prep_pipeline.py"

if not SOURCE.exists():
    SOURCE = PROJECT_DIR / "DATASET" / "files" / "prep_pipeline.py"

if not SOURCE.exists():
    matches = list(PROJECT_DIR.rglob("prep_pipeline.py"))
    if not matches:
        raise FileNotFoundError("prep_pipeline.py not found")
    SOURCE = matches[0]

lines = SOURCE.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

keywords = [
    "StandardScaler",
    "RobustScaler",
    "MinMaxScaler",
    "cgm_mean",
    "cgm_std",
    "model_ready",
    "to_csv",
    "schema",
    "scaler",
    "fast_insulin_last_4h",
    "mins_since_fast_insulin",
]

print("Source:", SOURCE)

for i, line in enumerate(lines):
    if any(word.lower() in line.lower() for word in keywords):
        start = max(0, i - 3)
        end = min(len(lines), i + 5)
        print(f"\n--- lines {start + 1}-{end} ---")
        for j in range(start, end):
            print(f"{j + 1:4}: {lines[j]}")
