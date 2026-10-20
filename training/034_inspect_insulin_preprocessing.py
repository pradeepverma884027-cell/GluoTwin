
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

candidates = [
    PROJECT_DIR / "files" / "prep_pipeline.py",
    PROJECT_DIR / "DATASET" / "files" / "prep_pipeline.py",
]

source_path = next(
    (path for path in candidates if path.exists()),
    None,
)

if source_path is None:
    matches = list(PROJECT_DIR.rglob("prep_pipeline.py"))
    source_path = matches[0] if matches else None

if source_path is None:
    print("ERROR: prep_pipeline.py was not found.")
    print("Expected it somewhere under:", PROJECT_DIR)
    raise SystemExit(1)

lines = source_path.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

keywords = [
    "sc_long_acting",
    "sc_units",
    "fast_insulin",
    "Insulin dose - s.c.",
    "mixed",
    "meal_logged",
]

print("Preprocessing source:", source_path)
print("Total source lines:", len(lines))

matched = set()

for i, line in enumerate(lines):
    if any(word.lower() in line.lower() for word in keywords):
        start = max(0, i - 4)
        end = min(len(lines), i + 6)
        matched.update(range(start, end))

print("\n========== RELEVANT SOURCE CODE ==========")

previous = -2

for i in sorted(matched):
    if i > previous + 1:
        print("\n--- ... ---")

    print(f"{i + 1:4}: {lines[i]}")
    previous = i

print("\nInspection complete. No files were modified.")
