from pathlib import Path

# ============================================================
# GLUCOTWIN - INSPECT ORIGINAL FEATURE PIPELINE
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent

SOURCE_FILE = PROJECT_DIR / "files" / "prep_pipeline.py"

FEATURE_TERMS = [
    "cgm_lag_15m",
    "cgm_delta_15",
    "cgm_roll_mean_60m",
    "cgm_roll_std_60m",
    "cgm_roll_min_120m",
    "fast_insulin_last_4h",
    "meal_grams_last_2h",
    "mins_since_meal",
    "mins_since_fast_insulin",
    "cgm_mean_6h",
    "cgm_std_6h",
    "cgm_mean_24h",
    "cgm_tir_24h",
    "cgm_tbr_24h",
    "iob_proxy",
    "cob_proxy",
    "hour_sin",
    "hour_cos",
    "target_cgm_t30",
    "target_cgm_t60",
]

print("=" * 70)
print("GLUCOTWIN - ORIGINAL FEATURE PIPELINE INSPECTION")
print("=" * 70)
print("Source:", SOURCE_FILE)

if not SOURCE_FILE.exists():
    print("\nSource file not found at the expected location.")
    print("Please check that prep_pipeline.py is in the files folder.")
    raise SystemExit(1)

lines = SOURCE_FILE.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

matched_lines = set()

for index, line in enumerate(lines):
    if any(term in line for term in FEATURE_TERMS):
        # Include surrounding lines for context.
        start = max(0, index - 3)
        end = min(len(lines), index + 5)

        matched_lines.update(range(start, end))

if not matched_lines:
    print("\nNo matching feature definitions were found.")
    print("The pipeline may use different names or helper functions.")
    raise SystemExit(0)

print("\nRelevant source excerpts:")
print("-" * 70)

previous = -2

for index in sorted(matched_lines):
    if index > previous + 1:
        print("\n...")

    print(f"{index + 1:04d}: {lines[index]}")
    previous = index

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
print("No source code was modified.")

