"""
GlucoTwin - Event Feature Timing Audit
--------------------------------------
Read-only source audit for the event-derived feature calculations used by the
15-minute CGM model. It prints relevant source excerpts and flags questions
that must be resolved before live inference.

No data, models, or source files are modified.
"""

from pathlib import Path
import re
import sys

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
PREP_FILE = PROJECT_ROOT / "files" / "prep_pipeline.py"

TERMS = [
    "fast_insulin", "csii_bolus", "sc_units", "iv_units",
    "meal_grams", "meal_logged", "mins_since_meal",
    "mins_since_fast_insulin", "iob_proxy", "cob_proxy",
    "csii_basal", "csii_suspend", "sc_long_acting",
    "EVENT0", "fillna(0", "ffill(", "ewm(", "rolling(",
    "interpolate", "sort_values", "groupby(\"seg\")",
]

def main():
    print("=" * 78)
    print("GLUCOTWIN - EVENT FEATURE TIMING AUDIT")
    print("=" * 78)
    print("Source:", PREP_FILE)

    if not PREP_FILE.is_file():
        raise FileNotFoundError(f"Preprocessing source not found:\n{PREP_FILE}")

    lines = PREP_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    for i, line in enumerate(lines):
        if any(term.lower() in line.lower() for term in TERMS):
            hits.append(i)

    ranges = []
    for hit in hits:
        start = max(0, hit - 4)
        end = min(len(lines), hit + 6)
        if ranges and start <= ranges[-1][1] + 2:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    print("\nRelevant source excerpts")
    print("-" * 78)
    total = 0
    for start, end in ranges:
        block = "\n".join(f"{i+1:04d}: {lines[i]}" for i in range(start, end))
        if total + len(block) > 26000:
            print("[Output truncated at 26,000 characters.]")
            break
        print("\n" + block)
        total += len(block)

    print("\n" + "=" * 78)
    print("MANUAL REVIEW QUESTIONS")
    print("=" * 78)
    questions = [
        "Are event timestamps available at or before each prediction timestamp?",
        "Are event quantities recorded at the event time, or entered/edited later?",
        "Does forward-filling basal rate represent what was known at that time?",
        "Can meal_logged be set without a valid meal timestamp or quantity?",
        "Do event rolling sums include the current timestamp, and is that intended?",
        "Are event recency features computed within the correct patient/segment?",
        "Are insulin and carbohydrate proxies initialized and reset at segment boundaries?",
        "Does the 96-row history filter address event timing, or only CGM interpolation?",
    ]
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")

    print("\nIMPORTANT")
    print("- This is a source-pattern audit, not a proof of causal correctness.")
    print("- Source code alone cannot establish whether original event timestamps")
    print("  reflect when the information became available to the model.")
    print("- Do not enable real-time prediction until these questions are resolved.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("AUDIT FAILED:", repr(exc))
        sys.exit(1)
