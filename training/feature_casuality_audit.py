
"""
GlucoTwin - Feature Causality Audit
Scans Python source files for potentially unsafe time-series preprocessing.

This is a static screening tool. Review each flagged line in context;
a match does not automatically mean data leakage.
"""

from pathlib import Path
import csv
import re

# Automatically find the project root:
# Expected location: <project_root>/training/013_feature_causality_audit.py
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = TRAINING_DIR.parent

OUTPUT_DIR = TRAINING_DIR / "results_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT = OUTPUT_DIR / "feature_causality_audit.txt"
REPORT_CSV = OUTPUT_DIR / "feature_causality_audit.csv"

# Skip generated files and environments.
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "site-packages", "node_modules", "results_corrected"
}

# Each pattern is a review trigger, not a definitive error.
RULES = [
    (
        "HIGH_REVIEW",
        "Backward fill may use future observations",
        re.compile(r"\.bfill\s*\(|\bbackfill\s*\("),
    ),
    (
        "HIGH_REVIEW",
        "Centered rolling window may use future observations",
        re.compile(r"\.rolling\s*\([^)]*center\s*=\s*True", re.I),
    ),
    (
        "HIGH_REVIEW",
        "Interpolation may use future observations",
        re.compile(r"\.interpolate\s*\("),
    ),
    (
        "REVIEW",
        "Shift operation: verify direction and grouping",
        re.compile(r"\.shift\s*\("),
    ),
    (
        "REVIEW",
        "Rolling window: verify window alignment and data availability",
        re.compile(r"\.rolling\s*\("),
    ),
    (
        "REVIEW",
        "Resampling: verify aggregation and timestamp alignment",
        re.compile(r"\.resample\s*\("),
    ),
    (
        "REVIEW",
        "Forward fill: verify values are available at prediction time",
        re.compile(r"\.ffill\s*\(|\bpad\s*\("),
    ),
    (
        "REVIEW",
        "Target column reference: verify targets are not model features",
        re.compile(
            r"\btarget_cgm_t(?:30|60|90|120)\b"
            r"|\btarget_hypo_60\b"
        ),
    ),
    (
        "REVIEW",
        "Potential train/validation/test preprocessing leakage",
        re.compile(
            r"\.fit_transform\s*\(|"
            r"\.fit\s*\([^)]*(?:X|data|df|features)",
            re.I,
        ),
    ),
]

def should_skip(path):
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return True

    return any(part in SKIP_DIRS for part in relative.parts)

def scan_python_files():
    findings = []
    scanned_files = 0
    unreadable_files = []

    for path in PROJECT_ROOT.rglob("*.py"):
        if should_skip(path):
            continue

        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            unreadable_files.append(f"{path}: {exc}")
            continue

        scanned_files += 1

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Ignore empty lines and full-line comments.
            if not stripped or stripped.startswith("#"):
                continue

            for severity, reason, pattern in RULES:
                if pattern.search(line):
                    findings.append({
                        "severity": severity,
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "line": line_number,
                        "reason": reason,
                        "source": stripped,
                    })

    return findings, scanned_files, unreadable_files

def main():
    print("=" * 65)
    print("GLUCOTWIN FEATURE CAUSALITY AUDIT")
    print("=" * 65)
    print(f"Project root: {PROJECT_ROOT}")
    print("Scanning Python files...")

    findings, scanned_files, unreadable_files = scan_python_files()

    # Save CSV report.
    columns = ["severity", "file", "line", "reason", "source"]

    with REPORT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(findings)

    # Save readable text report.
    with REPORT_TXT.open("w", encoding="utf-8") as file:
        file.write("GLUCOTWIN FEATURE CAUSALITY AUDIT\n")
        file.write(f"Project root: {PROJECT_ROOT}\n")
        file.write(f"Python files scanned: {scanned_files}\n")
        file.write(f"Potential review findings: {len(findings)}\n\n")

        if not findings:
            file.write(
                "No configured patterns matched. "
                "This does NOT prove the pipeline is leakage-free.\n"
            )

        for item in findings:
            entry = (
                f"[{item['severity']}] "
                f"{item['file']}:{item['line']}\n"
                f"  Reason: {item['reason']}\n"
                f"  Code:   {item['source']}\n\n"
            )
            file.write(entry)

        if unreadable_files:
            file.write("\nFILES THAT COULD NOT BE READ\n")
            file.write("\n".join(unreadable_files))

    print(f"\nPython files scanned: {scanned_files}")
    print(f"Review findings: {len(findings)}")
    print(f"\nText report:\n{REPORT_TXT}")
    print(f"\nCSV report:\n{REPORT_CSV}")

    if findings:
        print("\nFirst 15 findings:")
        for item in findings[:15]:
            print(
                f"\n[{item['severity']}] "
                f"{item['file']}:{item['line']}"
            )
            print(f"  {item['reason']}")
            print(f"  {item['source'][:180]}")
    else:
        print(
            "\nNo configured patterns matched. "
            "Manual review is still required."
        )

    print("\nAudit complete. No model files were modified.")

if __name__ == "__main__":
    main()
