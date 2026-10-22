
from pathlib import Path
import json

ROOT = Path(r"C:\Users\pradeep\Desktop\gluotwinv2")
BASE = ROOT / "dataset_from_beginning" / "DATASET"

SCHEMA = (
    BASE / "DATASET" / "processed" /
    "schema_cgm_t1dm_t2dm.json"
)

def main():
    print("=" * 65)
    print("GLUCOTWIN — CGM SCALING METADATA")
    print("=" * 65)

    if not SCHEMA.exists():
        print("Schema not found:", SCHEMA)
        return

    with open(SCHEMA, "r", encoding="utf-8") as f:
        schema = json.load(f)

    print("\nSchema keys:")
    for key in schema:
        print("-", key)

    print("\n--- SCALING METADATA ---")
    for key, value in schema.items():
        if any(word in key.lower() for word in [
            "scal", "mean", "std", "normal", "transform"
        ]):
            print(f"\n{key}:")
            print(json.dumps(value, indent=2)[:12000])

    print("\n--- CGM-RELATED METADATA ---")
    text = json.dumps(schema, indent=2)
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if "cgm" in line.lower() or "glucose" in line.lower():
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            print("\n".join(lines[start:end]))
            print("---")

    print("\nNo files modified.")

if __name__ == "__main__":
    main()
