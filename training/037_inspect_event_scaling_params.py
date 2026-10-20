
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

matches = list(
    PROJECT_DIR.rglob("schema_cgm_t1dm_t2dm.json")
)

if not matches:
    raise FileNotFoundError(
        f"Schema not found beneath {PROJECT_DIR}"
    )

SCHEMA = matches[0]
data = json.loads(SCHEMA.read_text(encoding="utf-8"))

print("Schema:", SCHEMA)
print("Top-level keys:", list(data.keys()))

params = data.get("scaled_cols_mean_std", {})

# Support alternate schema structures if necessary.
if not params:
    params = data.get("scaling_params", {})

print("\nInsulin/meal feature scaling parameters:")

for name in [
    "fast_insulin_last_4h",
    "meal_grams_last_2h",
    "mins_since_meal",
    "mins_since_fast_insulin",
    "iob_proxy",
    "cob_proxy",
]:
    print(f"{name}: {params.get(name, 'NOT FOUND')}")

print("\nAudit complete. No files modified.")
