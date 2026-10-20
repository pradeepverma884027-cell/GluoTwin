
from pathlib import Path
import joblib

from common import MODELS

path = MODELS / "linear__cgm_t1dm_t2dm__target_cgm_t30.joblib"
obj = joblib.load(path)

print("Loaded type:", type(obj))

if isinstance(obj, dict):
    print("Dictionary keys:", list(obj.keys()))

    for key, value in obj.items():
        print(f"\nKey: {key!r}")
        print("Type:", type(value))

        if hasattr(value, "predict"):
            print("This item supports predict().")
else:
    print("Has predict:", hasattr(obj, "predict"))
