from pathlib import Path
import joblib

TRAINING_DIR = Path(__file__).resolve().parent
MODELS_DIR = TRAINING_DIR / "results_causal" / "models"

print("=" * 65)
print("GLUCOTWIN - MODEL ARTIFACT INSPECTION")
print("=" * 65)

for horizon in [30, 60, 90, 120]:
    path = MODELS_DIR / f"lightgbm_causal_t{horizon}.joblib"

    print(f"\n{'-' * 55}")
    print(f"HORIZON: +{horizon} MINUTES")
    print(f"{'-' * 55}")

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    try:
        artifact = joblib.load(path)

        print("Artifact type:", type(artifact).__name__)

        if isinstance(artifact, dict):
            print("Dictionary keys:", list(artifact.keys()))

            for key, value in artifact.items():
                print(f"\n{key}:")
                print("  Type:", type(value).__name__)

                if hasattr(value, "n_features_in_"):
                    print("  Number of features:", value.n_features_in_)

                if hasattr(value, "feature_name_"):
                    print("  Feature names:", value.feature_name_)

                if isinstance(value, (list, tuple)):
                    print("  Length:", len(value))
                    print("  First items:", list(value)[:10])

        else:
            if hasattr(artifact, "n_features_in_"):
                print("Number of features:", artifact.n_features_in_)

            if hasattr(artifact, "feature_name_"):
                print("Feature names:", artifact.feature_name_)

            if hasattr(artifact, "get_params"):
                print("Model parameters:")
                print(artifact.get_params())

    except Exception as exc:
        print("Inspection error:", repr(exc))

print("\nInspection complete.")
