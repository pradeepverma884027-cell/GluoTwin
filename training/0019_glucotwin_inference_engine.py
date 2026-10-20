from pathlib import Path
import joblib

TRAINING_DIR = Path(__file__).resolve().parent
MODELS_DIR = TRAINING_DIR / "results_causal" / "models"

print("=" * 70)
print("GLUCOTWIN - LIVE INFERENCE PREPARATION")
print("=" * 70)

reference_features = None

for horizon in [30, 60, 90, 120]:
    path = MODELS_DIR / f"lightgbm_causal_t{horizon}.joblib"

    print(f"\n{'-' * 60}")
    print(f"HORIZON: +{horizon} MINUTES")
    print(f"{'-' * 60}")

    if not path.exists():
        print(f"MISSING MODEL: {path}")
        continue

    artifact = joblib.load(path)

    required_keys = [
        "model",
        "imputer",
        "features",
        "horizon_minutes",
        "cgm_mean",
        "cgm_std",
        "schema",
        "history_filter_steps",
        "target_scale",
    ]

    missing = [key for key in required_keys if key not in artifact]

    if missing:
        print("Missing artifact fields:", missing)
        continue

    features = artifact["features"]

    print("Task:", artifact.get("task"))
    print("Horizon:", artifact["horizon_minutes"])
    print("Number of features:", len(features))
    print("History filter steps:", artifact["history_filter_steps"])
    print("Target scale:", artifact["target_scale"])
    print("CGM mean:", artifact["cgm_mean"])
    print("CGM standard deviation:", artifact["cgm_std"])

    print("\nFull ordered feature list:")

    for index, feature in enumerate(features, start=1):
        print(f"{index:02d}. {feature}")

    if reference_features is None:
        reference_features = list(features)
    elif list(features) != reference_features:
        print("\nWARNING: Feature order differs from the first model.")
    else:
        print("\nFeature order matches the other models.")

    model = artifact["model"]
    imputer = artifact["imputer"]

    print("\nModel feature count:", model.n_features_in_)
    print("Imputer feature count:", imputer.n_features_in_)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
