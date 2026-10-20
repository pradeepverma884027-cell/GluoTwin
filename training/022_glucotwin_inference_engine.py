"""
GlucoTwin reusable inference engine
-----------------------------------
Loads existing causal LightGBM artifacts and returns predictions for
+30, +60, +90, and +120 minutes with calibrated conformal intervals.

IMPORTANT:
- This engine expects one row containing the exact feature values generated
  by GlucoTwin's causal feature pipeline.
- The caller must verify that the input history is valid and contains no
  prohibited interpolated readings under the project's 96-step history rule.
- This is a research / decision-support prototype, not a clinically validated
  medical device. Do not use its output to make treatment decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TRAINING_DIR / "results_causal"
MODELS_DIR = RESULTS_DIR / "models"

HORIZONS = (30, 60, 90, 120)

# Split-conformal radii from the project's validated calibration results.
# Values are in mg/dL. These are half-widths, not full interval widths.
CONFORMAL_RADII_MGDL = {
    30: {80: 15.39, 90: 22.11},
    60: {80: 28.10, 90: 40.92},
    90: {80: 37.61, 90: 54.62},
    120: {80: 43.82, 90: 63.89},
}


class GlucoTwinInferenceEngine:
    """Load once, then use predict() for each validated feature row."""

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.artifacts: dict[int, dict[str, Any]] = {}
        self.features: list[str] | None = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        for horizon in HORIZONS:
            path = self.models_dir / f"lightgbm_causal_t{horizon}.joblib"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing model artifact for +{horizon} minutes:\n{path}"
                )

            artifact = joblib.load(path)
            required_keys = {
                "model", "imputer", "features", "horizon_minutes",
                "task", "history_filter_steps", "target_scale",
            }
            missing = required_keys - set(artifact)
            if missing:
                raise ValueError(
                    f"Artifact {path.name} is missing keys: {sorted(missing)}"
                )

            artifact_features = list(artifact["features"])
            if self.features is None:
                self.features = artifact_features
            elif artifact_features != self.features:
                raise ValueError(
                    f"Feature order differs in artifact {path.name}. "
                    "All horizon models must use the same ordered feature list."
                )

            if int(artifact["horizon_minutes"]) != horizon:
                raise ValueError(
                    f"Artifact horizon mismatch in {path.name}: "
                    f"{artifact['horizon_minutes']}"
                )

            if str(artifact["target_scale"]).lower() != "mg/dl":
                raise ValueError(
                    f"Unexpected target_scale in {path.name}: "
                    f"{artifact['target_scale']!r}. Do not guess a conversion."
                )

            self.artifacts[horizon] = artifact

        if self.features is None:
            raise RuntimeError("No model features were loaded.")

    def health_check(self) -> dict[str, Any]:
        """Check that all artifacts and interval configurations are available."""
        return {
            "status": "ready",
            "horizons_minutes": list(HORIZONS),
            "feature_count": len(self.features or []),
            "history_filter_steps": {
                h: int(self.artifacts[h]["history_filter_steps"])
                for h in HORIZONS
            },
            "target_scale": "mg/dL",
            "interval_levels_percent": [80, 90],
            "model_files": [
                f"lightgbm_causal_t{h}.joblib" for h in HORIZONS
            ],
        }

    def predict(
        self,
        features: Mapping[str, Any],
        *,
        history_verified: bool = False,
    ) -> dict[str, Any]:
        """
        Predict all four horizons for one feature row.

        Parameters
        ----------
        features:
            Mapping of feature name -> numeric feature value. Must contain
            every feature in the model artifact. NaN is allowed only where
            the fitted imputer can handle it.
        history_verified:
            Must be True only after the caller has verified that the feature
            row was produced causally and passed the project's 96-step
            history/interpolation eligibility rule.
        """
        if not history_verified:
            raise ValueError(
                "Prediction blocked: set history_verified=True only after "
                "validating causal feature timing and the 96-step history rule."
            )

        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping of feature names to values.")

        assert self.features is not None
        missing = [name for name in self.features if name not in features]
        if missing:
            raise ValueError(
                "Required model features are missing: " + ", ".join(missing)
            )

        # Ignore extra metadata keys; select features in the exact artifact order.
        row_values = []
        for name in self.features:
            value = features[name]
            try:
                numeric_value = float(value) if value is not None else np.nan
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature {name!r} must be numeric or missing; got {value!r}"
                ) from exc

            if np.isinf(numeric_value):
                raise ValueError(f"Feature {name!r} cannot be +inf or -inf.")
            row_values.append(numeric_value)

        X = pd.DataFrame([row_values], columns=self.features)
        results: dict[str, Any] = {
            "unit": "mg/dL",
            "history_verified_by_caller": True,
            "predictions": [],
            "notice": (
                "Research prototype only; intervals reflect the existing "
                "calibration evaluation and may not guarantee coverage for "
                "new patients or distribution shifts."
            ),
        }

        for horizon in HORIZONS:
            artifact = self.artifacts[horizon]
            X_imputed = artifact["imputer"].transform(X)

            # Reproduction testing confirmed model outputs already match the
            # saved mg/dL predictions. Do not apply another inverse transform.
            prediction = float(np.asarray(artifact["model"].predict(X_imputed))[0])

            if not np.isfinite(prediction):
                raise RuntimeError(
                    f"Model returned a non-finite prediction at +{horizon} minutes."
                )

            interval_result: dict[str, dict[str, float]] = {}
            for level in (80, 90):
                radius = CONFORMAL_RADII_MGDL[horizon][level]
                interval_result[f"{level}%"] = {
                    "lower_mgdl": round(prediction - radius, 2),
                    "upper_mgdl": round(prediction + radius, 2),
                    "radius_mgdl": radius,
                    "width_mgdl": round(2 * radius, 2),
                }

            results["predictions"].append({
                "horizon_minutes": horizon,
                "prediction_mgdl": round(prediction, 2),
                "prediction_intervals": interval_result,
            })

        return results


def main() -> None:
    """Run an artifact-only check; does not invent a patient feature row."""
    engine = GlucoTwinInferenceEngine()
    report = engine.health_check()

    print("=" * 72)
    print("GLUCOTWIN - INFERENCE ENGINE HEALTH CHECK")
    print("=" * 72)
    print("Status:", report["status"])
    print("Horizons:", report["horizons_minutes"])
    print("Feature count:", report["feature_count"])
    print("Target scale:", report["target_scale"])
    print("History filter steps:", report["history_filter_steps"])
    print("Interval levels:", report["interval_levels_percent"], "%")
    print("\nLoaded model artifacts:")
    for name in report["model_files"]:
        print("  PASS:", name)
    print("\nHEALTH CHECK: PASS")
    print(
        "\nNo prediction was generated because no verified real feature row "
        "was supplied."
    )


if __name__ == "__main__":
    main()
