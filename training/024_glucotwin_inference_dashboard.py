"""
GlucoTwin - Inference Validation Dashboard
-----------------------------------------
A separate Streamlit app for validating the reusable inference engine against
previously eligible held-out rows. It does not modify the historical dashboard
and does not claim to be a live CGM integration.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


TRAINING_DIR = Path(__file__).resolve().parent
DATA_FILE = (
    TRAINING_DIR.parent
    / "DATASET"
    / "processed"
    / "cgm_t1dm_t2dm_model_ready.csv"
)
PREDICTIONS_FILE = (
    TRAINING_DIR / "results_causal" / "causal_test_predictions.csv"
)
ENGINE_FILE = TRAINING_DIR / "022_glucotwin_inference_engine.py"
HORIZONS = [30, 60, 90, 120]


@st.cache_resource
def load_engine():
    if not ENGINE_FILE.is_file():
        raise FileNotFoundError(f"Inference engine not found: {ENGINE_FILE}")

    spec = importlib.util.spec_from_file_location(
        "glucotwin_inference_engine", ENGINE_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load engine from {ENGINE_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GlucoTwinInferenceEngine()


@st.cache_data
def load_data():
    for path in (DATA_FILE, PREDICTIONS_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_FILE)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)
    df["split"] = df["split"].astype(str).str.lower()

    saved = pd.read_csv(PREDICTIONS_FILE)
    saved["split"] = saved["split"].astype(str).str.lower()
    saved["_row_index"] = pd.to_numeric(saved["_row_index"], errors="coerce")
    saved["horizon_minutes"] = pd.to_numeric(
        saved["horizon_minutes"], errors="coerce"
    )
    saved = saved.dropna(
        subset=["_row_index", "horizon_minutes", "prediction_mgdl"]
    ).copy()
    saved["_row_index"] = saved["_row_index"].astype(int)
    saved["horizon_minutes"] = saved["horizon_minutes"].astype(int)
    return df, saved


st.set_page_config(
    page_title="GlucoTwin Inference Validation",
    page_icon="🩺",
    layout="wide",
)

st.title("GlucoTwin — Inference Validation")
st.caption(
    "Research prototype · Retrospective held-out row testing · Not a live CGM feed"
)

st.warning(
    "This page re-runs predictions on previously eligible held-out dataset rows. "
    "It is not clinically validated and must not be used to make treatment decisions."
)

try:
    engine = load_engine()
    df, saved = load_data()
except Exception as exc:
    st.error(f"Could not initialize the inference view: {exc}")
    st.stop()

health = engine.health_check()

with st.expander("Engine status", expanded=False):
    st.json(health)

if not {"_row_index", "split", "horizon_minutes", "prediction_mgdl"}.issubset(
    saved.columns
):
    st.error("The saved predictions file is missing required columns.")
    st.stop()

test_saved = saved[saved["split"] == "test"].copy()
index_sets = [
    set(test_saved.loc[
        test_saved["horizon_minutes"] == horizon, "_row_index"
    ].tolist())
    for horizon in HORIZONS
]

if any(not indexes for indexes in index_sets):
    st.error("Saved test predictions are missing one or more horizons.")
    st.stop()

common_indices = sorted(set.intersection(*index_sets))
common_indices = [
    idx for idx in common_indices
    if 0 <= idx < len(df) and str(df.iloc[idx]["split"]).lower() == "test"
]

if not common_indices:
    st.error(
        "No common eligible held-out row was found across all four horizons. "
        "Check the row ordering used by the training pipeline."
    )
    st.stop()

st.subheader("Select an eligible held-out row")
selected_index = st.selectbox(
    "Dataset row index",
    options=common_indices,
    index=0,
    format_func=lambda idx: (
        f"Row {idx} · {df.iloc[idx]['patient_uid']} · "
        f"{df.iloc[idx]['dt']}"
    ),
)

source_row = df.iloc[int(selected_index)]

left, middle, right = st.columns(3)
left.metric("Split", str(source_row["split"]).upper())
middle.metric("Patient", str(source_row["patient_uid"]))
right.metric("Segment", str(source_row["segment_id"]))

st.write("**Timestamp:**", str(source_row["dt"]))
st.caption(
    "The selected row was retained by the prior causal training pipeline. "
    "This app inherits that eligibility decision; it does not independently "
    "audit all feature timestamps or interpolation history."
)

if st.button("Run inference and compare", type="primary"):
    try:
        feature_values = {
            name: source_row[name] for name in engine.features
        }
        result = engine.predict(
            feature_values,
            history_verified=True,
        )

        comparisons = []
        all_passed = True

        for item in result["predictions"]:
            horizon = int(item["horizon_minutes"])
            matches = test_saved[
                (test_saved["horizon_minutes"] == horizon)
                & (test_saved["_row_index"] == int(selected_index))
            ]

            if matches.empty:
                all_passed = False
                comparisons.append({
                    "Horizon": f"+{horizon} min",
                    "Engine prediction (mg/dL)": item["prediction_mgdl"],
                    "Saved prediction (mg/dL)": np.nan,
                    "Absolute difference (mg/dL)": np.nan,
                    "Reproduction": "MISSING",
                })
                continue

            expected = float(matches.iloc[0]["prediction_mgdl"])
            actual = float(item["prediction_mgdl"])
            difference = abs(actual - expected)
            passed = bool(np.isclose(actual, expected, rtol=1e-4, atol=0.05))
            all_passed = all_passed and passed

            comparisons.append({
                "Horizon": f"+{horizon} min",
                "Engine prediction (mg/dL)": round(actual, 4),
                "Saved prediction (mg/dL)": round(expected, 4),
                "Absolute difference (mg/dL)": round(difference, 6),
                "Reproduction": "PASS" if passed else "CHECK",
            })

        if all_passed:
            st.success("All four predictions reproduce the saved outputs within tolerance.")
        else:
            st.error("At least one horizon needs review.")

        st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)

        st.subheader("Predictions and calibrated intervals")
        cards = st.columns(4)
        for card, item in zip(cards, result["predictions"]):
            with card:
                horizon = item["horizon_minutes"]
                st.metric(
                    f"+{horizon} minutes",
                    f"{item['prediction_mgdl']:.2f} mg/dL",
                )
                intervals = item["prediction_intervals"]
                st.write(
                    f"**80% interval**  \n"
                    f"{intervals['80%']['lower_mgdl']:.2f}–"
                    f"{intervals['80%']['upper_mgdl']:.2f} mg/dL"
                )
                st.write(
                    f"**90% interval**  \n"
                    f"{intervals['90%']['lower_mgdl']:.2f}–"
                    f"{intervals['90%']['upper_mgdl']:.2f} mg/dL"
                )

        st.caption(result["notice"])

    except Exception as exc:
        st.error(f"Inference failed: {exc}")

st.divider()
st.caption(
    "Next engineering step for genuine live inference: implement and validate "
    "a causal feature builder that creates the exact 33 features from timestamped "
    "historical inputs, checks the 96-step history eligibility rule, and rejects "
    "stale, incomplete, or out-of-order input data."
)