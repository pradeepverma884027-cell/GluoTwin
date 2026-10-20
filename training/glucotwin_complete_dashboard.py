"""
GlucoTwin Complete Research Dashboard
-------------------------------------
Reads existing CSV research outputs only. Does NOT train or overwrite models.
Run from PowerShell:
    streamlit run glucotwin_complete_dashboard.py

Place this file inside:
    DATASET\training\glucotwin_complete_dashboard.py

This dashboard is for retrospective research and software demonstration only.
It is not a live CGM monitor, medical device, or clinical decision-support system.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="GlucoTwin | Research Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRAINING_DIR = Path(__file__).resolve().parent
OUT_DIR = TRAINING_DIR / "results_causal"
V3_DIR = OUT_DIR / "v3_model_comparison"
PREP_DIR = OUT_DIR / "preprocessing_v3"

FILES = {
    "metrics": OUT_DIR / "causal_multihorizon_metrics.csv",
    "predictions": OUT_DIR / "causal_test_predictions.csv",
    "intervals": OUT_DIR / "conformal_interval_results.csv",
    "hypo_patient": V3_DIR / "053_hypo_per_patient_metrics.csv",
    "hypo_macro": V3_DIR / "054_hypo_patient_macro_summary.csv",
    "alert_summary": V3_DIR / "062_earliest_alert_summary.csv",
    "alert_episodes": V3_DIR / "062_distinct_alert_episodes.csv",
    "event_matching": V3_DIR / "062_event_alert_matching.csv",
    "candidate_metrics": V3_DIR / "v3_candidate_metrics.csv",
    "preprocessed": PREP_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv",
}

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.15rem 1.35rem; border-radius: 14px;
        border: 1px solid rgba(128,128,128,.28);
        background: linear-gradient(120deg, rgba(35,112,145,.13), rgba(88,90,160,.08));
        margin-bottom: 1rem;
    }
    .muted {opacity: .78; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"__load_error__": str(exc)}

@st.cache_data(show_spinner=False)
def load_json(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"__load_error__": str(exc)}

DATA = {key: load_csv(str(path)) for key, path in FILES.items()}
metadata_path = OUT_DIR / "v3_run_metadata.json"
if not metadata_path.exists():
    metadata_path = V3_DIR / "v3_run_metadata.json"
RUN_METADATA = load_json(str(metadata_path))

def df(key):
    value = DATA.get(key)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()

def has_data(key):
    return isinstance(DATA.get(key), pd.DataFrame) and not DATA[key].empty

def safe_numeric(frame, columns):
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame

def first_existing(frame, candidates):
    return next((col for col in candidates if col in frame.columns), None)

def fmt(value, digits=3):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def missing_notice(keys):
    missing = [f"{key}: `{FILES[key]}`" for key in keys if not has_data(key)]
    if missing:
        with st.expander(f"Data availability: {len(missing)} file(s) unavailable", expanded=False):
            st.caption("These sections will be limited until the corresponding files are present.")
            for item in missing:
                st.write(f"- {item}")

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">🩸 GlucoTwin</h1>
      <p style="font-size:1.08rem;margin:.35rem 0 0 0;">
        Glucose forecasting · Hypoglycemia research · Patient-level evaluation
      </p>
      <p class="muted" style="margin:.45rem 0 0 0;">
        Retrospective research dashboard using saved experiment outputs. No training is performed.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "RESEARCH ONLY — This app displays historical datasets and offline model evaluations. "
    "It does not receive live CGM readings, diagnose conditions, or provide validated medical alerts. "
    "Do not use it to make treatment or insulin decisions."
)

with st.sidebar:
    st.header("Dashboard controls")
    st.caption("All inputs are saved CSVs; model files are not modified.")
    st.write(f"**Training folder**")
    st.code(str(TRAINING_DIR), language="text")
    section = st.radio(
        "Open section",
        [
            "Overview",
            "Glucose forecasts",
            "Forecast uncertainty",
            "Hypoglycemia model research",
            "Alert episode audit",
            "Patient-level diagnostics",
            "Data & file status",
        ],
    )
    st.divider()
    st.caption("A missing optional CSV hides or limits only its section.")

# Common data references
metrics = df("metrics")
pred = df("predictions")
intervals = df("intervals")
hypo_patient = df("hypo_patient")
hypo_macro = df("hypo_macro")
alert_summary = df("alert_summary")
alert_episodes = df("alert_episodes")
event_matching = df("event_matching")
candidate_metrics = df("candidate_metrics")

if section == "Overview":
    st.subheader("Research overview")
    pred_num = safe_numeric(pred, ["horizon_minutes", "prediction_mgdl", "baseline_mgdl", "actual_mgdl"])
    test_pred = pred_num
    if "split" in test_pred.columns:
        test_pred = test_pred[test_pred["split"].astype(str).str.lower().eq("test")]

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Saved forecast rows", f"{len(pred):,}" if not pred.empty else "—")
    if not pred_num.empty and "horizon_minutes" in pred_num.columns:
        horizons = sorted(pred_num["horizon_minutes"].dropna().unique())
        metric_card("Forecast horizons", ", ".join(f"{int(x)}m" for x in horizons) if horizons else "—")
    else:
        metric_card("Forecast horizons", "—")
    metric_card("Hypo patient rows", f"{len(hypo_patient):,}" if not hypo_patient.empty else "—")
    metric_card("Alert summary rows", f"{len(alert_summary):,}" if not alert_summary.empty else "—")

    st.markdown("### Available research outputs")
    status = []
    for key, path in FILES.items():
        status.append({
            "Output": key.replace("_", " ").title(),
            "Status": "Available" if has_data(key) else "Not found / empty",
            "File": str(path.relative_to(TRAINING_DIR)) if path.is_relative_to(TRAINING_DIR) else str(path),
        })
    status_df = pd.DataFrame(status)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    if not metrics.empty:
        st.markdown("### Saved regression evaluation")
        display_cols = [c for c in ["horizon_minutes", "model", "split", "rows", "MAE", "RMSE", "R2"] if c in metrics.columns]
        st.dataframe(metrics[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Forecast metric CSV not found. Other sections may still work.")

    if not hypo_macro.empty:
        st.markdown("### Patient-macro hypoglycemia results")
        st.dataframe(hypo_macro, use_container_width=True, hide_index=True)
    else:
        st.info("Patient-macro classification summary not found.")

    missing_notice(list(FILES.keys()))

elif section == "Glucose forecasts":
    st.subheader("Historical glucose forecasts")
    if pred.empty:
        st.error(f"Predictions CSV not found or empty: {FILES['predictions']}")
        st.stop()

    p = safe_numeric(pred, ["horizon_minutes", "prediction_mgdl", "baseline_mgdl", "actual_mgdl"])
    if "split" in p.columns:
        split_options = ["All"] + sorted(p["split"].dropna().astype(str).unique().tolist())
        selected_split = st.selectbox("Data split", split_options, index=split_options.index("test") if "test" in split_options else 0)
        if selected_split != "All":
            p = p[p["split"].astype(str).eq(selected_split)]
    if "horizon_minutes" in p.columns:
        horizon_options = sorted(p["horizon_minutes"].dropna().unique().tolist())
        if horizon_options:
            selected_horizon = st.selectbox("Forecast horizon (minutes)", horizon_options, index=0)
            p = p[p["horizon_minutes"].eq(selected_horizon)]

    target_options = sorted(p["_target"].dropna().astype(str).unique().tolist()) if "_target" in p.columns else []
    if target_options:
        selected_target = st.selectbox("Forecast target", target_options)
        p = p[p["_target"].astype(str).eq(selected_target)]

    if p.empty:
        st.warning("No rows match these filters.")
    else:
        st.caption("Rows are plotted in stored row order. They are not a continuous live time series unless the source provides timestamps.")
        chart = p.reset_index(drop=True).copy()
        chart["sample_order"] = np.arange(1, len(chart) + 1)
        max_points = st.slider("Maximum points to display", min_value=100, max_value=5000, value=min(1200, max(100, len(chart)), 5000), step=100)
        if len(chart) > max_points:
            chart = chart.iloc[np.linspace(0, len(chart) - 1, max_points).astype(int)]
        fig = go.Figure()
        if "actual_mgdl" in chart:
            fig.add_trace(go.Scatter(x=chart["sample_order"], y=chart["actual_mgdl"], mode="lines", name="Actual glucose"))
        if "prediction_mgdl" in chart:
            fig.add_trace(go.Scatter(x=chart["sample_order"], y=chart["prediction_mgdl"], mode="lines", name="Model prediction"))
        if "baseline_mgdl" in chart:
            fig.add_trace(go.Scatter(x=chart["sample_order"], y=chart["baseline_mgdl"], mode="lines", name="Baseline prediction"))
        fig.add_hline(y=70, line_dash="dash", annotation_text="70 mg/dL reference")
        fig.update_layout(xaxis_title="Displayed sample order", yaxis_title="Glucose (mg/dL)", height=480, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        if {"actual_mgdl", "prediction_mgdl"}.issubset(p.columns):
            valid = p.dropna(subset=["actual_mgdl", "prediction_mgdl"]).copy()
            valid["absolute_error"] = (valid["actual_mgdl"] - valid["prediction_mgdl"]).abs()
            valid["squared_error"] = (valid["actual_mgdl"] - valid["prediction_mgdl"]) ** 2
            a, b, c = st.columns(3)
            metric_card("Rows evaluated", f"{len(valid):,}")
            metric_card("MAE (mg/dL)", fmt(valid["absolute_error"].mean(), 2))
            metric_card("RMSE (mg/dL)", fmt(np.sqrt(valid["squared_error"].mean()), 2))
        with st.expander("Preview filtered prediction rows"):
            st.dataframe(p.head(500), use_container_width=True, hide_index=True)

    if not metrics.empty:
        st.markdown("### Saved metrics by horizon/model")
        metric_view = metrics.copy()
        display_cols = [c for c in ["horizon_minutes", "model", "split", "rows", "MAE", "RMSE", "R2"] if c in metric_view.columns]
        st.dataframe(metric_view[display_cols], use_container_width=True, hide_index=True)

elif section == "Forecast uncertainty":
    st.subheader("Conformal / forecast interval evaluation")
    if intervals.empty:
        st.error(f"Interval results CSV not found or empty: {FILES['intervals']}")
        st.stop()
    it = intervals.copy()
    numeric_cols = ["horizon_minutes", "target_coverage", "test_coverage", "calibration_quantile_mgdl", "mean_interval_width_mgdl"]
    it = safe_numeric(it, numeric_cols)
    if "horizon_minutes" in it.columns:
        opts = ["All"] + sorted(it["horizon_minutes"].dropna().unique().tolist())
        chosen = st.selectbox("Horizon", opts, key="interval_horizon")
        if chosen != "All":
            it = it[it["horizon_minutes"].eq(chosen)]
    if "interval" in it.columns:
        opts = ["All"] + sorted(it["interval"].dropna().astype(str).unique().tolist())
        chosen = st.selectbox("Interval configuration", opts, key="interval_kind")
        if chosen != "All":
            it = it[it["interval"].astype(str).eq(chosen)]
    st.dataframe(it, use_container_width=True, hide_index=True)

    if {"target_coverage", "test_coverage"}.issubset(it.columns):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=it.index.astype(str), y=it["target_coverage"], name="Target coverage"))
        fig.add_trace(go.Bar(x=it.index.astype(str), y=it["test_coverage"], name="Observed test coverage"))
        fig.update_layout(barmode="group", xaxis_title="Filtered interval row", yaxis_title="Coverage", yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig, use_container_width=True)
    if "mean_interval_width_mgdl" in it.columns:
        st.markdown("#### Mean interval width")
        st.bar_chart(it.set_index(it.index.astype(str))["mean_interval_width_mgdl"])

elif section == "Hypoglycemia model research":
    st.subheader("Hypoglycemia classification: patient-level research")
    st.caption("Classification metrics describe offline held-out results. Thresholds are exploratory, not medical alert thresholds.")
    if not hypo_macro.empty:
        st.markdown("### Macro summary across patients")
        st.dataframe(hypo_macro, use_container_width=True, hide_index=True)
        numeric_macro = [c for c in ["patient_macro_PR_AUC", "patient_macro_ROC_AUC", "patient_macro_Brier_all_patients"] if c in hypo_macro.columns]
        if numeric_macro and "model" in hypo_macro.columns:
            plot_data = hypo_macro.melt(id_vars=["model"], value_vars=numeric_macro, var_name="Metric", value_name="Value")
            fig = px.bar(plot_data, x="Metric", y="Value", color="model", barmode="group", title="Patient-macro metrics (Brier is lower-is-better)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Macro summary file not found.")

    if not hypo_patient.empty:
        hp = hypo_patient.copy()
        for col in ["PR_AUC", "ROC_AUC", "Brier", "prevalence", "sensitivity_at_exploratory_threshold", "specificity_at_exploratory_threshold"]:
            if col in hp.columns:
                hp[col] = pd.to_numeric(hp[col], errors="coerce")
        if "model" in hp.columns:
            models = sorted(hp["model"].dropna().astype(str).unique().tolist())
            selected_models = st.multiselect("Models", models, default=models)
            hp = hp[hp["model"].astype(str).isin(selected_models)]
        if "patient_uid" in hp.columns:
            patients = sorted(hp["patient_uid"].dropna().astype(str).unique().tolist())
            selected_patients = st.multiselect("Patients", patients, default=patients[:min(12, len(patients))])
            if selected_patients:
                hp = hp[hp["patient_uid"].astype(str).isin(selected_patients)]
        cols = [c for c in ["model", "patient_uid", "rows", "positive_rows", "prevalence", "PR_AUC", "ROC_AUC", "Brier", "sensitivity_at_exploratory_threshold", "specificity_at_exploratory_threshold", "has_both_classes"] if c in hp.columns]
        st.markdown("### Per-patient metrics")
        st.dataframe(hp[cols], use_container_width=True, hide_index=True)
        if {"patient_uid", "PR_AUC", "model"}.issubset(hp.columns):
            fig = px.bar(hp, x="patient_uid", y="PR_AUC", color="model", barmode="group", title="PR-AUC by patient")
            fig.update_layout(xaxis_title="Patient", yaxis_title="PR-AUC", height=450)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Per-patient classification metrics not found.")

    if not candidate_metrics.empty:
        with st.expander("Candidate model comparison CSV"):
            st.caption("This file may contain multiple task types. Review task/target columns before comparing values.")
            st.dataframe(candidate_metrics, use_container_width=True, hide_index=True)

elif section == "Alert episode audit":
    st.subheader("Offline alert-to-event audit")
    st.warning(
        "These are retrospective episode-matching results. An unmatched alert is not automatically a clinically false alert, "
        "and an episode match does not prove that a usable warning would be available in real time."
    )
    if alert_summary.empty:
        st.error(f"Alert summary CSV not found or empty: {FILES['alert_summary']}")
        st.stop()

    a = alert_summary.copy()
    a = safe_numeric(a, [
        "threshold", "horizon_minutes", "actual_low_episodes", "distinct_alert_episodes",
        "one_to_one_advance_matches", "events_without_matched_advance_alert",
        "median_lead_minutes", "maximum_lead_minutes", "late_alert_episodes", "unmatched_alert_episodes",
    ])
    if "model" in a.columns:
        model_options = ["All"] + sorted(a["model"].dropna().astype(str).unique().tolist())
        selected_model = st.selectbox("Model", model_options)
        if selected_model != "All":
            a = a[a["model"].astype(str).eq(selected_model)]
    if "patient_uid" in a.columns:
        patient_options = ["All"] + sorted(a["patient_uid"].dropna().astype(str).unique().tolist())
        selected_patient = st.selectbox("Patient", patient_options)
        if selected_patient != "All":
            a = a[a["patient_uid"].astype(str).eq(selected_patient)]
    if "threshold" in a.columns:
        threshold_options = ["All"] + sorted(a["threshold"].dropna().unique().tolist())
        selected_threshold = st.selectbox("Exploratory probability threshold", threshold_options)
        if selected_threshold != "All":
            a = a[a["threshold"].eq(selected_threshold)]

    st.dataframe(a, use_container_width=True, hide_index=True)
    if {"patient_uid", "one_to_one_advance_matches", "events_without_matched_advance_alert"}.issubset(a.columns):
        plot_cols = ["one_to_one_advance_matches", "events_without_matched_advance_alert"]
        if "model" in a.columns:
            plot_data = a.melt(id_vars=["patient_uid", "model"], value_vars=plot_cols, var_name="Outcome", value_name="Episodes")
            fig = px.bar(plot_data, x="patient_uid", y="Episodes", color="Outcome", facet_col="model", barmode="group", title="Matched vs unmatched actual low episodes")
        else:
            plot_data = a.melt(id_vars=["patient_uid"], value_vars=plot_cols, var_name="Outcome", value_name="Episodes")
            fig = px.bar(plot_data, x="patient_uid", y="Episodes", color="Outcome", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    if not alert_episodes.empty:
        with st.expander("Distinct alert episode rows"):
            st.dataframe(alert_episodes, use_container_width=True, hide_index=True)
    if not event_matching.empty:
        with st.expander("Event-to-alert matching rows"):
            st.dataframe(event_matching, use_container_width=True, hide_index=True)

elif section == "Patient-level diagnostics":
    st.subheader("Patient-level error and robustness diagnostics")
    diagnostic_files = {
        "Focus patient error summary": OUT_DIR / "055_focus_patient_error_summary.csv",
        "Monthly error segments": OUT_DIR / "055_monthly_error_segments.csv",
        "Patient calibration diagnostics": OUT_DIR / "055_patient_calibration_diagnostics.csv",
        "Probability-bin calibration": OUT_DIR / "055_probability_bin_calibration.csv",
        "Focus patient metrics": V3_DIR / "057_focus_patient_metrics.csv",
        "Focus monthly metrics": V3_DIR / "057_focus_monthly_metrics.csv",
        "Hypo label patient summary": V3_DIR / "056_hypo_label_patient_summary.csv",
        "Actual low episode lead time": V3_DIR / "060_actual_low_episode_lead_time.csv",
    }
    available = {}
    for label, path in diagnostic_files.items():
        frame = load_csv(str(path))
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            available[label] = (path, frame)
    if not available:
        st.info("No recognized diagnostic CSVs were found in results_causal / v3_model_comparison.")
    else:
        chosen_diag = st.selectbox("Diagnostic dataset", list(available.keys()))
        path, dframe = available[chosen_diag]
        st.caption(f"Source: `{path}`")
        st.dataframe(dframe, use_container_width=True, hide_index=True)
        numeric_cols = dframe.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            chosen_col = st.selectbox("Numeric column to chart", numeric_cols)
            if chosen_col:
                chart_df = dframe[[chosen_col]].dropna().reset_index(drop=True)
                chart_df.index.name = "Row"
                st.line_chart(chart_df[chosen_col])

elif section == "Data & file status":
    st.subheader("Data availability and provenance")
    rows = []
    for key, path in FILES.items():
        value = DATA.get(key)
        if isinstance(value, pd.DataFrame):
            state = "Available" if not value.empty else "Empty"
            row_count = len(value)
            col_count = len(value.columns)
            detail = ", ".join(value.columns[:12].astype(str)) + (" …" if len(value.columns) > 12 else "")
        elif isinstance(value, dict) and "__load_error__" in value:
            state = "Read error"
            row_count = "—"
            col_count = "—"
            detail = value["__load_error__"]
        else:
            state = "Missing"
            row_count = "—"
            col_count = "—"
            detail = ""
        rows.append({"Dataset key": key, "Status": state, "Rows": row_count, "Columns": col_count, "Path": str(path), "Column preview": detail})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Run metadata")
    if RUN_METADATA:
        st.json(RUN_METADATA)
    else:
        st.caption("No v3_run_metadata.json found at the expected paths.")

    st.markdown("### Important interpretation notes")
    st.markdown(
        """
        - `causal_test_predictions.csv` contains stored historical forecast predictions and does not contain patient IDs in the currently documented schema.
        - Hypoglycemia classifier metrics and alert episode matching are separate research outputs from glucose regression forecasts.
        - `target_hypo_60` is a retrospective label based on future CGM values; it is not a current-state glucose reading.
        - Do not infer patient identity, timestamp continuity, or clinical utility where the source CSV does not provide that information.
        - No model training, model saving, or artifact overwriting occurs in this dashboard.
        """
    )

st.divider()
st.caption(
    "GlucoTwin research prototype · Offline data only · Not for diagnosis, treatment, insulin dosing, or emergency decisions."
)
