from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# GLUCOTWIN - GLUCOSE FORECASTING DASHBOARD
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TRAINING_DIR / "results_causal"

METRICS_FILE = RESULTS_DIR / "causal_multihorizon_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "causal_test_predictions.csv"
INTERVALS_FILE = RESULTS_DIR / "conformal_interval_results.csv"

# -------------------- Page configuration --------------------

st.set_page_config(
    page_title="GlucoTwin | Forecast Dashboard",
    page_icon="🩸",
    layout="wide",
)

# -------------------- Custom styling --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220, #101b2d);
        color: #f1f5f9;
    }

    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.80);
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 14px;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(120deg, #123047, #1d3557);
        border: 1px solid #36536f;
        margin-bottom: 20px;
    }

    .muted {
        color: #a8b7ca;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Header --------------------

st.markdown(
    """
    <div class="hero">
        <h1>🩸 GlucoTwin</h1>
        <h3>Glucose Forecasting & Uncertainty Analytics</h3>
        <p class="muted">
            Multi-horizon LightGBM · Model evaluation · Prediction intervals
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research prototype only. This dashboard displays saved evaluation "
    "predictions, not live personal glucose forecasts. It is not validated "
    "for diagnosis, treatment, insulin dosing, or other clinical decisions."
)

# -------------------- Check files --------------------

st.sidebar.title("Dashboard controls")

missing_files = [
    path for path in [
        METRICS_FILE,
        PREDICTIONS_FILE,
        INTERVALS_FILE,
    ]
    if not path.exists()
]

if missing_files:
    st.error("Required result files were not found:")
    for path in missing_files:
        st.code(str(path))
    st.stop()

# -------------------- Load data --------------------

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


metrics = load_csv(str(METRICS_FILE))
predictions = load_csv(str(PREDICTIONS_FILE))
intervals = load_csv(str(INTERVALS_FILE))

required_metrics = {
    "horizon_minutes", "model", "split", "rows", "MAE", "RMSE", "R2"
}

required_predictions = {
    "horizon_minutes", "split", "prediction_mgdl", "actual_mgdl"
}

required_intervals = {
    "horizon_minutes", "interval", "target_coverage",
    "test_coverage", "calibration_quantile_mgdl",
    "mean_interval_width_mgdl"
}

if not required_metrics.issubset(metrics.columns):
    st.error("Unexpected metrics CSV columns.")
    st.write(list(metrics.columns))
    st.stop()

if not required_predictions.issubset(predictions.columns):
    st.error("Unexpected predictions CSV columns.")
    st.write(list(predictions.columns))
    st.stop()

if not required_intervals.issubset(intervals.columns):
    st.error("Unexpected interval CSV columns.")
    st.write(list(intervals.columns))
    st.stop()

# Normalize data types and split labels.
for frame in [metrics, predictions]:
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["horizon_minutes"] = pd.to_numeric(
        frame["horizon_minutes"], errors="coerce"
    )

for col in ["MAE", "RMSE", "R2", "rows"]:
    metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

for col in ["prediction_mgdl", "actual_mgdl"]:
    predictions[col] = pd.to_numeric(
        predictions[col], errors="coerce"
    )

intervals["horizon_minutes"] = pd.to_numeric(
    intervals["horizon_minutes"], errors="coerce"
)

# -------------------- Sidebar filters --------------------

horizons = sorted(
    int(value)
    for value in metrics["horizon_minutes"].dropna().unique()
)

if not horizons:
    st.error("No forecast horizons were found in the metrics file.")
    st.stop()

selected_horizon = st.sidebar.selectbox(
    "Forecast horizon",
    horizons,
    format_func=lambda value: f"+{value} minutes",
)

selected_split = st.sidebar.selectbox(
    "Evaluation split",
    ["test", "validation"],
    index=0,
)

selected_interval = st.sidebar.selectbox(
    "Prediction interval",
    ["80%", "90%"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.caption("Source: saved GlucoTwin model evaluation files")

# -------------------- Selected model metrics --------------------

horizon_metrics = metrics[
    (metrics["horizon_minutes"] == selected_horizon)
    & (metrics["split"] == selected_split)
].copy()

lightgbm_rows = horizon_metrics[
    horizon_metrics["model"].astype(str).str.contains(
        "LightGBM", case=False, na=False
    )
]

baseline_rows = horizon_metrics[
    horizon_metrics["model"].astype(str).str.contains(
        "Baseline", case=False, na=False
    )
]

if lightgbm_rows.empty:
    st.error("No LightGBM metrics found for this horizon and split.")
    st.stop()

model_row = lightgbm_rows.iloc[0]

baseline_row = (
    baseline_rows.iloc[0]
    if not baseline_rows.empty
    else None
)

# -------------------- Performance cards --------------------

st.subheader(f"Model performance · +{selected_horizon} minutes")

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", f'{model_row["MAE"]:.2f} mg/dL')
c2.metric("RMSE", f'{model_row["RMSE"]:.2f} mg/dL')
c3.metric("R²", f'{model_row["R2"]:.4f}')
c4.metric("Evaluation rows", f'{int(model_row["rows"]):,}')

if baseline_row is not None and baseline_row["MAE"] > 0:
    improvement = (
        (baseline_row["MAE"] - model_row["MAE"])
        / baseline_row["MAE"]
    ) * 100

    st.info(
        f"LightGBM MAE is {improvement:.1f}% lower than the "
        f"last-CGM baseline on the {selected_split} split."
    )

# -------------------- Prediction quality charts --------------------

st.markdown("---")
st.subheader("Prediction quality analysis")

plot_df = predictions[
    (predictions["horizon_minutes"] == selected_horizon)
    & (predictions["split"] == selected_split)
].copy()

plot_df = plot_df.dropna(
    subset=["prediction_mgdl", "actual_mgdl"]
)

if plot_df.empty:
    st.warning("No prediction rows are available for this selection.")
    st.stop()

actual = plot_df["actual_mgdl"]
predicted = plot_df["prediction_mgdl"]
errors = actual - predicted

recomputed_mae = errors.abs().mean()
recomputed_rmse = (errors.pow(2).mean()) ** 0.5
mean_error = errors.mean()

q1, q2, q3 = st.columns(3)

q1.metric("Rows analyzed", f"{len(plot_df):,}")
q2.metric("Recomputed MAE", f"{recomputed_mae:.2f} mg/dL")
q3.metric("Mean error", f"{mean_error:+.2f} mg/dL")

st.caption(
    "Mean error is actual minus predicted. Positive values indicate "
    "average underprediction; negative values indicate average overprediction."
)

# 1. Actual versus predicted scatter plot.

scatter_df = plot_df

if len(scatter_df) > 4000:
    scatter_df = scatter_df.sample(
        n=4000, random_state=42
    )

lower = min(
    scatter_df["actual_mgdl"].min(),
    scatter_df["prediction_mgdl"].min(),
)

upper = max(
    scatter_df["actual_mgdl"].max(),
    scatter_df["prediction_mgdl"].max(),
)

fig_scatter = go.Figure()

fig_scatter.add_trace(
    go.Scatter(
        x=scatter_df["actual_mgdl"],
        y=scatter_df["prediction_mgdl"],
        mode="markers",
        name="Predictions",
        marker=dict(size=5, opacity=0.45),
    )
)

fig_scatter.add_trace(
    go.Scatter(
        x=[lower, upper],
        y=[lower, upper],
        mode="lines",
        name="Perfect prediction",
        line=dict(dash="dash"),
    )
)

fig_scatter.update_layout(
    template="plotly_dark",
    title="Actual vs predicted glucose",
    xaxis_title="Actual glucose (mg/dL)",
    yaxis_title="Predicted glucose (mg/dL)",
    height=470,
)

st.plotly_chart(fig_scatter, use_container_width=True)

# 2. Prediction error distribution.

fig_error = go.Figure()

fig_error.add_trace(
    go.Histogram(
        x=errors,
        nbinsx=50,
        name="Prediction errors",
    )
)

fig_error.add_vline(
    x=0,
    line_dash="dash",
    annotation_text="Zero error",
)

fig_error.update_layout(
    template="plotly_dark",
    title="Prediction error distribution",
    xaxis_title="Actual - predicted (mg/dL)",
    yaxis_title="Number of predictions",
    height=400,
)

st.plotly_chart(fig_error, use_container_width=True)

# -------------------- Calibrated uncertainty intervals --------------------

st.markdown("---")
st.subheader("Calibrated prediction intervals")

interval_match = intervals[
    (intervals["horizon_minutes"] == selected_horizon)
    & (intervals["interval"] == selected_interval)
]

if interval_match.empty:
    st.warning("No calibration result found for this selection.")
else:
    interval_row = interval_match.iloc[0]

    radius = float(
        interval_row["calibration_quantile_mgdl"]
    )

    lower_bound = predicted - radius
    upper_bound = predicted + radius

    if selected_split == "test":
        covered = (
            (actual >= lower_bound)
            & (actual <= upper_bound)
        )

        observed_coverage = covered.mean()

        a, b, c = st.columns(3)

        a.metric(
            "Target coverage",
            f'{float(interval_row["target_coverage"]):.1%}',
        )

        b.metric(
            "Recomputed test coverage",
            f"{observed_coverage:.1%}",
        )

        c.metric(
            "Interval radius",
            f"±{radius:.2f} mg/dL",
        )

        st.metric(
            "Mean full interval width",
            f"{2 * radius:.2f} mg/dL",
        )

        # Visualize a subset of the interval predictions.
        interval_plot = plot_df.copy()

        if "_row_index" in interval_plot.columns:
            interval_plot = interval_plot.sort_values("_row_index")

        if len(interval_plot) > 1000:
            positions = pd.Series(
                range(len(interval_plot))
            ).iloc[
                list(
                    range(
                        0,
                        len(interval_plot),
                        max(1, len(interval_plot) // 1000),
                    )
                )
            ].tolist()

            interval_plot = interval_plot.iloc[positions[:1000]]

        interval_actual = interval_plot["actual_mgdl"]
        interval_prediction = interval_plot["prediction_mgdl"]

        fig_interval = go.Figure()

        x_values = list(range(len(interval_plot)))

        fig_interval.add_trace(
            go.Scatter(
                x=x_values,
                y=interval_prediction + radius,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig_interval.add_trace(
            go.Scatter(
                x=x_values,
                y=interval_prediction - radius,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.20)",
                name=f"{selected_interval} interval",
            )
        )

        fig_interval.add_trace(
            go.Scatter(
                x=x_values,
                y=interval_actual,
                mode="markers",
                marker=dict(size=4),
                name="Actual glucose",
            )
        )

        fig_interval.add_trace(
            go.Scatter(
                x=x_values,
                y=interval_prediction,
                mode="lines",
                name="Prediction",
            )
        )

        fig_interval.update_layout(
            template="plotly_dark",
            title="Held-out test predictions and uncertainty",
            xaxis_title="Evaluation sample index (not elapsed time)",
            yaxis_title="Glucose (mg/dL)",
            height=450,
        )

        st.plotly_chart(
            fig_interval,
            use_container_width=True,
        )

    else:
        st.info(
            "The intervals were calibrated using validation residuals. "
            "Switch to the test split to evaluate their held-out coverage."
        )

    st.caption(
        "Split conformal coverage is empirical. It does not guarantee "
        "coverage for every patient and is not a clinical safety guarantee."
    )

# -------------------- Compare all horizons --------------------

st.markdown("---")
st.subheader("Performance across forecast horizons")

comparison = metrics[
    (metrics["split"] == selected_split)
    & metrics["model"].astype(str).str.contains(
        "LightGBM|Baseline", case=False, na=False
    )
].copy()

if not comparison.empty:
    comparison["display_model"] = comparison["model"].apply(
        lambda name: (
            "LightGBM"
            if "lightgbm" in str(name).lower()
            else "Last-CGM baseline"
        )
    )

    fig_comparison = go.Figure()

    for name in ["LightGBM", "Last-CGM baseline"]:
        subset = comparison[
            comparison["display_model"] == name
        ].sort_values("horizon_minutes")

        if not subset.empty:
            fig_comparison.add_trace(
                go.Scatter(
                    x=subset["horizon_minutes"],
                    y=subset["MAE"],
                    mode="lines+markers",
                    name=name,
                )
            )

    fig_comparison.update_layout(
        template="plotly_dark",
        title="MAE by forecast horizon",
        xaxis_title="Forecast horizon (minutes)",
        yaxis_title="MAE (mg/dL)",
        height=400,
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True,
    )

# -------------------- Data tables --------------------

with st.expander("View all model evaluation metrics"):
    st.dataframe(
        metrics.sort_values(
            ["horizon_minutes", "split", "model"]
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("View uncertainty calibration results"):
    st.dataframe(
        intervals.sort_values(
            ["horizon_minutes", "interval"]
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Inspect saved prediction examples"):
    st.dataframe(
        plot_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

# -------------------- Footer --------------------

st.markdown("---")

st.markdown(
    """
    <p class="muted">
    GlucoTwin · Research prototype · Historical evaluation only.<br>
    Not intended for diagnosis, treatment, insulin dosing, or other clinical
    decisions. Performance may vary across patients and settings.
    </p>
    """,
    unsafe_allow_html=True,
)
