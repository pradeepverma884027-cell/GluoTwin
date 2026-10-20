from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
PREP = BASE / "results_causal" / "preprocessing_v3"
OUT = BASE / "results_causal" / "v3_model_comparison"

DATA = PREP / "cgm_t1dm_t2dm_causal_model_ready.csv"
PRED = OUT / "058_focus_test_predictions_full.csv"

EVENT_OUT = OUT / "060_actual_low_episode_lead_time.csv"
ALERT_OUT = OUT / "060_alert_classification.csv"
SUMMARY_OUT = OUT / "060_lead_time_summary.csv"

FOCUS = ["T2DM_2072", "T2DM_2077", "T2DM_2078"]
CGM_MEAN = 144.94967492045924
CGM_STD = 52.47014432357027
LOW = 70.0
THRESHOLD = 0.20
HORIZON = pd.Timedelta(minutes=60)
GRID = pd.Timedelta(minutes=15)

def main():
    for p in [DATA, PRED]:
        if not p.exists():
            raise FileNotFoundError(p)

    cols = ["patient_uid", "segment_id", "dt", "split", "cgm"]
    d = pd.read_csv(DATA, usecols=cols, parse_dates=["dt"])
    d = d[
        d["patient_uid"].isin(FOCUS)
        & (d["split"].astype(str).str.lower() == "test")
    ].copy()

    d["cgm_mgdl_reconstructed"] = d["cgm"] * CGM_STD + CGM_MEAN

    p = pd.read_csv(PRED, parse_dates=["dt"])
    p = p[
        p["patient_uid"].isin(FOCUS)
        & (p["split"].astype(str).str.lower() == "test")
    ].copy()

    keys = ["patient_uid", "segment_id", "dt"]
    if d.duplicated(keys).any() or p.duplicated(keys).any():
        raise ValueError("Duplicate patient/segment/timestamp keys detected.")

    df = d.merge(
        p[keys + ["xgb_probability", "lgb_probability"]],
        on=keys, how="left", validate="one_to_one"
    )
    if df[["xgb_probability", "lgb_probability"]].isna().any().any():
        raise ValueError("Some test rows did not match prediction timestamps.")

    df = df.sort_values(keys).reset_index(drop=True)
    df["is_low"] = df["cgm_mgdl_reconstructed"] < LOW

    # Create actual-low episodes, splitting at gaps and segment boundaries.
    grp = df.groupby(["patient_uid", "segment_id"], sort=False)
    prev_low = grp["is_low"].shift()
    prev_dt = grp["dt"].shift()

    starts = (
        df["is_low"]
        & (
            prev_low.ne(True)
            | df["dt"].sub(prev_dt).ne(GRID)
        )
    )
    df["low_start"] = starts

    episode_rows = []
    for (patient, segment), part in df.groupby(
        ["patient_uid", "segment_id"], sort=False
    ):
        part = part.sort_values("dt").copy()
        part["episode_no"] = part["low_start"].cumsum()

        lows = part[part["is_low"]]
        for episode_no, ep in lows.groupby("episode_no", sort=False):
            episode_rows.append({
                "patient_uid": patient,
                "segment_id": segment,
                "episode_no": int(episode_no),
                "start_dt": ep["dt"].iloc[0],
                "end_dt": ep["dt"].iloc[-1],
                "readings_below_70": len(ep),
                "min_reconstructed_cgm_mgdl":
                    ep["cgm_mgdl_reconstructed"].min(),
            })

    events = pd.DataFrame(episode_rows)
    if events.empty:
        raise ValueError("No actual-low episodes found.")

    # Match alerts to upcoming episode starts in the next 60 minutes.
    # Current glucose must not already be low for an advance warning.
    alert_rows = []
    for model in ["xgb", "lgb"]:
        prob = f"{model}_probability"
        df[f"{model}_alert"] = df[prob] >= THRESHOLD

        for _, r in df[df[f"{model}_alert"]].iterrows():
            patient_events = events[
                (events["patient_uid"] == r["patient_uid"])
                & (events["segment_id"] == r["segment_id"])
            ]
            future = patient_events[
                (patient_events["start_dt"] > r["dt"])
                & (patient_events["start_dt"] <= r["dt"] + HORIZON)
            ]
            currently_low = bool(r["is_low"])

            if currently_low:
                category = "late_warning_currently_low"
                matched_start = pd.NaT
                lead_minutes = np.nan
            elif not future.empty:
                next_event = future.sort_values("start_dt").iloc[0]
                category = "advance_warning"
                matched_start = next_event["start_dt"]
                lead_minutes = (
                    matched_start - r["dt"]
                ).total_seconds() / 60
            else:
                category = "unmatched_alert_no_low_start_within_60m"
                matched_start = pd.NaT
                lead_minutes = np.nan

            alert_rows.append({
                "patient_uid": r["patient_uid"],
                "segment_id": r["segment_id"],
                "alert_dt": r["dt"],
                "model": model.upper(),
                "probability": r[prob],
                "current_reconstructed_cgm_mgdl":
                    r["cgm_mgdl_reconstructed"],
                "alert_category": category,
                "matched_low_start_dt": matched_start,
                "lead_minutes": lead_minutes,
            })

    alerts = pd.DataFrame(alert_rows)
    if alerts.empty:
        alerts = pd.DataFrame(columns=[
            "patient_uid", "segment_id", "alert_dt", "model",
            "probability", "current_reconstructed_cgm_mgdl",
            "alert_category", "matched_low_start_dt", "lead_minutes"
        ])

    # Event-level detection: at least one eligible alert in the 60 minutes
    # before onset, while reconstructed current CGM is >= 70.
    for model in ["xgb", "lgb"]:
        detected = []
        lead_values = []
        for _, event in events.iterrows():
            candidates = df[
                (df["patient_uid"] == event["patient_uid"])
                & (df["segment_id"] == event["segment_id"])
                & (df["dt"] < event["start_dt"])
                & (df["dt"] >= event["start_dt"] - HORIZON)
                & (df["cgm_mgdl_reconstructed"] >= LOW)
                & (df[f"{model}_alert"])
            ]
            if candidates.empty:
                detected.append(False)
                lead_values.append(np.nan)
            else:
                first = candidates.sort_values("dt").iloc[0]
                detected.append(True)
                lead_values.append(
                    (event["start_dt"] - first["dt"]).total_seconds() / 60
                )

        events[f"{model}_advance_detected"] = detected
        events[f"{model}_lead_minutes"] = lead_values

    events.to_csv(EVENT_OUT, index=False)
    alerts.to_csv(ALERT_OUT, index=False)

    summaries = []
    for patient in FOCUS:
        pe = events[events["patient_uid"] == patient]
        pa = alerts[alerts["patient_uid"] == patient]

        for model in ["XGB", "LGB"]:
            col = f"{model.lower()}_advance_detected"
            leads = pe.loc[pe[col], f"{model.lower()}_lead_minutes"]
            ma = pa[pa["model"] == model]
            counts = ma["alert_category"].value_counts()

            summaries.append({
                "patient_uid": patient,
                "model": model,
                "actual_low_episodes": len(pe),
                "episodes_with_advance_warning": int(pe[col].sum()),
                "episodes_without_advance_warning": int((~pe[col]).sum()),
                "advance_detection_rate":
                    float(pe[col].mean()) if len(pe) else np.nan,
                "median_lead_minutes":
                    float(leads.median()) if len(leads) else np.nan,
                "advance_warning_alerts":
                    int(counts.get("advance_warning", 0)),
                "late_warning_alerts":
                    int(counts.get("late_warning_currently_low", 0)),
                "unmatched_alerts":
                    int(counts.get(
                        "unmatched_alert_no_low_start_within_60m", 0
                    )),
                "threshold": THRESHOLD,
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("=" * 70)
    print("READ-ONLY ACTUAL-LOW LEAD-TIME AUDIT")
    print("=" * 70)
    print("CGM values are reconstructed from the saved standardization schema.")
    print("Threshold:", THRESHOLD)
    print("\nSummary:")
    print(summary.round(3).to_string(index=False))
    print("\nSaved:")
    print(EVENT_OUT)
    print(ALERT_OUT)
    print(SUMMARY_OUT)
    print("\nNo models trained or modified.")

if __name__ == "__main__":
    main()
