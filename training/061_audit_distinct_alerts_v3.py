from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
PREP = BASE / "results_causal" / "preprocessing_v3"
OUT = BASE / "results_causal" / "v3_model_comparison"

DATA = PREP / "cgm_t1dm_t2dm_causal_model_ready.csv"
PRED = OUT / "058_focus_test_predictions_full.csv"

ALERTS_OUT = OUT / "061_distinct_alert_episodes.csv"
MATCH_OUT = OUT / "061_event_alert_matching.csv"
SUMMARY_OUT = OUT / "061_corrected_alert_summary.csv"

FOCUS = ["T2DM_2072", "T2DM_2077", "T2DM_2078"]
CGM_MEAN = 144.94967492045924
CGM_STD = 52.47014432357027
LOW = 70.0
THRESHOLD = 0.20
HORIZON_MIN = 60
GRID_MIN = 15

def main():
    for path in [DATA, PRED]:
        if not path.exists():
            raise FileNotFoundError(path)

    d = pd.read_csv(
        DATA,
        usecols=["patient_uid", "segment_id", "dt", "split", "cgm"],
        parse_dates=["dt"]
    )
    d = d[
        d["patient_uid"].isin(FOCUS)
        & d["split"].astype(str).str.lower().eq("test")
    ].copy()
    d["cgm_mgdl_reconstructed"] = d["cgm"] * CGM_STD + CGM_MEAN

    p = pd.read_csv(PRED, parse_dates=["dt"])
    p = p[
        p["patient_uid"].isin(FOCUS)
        & p["split"].astype(str).str.lower().eq("test")
    ].copy()

    keys = ["patient_uid", "segment_id", "dt"]
    if d.duplicated(keys).any() or p.duplicated(keys).any():
        raise ValueError("Duplicate patient/segment/timestamp keys found.")

    df = d.merge(
        p[keys + ["xgb_probability", "lgb_probability"]],
        on=keys, how="left", validate="one_to_one"
    )
    if df[["xgb_probability", "lgb_probability"]].isna().any().any():
        raise ValueError("Some rows have no matching predictions.")

    df = df.sort_values(keys).reset_index(drop=True)
    df["is_low"] = df["cgm_mgdl_reconstructed"] < LOW

    # Create contiguous actual-low episodes, splitting across time gaps.
    group = df.groupby(["patient_uid", "segment_id"], sort=False)
    prev_low = group["is_low"].shift()
    prev_dt = group["dt"].shift()
    df["low_start"] = (
        df["is_low"]
        & (prev_low.ne(True)
           | df["dt"].sub(prev_dt).dt.total_seconds().div(60).ne(GRID_MIN))
    )

    event_rows = []
    for (patient, segment), part in df.groupby(
        ["patient_uid", "segment_id"], sort=False
    ):
        part = part.sort_values("dt").copy()
        part["episode_no"] = part["low_start"].cumsum()
        lows = part[part["is_low"]]
        for no, ep in lows.groupby("episode_no", sort=False):
            event_rows.append({
                "patient_uid": patient,
                "segment_id": segment,
                "event_id": f"{patient}_{segment}_E{int(no)}",
                "event_start": ep["dt"].iloc[0],
                "event_end": ep["dt"].iloc[-1],
                "low_rows": len(ep),
                "minimum_cgm_mgdl": ep["cgm_mgdl_reconstructed"].min(),
            })
    events = pd.DataFrame(event_rows)
    if events.empty:
        raise ValueError("No actual-low episodes found.")

    all_alerts = []
    matches = []
    summaries = []

    for model in ["xgb", "lgb"]:
        prob_col = f"{model}_probability"
        df["alert"] = df[prob_col] >= THRESHOLD

        # Collapse consecutive alert rows into distinct alert episodes.
        alert_parts = []
        for (patient, segment), part in df.groupby(
            ["patient_uid", "segment_id"], sort=False
        ):
            part = part.sort_values("dt").copy()
            prev_alert = part["alert"].shift()
            prev_dt = part["dt"].shift()
            part["alert_start"] = (
                part["alert"]
                & (prev_alert.ne(True)
                   | part["dt"].sub(prev_dt).dt.total_seconds().div(60).ne(GRID_MIN))
            )
            part["alert_id_num"] = part["alert_start"].cumsum()

            for no, a in part[part["alert"]].groupby(
                "alert_id_num", sort=False
            ):
                alert_parts.append({
                    "patient_uid": patient,
                    "segment_id": segment,
                    "model": model.upper(),
                    "alert_id": f"{patient}_{segment}_{model.upper()}_A{int(no)}",
                    "alert_start": a["dt"].iloc[0],
                    "alert_end": a["dt"].iloc[-1],
                    "alert_rows": len(a),
                    "max_probability": a[prob_col].max(),
                    "cgm_at_alert_start_mgdl":
                        a["cgm_mgdl_reconstructed"].iloc[0],
                    "already_low_at_alert_start": bool(a["is_low"].iloc[0]),
                })

        alerts = pd.DataFrame(alert_parts)
        if alerts.empty:
            continue

        # Candidate advance matches: alert begins before event onset,
        # within the 60-minute horizon, while CGM is not already low.
        candidates = []
        for _, a in alerts.iterrows():
            if a["already_low_at_alert_start"]:
                continue
            possible = events[
                (events["patient_uid"] == a["patient_uid"])
                & (events["segment_id"] == a["segment_id"])
                & (events["event_start"] > a["alert_start"])
                & (
                    events["event_start"]
                    <= a["alert_start"] + pd.Timedelta(minutes=HORIZON_MIN)
                )
            ]
            for _, e in possible.iterrows():
                candidates.append({
                    "model": model.upper(),
                    "alert_id": a["alert_id"],
                    "event_id": e["event_id"],
                    "patient_uid": a["patient_uid"],
                    "alert_start": a["alert_start"],
                    "event_start": e["event_start"],
                    "lead_minutes": (
                        e["event_start"] - a["alert_start"]
                    ).total_seconds() / 60,
                })

        # Greedy one-to-one matching: shortest positive lead first.
        # Each alert and each low event can be matched at most once per model.
        cand = pd.DataFrame(candidates)
        chosen = []
        used_alerts = set()
        used_events = set()
        if not cand.empty:
            cand = cand.sort_values(
                ["lead_minutes", "alert_start", "event_start"]
            )
            for _, c in cand.iterrows():
                if c["alert_id"] in used_alerts or c["event_id"] in used_events:
                    continue
                chosen.append(c.to_dict())
                used_alerts.add(c["alert_id"])
                used_events.add(c["event_id"])

        matched = pd.DataFrame(chosen)
        alerts["matched_event_id"] = ""
        alerts["alert_status"] = np.where(
            alerts["already_low_at_alert_start"],
            "late_alert_currently_low",
            "unmatched_alert"
        )

        if not matched.empty:
            mapping = matched.set_index("alert_id")["event_id"].to_dict()
            alerts["matched_event_id"] = alerts["alert_id"].map(mapping).fillna("")
            alerts.loc[
                alerts["matched_event_id"].ne(""), "alert_status"
            ] = "matched_advance_alert"

        for _, a in alerts.iterrows():
            all_alerts.append(a.to_dict())

        for _, e in events.iterrows():
            m = matched[matched["event_id"] == e["event_id"]] if not matched.empty else pd.DataFrame()
            matches.append({
                **e.to_dict(),
                "model": model.upper(),
                "advance_alert_matched": not m.empty,
                "matched_alert_id": m.iloc[0]["alert_id"] if not m.empty else "",
                "lead_minutes": m.iloc[0]["lead_minutes"] if not m.empty else np.nan,
            })

        for patient in FOCUS:
            pa = alerts[alerts["patient_uid"] == patient]
            pe = events[events["patient_uid"] == patient]
            pm = matched[matched["patient_uid"] == patient] if not matched.empty else pd.DataFrame()
            late = int((pa["alert_status"] == "late_alert_currently_low").sum())
            unmatched = int((pa["alert_status"] == "unmatched_alert").sum())
            leads = pm["lead_minutes"] if not pm.empty else pd.Series(dtype=float)

            summaries.append({
                "patient_uid": patient,
                "model": model.upper(),
                "threshold": THRESHOLD,
                "actual_low_episodes": len(pe),
                "distinct_alert_episodes": len(pa[pa["patient_uid"] == patient]),
                "one_to_one_advance_matches": len(pm),
                "events_without_matched_advance_alert": len(pe) - len(pm),
                "median_lead_minutes": leads.median() if len(leads) else np.nan,
                "late_alert_episodes": late,
                "unmatched_alert_episodes": unmatched,
            })

    alert_df = pd.DataFrame(all_alerts)
    match_df = pd.DataFrame(matches)
    summary_df = pd.DataFrame(summaries)

    alert_df.to_csv(ALERTS_OUT, index=False)
    match_df.to_csv(MATCH_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("=" * 70)
    print("CORRECTED DISTINCT-ALERT AUDIT (EXPLORATORY)")
    print("=" * 70)
    print(summary_df.round(3).to_string(index=False))
    print("\nSaved:")
    print(ALERTS_OUT)
    print(MATCH_OUT)
    print(SUMMARY_OUT)
    print("\nRead-only analysis: no models trained or modified.")

if __name__ == "__main__":
    main()
