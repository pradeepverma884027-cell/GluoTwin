
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BASE = Path(__file__).resolve().parent
PREP = BASE / "results_causal" / "preprocessing_v3"
OUT = BASE / "results_causal" / "v3_model_comparison"

DATA = PREP / "cgm_t1dm_t2dm_causal_model_ready.csv"
PRED = OUT / "058_focus_test_predictions_full.csv"

ALERTS_OUT = OUT / "062_distinct_alert_episodes.csv"
MATCH_OUT = OUT / "062_event_alert_matching.csv"
SUMMARY_OUT = OUT / "062_earliest_alert_summary.csv"

FOCUS = ["T2DM_2072", "T2DM_2077", "T2DM_2078"]

CGM_MEAN = 144.94967492045924
CGM_STD = 52.47014432357027
LOW = 70.0
THRESHOLD = 0.20
HORIZON_MIN = 60
GRID_MIN = 15


def create_episodes(df, flag_col, prefix, model, probability_col=None):
    """Collapse consecutive flagged rows into distinct episodes."""
    episodes = []

    for (patient, segment), part in df.groupby(
        ["patient_uid", "segment_id"], sort=False
    ):
        part = part.sort_values("dt").copy()

        previous_flag = part[flag_col].shift()
        previous_dt = part["dt"].shift()

        starts = (
            part[flag_col]
            & (
                previous_flag.ne(True)
                | part["dt"].sub(previous_dt)
                .dt.total_seconds().div(60).ne(GRID_MIN)
            )
        )

        part["episode_no"] = starts.cumsum()

        for number, ep in part[part[flag_col]].groupby(
            "episode_no", sort=False
        ):
            record = {
                "patient_uid": patient,
                "segment_id": segment,
                f"{prefix}_id": (
                    f"{patient}_{segment}_{model}_{prefix.upper()}{int(number)}"
                ),
                f"{prefix}_start": ep["dt"].iloc[0],
                f"{prefix}_end": ep["dt"].iloc[-1],
                f"{prefix}_rows": len(ep),
            }

            if probability_col:
                record["max_probability"] = ep[probability_col].max()
                record["cgm_at_alert_start_mgdl"] = (
                    ep["cgm_mgdl_reconstructed"].iloc[0]
                )
                record["already_low_at_alert_start"] = bool(
                    ep["is_low"].iloc[0]
                )

            if prefix == "event":
                record["minimum_cgm_mgdl"] = (
                    ep["cgm_mgdl_reconstructed"].min()
                )

            episodes.append(record)

    return pd.DataFrame(episodes)


def match_alerts_to_events(alerts, events, model):
    """
    Globally maximize one-to-one matches first, then total lead time.
    Only alerts before event onset, within the horizon, and not already
    low at alert onset are eligible.
    """
    alert_col = "alert_start"
    event_col = "event_start"

    candidates = []

    for _, alert in alerts.iterrows():
        if alert["already_low_at_alert_start"]:
            continue

        possible = events[
            (events["patient_uid"] == alert["patient_uid"])
            & (events["segment_id"] == alert["segment_id"])
            & (events[event_col] > alert[alert_col])
            & (
                events[event_col]
                <= alert[alert_col] + pd.Timedelta(minutes=HORIZON_MIN)
            )
        ]

        for _, event in possible.iterrows():
            lead = (
                event[event_col] - alert[alert_col]
            ).total_seconds() / 60

            candidates.append({
                "model": model.upper(),
                "patient_uid": alert["patient_uid"],
                "segment_id": alert["segment_id"],
                "alert_id": alert["alert_id"],
                "event_id": event["event_id"],
                "alert_start": alert[alert_col],
                "event_start": event[event_col],
                "lead_minutes": lead,
            })

    matched_columns = [
        "model", "patient_uid", "segment_id", "alert_id",
        "event_id", "alert_start", "event_start", "lead_minutes"
    ]

    if not candidates:
        return pd.DataFrame(columns=matched_columns)

    cand = pd.DataFrame(candidates)

    # Build a bipartite assignment problem.
    alert_ids = alerts["alert_id"].tolist()
    event_ids = events["event_id"].tolist()

    alert_index = {value: i for i, value in enumerate(alert_ids)}
    event_index = {value: i for i, value in enumerate(event_ids)}

    n_alerts = len(alert_ids)
    n_events = len(event_ids)

    # Dummy columns let alerts remain unmatched.
    # A large reward prioritizes match count; lead breaks ties.
    reward = 1_000_000.0
    impossible = -1_000_000_000.0

    score = np.zeros((n_alerts, n_events + n_alerts), dtype=float)
    score[:, :n_events] = impossible

    for _, candidate in cand.iterrows():
        i = alert_index[candidate["alert_id"]]
        j = event_index[candidate["event_id"]]

        score[i, j] = reward + candidate["lead_minutes"]

    rows, cols = linear_sum_assignment(score, maximize=True)

    chosen = []

    for i, j in zip(rows, cols):
        if j >= n_events or score[i, j] <= 0:
            continue

        candidate = cand[
            (cand["alert_id"] == alert_ids[i])
            & (cand["event_id"] == event_ids[j])
        ]

        if not candidate.empty:
            chosen.append(candidate.iloc[0].to_dict())

    return pd.DataFrame(chosen, columns=matched_columns)


def main():
    for path in [DATA, PRED]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    # Read-only inputs.
    d = pd.read_csv(
        DATA,
        usecols=["patient_uid", "segment_id", "dt", "split", "cgm"],
        parse_dates=["dt"],
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

    probability_columns = keys + ["xgb_probability", "lgb_probability"]

    df = d.merge(
        p[probability_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if df[["xgb_probability", "lgb_probability"]].isna().any().any():
        raise ValueError("Some rows have no matching predictions.")

    df = df.sort_values(keys).reset_index(drop=True)
    df["is_low"] = df["cgm_mgdl_reconstructed"] < LOW

    events = create_episodes(
        df, "is_low", "event", "ACTUAL"
    ).rename(columns={
        "event_id": "event_id",
        "event_start": "event_start",
        "event_end": "event_end",
        "event_rows": "low_rows",
    })

    if events.empty:
        raise ValueError("No actual-low episodes found.")

    all_alerts = []
    all_matches = []
    summaries = []

    for model in ["xgb", "lgb"]:
        probability_col = f"{model}_probability"
        df["alert_flag"] = df[probability_col] >= THRESHOLD

        alerts = create_episodes(
            df,
            "alert_flag",
            "alert",
            model.upper(),
            probability_col,
        )

        if alerts.empty:
            alerts = pd.DataFrame(columns=[
                "patient_uid", "segment_id", "alert_id",
                "alert_start", "alert_end", "alert_rows",
                "max_probability", "cgm_at_alert_start_mgdl",
                "already_low_at_alert_start",
            ])
        else:
            alerts["alert_status"] = np.where(
                alerts["already_low_at_alert_start"],
                "late_alert_currently_low",
                "unmatched_alert",
            )

        matched = match_alerts_to_events(alerts, events, model)

        if not alerts.empty and not matched.empty:
            mapping = matched.set_index("alert_id")["event_id"].to_dict()
            alerts["matched_event_id"] = (
                alerts["alert_id"].map(mapping).fillna("")
            )
            alerts.loc[
                alerts["matched_event_id"].ne(""), "alert_status"
            ] = "matched_advance_alert"

        elif not alerts.empty:
            alerts["matched_event_id"] = ""
            alerts["alert_status"] = np.where(
                alerts["already_low_at_alert_start"],
                "late_alert_currently_low",
                "unmatched_alert",
            )

        if not alerts.empty:
            alerts["model"] = model.upper()
            all_alerts.extend(alerts.to_dict("records"))

        # One output row for every actual low episode and model.
        for _, event in events.iterrows():
            event_match = (
                matched[matched["event_id"] == event["event_id"]]
                if not matched.empty
                else pd.DataFrame()
            )

            all_matches.append({
                **event.to_dict(),
                "model": model.upper(),
                "advance_alert_matched": not event_match.empty,
                "matched_alert_id": (
                    event_match.iloc[0]["alert_id"]
                    if not event_match.empty else ""
                ),
                "matched_alert_start": (
                    event_match.iloc[0]["alert_start"]
                    if not event_match.empty else pd.NaT
                ),
                "lead_minutes": (
                    event_match.iloc[0]["lead_minutes"]
                    if not event_match.empty else np.nan
                ),
            })

        # Per-patient summaries.
        for patient in FOCUS:
            patient_alerts = alerts[
                alerts["patient_uid"] == patient
            ]
            patient_events = events[
                events["patient_uid"] == patient
            ]

            patient_matches = (
                matched[matched["patient_uid"] == patient]
                if not matched.empty
                else pd.DataFrame()
            )

            leads = (
                patient_matches["lead_minutes"]
                if not patient_matches.empty
                else pd.Series(dtype=float)
            )

            late_count = (
                int((patient_alerts["alert_status"] ==
                     "late_alert_currently_low").sum())
                if not patient_alerts.empty else 0
            )

            unmatched_count = (
                int((patient_alerts["alert_status"] ==
                     "unmatched_alert").sum())
                if not patient_alerts.empty else 0
            )

            summaries.append({
                "patient_uid": patient,
                "model": model.upper(),
                "threshold": THRESHOLD,
                "horizon_minutes": HORIZON_MIN,
                "actual_low_episodes": len(patient_events),
                "distinct_alert_episodes": len(patient_alerts),
                "one_to_one_advance_matches": len(patient_matches),
                "events_without_matched_advance_alert": (
                    len(patient_events) - len(patient_matches)
                ),
                "median_lead_minutes": (
                    leads.median() if len(leads) else np.nan
                ),
                "maximum_lead_minutes": (
                    leads.max() if len(leads) else np.nan
                ),
                "late_alert_episodes": late_count,
                "unmatched_alert_episodes": unmatched_count,
            })

    alert_df = pd.DataFrame(all_alerts)
    match_df = pd.DataFrame(all_matches)
    summary_df = pd.DataFrame(summaries)

    # Write only new 062 output files.
    alert_df.to_csv(ALERTS_OUT, index=False)
    match_df.to_csv(MATCH_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("=" * 76)
    print("EARLIEST-ALERT ONE-TO-ONE AUDIT (EXPLORATORY)")
    print("=" * 76)
    print(summary_df.round(3).to_string(index=False))

    print("\nSaved:")
    print(ALERTS_OUT)
    print(MATCH_OUT)
    print(SUMMARY_OUT)

    print("\nRead-only analysis: no models trained or modified.")
    print("Note: CGM values are reconstructed from standardized features.")
    print("Results are exploratory, not clinical validation.")


if __name__ == "__main__":
    main()
