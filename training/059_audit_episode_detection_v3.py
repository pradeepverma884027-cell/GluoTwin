from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
INPUT = BASE / "results_causal" / "v3_model_comparison" / "058_focus_test_predictions_full.csv"
OUT_DIR = BASE / "results_causal" / "v3_model_comparison"

EPISODE_OUT = OUT_DIR / "059_target_positive_episode_results.csv"
SUMMARY_OUT = OUT_DIR / "059_episode_detection_summary.csv"

THRESHOLD = 0.20
EXPECTED_GAP = pd.Timedelta(minutes=15)

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT, parse_dates=["dt"])

    required = [
        "patient_uid", "segment_id", "dt", "split", "y_true",
        "xgb_probability", "lgb_probability"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise").astype(int)

    for model in ["xgb", "lgb"]:
        prob_col = f"{model}_probability"
        df[prob_col] = pd.to_numeric(df[prob_col], errors="raise")
        df[f"{model}_pred"] = (df[prob_col] >= THRESHOLD).astype(int)

    df = df.sort_values(["patient_uid", "segment_id", "dt"]).reset_index(drop=True)

    # Start a new run at a label change, segment/patient boundary,
    # or a gap in the 15-minute time grid.
    group_cols = ["patient_uid", "segment_id"]
    previous_label = df.groupby(group_cols)["y_true"].shift()
    previous_dt = df.groupby(group_cols)["dt"].shift()

    new_run = (
        previous_label.isna()
        | df["y_true"].ne(previous_label)
        | df["dt"].sub(previous_dt).ne(EXPECTED_GAP)
    )

    df["run_id"] = new_run.groupby(
        [df["patient_uid"], df["segment_id"]]
    ).cumsum()

    positives = df[df["y_true"] == 1].copy()

    rows = []
    for (patient, segment, run_id), run in positives.groupby(
        ["patient_uid", "segment_id", "run_id"], sort=False
    ):
        run = run.sort_values("dt")
        row = {
            "patient_uid": patient,
            "segment_id": segment,
            "run_id": int(run_id),
            "start_dt": run["dt"].iloc[0],
            "end_dt": run["dt"].iloc[-1],
            "positive_rows": len(run),
            "duration_minutes_between_first_last":
                (run["dt"].iloc[-1] - run["dt"].iloc[0]).total_seconds() / 60,
        }

        for model in ["xgb", "lgb"]:
            pred = run[f"{model}_pred"].to_numpy()
            detected = int(pred.sum())

            if detected == 0:
                status = "fully_missed"
            elif detected == len(pred):
                status = "all_rows_detected"
            else:
                status = "partially_detected"

            row[f"{model}_positive_prediction_rows"] = detected
            row[f"{model}_episode_status"] = status
            row[f"{model}_detected_anywhere"] = detected > 0

        rows.append(row)

    episodes = pd.DataFrame(rows)
    if episodes.empty:
        raise ValueError("No positive-label episodes found.")

    episodes.to_csv(EPISODE_OUT, index=False)

    summary_rows = []
    for patient, part in episodes.groupby("patient_uid"):
        for model in ["xgb", "lgb"]:
            counts = part[f"{model}_episode_status"].value_counts()
            total = len(part)
            detected = int(part[f"{model}_detected_anywhere"].sum())
            missed = int(counts.get("fully_missed", 0))
            partial = int(counts.get("partially_detected", 0))
            all_rows = int(counts.get("all_rows_detected", 0))

            summary_rows.append({
                "patient_uid": patient,
                "model": model.upper(),
                "threshold": THRESHOLD,
                "positive_label_episodes": total,
                "fully_missed_episodes": missed,
                "partially_detected_episodes": partial,
                "all_rows_detected_episodes": all_rows,
                "episodes_with_any_positive_prediction": detected,
                "episode_any_detection_rate": detected / total,
                "fully_missed_episode_rate": missed / total,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("=" * 68)
    print("TARGET-POSITIVE EPISODE AUDIT - READ ONLY")
    print("=" * 68)
    print(f"Input rows: {len(df)}")
    print(f"Positive-label episodes: {len(episodes)}")
    print(f"Threshold: {THRESHOLD}")
    print("\nEpisode detection summary:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved episode details:", EPISODE_OUT)
    print("Saved summary:", SUMMARY_OUT)
    print("\nNo models trained or modified.")

if __name__ == "__main__":
    main()
