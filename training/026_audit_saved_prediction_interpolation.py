"""
GlucoTwin - Saved prediction interpolation eligibility audit
-------------------------------------------------------------
Read-only check of the existing saved test predictions against the processed
dataset. Checks recent history interpolation and future target interpolation
within patient/segment boundaries.

This is a diagnostic audit, not a complete proof of causal validity.
"""

from pathlib import Path
import numpy as np
import pandas as pd

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
HORIZON_STEPS = {30: 2, 60: 4, 90: 6, 120: 8}
HISTORY_STEPS = 96


def main():
    print("=" * 78)
    print("GLUCOTWIN - INTERPOLATION ELIGIBILITY AUDIT")
    print("=" * 78)

    for path in (DATA_FILE, PREDICTIONS_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found:\n{path}")
        print("FOUND:", path)

    df = pd.read_csv(DATA_FILE)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(
        ["patient_uid", "segment_id", "dt"]
    ).reset_index(drop=True)
    df["split"] = df["split"].astype(str).str.lower()

    if "cgm_interpolated" not in df.columns:
        raise ValueError("Dataset has no cgm_interpolated feature.")

    # The processed feature is documented as a binary/unscaled flag.
    interp = pd.to_numeric(df["cgm_interpolated"], errors="coerce").fillna(0)
    # Be defensive if the flag is represented as text or numeric.
    df["_interp_flag"] = (interp > 0.5).astype(int)

    saved = pd.read_csv(PREDICTIONS_FILE)
    required = {
        "_row_index", "split", "horizon_minutes", "prediction_mgdl"
    }
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(
            f"Saved predictions CSV lacks columns: {sorted(missing)}"
        )

    saved["split"] = saved["split"].astype(str).str.lower()
    saved["_row_index"] = pd.to_numeric(saved["_row_index"], errors="coerce")
    saved["horizon_minutes"] = pd.to_numeric(
        saved["horizon_minutes"], errors="coerce"
    )
    saved = saved.dropna(subset=["_row_index", "horizon_minutes"]).copy()
    saved["_row_index"] = saved["_row_index"].astype(int)
    saved["horizon_minutes"] = saved["horizon_minutes"].astype(int)
    saved = saved[saved["split"] == "test"].copy()

    print("\nDataset rows:", len(df))
    print("Saved test prediction rows:", len(saved))
    print("History rule being checked:", HISTORY_STEPS, "rows")
    print("Horizon target steps:", HORIZON_STEPS)

    out_of_bounds = (
        (saved["_row_index"] < 0)
        | (saved["_row_index"] >= len(df))
    )
    print("Out-of-bounds saved row indices:", int(out_of_bounds.sum()))
    saved = saved.loc[~out_of_bounds].copy()

    # Check row alignment first.
    selected_rows = df.iloc[saved["_row_index"].to_numpy()]
    split_mismatch = (
        selected_rows["split"].to_numpy() != saved["split"].to_numpy()
    )
    print("Saved-index split mismatches:", int(split_mismatch.sum()))
    if split_mismatch.any():
        print(
            "\nWARNING: Row indices do not align with the sorted processed dataset. "
            "Do not interpret history/target results until ordering is reconciled."
        )
        return

    # Build positions within each patient/segment, respecting boundaries.
    df["_position_in_segment"] = df.groupby(
        ["patient_uid", "segment_id"], dropna=False
    ).cumcount()

    # For each row, count interpolated values among current row and previous
    # 95 rows in the same segment. Do not cross patient/segment boundaries.
    df["_interp_in_recent_history"] = (
        df.groupby(["patient_uid", "segment_id"], dropna=False)["_interp_flag"]
        .transform(lambda s: s.rolling(HISTORY_STEPS, min_periods=1).sum())
    )

    history_bad = 0
    target_bad = 0
    boundary_short = 0
    checked = 0
    target_checked = 0
    per_horizon = []

    for horizon, steps in HORIZON_STEPS.items():
        part = saved[saved["horizon_minutes"] == horizon].copy()
        if part.empty:
            per_horizon.append({
                "horizon": horizon,
                "rows": 0,
                "history_bad": np.nan,
                "target_interpolated": np.nan,
                "target_unavailable": np.nan,
            })
            continue

        row_indices = part["_row_index"].to_numpy()
        current = df.iloc[row_indices]
        h_bad = int(
            (current["_interp_in_recent_history"].to_numpy() > 0).sum()
        )
        history_bad += h_bad
        checked += len(part)

        # Target row is horizon steps ahead in the same segment.
        target_interpolated = 0
        target_unavailable = 0
        for idx in row_indices:
            row = df.iloc[int(idx)]
            same_segment = df[
                (df["patient_uid"] == row["patient_uid"])
                & (df["segment_id"] == row["segment_id"])
            ]
            pos = int(row["_position_in_segment"])
            target_pos = pos + steps
            if target_pos >= len(same_segment):
                target_unavailable += 1
                continue

            target_row = same_segment.iloc[target_pos]
            # Verify target timestamp is exactly the requested 15-minute step.
            expected_dt = row["dt"] + pd.Timedelta(minutes=horizon)
            if pd.isna(target_row["dt"]) or target_row["dt"] != expected_dt:
                target_unavailable += 1
                continue

            target_checked += 1
            if int(target_row["_interp_flag"]) == 1:
                target_interpolated += 1

        target_bad += target_interpolated
        per_horizon.append({
            "horizon": horizon,
            "rows": len(part),
            "history_bad": h_bad,
            "target_interpolated": target_interpolated,
            "target_unavailable": target_unavailable,
        })

    print("\nPer-horizon results")
    print("-" * 78)
    print(
        f"{'Horizon':<12}{'Rows':>10}{'History bad':>16}"
        f"{'Target interp.':>18}{'Target unavailable':>22}"
    )
    for row in per_horizon:
        print(
            f"+{row['horizon']} min".ljust(12)
            + f"{row['rows']:>10}"
            + f"{row['history_bad']:>16}"
            + f"{row['target_interpolated']:>18}"
            + f"{row['target_unavailable']:>22}"
        )

    print("\nSummary")
    print("-" * 78)
    print("Rows checked for history:", checked)
    print("Rows with interpolation in current/prior history window:", history_bad)
    print("Targets checked at exact horizon timestamp:", target_checked)
    print("Interpolated target rows:", target_bad)
    print("Interpretation:")
    if history_bad == 0 and target_bad == 0:
        print(
            "PASS for these two interpolation checks on saved test predictions. "
            "This does not prove every feature is causal."
        )
    else:
        print(
            "REVIEW REQUIRED: at least one saved prediction row appears to have "
            "interpolation in its recent history or target. Confirm training row "
            "ordering, filter semantics, and exact target-selection logic."
        )

    print(
        "\nLimitations: this checks the processed dataset and saved row indices. "
        "It does not audit event timestamps, every derived feature, model selection, "
        "or external validity."
    )


if __name__ == "__main__":
    main()
