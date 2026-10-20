"""
GlucoTwin v3 hypoglycemia-label and episode audit (READ ONLY)

Place this script in:
  DATASET/training/

Run:
  python -u 056_audit_hypo_label_episodes_v3.py

Uses the audited v3 clean/model-ready CSVs and raw CGM files.
Does NOT train models or modify datasets/models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
RAW_DIR = DATASET_ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
ROOT = SCRIPT_DIR / "results_causal" / "preprocessing_v3"
CLEAN_PATH = ROOT / "cgm_t1dm_t2dm_causal_clean.csv"
MODEL_PATH = ROOT / "cgm_t1dm_t2dm_causal_model_ready.csv"
OUT_DIR = SCRIPT_DIR / "results_causal" / "v3_model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOCUS_PATIENTS = ["T2DM_2077", "T2DM_2072", "T2DM_2078"]
GRID_MINUTES = 15
MAX_GAP_MINUTES = 45
HYPO_THRESHOLD = 70.0
FUTURE_MINUTES = [15, 30, 45, 60]


def find_col(df, names):
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.strip().lower() in mapping:
            return mapping[name.strip().lower()]
    return None


def load_raw_slots():
    parts = []
    for filename, dtype in [("t1dm.csv", "T1DM"), ("t2dm.csv", "T2DM")]:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Raw file not found: {path}")
        raw = pd.read_csv(path, low_memory=False)
        date_col = find_col(raw, ["Date", "dt", "datetime", "timestamp"])
        pid_col = find_col(raw, ["patient_id", "Patient ID", "Patient_ID"])
        cgm_col = find_col(raw, ["CGM (mg / dl)", "cgm"])
        if not all([date_col, pid_col, cgm_col]):
            raise ValueError(
                f"Required columns not found in {path}. "
                f"Columns={list(raw.columns)}"
            )
        parts.append(pd.DataFrame({
            "patient_uid": dtype + "_" + raw[pid_col].astype("string").str.strip().fillna(""),
            "source_dt": pd.to_datetime(raw[date_col], errors="coerce"),
            "cgm": pd.to_numeric(raw[cgm_col], errors="coerce"),
        }))

    raw = pd.concat(parts, ignore_index=True)
    raw = raw.loc[raw["source_dt"].notna() & raw["cgm"].notna()].copy()
    raw = raw.sort_values(["patient_uid", "source_dt"])
    raw = raw.drop_duplicates(["patient_uid", "source_dt"], keep="first").reset_index(drop=True)

    gap = raw.groupby("patient_uid")["source_dt"].diff().dt.total_seconds() / 60
    raw["new_segment"] = gap.isna() | (gap > MAX_GAP_MINUTES)
    raw["segment_id"] = raw.groupby("patient_uid")["new_segment"].cumsum().astype(int)
    raw["slot_k"] = 0
    raw["grid_dt"] = pd.NaT

    for (_, _), group in raw.groupby(["patient_uid", "segment_id"], sort=False):
        start = group["source_dt"].iloc[0]
        elapsed_seconds = (group["source_dt"] - start).dt.total_seconds()
        slot_k = np.ceil(elapsed_seconds / (GRID_MINUTES * 60) - 1e-12).astype(int)
        raw.loc[group.index, "slot_k"] = slot_k
        raw.loc[group.index, "grid_dt"] = (
            start + pd.to_timedelta(slot_k * GRID_MINUTES, unit="m")
        ).values

    raw["grid_dt"] = pd.to_datetime(raw["grid_dt"])
    raw = raw.sort_values(["patient_uid", "segment_id", "slot_k", "source_dt"])
    slots = raw.groupby(
        ["patient_uid", "segment_id", "grid_dt"], as_index=False
    ).tail(1)
    return slots[["patient_uid", "segment_id", "grid_dt", "source_dt", "cgm"]].copy()


def episodes_for_patient(frame, patient_uid, label_col, value_col=None):
    g = frame.loc[frame["patient_uid"].astype(str) == patient_uid].copy()
    if g.empty:
        return []
    g = g.sort_values(["segment_id", "dt"]).reset_index(drop=True)
    episodes = []
    active = None
    prev_seg = None
    prev_dt = None

    for row in g.itertuples(index=False):
        rowd = row._asdict()
        label = int(rowd[label_col]) if pd.notna(rowd[label_col]) else 0
        dt = rowd["dt"]
        seg = rowd["segment_id"]
        contiguous = (
            active is not None and seg == prev_seg and
            (dt - prev_dt).total_seconds() / 60 == GRID_MINUTES
        )
        if label == 1:
            if not contiguous:
                if active is not None:
                    episodes.append(active)
                active = {
                    "patient_uid": patient_uid,
                    "segment_id": int(seg),
                    "start_dt": str(dt),
                    "end_dt": str(dt),
                    "positive_slots": 1,
                    "duration_minutes_between_first_last_samples": 0,
                    "min_cgm_in_episode": float(rowd[value_col]) if value_col and pd.notna(rowd[value_col]) else None,
                }
            else:
                active["end_dt"] = str(dt)
                active["positive_slots"] += 1
                active["duration_minutes_between_first_last_samples"] = (
                    active["positive_slots"] - 1
                ) * GRID_MINUTES
                if value_col and pd.notna(rowd[value_col]):
                    active["min_cgm_in_episode"] = min(
                        active["min_cgm_in_episode"], float(rowd[value_col])
                    )
        else:
            if active is not None:
                episodes.append(active)
                active = None
        prev_seg, prev_dt = seg, dt

    if active is not None:
        episodes.append(active)
    return episodes


def main():
    for path in (CLEAN_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required v3 file missing: {path}")

    clean = pd.read_csv(CLEAN_PATH, parse_dates=["dt"], low_memory=False)
    model = pd.read_csv(MODEL_PATH, parse_dates=["dt"], low_memory=False)
    slots = load_raw_slots()

    required_clean = {"patient_uid", "segment_id", "dt", "cgm"}
    required_model = {"patient_uid", "segment_id", "dt", "target_hypo_60"}
    if not required_clean.issubset(clean.columns):
        raise ValueError(f"Clean CSV missing columns: {sorted(required_clean - set(clean.columns))}")
    if not required_model.issubset(model.columns):
        raise ValueError(f"Model CSV missing columns: {sorted(required_model - set(model.columns))}")

    # Independently reconstruct each model row's label from the next four grid CGM values.
    slot_lookup = slots.set_index(["patient_uid", "segment_id", "grid_dt"])["cgm"]
    recon = []
    for pid, seg, dt, label in model[
        ["patient_uid", "segment_id", "dt", "target_hypo_60"]
    ].itertuples(index=False, name=None):
        future = []
        for mins in FUTURE_MINUTES:
            key = (pid, seg, dt + pd.Timedelta(minutes=mins))
            future.append(slot_lookup.get(key, np.nan))
        complete = all(pd.notna(v) for v in future)
        reconstructed = int(min(future) < HYPO_THRESHOLD) if complete else np.nan
        recon.append({
            "patient_uid": pid,
            "segment_id": seg,
            "dt": dt,
            "stored_target_hypo_60": label,
            "reconstructed_target_hypo_60": reconstructed,
            "all_four_future_slots_available": complete,
            "future_min_cgm": float(min(future)) if complete else np.nan,
        })
    label_check = pd.DataFrame(recon)
    label_check["label_mismatch"] = (
        label_check["all_four_future_slots_available"] &
        (label_check["stored_target_hypo_60"].astype(float) !=
         label_check["reconstructed_target_hypo_60"].astype(float))
    )

    summary_rows = []
    episode_rows = []
    for pid in sorted(set(FOCUS_PATIENTS) | set(model["patient_uid"].astype(str))):
        subset = label_check.loc[label_check["patient_uid"].astype(str) == pid]
        if subset.empty:
            continue
        summary_rows.append({
            "patient_uid": pid,
            "model_rows": int(len(subset)),
            "positive_labels": int((subset["stored_target_hypo_60"] == 1).sum()),
            "label_prevalence": float(subset["stored_target_hypo_60"].mean()),
            "complete_future_windows": int(subset["all_four_future_slots_available"].sum()),
            "label_mismatches": int(subset["label_mismatch"].sum()),
            "mean_future_min_cgm": float(subset["future_min_cgm"].mean()),
            "p05_future_min_cgm": float(subset["future_min_cgm"].quantile(.05)),
            "min_future_min_cgm": float(subset["future_min_cgm"].min()),
        })

    # Episodes of actual low CGM on the clean grid; this differs from target label episodes.
    clean["actual_cgm_below_70"] = (pd.to_numeric(clean["cgm"], errors="coerce") < HYPO_THRESHOLD).astype(int)
    for pid in FOCUS_PATIENTS:
        actual_eps = episodes_for_patient(clean, pid, "actual_cgm_below_70", "cgm")
        for ep in actual_eps:
            ep["episode_type"] = "actual_cgm_below_70_on_grid"
            episode_rows.append(ep)

        label_frame = model[["patient_uid", "segment_id", "dt", "target_hypo_60"]].copy()
        label_frame["target_hypo_60"] = pd.to_numeric(label_frame["target_hypo_60"], errors="coerce").fillna(0).astype(int)
        label_eps = episodes_for_patient(label_frame, pid, "target_hypo_60", None)
        for ep in label_eps:
            ep["episode_type"] = "future_hypo60_target_positive"
            ep["min_cgm_in_episode"] = None
            episode_rows.append(ep)

    summary_df = pd.DataFrame(summary_rows)
    episodes_df = pd.DataFrame(episode_rows)
    mismatch_df = label_check.loc[label_check["label_mismatch"]].copy()

    summary_path = OUT_DIR / "056_hypo_label_patient_summary.csv"
    episodes_path = OUT_DIR / "056_hypo_label_episodes_focus_patients.csv"
    mismatch_path = OUT_DIR / "056_hypo_label_mismatches.csv"
    report_path = OUT_DIR / "056_hypo_label_episode_audit.json"

    summary_df.to_csv(summary_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)
    mismatch_df.to_csv(mismatch_path, index=False)

    focus = summary_df.loc[summary_df["patient_uid"].isin(FOCUS_PATIENTS)]
    report = {
        "audit": "GlucoTwin v3 hypoglycemia label and episode audit",
        "read_only": True,
        "label_definition_reconstructed": (
            "target_hypo_60 = 1 if the minimum of the next 4 grid CGM values "
            "at +15, +30, +45, +60 minutes is below 70 mg/dL; otherwise 0."
        ),
        "hypoglycemia_threshold_mg_dl": HYPO_THRESHOLD,
        "model_rows": int(len(model)),
        "raw_source_slot_rows": int(len(slots)),
        "complete_future_windows": int(label_check["all_four_future_slots_available"].sum()),
        "incomplete_future_windows": int((~label_check["all_four_future_slots_available"]).sum()),
        "stored_vs_reconstructed_label_mismatches": int(label_check["label_mismatch"].sum()),
        "focus_patients": FOCUS_PATIENTS,
        "focus_patient_summary": focus.to_dict(orient="records"),
        "limitations": [
            "This reconstructs the label from the dataset's grid CGM values; it does not validate clinical event definitions.",
            "A positive target means at least one of four future grid readings is below 70 mg/dL, not necessarily a sustained episode.",
            "A sequence of positive target labels can overlap the same low-CGM event and should not be counted as independent events.",
            "Patient-specific prevalence may reflect real differences, data collection patterns, sensor artifacts, or cohort composition; this audit alone cannot distinguish them.",
            "No model was trained and no source dataset or model was modified."
        ],
        "outputs": {
            "patient_summary_csv": str(summary_path),
            "focus_episodes_csv": str(episodes_path),
            "label_mismatches_csv": str(mismatch_path),
            "json_report": str(report_path)
        }
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 72)
    print("GLUCOTWIN - HYPOGLYCEMIA LABEL / EPISODE AUDIT (READ ONLY)")
    print("=" * 72)
    print(f"Model rows: {len(model):,}")
    print(f"Complete future windows: {int(label_check['all_four_future_slots_available'].sum()):,}")
    print(f"Incomplete future windows: {int((~label_check['all_four_future_slots_available']).sum()):,}")
    print(f"Stored-vs-reconstructed label mismatches: {int(label_check['label_mismatch'].sum()):,}")
    print("\nFocus-patient summary:")
    if focus.empty:
        print("No focus-patient rows found.")
    else:
        print(focus.to_string(index=False))
    print("\nOutputs:")
    print(summary_path)
    print(episodes_path)
    print(mismatch_path)
    print(report_path)
    print("\nRead-only: no dataset or model was modified.")


if __name__ == "__main__":
    main()
