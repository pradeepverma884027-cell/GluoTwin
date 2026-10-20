"""
GlucoTwin v3 target timestamp audit (READ ONLY)
Place in training beside scripts 046-048 and run:
python -u 049_audit_target_timestamps_v3.py
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
OUT_PATH = ROOT / "049_target_timestamp_audit.json"
GRID_MINUTES = 15
MAX_GAP_MINUTES = 45

def find_col(df, names):
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in mapping:
            return mapping[n.lower()]
    return None

def load_raw():
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
            raise ValueError(f"Required columns missing in {path}")
        part = pd.DataFrame({
            "patient_uid": dtype + "_" + raw[pid_col].astype("string").str.strip().fillna(""),
            "source_dt": pd.to_datetime(raw[date_col], errors="coerce"),
            "cgm": pd.to_numeric(raw[cgm_col], errors="coerce")
        })
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df = df.loc[df.source_dt.notna() & df.cgm.notna() & df.patient_uid.notna()].copy()
    df = df.sort_values(["patient_uid", "source_dt"]).drop_duplicates(["patient_uid", "source_dt"], keep="first")
    df = df.reset_index(drop=True)
    gap = df.groupby("patient_uid")["source_dt"].diff().dt.total_seconds() / 60
    df["new_segment"] = gap.isna() | (gap > MAX_GAP_MINUTES)
    df["segment_id"] = df.groupby("patient_uid")["new_segment"].cumsum().astype(int)
    # Recreate the v3 ceiling assignment; record original measurement timestamp.
    df["slot_k"] = 0
    for (_, _), g in df.groupby(["patient_uid", "segment_id"], sort=False):
        start = g["source_dt"].iloc[0]
        elapsed = (g["source_dt"] - start).dt.total_seconds()
        df.loc[g.index, "slot_k"] = np.ceil(elapsed / (GRID_MINUTES * 60) - 1e-12).astype(int)
        df.loc[g.index, "grid_dt"] = (start + pd.to_timedelta(df.loc[g.index, "slot_k"] * GRID_MINUTES, unit="m")).values
    df["grid_dt"] = pd.to_datetime(df["grid_dt"])
    # v3 builder chooses the last source observation in each slot.
    df = df.sort_values(["patient_uid", "segment_id", "slot_k", "source_dt"])
    selected = df.groupby(["patient_uid", "segment_id", "grid_dt"], as_index=False).tail(1)
    selected = selected[["patient_uid", "segment_id", "grid_dt", "source_dt", "cgm"]]
    return selected

def main():
    for p in (CLEAN_PATH, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required v3 file missing: {p}")
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["dt"], low_memory=False)
    model = pd.read_csv(MODEL_PATH, parse_dates=["dt"], low_memory=False)
    source = load_raw()
    merged = clean[["patient_uid", "segment_id", "dt", "cgm"]].merge(
        source, left_on=["patient_uid", "segment_id", "dt"],
        right_on=["patient_uid", "segment_id", "grid_dt"], how="left",
        suffixes=("_clean", "_source")
    )
    # For model rows, identify actual source observation timestamps for target slots.
    slots = source.set_index(["patient_uid", "segment_id", "grid_dt"])
    checked = {"t30": 0, "t60": 0, "hypo60": 0}
    missing = {"t30": 0, "t60": 0, "hypo60": 0}
    horizon_minutes = {"t30": [], "t60": []}
    target_mismatches = {"t30": 0, "t60": 0, "hypo60": 0}
    examples = []
    for row in model[["patient_uid", "segment_id", "dt", "target_cgm_t30", "target_cgm_t60", "target_hypo_60"]].itertuples(index=False, name=None):
        pid, seg, dt, y30, y60, yh = row
        current_key = (pid, seg, dt)
        if current_key not in slots.index:
            continue
        for name, mins, y in [("t30", 30, y30), ("t60", 60, y60)]:
            key = (pid, seg, dt + pd.Timedelta(minutes=mins))
            if key not in slots.index:
                missing[name] += 1
                continue
            target_row = slots.loc[key]
            if isinstance(target_row, pd.DataFrame):
                target_row = target_row.iloc[-1]
            checked[name] += 1
            effective = (target_row["source_dt"] - dt).total_seconds() / 60
            horizon_minutes[name].append(effective)
            if not np.isclose(float(y), float(target_row["cgm"]), atol=1e-5):
                target_mismatches[name] += 1
            if len(examples) < 12 and (effective < mins or effective > mins + GRID_MINUTES):
                examples.append({
                    "patient_uid": pid, "segment_id": int(seg), "prediction_grid_dt": str(dt),
                    "target": name, "nominal_target_grid_dt": str(key[2]),
                    "actual_target_measurement_dt": str(target_row["source_dt"]),
                    "effective_horizon_minutes": effective
                })
        future = []
        for mins in (15, 30, 45, 60):
            key = (pid, seg, dt + pd.Timedelta(minutes=mins))
            target_row = slots.loc[key] if key in slots.index else None
            if isinstance(target_row, pd.DataFrame):
                target_row = target_row.iloc[-1]
            future.append(np.nan if target_row is None else target_row["cgm"])
        if all(pd.notna(x) for x in future):
            checked["hypo60"] += 1
            if not np.isclose(float(yh), float(min(future) < 70.0)):
                target_mismatches["hypo60"] += 1
        else:
            missing["hypo60"] += 1

    def summary(vals):
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {"min": float(np.min(arr)), "p05": float(np.quantile(arr, .05)),
                "median": float(np.median(arr)), "p95": float(np.quantile(arr, .95)),
                "max": float(np.max(arr)), "count_below_nominal": int(np.sum(arr < (30 if len(vals)==-1 else -1)))}
    report = {
        "clean_rows": int(len(clean)), "model_rows": int(len(model)),
        "source_slot_rows": int(len(source)),
        "target_rows_checked": checked, "target_slots_missing_from_source": missing,
        "target_value_mismatches_against_raw_slot": target_mismatches,
        "effective_horizon_minutes": {
            name: ({"min": float(np.min(vals)), "p05": float(np.quantile(vals, .05)),
                    "median": float(np.median(vals)), "p95": float(np.quantile(vals, .95)),
                    "max": float(np.max(vals)),
                    "below_nominal_count": int(np.sum(np.asarray(vals) < (30 if name=="t30" else 60)))}
                   if vals else None)
            for name, vals in horizon_minutes.items()
        },
        "examples_outside_nominal_horizon_to_plus_15m": examples,
        "limitations": [
            "Read-only audit; does not modify candidate files or models.",
            "Reconstructs source timestamps using the same v3 ceiling slot mapping and last-observation selection.",
            "Checks target observation timing but does not establish clinical validity."
        ]
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=" * 70)
    print("GLUCOTWIN - V3 TARGET TIMESTAMP AUDIT")
    print("=" * 70)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"Report saved: {OUT_PATH}")
    print("Read-only: no dataset or model was modified.")

if __name__ == "__main__":
    main()
