"""
GlucoTwin v3 raw-to-grid event audit (READ ONLY)
Place beside scripts 044/046/047 in training, then run:
python -u 048_audit_event_causality_v3.py

Reads raw source CSVs and v3 clean CSV. Does not modify datasets or models.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
RAW_DIR = DATASET_ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
CLEAN_PATH = SCRIPT_DIR / "results_causal" / "preprocessing_v3" / "cgm_t1dm_t2dm_causal_clean.csv"
OUT_PATH = SCRIPT_DIR / "results_causal" / "preprocessing_v3" / "048_event_causality_audit.json"
GRID_MINUTES = 15
MAX_GAP_MINUTES = 45
NUM = r"(\d+(?:\.\d+)?)"
UNAVAILABLE = re.compile(r"data\s+not\s+available", re.I)

def find_col(df, names):
    m = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None

def sum_num(value, pattern):
    if pd.isna(value):
        return np.nan
    matches = re.findall(pattern, str(value), flags=re.I)
    if not matches:
        return np.nan
    total = 0.0
    for item in matches:
        if isinstance(item, tuple):
            item = next((x for x in item if x), "")
        try:
            total += float(item)
        except (ValueError, TypeError):
            pass
    return total

def load_raw():
    parts = []
    for fname, dtype in [("t1dm.csv", "T1DM"), ("t2dm.csv", "T2DM")]:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Raw file not found: {path}")
        raw = pd.read_csv(path, low_memory=False)
        date = find_col(raw, ["Date", "dt", "datetime", "timestamp"])
        pid = find_col(raw, ["patient_id", "Patient ID", "Patient_ID"])
        cgm = find_col(raw, ["CGM (mg / dl)", "cgm"])
        if not all([date, pid, cgm]):
            raise ValueError(f"Required columns missing in {path}")
        diet = find_col(raw, ["Dietary intake", "diet_text"])
        sc = find_col(raw, ["Insulin dose - s.c.", "sc_insulin_text"])
        bolus = find_col(raw, ["CSII - bolus insulin (Novolin R, IU)", "csii_bolus_raw"])
        basal = find_col(raw, ["CSII - basal insulin (Novolin R, IU / H)", "csii_basal_raw"])
        iv = find_col(raw, ["Insulin dose - i.v.", "iv_insulin_text"])
        part = pd.DataFrame({
            "patient_uid": dtype + "_" + raw[pid].astype("string").str.strip().fillna(""),
            "dt": pd.to_datetime(raw[date], errors="coerce"),
            "cgm": pd.to_numeric(raw[cgm], errors="coerce"),
            "diet": raw[diet].astype("string") if diet else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "sc": raw[sc].astype("string") if sc else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "bolus": raw[bolus] if bolus else pd.Series(np.nan, index=raw.index),
            "basal": raw[basal] if basal else pd.Series(np.nan, index=raw.index),
            "iv": raw[iv].astype("string") if iv else pd.Series(pd.NA, index=raw.index, dtype="string"),
        })
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df = df.loc[df.dt.notna() & df.cgm.notna() & df.patient_uid.notna()].copy()
    df = df.sort_values(["patient_uid", "dt"]).drop_duplicates(["patient_uid", "dt"], keep="first")
    df = df.reset_index(drop=True)
    df["gap_minutes"] = df.groupby("patient_uid").dt.diff().dt.total_seconds() / 60
    df["new_segment"] = df.gap_minutes.isna() | (df.gap_minutes > MAX_GAP_MINUTES)
    df["segment_id"] = df.groupby("patient_uid").new_segment.cumsum().astype(int)
    diet = df.diet.astype("string").str.strip()
    df["meal_unknown"] = diet.str.contains(UNAVAILABLE, na=False).astype(int)
    has_text = diet.notna() & diet.ne("") & diet.str.lower().ne("nan")
    df["meal_logged"] = (has_text & df.meal_unknown.eq(0)).astype(int)
    df["meal_grams"] = diet.map(lambda x: sum_num(x, NUM + r"\s*g\b"))
    df.loc[df.meal_unknown.eq(1), "meal_grams"] = np.nan
    bolus_text = df.bolus.astype("string")
    df["bolus_num"] = pd.to_numeric(df.bolus, errors="coerce")
    df["sc_units"] = df.sc.map(lambda x: sum_num(x, NUM + r"\s*IU"))
    df["sc_long_acting"] = df.sc.astype("string").str.contains(r"glargine|detemir|degludec", case=False, na=False).astype(int)
    df["iv_units"] = df.iv.map(lambda x: sum_num(x, NUM + r"\s*IU"))
    df["fast_insulin"] = (
        df.bolus_num.fillna(0) +
        np.where(df.sc_long_acting.eq(0), df.sc_units.fillna(0), 0.0) +
        df.iv_units.fillna(0)
    ).clip(lower=0)
    df["event_present"] = (
        df.meal_logged.eq(1) | df.meal_unknown.eq(1) |
        df.fast_insulin.gt(0) |
        bolus_text.str.contains("suspend", case=False, na=False) |
        df.basal.astype("string").str.contains("suspend", case=False, na=False)
    )
    return df

def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"v3 clean CSV not found: {CLEAN_PATH}")
    raw = load_raw()
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["dt"], low_memory=False)
    clean_keys = set(zip(clean.patient_uid, clean.segment_id.astype(int), clean.dt))
    checked = 0
    timestamp_violations = []
    missing_slots = []
    expected_events = 0
    event_slot_count = 0
    meal_flag_mismatch = 0
    meal_grams_mismatch = 0
    fast_insulin_mismatch = 0
    event_rows = raw.loc[raw.event_present].copy()
    for (pid, seg), g in raw.groupby(["patient_uid", "segment_id"], sort=False):
        g = g.sort_values("dt")
        start = g.dt.iloc[0]
        elapsed = (g.dt - start).dt.total_seconds()
        k = np.ceil(elapsed / (GRID_MINUTES * 60) - 1e-12).astype(int)
        g = g.assign(k=k)
        for row in g.loc[g.event_present].itertuples(index=False):
            expected_events += 1
            slot_dt = start + pd.Timedelta(minutes=int(row.k) * GRID_MINUTES)
            checked += 1
            if slot_dt < row.dt:
                timestamp_violations.append({
                    "patient_uid": pid, "segment_id": int(seg),
                    "source_dt": str(row.dt), "assigned_slot_dt": str(slot_dt)
                })
            if (pid, int(seg), slot_dt) not in clean_keys:
                missing_slots.append({
                    "patient_uid": pid, "segment_id": int(seg),
                    "source_dt": str(row.dt), "expected_slot_dt": str(slot_dt)
                })

    # Independently compare aggregated meal/insulin signals at slots against the clean file.
    # Values use same ceiling mapping and event parsing as builder.
    raw["slot_k"] = 0
    for (pid, seg), g in raw.groupby(["patient_uid", "segment_id"], sort=False):
        start = g.dt.iloc[0]
        elapsed = (g.dt - start).dt.total_seconds()
        raw.loc[g.index, "slot_k"] = np.ceil(elapsed / (GRID_MINUTES * 60) - 1e-12).astype(int)
        raw.loc[g.index, "slot_dt"] = (start + pd.to_timedelta(raw.loc[g.index, "slot_k"] * GRID_MINUTES, unit="m")).values
    raw["slot_dt"] = pd.to_datetime(raw["slot_dt"])
    agg = raw.groupby(["patient_uid", "segment_id", "slot_dt"], as_index=False).agg(
        exp_meal_logged=("meal_logged", "max"),
        exp_meal_unknown=("meal_unknown", "max"),
        exp_meal_grams=("meal_grams", lambda x: x.sum(min_count=1)),
        exp_fast_insulin=("fast_insulin", "sum")
    )
    merged = agg.merge(
        clean[["patient_uid", "segment_id", "dt", "meal_logged", "meal_unknown", "meal_grams", "fast_insulin"]]
            if "fast_insulin" in clean.columns else clean.assign(fast_insulin=np.nan)[["patient_uid","segment_id","dt","meal_logged","meal_unknown","meal_grams","fast_insulin"]],
        left_on=["patient_uid", "segment_id", "slot_dt"], right_on=["patient_uid", "segment_id", "dt"],
        how="left", suffixes=("_expected", "_actual")
    )
    meal_flag_mismatch = int(((merged.exp_meal_logged.fillna(0) != merged.meal_logged.fillna(0)) |
                              (merged.exp_meal_unknown.fillna(0) != merged.meal_unknown.fillna(0))).sum())
    grams = merged[["exp_meal_grams", "meal_grams"]].dropna()
    meal_grams_mismatch = int((~np.isclose(grams.exp_meal_grams, grams.meal_grams, atol=1e-5)).sum())
    ins = merged[["exp_fast_insulin", "fast_insulin"]].dropna()
    fast_insulin_mismatch = int((~np.isclose(ins.exp_fast_insulin, ins.fast_insulin, atol=1e-5)).sum())

    report = {
        "raw_valid_rows": int(len(raw)),
        "v3_clean_rows": int(len(clean)),
        "raw_rows_with_event_signal": int(raw.event_present.sum()),
        "event_rows_checked_for_no_backward_assignment": checked,
        "event_timestamp_violations": len(timestamp_violations),
        "event_expected_slots_missing_from_clean_grid": len(missing_slots),
        "meal_flag_slot_mismatches": meal_flag_mismatch,
        "meal_grams_slot_mismatches": meal_grams_mismatch,
        "fast_insulin_slot_mismatches": fast_insulin_mismatch,
        "timestamp_violation_examples": timestamp_violations[:10],
        "missing_slot_examples": missing_slots[:10],
        "limitations": [
            "Read-only audit; does not modify dataset or models.",
            "Checks event slot timing and selected raw-to-grid aggregates, not clinical validity.",
            "Basal-state semantics and every model feature still need separate code review."
        ]
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=" * 70)
    print("GLUCOTWIN - V3 RAW-TO-GRID EVENT CAUSALITY AUDIT")
    print("=" * 70)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"Report saved: {OUT_PATH}")
    print("Read-only: no dataset or model was modified.")

if __name__ == "__main__":
    main()
