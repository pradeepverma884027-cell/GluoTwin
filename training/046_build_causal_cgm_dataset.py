"""
GlucoTwin v2 - strictly causal CGM preprocessing candidate
=========================================================

Reads the Shanghai-style T1DM/T2DM raw CSVs and writes NEW files only to:
  training/results_causal/preprocessing_v2/

It does NOT overwrite the existing clean/model-ready CSV, schema, or models.

Important design choices:
- "data not available" is NOT treated as a logged meal.
- No linear interpolation or backward fill of CGM is used.
- CGM features use current/past values only (right-aligned rolling windows).
- Future CGM is used only to construct prediction targets.
- Patient-level train/val/test split; scaling parameters fit on train only.
- Rows without a complete history or required future targets are excluded.
- Missing CGM grid points remain missing; no synthetic CGM is invented.

Run from any directory:
  python 044_build_causal_cgm_dataset.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
GRID_MINUTES = 15
MAX_SEGMENT_GAP_MINUTES = 45
HYPO_THRESHOLD = 70.0
NUM = r"(\d+(?:\.\d+)?)"
UNAVAILABLE_RE = re.compile(r"data\s+not\s+available", re.IGNORECASE)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
RAW_DIR = DATASET_ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
OUT_DIR = SCRIPT_DIR / "results_causal" / "preprocessing_v3"

RAW_FILES = [("t1dm", "T1DM"), ("t2dm", "T2DM")]

# The 33 feature names/order used by the existing GlucoTwin schema.
LAG_COLS = [f"cgm_lag_{15 * lag}m" for lag in (1, 2, 3, 4, 6, 8)]
CONT_FEATURES = [
    "cgm", *LAG_COLS,
    "cgm_delta_15", "cgm_delta_30", "cgm_delta_60",
    "cgm_roll_mean_60m", "cgm_roll_std_60m", "cgm_roll_min_120m",
    "csii_basal", "fast_insulin_last_4h", "meal_grams_last_2h",
    "mins_since_meal", "mins_since_fast_insulin", "oral_agent_count",
    "cgm_mean_6h", "cgm_std_6h", "cgm_mean_24h",
    "cgm_tir_24h", "cgm_tbr_24h", "iob_proxy", "cob_proxy",
]
BINARY_FEATURES = [
    "csii_suspend", "sc_long_acting", "meal_logged",
    "cgm_interpolated", "is_t2dm", "hour_sin", "hour_cos",
]
FEATURES = CONT_FEATURES + BINARY_FEATURES
TARGETS = ["target_cgm_t30", "target_cgm_t60", "target_hypo_60"]
META = ["patient_uid", "diabetes_type", "segment_id", "dt", "split"]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in mapping:
            return mapping[name.lower()]
    return None


def sum_num(series: pd.Series, pattern: str) -> pd.Series:
    """Sum every matching numeric amount in each text cell."""
    rx = re.compile(pattern, flags=re.IGNORECASE)
    def parse(value):
        if pd.isna(value):
            return np.nan
        text = str(value)
        matches = rx.findall(text)
        if not matches:
            return np.nan
        total = 0.0
        for match in matches:
            if isinstance(match, tuple):
                match = next((x for x in match if x), "")
            try:
                total += float(match)
            except (TypeError, ValueError):
                pass
        return total if matches else np.nan
    return series.map(parse).astype(float)


def split_patient_groups(df: pd.DataFrame) -> pd.Series:
    """Deterministic group split, stratified by diabetes type at patient level."""
    rng = np.random.default_rng(SEED)
    group_type = df.groupby("patient_uid")["diabetes_type"].first()
    assignments = {}
    for dtype in sorted(group_type.unique()):
        ids = group_type.index[group_type.eq(dtype)].to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = max(0, n - n_val - n_test)
        for pid in ids[:n_train]:
            assignments[pid] = "train"
        for pid in ids[n_train:n_train + n_val]:
            assignments[pid] = "val"
        for pid in ids[n_train + n_val:]:
            assignments[pid] = "test"
    return df["patient_uid"].map(assignments)


def read_raw() -> pd.DataFrame:
    parts = []
    for filename, dtype in RAW_FILES:
        path = RAW_DIR / f"{filename}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Raw input not found: {path}")
        raw = pd.read_csv(path, low_memory=False)
        date_col = find_col(raw, ["Date", "dt", "datetime", "timestamp"])
        pid_col = find_col(raw, ["patient_id", "Patient ID", "Patient_ID"])
        cgm_col = find_col(raw, ["CGM (mg / dl)", "cgm"])
        if not all([date_col, pid_col, cgm_col]):
            raise ValueError(f"Missing required Date/patient/CGM columns in {path}. Columns: {list(raw.columns)}")

        def get_series(candidates):
            col = find_col(raw, candidates)
            return raw[col] if col else pd.Series(np.nan, index=raw.index)

        part = pd.DataFrame({
            "diabetes_type": dtype,
            "patient_id": raw[pid_col].astype("string").str.strip(),
            "dt": pd.to_datetime(raw[date_col], errors="coerce"),
            "cgm": pd.to_numeric(raw[cgm_col], errors="coerce"),
            "cbg": pd.to_numeric(get_series(["CBG (mg / dl)", "cbg"]), errors="coerce"),
            "ketone": pd.to_numeric(get_series(["Blood Ketone (mmol / L)", "ketone"]), errors="coerce"),
            "diet_text": get_series(["Dietary intake", "diet_text"]).astype("string"),
            "sc_insulin_text": get_series(["Insulin dose - s.c.", "sc_insulin_text"]).astype("string"),
            "oral_agent_text": get_series(["Non-insulin hypoglycemic agents", "oral_agent_text"]).astype("string"),
            "csii_bolus_raw": get_series(["CSII - bolus insulin (Novolin R, IU)", "csii_bolus_raw"]),
            "csii_basal_raw": get_series(["CSII - basal insulin (Novolin R, IU / H)", "csii_basal_raw"]),
            "iv_insulin_text": get_series(["Insulin dose - i.v.", "iv_insulin_text"]).astype("string"),
        })
        part["patient_uid"] = dtype + "_" + part["patient_id"].fillna("")
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df = df.loc[df["dt"].notna() & df["cgm"].notna() & df["patient_id"].notna()].copy()
    df = df.sort_values(["patient_uid", "dt"]).drop_duplicates(["patient_uid", "dt"], keep="first")
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after parsing timestamps and CGM values.")
    return df


def make_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bolus_text = df["csii_bolus_raw"].astype("string")
    basal_text = df["csii_basal_raw"].astype("string")
    sc_text = df["sc_insulin_text"].astype("string")
    diet_text = df["diet_text"].astype("string").str.strip()

    df["csii_suspend"] = (
        bolus_text.str.contains("suspend", case=False, na=False)
        | basal_text.str.contains("suspend", case=False, na=False)
    ).astype(int)
    df["csii_bolus"] = pd.to_numeric(df["csii_bolus_raw"], errors="coerce")
    df["csii_basal"] = pd.to_numeric(df["csii_basal_raw"], errors="coerce")
    df.loc[basal_text.str.contains("suspend", case=False, na=False), "csii_basal"] = 0.0

    # SC insulin quantity; classify long-acting by medication name.
    df["sc_units"] = sum_num(sc_text, NUM + r"\s*IU")
    df["sc_long_acting"] = sc_text.str.contains(
        r"glargine|detemir|degludec", case=False, na=False
    ).astype(int)
    df["iv_units"] = sum_num(df["iv_insulin_text"], NUM + r"\s*IU")

    oral = df["oral_agent_text"].astype("string").str.strip()
    df["oral_agent_count"] = np.where(
        oral.notna() & oral.ne("") & oral.str.lower().ne("nan"),
        oral.str.count(",").fillna(0) + 1,
        np.nan,
    )

    # Critical correction: "data not available" is not a meal log.
    df["meal_unknown"] = diet_text.str.contains(UNAVAILABLE_RE, na=False).astype(int)
    has_text = diet_text.notna() & diet_text.ne("") & diet_text.str.lower().ne("nan")
    df["meal_logged"] = (has_text & df["meal_unknown"].eq(0)).astype(int)
    df["meal_grams"] = sum_num(diet_text, NUM + r"\s*g\b")
    df.loc[df["meal_unknown"].eq(1), "meal_grams"] = np.nan

    df["fast_insulin"] = (
        pd.to_numeric(df["csii_bolus"], errors="coerce").fillna(0)
        + np.where(df["sc_long_acting"].eq(0), df["sc_units"].fillna(0), 0.0)
        + df["iv_units"].fillna(0)
    )
    df["fast_insulin"] = df["fast_insulin"].clip(lower=0)
    return df


def build_grid_and_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = make_events(raw)
    raw["gap_minutes"] = raw.groupby("patient_uid")["dt"].diff().dt.total_seconds() / 60.0
    raw["new_segment"] = raw["gap_minutes"].isna() | (raw["gap_minutes"] > MAX_SEGMENT_GAP_MINUTES)
    raw["segment_id"] = raw.groupby("patient_uid")["new_segment"].cumsum().astype(int)

    # Segment-anchored 15-minute grid. Assign each observation/event to the
    # first grid timestamp AT OR AFTER its real timestamp (ceiling, never nearest).
    # Multiple records in one interval are aggregated rather than discarded.
    event_cols = [
        "cbg", "ketone", "csii_bolus", "csii_basal", "csii_suspend", "sc_units",
        "sc_long_acting", "iv_units", "oral_agent_count", "meal_grams",
        "meal_logged", "meal_unknown", "fast_insulin",
    ]
    parts = []
    for (pid, seg_id), g in raw.groupby(["patient_uid", "segment_id"], sort=False):
        g = g.sort_values("dt").copy()
        start = g["dt"].iloc[0]
        step_seconds = GRID_MINUTES * 60
        elapsed_seconds = (g["dt"] - start).dt.total_seconds()
        g["k"] = np.ceil(elapsed_seconds / step_seconds - 1e-12).astype(int)

        # Aggregate every source row assigned to a slot. CGM/continuous measurements
        # use the last observation in the slot; event quantities are summed and
        # event flags use max, so simultaneous records are not silently discarded.
        g["_source_order"] = np.arange(len(g))
        g = g.sort_values(["k", "dt", "_source_order"])
        grouped = g.groupby("k", sort=True)
        aggregated = grouped.agg(
            cgm=("cgm", "last"),
            cbg=("cbg", "last"),
            ketone=("ketone", "last"),
            csii_bolus=("csii_bolus", "sum"),
            csii_basal=("csii_basal", "last"),
            csii_suspend=("csii_suspend", "max"),
            sc_units=("sc_units", "sum"),
            sc_long_acting=("sc_long_acting", "max"),
            iv_units=("iv_units", "sum"),
            oral_agent_count=("oral_agent_count", "max"),
            meal_grams=("meal_grams", lambda x: x.sum(min_count=1)),
            meal_logged=("meal_logged", "max"),
            meal_unknown=("meal_unknown", "max"),
            fast_insulin=("fast_insulin", "sum"),
        ).reset_index()

        max_k = int(aggregated["k"].max())
        grid = pd.DataFrame({"k": np.arange(max_k + 1, dtype=int)})
        grid["dt"] = start + pd.to_timedelta(grid["k"] * GRID_MINUTES, unit="m")
        grid = grid.merge(aggregated, on="k", how="left", validate="one_to_one")
        grid["patient_uid"] = pid
        grid["segment_id"] = seg_id
        grid["diabetes_type"] = g["diabetes_type"].iloc[0]
        grid["patient_id"] = g["patient_id"].iloc[0]

        # Event quantities/flags are events at the current or earlier-known slot.
        for col in ["csii_bolus", "csii_suspend", "sc_units", "sc_long_acting", "iv_units",
                    "oral_agent_count", "meal_grams", "meal_logged", "meal_unknown", "fast_insulin"]:
            grid[col] = grid[col].fillna(0.0)
        # Basal rate is a state: carry forward only the last known value.
        grid["csii_basal"] = grid["csii_basal"].ffill().fillna(0.0)
        parts.append(grid)

    df = pd.concat(parts, ignore_index=True).sort_values(["patient_uid", "segment_id", "dt"]).reset_index(drop=True)
    group_keys = ["patient_uid", "segment_id"]
    grouped = df.groupby(group_keys, sort=False, group_keys=False)

    # No interpolation: missing grid CGM remains NaN. Every CGM feature is right-aligned.
    for lag in (1, 2, 3, 4, 6, 8):
        df[f"cgm_lag_{15 * lag}m"] = grouped["cgm"].shift(lag)
    df["cgm_delta_15"] = df["cgm"] - df["cgm_lag_15m"]
    df["cgm_delta_30"] = df["cgm"] - df["cgm_lag_30m"]
    df["cgm_delta_60"] = df["cgm"] - df["cgm_lag_60m"]

    def roll(series_name, window, stat, min_periods):
        return df.groupby(group_keys, sort=False)[series_name].transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).agg(stat)
        )

    df["cgm_roll_mean_60m"] = roll("cgm", 4, "mean", 4)
    df["cgm_roll_std_60m"] = roll("cgm", 4, "std", 4)
    df["cgm_roll_min_120m"] = roll("cgm", 8, "min", 8)

    df["fast_insulin_last_4h"] = roll("fast_insulin", 16, "sum", 1)
    df["meal_grams_last_2h"] = roll("meal_grams", 8, "sum", 1)

    # Recency uses only current/past event rows; cap at 360 min.
    positions = df.groupby(group_keys, sort=False).cumcount()
    for event_col, out_col in [
        ("meal_logged", "mins_since_meal"),
        ("fast_insulin", "mins_since_fast_insulin"),
    ]:
        last_pos = positions.where(df[event_col] > 0).groupby(
            [df["patient_uid"], df["segment_id"]], sort=False
        ).ffill()
        df[out_col] = ((positions - last_pos) * GRID_MINUTES).fillna(360).clip(upper=360)

    tod = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * tod / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * tod / 24.0)
    df["is_t2dm"] = df["diabetes_type"].eq("T2DM").astype(int)
    df["cgm_interpolated"] = 0  # no CGM interpolation is performed

    df["cgm_mean_6h"] = roll("cgm", 24, "mean", 16)
    df["cgm_std_6h"] = roll("cgm", 24, "std", 16)
    df["cgm_mean_24h"] = roll("cgm", 96, "mean", 48)
    in_range = df["cgm"].between(70, 180).astype(float).where(df["cgm"].notna())
    below_range = df["cgm"].lt(70).astype(float).where(df["cgm"].notna())
    df["_in_range"] = in_range
    df["_below_range"] = below_range
    df["cgm_tir_24h"] = roll("_in_range", 96, "mean", 48)
    df["cgm_tbr_24h"] = roll("_below_range", 96, "mean", 48)

    alpha_i = 1 - 0.5 ** (GRID_MINUTES / 90.0)
    alpha_c = 1 - 0.5 ** (GRID_MINUTES / 60.0)
    df["iob_proxy"] = df.groupby(group_keys, sort=False)["fast_insulin"].transform(
        lambda s: s.ewm(alpha=alpha_i, adjust=False).mean() / alpha_i
    )
    df["cob_proxy"] = df.groupby(group_keys, sort=False)["meal_grams"].transform(
        lambda s: s.ewm(alpha=alpha_c, adjust=False).mean() / alpha_c
    )

    # Future shifts are used ONLY for targets, never as features.
    df["target_cgm_t30"] = df.groupby(group_keys, sort=False)["cgm"].shift(-2)
    df["target_cgm_t60"] = df.groupby(group_keys, sort=False)["cgm"].shift(-4)
    future = pd.concat(
        [df.groupby(group_keys, sort=False)["cgm"].shift(-i) for i in range(1, 5)],
        axis=1,
    )
    df["target_hypo_60"] = np.where(
        future.notna().all(axis=1),
        (future.min(axis=1) < HYPO_THRESHOLD).astype(float),
        np.nan,
    )
    df = df.drop(columns=["_in_range", "_below_range"], errors="ignore")
    return df


def main():
    print("=" * 72)
    print("GLUCOTWIN - CAUSAL PREPROCESSING CANDIDATE")
    print("=" * 72)
    print(f"Raw directory: {RAW_DIR}")
    print(f"Output directory (new): {OUT_DIR}")
    print("Existing datasets/models will not be overwritten.")

    raw = read_raw()
    print(f"Raw rows after timestamp/CGM validation: {len(raw):,}")
    df = build_grid_and_features(raw)
    print(f"Rows on causal 15-minute grid: {len(df):,}")

    # Require complete feature history and complete target values.
    required = CONT_FEATURES + TARGETS
    model = df.dropna(subset=required).copy()
    if model.empty:
        raise ValueError("No rows remain after requiring complete causal features and targets.")

    model["split"] = split_patient_groups(model)
    # Train-only z-score scaling. Use population standard deviation; safe fallback for constants.
    scale_params = {}
    for col in CONT_FEATURES:
        train_values = model.loc[model["split"].eq("train"), col]
        mean = float(train_values.mean())
        std = float(train_values.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        scale_params[col] = {"mean": mean, "std": std}
        model[col] = (model[col] - mean) / std

    # Explicit schema order and metadata. Targets stay in original units.
    model = model.rename(columns={"segment_id": "segment_id"})
    out = model[META + FEATURES + TARGETS].copy()

    # Basic integrity checks before writing.
    assert out[FEATURES].notna().all().all(), "Feature matrix contains missing values."
    assert out.groupby("patient_uid")["split"].nunique().max() == 1, "Patient leakage across splits."
    assert out["dt"].notna().all()
    assert out["target_hypo_60"].dropna().isin([0.0, 1.0]).all()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_cols = [
        "patient_uid", "diabetes_type", "patient_id", "segment_id", "dt", "cgm",
        "cgm_interpolated", "cbg", "ketone", "csii_basal", "csii_bolus", "csii_suspend",
        "sc_units", "sc_long_acting", "iv_units", "oral_agent_count",
        "meal_logged", "meal_unknown", "meal_grams",
    ]
    clean_path = OUT_DIR / "cgm_t1dm_t2dm_causal_clean.csv"
    model_path = OUT_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
    schema_path = OUT_DIR / "schema_cgm_t1dm_t2dm_causal.json"
    audit_path = OUT_DIR / "causal_preprocessing_audit.json"

    df[clean_cols].to_csv(clean_path, index=False)
    out.to_csv(model_path, index=False)
    schema = {
        "dataset": "cgm_t1dm_t2dm_causal_candidate",
        "id_cols": ["patient_uid", "segment_id", "dt"],
        "group_col": "patient_uid",
        "split_method": "patient-level 70/15/15 approximately, stratified by diabetes_type",
        "features": FEATURES,
        "targets": TARGETS,
        "scaled_cols_mean_std_train_only": scale_params,
        "unscaled_cols": BINARY_FEATURES,
        "target_units": {"target_cgm_t30": "mg/dL", "target_cgm_t60": "mg/dL", "target_hypo_60": "binary"},
        "causal_guarantees": [
            "No interpolation or backward fill of CGM",
            "CGM-derived input features use current/past rows only",
            "Future CGM shifts are used only to create targets",
            "Patient-level split prevents one patient appearing in multiple splits",
            "Scaling parameters are fitted on training rows only",
            "'data not available' is excluded from meal_logged and meal_grams",
        ],
        "notes": [
            "Candidate dataset; do not replace current production dataset or models until reviewed.",
            "Observations and events are assigned to the first 15-minute grid timestamp at or after their source timestamp; missing CGM grid values remain missing.",
            "Event logs are assigned to the first grid slot at or after their source timestamp; multiple events in a slot are aggregated.",
            "CGM targets require actual non-missing future CGM grid values; no synthetic future CGM is created.",
        ],
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    split_rows = out["split"].value_counts().to_dict()
    split_patients = out.groupby("split")["patient_uid"].nunique().to_dict()
    unavailable_raw = int(raw["diet_text"].astype("string").str.contains(UNAVAILABLE_RE, na=False).sum())
    audit = {
        "raw_rows_after_validation": int(len(raw)),
        "causal_grid_rows": int(len(df)),
        "model_ready_rows": int(len(out)),
        "features_count": int(len(FEATURES)),
        "targets": TARGETS,
        "rows_by_split": {str(k): int(v) for k, v in split_rows.items()},
        "patients_by_split": {str(k): int(v) for k, v in split_patients.items()},
        "raw_unavailable_meal_text_rows": unavailable_raw,
        "model_rows_with_meal_logged_one_and_meal_unknown_one": int(
            ((df["meal_logged"] == 1) & (df["meal_unknown"] == 1)).sum()
        ),
        "cgm_interpolated_rows": int(df["cgm_interpolated"].sum()),
        "feature_matrix_missing_cells": int(out[FEATURES].isna().sum().sum()),
        "patient_split_overlap_check": "PASS",
        "note": "This is a newly generated candidate; existing artifacts were not changed.",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nBUILD COMPLETE")
    print(f"Model-ready shape: {out.shape}")
    print(f"Feature count: {len(FEATURES)}")
    print(f"Rows by split: {split_rows}")
    print(f"Patients by split: {split_patients}")
    print(f"Unavailable meal text in raw records: {unavailable_raw:,}")
    print(f"Rows where meal_logged=1 and meal_unknown=1: {audit['model_rows_with_meal_logged_one_and_meal_unknown_one']}")
    print(f"Interpolated CGM rows: {audit['cgm_interpolated_rows']} (expected 0)")
    print("\nCreated new files:")
    print(f"  {clean_path}")
    print(f"  {model_path}")
    print(f"  {schema_path}")
    print(f"  {audit_path}")
    print("\nNo model was trained. Review this candidate and audit JSON before any retraining.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print("No existing datasets or models were overwritten.")
        raise
