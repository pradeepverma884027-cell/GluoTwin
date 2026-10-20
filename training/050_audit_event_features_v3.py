from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GRID_MINUTES = 15
MAX_SEGMENT_GAP_MINUTES = 45
NUM = r"(\d+(?:\.\d+)?)"
UNAVAILABLE_RE = re.compile(r"data\s+not\s+available", re.IGNORECASE)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
RAW_DIR = DATASET_ROOT / "DATASET" / "raw" / "SHAHNGHAI-DATASET"
OUT_DIR = SCRIPT_DIR / "results_causal" / "preprocessing_v3"
REPORT_PATH = OUT_DIR / "050_event_feature_audit.json"


def find_col(df, candidates):
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in mapping:
            return mapping[name.lower()]
    return None


def sum_num(series, pattern):
    rx = re.compile(pattern, flags=re.IGNORECASE)

    def parse(value):
        if pd.isna(value):
            return np.nan
        matches = rx.findall(str(value))
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


def get_series(raw, candidates):
    col = find_col(raw, candidates)
    return raw[col] if col else pd.Series(np.nan, index=raw.index)


def load_raw():
    parts = []
    for filename, dtype in [("t1dm", "T1DM"), ("t2dm", "T2DM")]:
        path = RAW_DIR / f"{filename}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Raw input not found: {path}")
        raw = pd.read_csv(path, low_memory=False)
        date_col = find_col(raw, ["Date", "dt", "datetime", "timestamp"])
        pid_col = find_col(raw, ["patient_id", "Patient ID", "Patient_ID"])
        cgm_col = find_col(raw, ["CGM (mg / dl)", "cgm"])
        if not all([date_col, pid_col, cgm_col]):
            raise ValueError(f"Missing Date/patient/CGM columns in {path}")

        part = pd.DataFrame({
            "diabetes_type": dtype,
            "patient_id": raw[pid_col].astype("string").str.strip(),
            "dt": pd.to_datetime(raw[date_col], errors="coerce"),
            "cgm": pd.to_numeric(raw[cgm_col], errors="coerce"),
            "diet_text": get_series(raw, ["Dietary intake", "diet_text"]).astype("string"),
            "sc_text": get_series(raw, ["Insulin dose - s.c.", "sc_insulin_text"]).astype("string"),
            "csii_bolus_raw": get_series(raw, ["CSII - bolus insulin (Novolin R, IU)", "csii_bolus_raw"]),
            "csii_basal_raw": get_series(raw, ["CSII - basal insulin (Novolin R, IU / H)", "csii_basal_raw"]),
            "iv_text": get_series(raw, ["Insulin dose - i.v.", "iv_insulin_text"]).astype("string"),
        })
        part["patient_uid"] = dtype + "_" + part["patient_id"].fillna("")
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df = df.loc[df["dt"].notna() & df["cgm"].notna() & df["patient_id"].notna()].copy()
    df = df.sort_values(["patient_uid", "dt"]).drop_duplicates(["patient_uid", "dt"], keep="first").reset_index(drop=True)

    bolus_text = df["csii_bolus_raw"].astype("string")
    basal_text = df["csii_basal_raw"].astype("string")
    sc_text = df["sc_text"].astype("string")
    diet_text = df["diet_text"].astype("string").str.strip()

    df["csii_bolus"] = pd.to_numeric(df["csii_bolus_raw"], errors="coerce")
    df["csii_basal"] = pd.to_numeric(df["csii_basal_raw"], errors="coerce")
    df.loc[basal_text.str.contains("suspend", case=False, na=False), "csii_basal"] = 0.0
    df["csii_suspend"] = (
        bolus_text.str.contains("suspend", case=False, na=False)
        | basal_text.str.contains("suspend", case=False, na=False)
    ).astype(int)
    df["sc_units"] = sum_num(sc_text, NUM + r"\s*IU")
    df["sc_long_acting"] = sc_text.str.contains(r"glargine|detemir|degludec", case=False, na=False).astype(int)
    df["iv_units"] = sum_num(df["iv_text"], NUM + r"\s*IU")
    df["meal_unknown"] = diet_text.str.contains(UNAVAILABLE_RE, na=False).astype(int)
    has_text = diet_text.notna() & diet_text.ne("") & diet_text.str.lower().ne("nan")
    df["meal_logged"] = (has_text & df["meal_unknown"].eq(0)).astype(int)
    df["meal_grams"] = sum_num(diet_text, NUM + r"\s*g\b")
    df.loc[df["meal_unknown"].eq(1), "meal_grams"] = np.nan
    df["fast_insulin"] = (
        df["csii_bolus"].fillna(0)
        + np.where(df["sc_long_acting"].eq(0), df["sc_units"].fillna(0), 0.0)
        + df["iv_units"].fillna(0)
    ).clip(lower=0)

    gap = df.groupby("patient_uid")["dt"].diff().dt.total_seconds() / 60.0
    df["new_segment"] = gap.isna() | (gap > MAX_SEGMENT_GAP_MINUTES)
    df["segment_id"] = df.groupby("patient_uid")["new_segment"].cumsum().astype(int)

    rows = []
    for (pid, seg_id), g in df.groupby(["patient_uid", "segment_id"], sort=False):
        g = g.sort_values("dt").copy()
        start = g["dt"].iloc[0]
        elapsed = (g["dt"] - start).dt.total_seconds()
        g["k"] = np.ceil(elapsed / (GRID_MINUTES * 60) - 1e-12).astype(int)
        g["_source_order"] = np.arange(len(g))
        g = g.sort_values(["k", "dt", "_source_order"])
        agg = g.groupby("k", sort=True).agg(
            cgm=("cgm", "last"),
            csii_bolus=("csii_bolus", "sum"),
            csii_basal=("csii_basal", "last"),
            csii_suspend=("csii_suspend", "max"),
            sc_units=("sc_units", "sum"),
            sc_long_acting=("sc_long_acting", "max"),
            iv_units=("iv_units", "sum"),
            meal_grams=("meal_grams", lambda x: x.sum(min_count=1)),
            meal_logged=("meal_logged", "max"),
            meal_unknown=("meal_unknown", "max"),
            fast_insulin=("fast_insulin", "sum"),
        ).reset_index()
        grid = pd.DataFrame({"k": np.arange(int(agg["k"].max()) + 1, dtype=int)})
        grid["dt"] = start + pd.to_timedelta(grid["k"] * GRID_MINUTES, unit="m")
        grid = grid.merge(agg, on="k", how="left", validate="one_to_one")
        grid["patient_uid"] = pid
        grid["segment_id"] = seg_id
        grid["csii_basal"] = grid["csii_basal"].ffill().fillna(0.0)
        for col in ["csii_bolus", "csii_suspend", "sc_units", "sc_long_acting", "iv_units",
                    "meal_logged", "meal_unknown", "meal_grams", "fast_insulin"]:
            grid[col] = grid[col].fillna(0.0)
        rows.append(grid)
    clean_rebuilt = pd.concat(rows, ignore_index=True)
    return clean_rebuilt


def main():
    print("=" * 70)
    print("GLUCOTWIN V3 - EVENT AND DERIVED-FEATURE AUDIT (READ ONLY)")
    print("=" * 70)
    clean_path = OUT_DIR / "cgm_t1dm_t2dm_causal_clean.csv"
    model_path = OUT_DIR / "cgm_t1dm_t2dm_causal_model_ready.csv"
    schema_path = OUT_DIR / "schema_cgm_t1dm_t2dm_causal.json"
    for path in [clean_path, model_path, schema_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required v3 artifact missing: {path}")

    clean = pd.read_csv(clean_path, parse_dates=["dt"], low_memory=False)
    model = pd.read_csv(model_path, parse_dates=["dt"], low_memory=False)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    raw_grid = load_raw()

    key = ["patient_uid", "segment_id", "dt"]
    for frame_name, frame in [("clean", clean), ("model", model), ("raw_grid", raw_grid)]:
        if frame.duplicated(key).any():
            raise ValueError(f"{frame_name} contains duplicate patient/segment/timestamp keys")

    joined = clean.merge(raw_grid, on=key, how="left", suffixes=("_clean", "_raw"), validate="one_to_one", indicator=True)
    matched = joined["_merge"].eq("both")
    event_checks = {}
    for col in ["meal_logged", "meal_unknown", "csii_suspend", "sc_long_acting"]:
        a, b = joined[f"{col}_clean"], joined[f"{col}_raw"]
        event_checks[f"{col}_mismatches"] = int((matched & a.fillna(-999).ne(b.fillna(-999))).sum())
    for col in ["meal_grams", "csii_bolus", "sc_units", "iv_units", "fast_insulin"]:
        clean_col = col if col in clean.columns else None
        raw_col = f"{col}_raw"
        if clean_col is None:
            event_checks[f"{col}_comparison"] = "NOT CHECKABLE: column absent from clean CSV"
        else:
            a, b = joined[f"{col}_clean"], joined[raw_col]
            event_checks[f"{col}_mismatches"] = int((matched & ~np.isclose(a, b, atol=1e-8, rtol=1e-7, equal_nan=True)).sum())

    unknown_rows = clean["meal_unknown"].eq(1)
    meal_flag_conflicts = int((unknown_rows & clean["meal_logged"].eq(1)).sum())
    unknown_nonzero_grams = int((unknown_rows & clean["meal_grams"].fillna(0).ne(0)).sum())

    # Independently recompute event-derived features from reconstructed raw-grid events.
    derived = raw_grid.sort_values(key).copy()
    group = derived.groupby(["patient_uid", "segment_id"], sort=False)
    derived["fast_insulin_last_4h_unscaled"] = group["fast_insulin"].transform(lambda s: s.rolling(16, min_periods=1).sum())
    derived["meal_grams_last_2h_unscaled"] = group["meal_grams"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    positions = derived.groupby(["patient_uid", "segment_id"], sort=False).cumcount()
    for event_col, out_col in [("meal_logged", "mins_since_meal_unscaled"),
                               ("fast_insulin", "mins_since_fast_insulin_unscaled")]:
        last_pos = positions.where(derived[event_col] > 0).groupby(
            [derived["patient_uid"], derived["segment_id"]], sort=False
        ).ffill()
        derived[out_col] = ((positions - last_pos) * GRID_MINUTES).fillna(360).clip(upper=360)

    alpha_i = 1 - 0.5 ** (GRID_MINUTES / 90.0)
    alpha_c = 1 - 0.5 ** (GRID_MINUTES / 60.0)
    derived["iob_proxy_unscaled"] = derived.groupby(["patient_uid", "segment_id"], sort=False)["fast_insulin"].transform(
        lambda s: s.ewm(alpha=alpha_i, adjust=False).mean() / alpha_i
    )
    derived["cob_proxy_unscaled"] = derived.groupby(["patient_uid", "segment_id"], sort=False)["meal_grams"].transform(
        lambda s: s.fillna(0).ewm(alpha=alpha_c, adjust=False).mean() / alpha_c
    )

    model_check = model.merge(derived[key + [
        "fast_insulin_last_4h_unscaled", "meal_grams_last_2h_unscaled",
        "mins_since_meal_unscaled", "mins_since_fast_insulin_unscaled",
        "iob_proxy_unscaled", "cob_proxy_unscaled"
    ]], on=key, how="left", validate="one_to_one", indicator=True)
    feature_mapping = {
        "fast_insulin_last_4h": "fast_insulin_last_4h_unscaled",
        "meal_grams_last_2h": "meal_grams_last_2h_unscaled",
        "mins_since_meal": "mins_since_meal_unscaled",
        "mins_since_fast_insulin": "mins_since_fast_insulin_unscaled",
        "iob_proxy": "iob_proxy_unscaled",
        "cob_proxy": "cob_proxy_unscaled",
    }
    feature_checks = {}
    scaling = schema.get("scaled_cols_mean_std_train_only", {})
    for feature, expected_col in feature_mapping.items():
        if feature not in model.columns:
            feature_checks[feature] = {"status": "MISSING_MODEL_FEATURE"}
            continue
        if feature not in scaling:
            feature_checks[feature] = {"status": "MISSING_SCALING_METADATA"}
            continue
        mean = float(scaling[feature]["mean"])
        std = float(scaling[feature]["std"])
        observed = pd.to_numeric(model[feature], errors="coerce") * std + mean
        expected = pd.to_numeric(model_check[expected_col], errors="coerce")
        valid = model_check["_merge"].eq("both") & observed.notna() & expected.notna()
        diffs = (observed[valid] - expected[valid]).abs()
        feature_checks[feature] = {
            "rows_compared": int(valid.sum()),
            "max_abs_error_after_inverse_scaling": float(diffs.max()) if len(diffs) else None,
            "mismatches_at_1e-5": int((diffs > 1e-5).sum()) if len(diffs) else None,
            "status": "PASS" if len(diffs) and (diffs <= 1e-5).all() else "CHECK",
        }

    report = {
        "read_only": True,
        "clean_rows": int(len(clean)),
        "model_rows": int(len(model)),
        "raw_reconstructed_grid_rows": int(len(raw_grid)),
        "clean_rows_without_raw_grid_match": int((~matched).sum()),
        "event_checks": event_checks,
        "unknown_meal_rows": int(unknown_rows.sum()),
        "unknown_meal_rows_with_meal_logged_1": meal_flag_conflicts,
        "unknown_meal_rows_with_nonzero_grams": unknown_nonzero_grams,
        "derived_model_feature_checks": feature_checks,
        "limitations": [
            "Raw event quantities are reconstructed using the same parsing and slot aggregation rules as the candidate builder.",
            "Derived feature checks compare inverse-scaled model columns against independently recomputed raw-grid features.",
            "Passing confirms implementation consistency, not clinical validity or predictive performance.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {REPORT_PATH}")
    print("Read-only: no dataset, schema, or model was modified.")


if __name__ == "__main__":
    main()
