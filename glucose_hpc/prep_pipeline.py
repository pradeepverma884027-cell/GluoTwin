#!/usr/bin/env python3
"""
Data preparation pipeline (clean -> engineer -> split -> scale) for:
  GlucoTwin.csv, smart_wearable.csv, t1dm.csv + t2dm.csv (Shanghai CGM style), ohhim.csv (OhioT1DM style)

For every dataset it writes into OUT:
  <name>_clean.csv        cleaned, readable, human-friendly table
  <name>_model_ready.csv  encoded + scaled (train-only stats) + `split` column, NO NaN in features
  schema_<name>.json      id / feature / target columns, dropped columns, scaling params
Run:  python3 prep_pipeline.py
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

IN = Path("data")          # folder with the ORIGINAL csv files (set by run_all)
OUT = Path("processed")    # where cleaned / model-ready files are written (set by run_all)
SEED = 42
LOG = []


def log(ds, msg):
    LOG.append((ds, msg))
    print(f"[{ds}] {msg}")


# ----------------------------------------------------------------- helpers
def split_groups(df, gcol, scol=None, frac=(0.70, 0.15), seed=SEED):
    """Group-wise split (a group never lands in two splits), stratified by `scol` at group level."""
    rng = np.random.default_rng(seed)
    strata = (pd.Series("all", index=pd.Index(df[gcol].unique())) if scol is None
              else df.groupby(gcol)[scol].first())
    assign = {}
    for s in strata.unique():
        ids = list(strata.index[strata == s])
        rng.shuffle(ids)
        n = len(ids)
        n_va = max(1, round(n * frac[1]))
        n_te = max(1, round(n * (1 - frac[0] - frac[1])))
        n_tr = n - n_va - n_te
        for i in ids[:n_tr]: assign[i] = "train"
        for i in ids[n_tr:n_tr + n_va]: assign[i] = "val"
        for i in ids[n_tr + n_va:]: assign[i] = "test"
    return df[gcol].map(assign)


def split_rows(y, frac=(0.70, 0.15), seed=SEED):
    """Row-wise stratified split."""
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=y.index)
    for c in y.unique():
        idx = np.array(y.index[y == c]); rng.shuffle(idx); n = len(idx)
        a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
        out.loc[idx[a:b]] = "val"; out.loc[idx[b:]] = "test"
    return out


def zscore(df, cols, train_mask):
    """Fit mean/std on TRAIN rows only, apply to all rows. Returns params."""
    params = {}
    for c in cols:
        m = df.loc[train_mask, c].mean(); s = df.loc[train_mask, c].std(ddof=0)
        s = float(s) if np.isfinite(s) and s > 0 else 1.0
        params[c] = [float(m), s]
        df[c] = (df[c] - m) / s
    return params


def prune_corr(train, cols, thr=0.95, protect=()):
    cand = [c for c in cols if not c.startswith(protect)]
    corr = train[cand].corr().abs()
    drop, pairs = [], []
    for i, c in enumerate(cand):
        if c in drop: continue
        for d in cand[i + 1:]:
            if d not in drop and corr.loc[c, d] > thr:
                drop.append(d); pairs.append((c, d, round(float(corr.loc[c, d]), 3)))
    return drop, pairs


def interp_short(s, max_gap):
    """Linear-interpolate only NaN runs of length <= max_gap that have data on both sides."""
    isna = s.isna()
    run = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run).transform("sum")
    filled = s.interpolate(limit_area="inside")
    out = s.copy()
    m = isna & (run_len <= max_gap)
    out[m] = filled[m]
    return out


def sum_num(s, pat):
    """Sum all regex numeric matches inside free-text cells. NaN where cell empty or nothing matched."""
    s = s.astype(str).where(s.notna())
    m = s.str.extractall(pat)
    if m.empty:
        return pd.Series(np.nan, index=s.index)
    return m[0].astype(float).groupby(level=0).sum().reindex(s.index)


def write(df, name, suffix):
    p = OUT / f"{name}_{suffix}.csv"
    df.to_csv(p, index=False)
    log(name, f"wrote {p.name}  shape={df.shape}")


def write_schema(name, **kw):
    (OUT / f"schema_{name}.json").write_text(json.dumps(kw, indent=2, default=str))


def check_model_ready(name, df, feats, targets, group=None):
    assert df[feats].isna().sum().sum() == 0, f"{name}: NaN in features"
    assert np.isfinite(df[feats].to_numpy(dtype=float)).all(), f"{name}: inf in features"
    assert df[targets].isna().sum().sum() == 0, f"{name}: NaN in targets"
    assert set(df["split"].unique()) == {"train", "val", "test"}, f"{name}: bad splits"
    if group:
        multi = df.groupby(group)["split"].nunique().max()
        assert multi == 1, f"{name}: group leaks across splits"
    log(name, f"CHECK OK  split rows = {df['split'].value_counts().to_dict()}")


# ================================================================ 1. GlucoTwin
def prep_glucotwin():
    ds = "glucotwin"
    df = pd.read_csv(IN / "GlucoTwin.csv")
    ren = {"#1 Contour Fingerstick GLU": "Fingerstick_GLU_1", "#2 Contour Fingerstick GLU": "Fingerstick_GLU_2",
           "#3 Contour Fingerstick GLU": "Fingerstick_GLU_3", "Time (t) #1": "Fingerstick_Time_1",
           "Time (t) #2": "Fingerstick_Time_2", "Time (t) #3": "Fingerstick_Time_3",
           "Fasting GLU - PDL (Lab)": "Fasting_GLU", "Self-identify": "Self_identify"}
    df = df.rename(columns=ren)
    df.columns = [re.sub(r"[^0-9A-Za-z_+\-]+", "_", c).strip("_") for c in df.columns]
    log(ds, f"loaded {df.shape}; missing cells = {int(df.isna().sum().sum())}")

    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(["Participant_ID", "Timestamp"])
    log(ds, f"duplicate rows removed: {n0 - len(df)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M")  # verified mm/dd/yyyy
    df["Is_Weekend"] = (df["Timestamp"].dt.dayofweek >= 5).astype(int)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour_of_Day"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour_of_Day"] / 24)
    before = df["Self_identify"].nunique()
    df["Self_identify"] = df["Self_identify"].replace({"African American Black": "African American"})
    log(ds, f"Self_identify categories merged {before} -> {df['Self_identify'].nunique()}")
    for c in ["Collection_time_PDL_Lab", "Fingerstick_Time_1", "Fingerstick_Time_2", "Fingerstick_Time_3"]:
        t = pd.to_datetime(df[c], format="%H:%M")
        df[c + "_min"] = t.dt.hour * 60 + t.dt.minute
    df["Is_Male"] = (df["Gender"] == "M").astype(int)
    df["Diabetes_Status_code"] = df["Diabetes_Status"].map({"Healthy": 0, "Prediabetes": 1, "T2D": 2})
    tg = [c for c in df if c.startswith("Target_Glucose")]
    df["Sensor_Ceiling_Hit"] = (df[tg] >= 400).any(axis=1).astype(int)
    log(ds, f"rows touching CGM ceiling (400 mg/dL): {int(df['Sensor_Ceiling_Hit'].sum())} (kept, flagged)")

    # split by participant, stratified by diabetes status
    df["split"] = split_groups(df, "Participant_ID", "Diabetes_Status")
    tr = df["split"] == "train"

    # winsorise wearable-derived columns at train 0.5/99.5 pct
    for c in ["HR_Mean_30", "HR_Mean_60", "Mets_Mean_30", "Mets_Max_60", "ActCal_Sum_60"]:
        lo, hi = df.loc[tr, c].quantile([0.005, 0.995])
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lo, hi)
        log(ds, f"winsorised {c}: {n} values clipped to [{lo:.1f}, {hi:.1f}]")

    clean = df.copy()

    # ---- model-ready
    leak = ["Post_HR_Mean_60", "Post_Mets_Mean_60", "Next_Meal_Within_120min"]  # info from AFTER the meal
    targets = [c for c in df if c.startswith("Target_")]
    admin = ["Timestamp", "Gender", "Diabetes_Status", "Diabetes_Status_code", "Meal_Type", "Self_identify",
             "Hour_of_Day", "Day_Index", "Collection_time_PDL_Lab", "Fingerstick_Time_1", "Fingerstick_Time_2",
             "Fingerstick_Time_3", "Collection_time_PDL_Lab_min", "Fingerstick_Time_1_min",
             "Fingerstick_Time_2_min", "Fingerstick_Time_3_min", "Sensor_Ceiling_Hit"]
    m = df.drop(columns=leak + admin)
    m = pd.concat([m, pd.get_dummies(df["Meal_Type"], prefix="Meal", dtype=int),
                   pd.get_dummies(df["Self_identify"].str.replace(r"\W+", "_", regex=True),
                                  prefix="Eth", dtype=int),
                   df[["Diabetes_Status_code"]].rename(columns={"Diabetes_Status_code": "AUX_Diabetes_Status_code"})],
                  axis=1)
    meta = ["Participant_ID", "split"]
    aux = ["AUX_Diabetes_Status_code"]
    # extra engineered features (all row-wise, no leakage)
    m["Dexcom_minus_Fasting"] = m["Dexcom_t0"] - m["Fasting_GLU"]
    m["Carb_Fiber_Ratio"] = m["Carbs"] / (m["Fiber"] + 1.0)
    m["Carb_Cal_Share"] = (m["Carbs"] * 4) / (m["Calories"] + 1.0)
    m["TG_HDL_Ratio"] = m["Triglycerides"] / (m["HDL"] + 1e-6)
    m["HOMA_IR"] = m["Insulin"] * m["Fasting_GLU"] / 405.0
    dex = [c for c in m if c.startswith("Dexcom_t")]
    m["Glucose_Range_60"] = m[dex].max(axis=1) - m[dex].min(axis=1)
    m["Rise_From_Min_60"] = m["Dexcom_t0"] - m[dex].min(axis=1)
    feats = [c for c in m.columns if c not in meta + targets + aux]
    trm = m["split"] == "train"

    const = [c for c in feats if m.loc[trm, c].nunique() <= 1]
    drop_corr, pairs = prune_corr(m.loc[trm], [c for c in feats if c not in const], 0.95,
                                  protect=("Dexcom_", "Libre_", "Glucose_", "Meal_", "Eth_"))
    for a, b, r in pairs:
        log(ds, f"redundant feature dropped: {b} (|r|={r} with {a})")
    feats = [c for c in feats if c not in const + drop_corr]
    # 22 microbiome scores (1-3) are per-person constants -> compress to 5 PCA parts (fit on train only)
    from sklearn.decomposition import PCA
    mic = [c for c in feats if set(m[c].unique()) <= {1, 2, 3}]
    pca = PCA(n_components=5, random_state=SEED).fit(m.loc[trm, mic])
    pcs = pca.transform(m[mic])
    for i in range(5): m[f"Microbiome_PC{i + 1}"] = pcs[:, i]
    log(ds, f"microbiome: {len(mic)} score columns -> 5 PCA components, explained var {pca.explained_variance_ratio_.sum():.2f}")
    feats = [c for c in feats if c not in mic] + [f"Microbiome_PC{i + 1}" for i in range(5)]

    binary = [c for c in feats if m[c].nunique() <= 3]  # binary + 1-3 microbiome scores stay raw
    cont = [c for c in feats if c not in binary and c not in ("Hour_sin", "Hour_cos")]
    params = zscore(m, cont, trm)
    out = m[meta + aux + feats + targets]
    check_model_ready(ds, out, feats, targets, group="Participant_ID")

    write(clean, ds, "clean"); write(out, ds, "model_ready")
    write_schema(ds, id_cols=["Participant_ID"], group_col="Participant_ID", split_method="participant-level 70/15/15, stratified by Diabetes_Status",
                 features=feats, targets=targets, aux_targets=aux, dropped_leakage=leak, dropped_redundant=drop_corr, microbiome_replaced_by_pca=mic,
                 scaled_cols_mean_std=params, unscaled_cols=[c for c in feats if c not in cont],
                 note="Targets are in original units (mg/dL etc). Scaling params fit on train rows only.")


# ============================================================ 2. smart_wearable
def prep_wearable():
    ds = "smart_wearable"
    df = pd.read_csv(IN / "smart_wearable.csv")
    log(ds, f"loaded {df.shape}; missing cells = {int(df.isna().sum().sum())}")
    n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
    log(ds, f"duplicate rows removed: {n0 - len(df)}")

    n = int((df["blood_oxygen_level"] > 100).sum())
    df["blood_oxygen_level"] = df["blood_oxygen_level"].clip(upper=100.0)
    log(ds, f"SpO2 > 100% is impossible: {n} values capped at 100")
    bad_bmi = df["bmi"] < 15
    log(ds, f"BMI < 15 is implausible: {int(bad_bmi.sum())} values set to NaN then median-imputed (train median)")
    df["bmi_was_imputed"] = bad_bmi.astype(int)
    df.loc[bad_bmi, "bmi"] = np.nan
    assert (df["systolic_bp"] > df["diastolic_bp"]).all()

    df["split"] = split_rows(df["health_risk"])
    trm = df["split"] == "train"
    df["bmi"] = df["bmi"].fillna(df.loc[trm, "bmi"].median())
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["steps_x_stress"] = df["avg_daily_steps"] * df["stress_index"]
    df["short_sleep"] = (df["sleep_hours"] < 6).astype(int)
    df["obese"] = (df["bmi"] >= 30).astype(int)
    for c in ["resting_heart_rate", "avg_daily_steps", "sleep_hours", "stress_index", "systolic_bp"]:
        q1, q3 = df.loc[trm, c].quantile([.25, .75]); iqr = q3 - q1
        k = int(((df[c] < q1 - 3 * iqr) | (df[c] > q3 + 3 * iqr)).sum())
        if k: log(ds, f"extreme outliers (3xIQR) in {c}: {k}")
    log(ds, f"class balance health_risk: {df['health_risk'].value_counts(normalize=True).round(3).to_dict()} -> use stratified CV / class_weight")
    clean = df.copy()

    drop = ["year"]  # collection year: not a causal feature, corr ~ -0.05
    m = df.drop(columns=drop)
    target = ["health_risk"]; meta = ["split"]
    feats = [c for c in m.columns if c not in target + meta]
    binary = ["bmi_was_imputed", "short_sleep", "obese"]
    cont = [c for c in feats if c not in binary]
    params = zscore(m, cont, m["split"] == "train")
    out = m[meta + feats + target]
    check_model_ready(ds, out, feats, target)
    write(clean, ds, "clean"); write(out, ds, "model_ready")
    write_schema(ds, features=feats, targets=target, split_method="row-level stratified 70/15/15", dropped=drop,
                 scaled_cols_mean_std=params, unscaled_cols=binary)


# ============================================================ 3. T1DM + T2DM CGM
CGM_RENAME = {"CGM (mg / dl)": "cgm", "CBG (mg / dl)": "cbg", "Blood Ketone (mmol / L)": "ketone",
              "Dietary intake": "diet_text", "Insulin dose - s.c.": "sc_insulin_text",
              "Non-insulin hypoglycemic agents": "oral_agent_text",
              "CSII - bolus insulin (Novolin R, IU)": "csii_bolus_raw",
              "CSII - basal insulin (Novolin R, IU / H)": "csii_basal_raw",
              "Insulin dose - i.v.": "iv_insulin_text"}
TEXTCOLS = ["CSII - bolus insulin (Novolin R, IU)", "CSII - basal insulin (Novolin R, IU / H)", "Insulin dose - i.v.",
            "Insulin dose - s.c.", "Non-insulin hypoglycemic agents", "Dietary intake", "饮食"]
NUM = r"(\d+(?:\.\d+)?)"


def prep_cgm():
    ds = "cgm_t1dm_t2dm"
    frames = []
    for name, typ in (("t1dm", "T1DM"), ("t2dm", "T2DM")):
        d = pd.read_csv(IN / f"{name}.csv", low_memory=False, dtype={c: str for c in TEXTCOLS})
        d["diabetes_type"] = typ; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    log(ds, f"loaded T1DM+T2DM {df.shape}")
    df = df.drop(columns=["饮食", "source_file", "sheet"], errors="ignore").rename(columns=CGM_RENAME)
    log(ds, "dropped Chinese duplicate of Dietary intake (饮食), source_file, sheet")

    df["dt"] = pd.to_datetime(df["Date"], errors="coerce")
    bad = df["dt"].isna() | df["cgm"].isna()
    log(ds, f"rows with unparseable date or missing CGM dropped: {int(bad.sum())}")
    df = df.loc[~bad].drop(columns="Date")
    df["patient_uid"] = df["diabetes_type"] + "_" + df["patient_id"].astype(str)
    df = df.sort_values(["patient_uid", "dt"]).drop_duplicates(["patient_uid", "dt"]).reset_index(drop=True)

    # ---- parse free-text event columns into numbers + flags
    df["csii_suspend"] = (df["csii_bolus_raw"].str.contains("suspend", case=False, na=False) |
                          df["csii_basal_raw"].str.contains("suspend", case=False, na=False)).astype(int)
    df["csii_bolus"] = pd.to_numeric(df["csii_bolus_raw"], errors="coerce")
    df["csii_basal"] = pd.to_numeric(df["csii_basal_raw"], errors="coerce")
    wrong = ((df["csii_bolus_raw"].notna() & df["csii_bolus"].isna() & (df["csii_suspend"] == 0)) |
             (df["csii_basal_raw"].notna() & df["csii_basal"].isna() & (df["csii_suspend"] == 0)))
    log(ds, f"CSII cells holding wrong text (e.g. a drug name) set to NaN: {int(wrong.sum())}")
    df.loc[df["csii_basal_raw"].str.contains("suspend", case=False, na=False), "csii_basal"] = 0.0
    df["sc_units"] = sum_num(df["sc_insulin_text"], NUM + r"\s*IU")
    df["sc_long_acting"] = df["sc_insulin_text"].str.contains("glargine|detemir|degludec", case=False, na=False).astype(int)
    df["iv_units"] = sum_num(df["iv_insulin_text"], NUM + r"\s*IU")
    # drug doses in mixed units / mixed drugs cannot be summed meaningfully -> use number of oral agents given
    txt = df["oral_agent_text"].str.strip()
    df["oral_agent_count"] = np.where(txt.notna() & (txt != ""), txt.str.count(",").fillna(0) + 1, np.nan)
    df["meal_grams"] = sum_num(df["diet_text"], NUM + r"\s*g\b")
    df["meal_logged"] = df["diet_text"].notna().astype(int)
    df["meal_unknown"] = df["diet_text"].str.contains("data not available", case=False, na=False).astype(int)
    for col, txt in (("sc_units", "sc_insulin_text"), ("iv_units", "iv_insulin_text"), ("oral_agent_count", "oral_agent_text")):
        k = int((df[txt].notna() & df[col].isna()).sum())
        if k: log(ds, f"{txt}: {k} text cells without a parsable number (flag kept, dose=0)")

    # quality flags (kept, not removed)
    df["cgm_at_floor"] = (df["cgm"] <= 39.6).astype(int)
    df["cgm_at_ceiling"] = (df["cgm"] >= 450).astype(int)
    log(ds, f"CGM at sensor floor 39.6: {int(df['cgm_at_floor'].sum())}, at/above 450: {int(df['cgm_at_ceiling'].sum())} (flagged, kept)")
    k = int((df["ketone"] > 3).sum())
    log(ds, f"ketone > 3 mmol/L (possible DKA, real clinical value): {k} rows kept; ketone & CBG too sparse (97-100% missing) -> not model inputs")

    # ---- segments: new segment when gap > 45 min; short gaps (<=2 missing steps) interpolated
    gap = df.groupby("patient_uid")["dt"].diff().dt.total_seconds() / 60
    df["seg"] = (gap.isna() | (gap > 45)).cumsum()
    EVENT0 = ["sc_units", "iv_units", "oral_agent_count", "meal_grams", "csii_bolus", "csii_suspend", "sc_long_acting",
              "meal_logged", "meal_unknown", "cgm_at_floor", "cgm_at_ceiling"]
    STATIC = ["patient_uid", "diabetes_type", "patient_id", "file_no", "seg"]
    parts = []
    for seg, g in df.groupby("seg", sort=False):
        start = g["dt"].iloc[0]
        k = ((g["dt"] - start).dt.total_seconds() / 900).round().astype(int)
        g = g.assign(k=k.values).drop_duplicates("k")
        m = pd.DataFrame({"k": np.arange(g["k"].min(), g["k"].max() + 1)}).merge(g, on="k", how="left")
        m["cgm_interpolated"] = m["cgm"].isna().astype(int)
        m["cgm"] = m["cgm"].interpolate()
        m["dt"] = start + pd.to_timedelta(m["k"] * 15, unit="min")
        for c in STATIC: m[c] = m[c].ffill()
        for c in EVENT0: m[c] = m[c].fillna(0)
        m["csii_basal"] = m["csii_basal"].ffill().fillna(0.0)  # basal RATE carried forward until it changes
        parts.append(m.drop(columns=["k"]))
    df = pd.concat(parts, ignore_index=True)
    log(ds, f"segments: {df['seg'].nunique()}; interpolated rows: {int(df['cgm_interpolated'].sum())}")

    # rate-of-change sanity
    df["cgm_roc15"] = df.groupby("seg")["cgm"].diff()
    nroc = int((df["cgm_roc15"].abs() > 100).sum())
    log(ds, f"|CGM change| > 100 mg/dL per 15 min: {nroc} rows (flagged `cgm_roc_suspect`, kept)")
    df["cgm_roc_suspect"] = (df["cgm_roc15"].abs() > 100).astype(int)

    # ---- features (all computed inside a segment, so no window crosses a time gap)
    g = df.groupby("seg")
    for L in (1, 2, 3, 4, 6, 8):
        df[f"cgm_lag_{15 * L}m"] = g["cgm"].shift(L)
    df["cgm_delta_15"] = df["cgm"] - df["cgm_lag_15m"]
    df["cgm_delta_30"] = df["cgm"] - df["cgm_lag_30m"]
    df["cgm_delta_60"] = df["cgm"] - df["cgm_lag_60m"]

    def roll(col, w, fn, mp=None):
        return getattr(g[col].rolling(w, min_periods=mp or w), fn)().droplevel(0)

    df["cgm_roll_mean_60m"] = roll("cgm", 4, "mean")
    df["cgm_roll_std_60m"] = roll("cgm", 4, "std")
    df["cgm_roll_min_120m"] = roll("cgm", 8, "min")
    df["fast_insulin"] = df["csii_bolus"] + np.where(df["sc_long_acting"] == 0, df["sc_units"], 0.0) + df["iv_units"]
    df["fast_insulin_last_4h"] = roll("fast_insulin", 16, "sum", 1)
    df["meal_grams_last_2h"] = roll("meal_grams", 8, "sum", 1)
    pos = pd.Series(df.index, index=df.index)
    for ev, col in (((df["meal_logged"] > 0), "mins_since_meal"), ((df["fast_insulin"] > 0), "mins_since_fast_insulin")):
        last = pos.where(ev).groupby(df["seg"]).ffill()
        df[col] = ((pos - last) * 15).fillna(360).clip(upper=360)
    tod = df["dt"].dt.hour + df["dt"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * tod / 24); df["hour_cos"] = np.cos(2 * np.pi * tod / 24)
    df["is_t2dm"] = (df["diabetes_type"] == "T2DM").astype(int)
    # long context + decayed insulin/carb load (all causal: only past values)
    df["in_range"] = ((df["cgm"] >= 70) & (df["cgm"] <= 180)).astype(float)
    df["below_range"] = (df["cgm"] < 70).astype(float)
    df["cgm_mean_6h"] = roll("cgm", 24, "mean", 8)
    df["cgm_std_6h"] = roll("cgm", 24, "std", 8)
    df["cgm_mean_24h"] = roll("cgm", 96, "mean", 8)
    df["cgm_tir_24h"] = roll("in_range", 96, "mean", 8)
    df["cgm_tbr_24h"] = roll("below_range", 96, "mean", 8)
    a_i, a_c = 1 - 0.5 ** (15 / 90), 1 - 0.5 ** (15 / 60)   # half-life: insulin 90 min, carbs 60 min
    df["iob_proxy"] = g["fast_insulin"].ewm(alpha=a_i, adjust=False).mean().droplevel(0) / a_i
    df["cob_proxy"] = g["meal_grams"].ewm(alpha=a_c, adjust=False).mean().droplevel(0) / a_c

    # ---- targets (cgm in 30 / 60 min, hypo in next 60 min)
    df["target_cgm_t30"] = g["cgm"].shift(-2)
    df["target_cgm_t60"] = g["cgm"].shift(-4)
    fut = pd.concat([g["cgm"].shift(-i) for i in range(1, 5)], axis=1)
    df["target_hypo_60"] = np.where(fut.isna().any(axis=1), np.nan, (fut.min(axis=1) < 70).astype(float))

    clean_cols = ["patient_uid", "diabetes_type", "patient_id", "file_no", "seg", "dt", "cgm", "cgm_interpolated",
                  "cgm_at_floor", "cgm_at_ceiling", "cgm_roc_suspect", "cbg", "ketone", "csii_basal", "csii_bolus",
                  "csii_suspend", "sc_units", "sc_long_acting", "iv_units", "oral_agent_count", "meal_logged", "meal_unknown",
                  "meal_grams"]
    clean = df[clean_cols].rename(columns={"seg": "segment_id"})

    # ---- model-ready
    lag_cols = [f"cgm_lag_{15 * L}m" for L in (1, 2, 3, 4, 6, 8)]
    cont = ["cgm"] + lag_cols + ["cgm_delta_15", "cgm_delta_30", "cgm_delta_60", "cgm_roll_mean_60m",
                                 "cgm_roll_std_60m", "cgm_roll_min_120m", "csii_basal", "fast_insulin_last_4h",
                                 "meal_grams_last_2h", "mins_since_meal", "mins_since_fast_insulin", "oral_agent_count", "cgm_mean_6h", "cgm_std_6h",
                                 "cgm_mean_24h", "cgm_tir_24h", "cgm_tbr_24h", "iob_proxy", "cob_proxy"]
    binary = ["csii_suspend", "sc_long_acting", "meal_logged", "cgm_interpolated", "is_t2dm", "hour_sin", "hour_cos"]
    feats = cont + binary
    targets = ["target_cgm_t30", "target_cgm_t60", "target_hypo_60"]
    meta = ["patient_uid", "diabetes_type", "seg", "dt", "split"]
    m = df.dropna(subset=cont + targets).copy()
    log(ds, f"model rows after dropping incomplete windows: {len(m)} of {len(df)}")
    m["split"] = split_groups(m, "patient_uid", "diabetes_type")
    params = zscore(m, cont, m["split"] == "train")
    m = m.rename(columns={"seg": "segment_id"})
    meta = ["patient_uid", "diabetes_type", "segment_id", "dt", "split"]
    out = m[meta + feats + targets]
    check_model_ready(ds, out, feats, targets, group="patient_uid")
    log(ds, f"hypo (<70 in next 60 min) rate: {out['target_hypo_60'].mean():.3f}")
    write(clean, ds, "clean"); write(out, ds, "model_ready")
    write_schema(ds, id_cols=["patient_uid", "segment_id", "dt"], group_col="patient_uid",
                 split_method="patient-level 70/15/15, stratified by diabetes_type", features=feats, targets=targets,
                 not_used_as_features=["cbg", "ketone", "csii_basal_raw/diet_text etc. (raw text)"],
                 scaled_cols_mean_std=params, unscaled_cols=binary,
                 note="15-min grid. Windows never cross a gap > 45 min. Targets unscaled (mg/dL; hypo = 0/1).")


# ============================================================ 4. ohhim (OhioT1DM style)
OHIO_ROWS = ["glucose_level", "finger_stick", "basal", "temp_basal", "bolus", "meal", "sleep", "work", "stressors",
             "hypo_event", "illness", "exercise", "basis_heart_rate", "basis_gsr", "basis_skin_temperature",
             "basis_air_temperature", "basis_steps", "basis_sleep"]
TS_COLS = ("ts", "ts_begin", "ts_end", "tbegin", "tend")


def ohio_events(row):
    ev = row["event"]
    if isinstance(ev, str) and ev.strip():
        recs = json.loads(ev)
        return [recs] if isinstance(recs, dict) else recs
    flat = {c.replace("event.", ""): row[c] for c in row.index if c.startswith("event.@") and pd.notna(row[c])}
    return [flat] if flat else []   # single-event rows come flattened into event.@* columns


def prep_ohio():
    ds = "ohhim"
    raw = pd.read_csv(IN / "ohhim.csv")
    files = list(dict.fromkeys(raw["source_file"]))
    pid = {f: f"P{i + 1}" for i, f in enumerate(files)}
    log(ds, f"{len(files)} source files -> patients {list(pid.values())} (file names carry no real patient id); "
            f"event type per row inferred from fixed 18-row layout")
    T = {}
    for _, r in raw.iterrows():
        d = pd.DataFrame(ohio_events(r))
        d.columns = [c.lstrip("@") for c in d.columns]
        for c in TS_COLS:
            if c in d: d[c] = pd.to_datetime(d[c], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        T[(pid[r["source_file"]], OHIO_ROWS[int(r["row_in_file"])])] = d

    def series(p, name, col="value", agg="mean", lo=-np.inf, hi=np.inf, f2c=False):
        d = T[(p, name)]
        if d.empty or "ts" not in d: return None
        v = pd.to_numeric(d[col], errors="coerce")
        if f2c: v = (v - 32) * 5 / 9
        v = v.where((v >= lo) & (v <= hi))
        s = pd.DataFrame({"b": d["ts"].dt.floor("5min"), "v": v}).dropna().groupby("b")["v"].agg(agg)
        return s

    def interval(grid_idx, d, b="ts_begin", e="ts_end", val=None):
        out = pd.Series(0.0, index=grid_idx)
        if d.empty or b not in d: return out
        for _, r in d.dropna(subset=[b, e]).iterrows():
            s, t = sorted([r[b], r[e]])  # data quirk: some sleep intervals have end < begin
            v = 1.0 if val is None else pd.to_numeric(r[val], errors="coerce")
            if pd.isna(v): v = 1.0
            out[(grid_idx >= s) & (grid_idx <= t)] = np.maximum(out[(grid_idx >= s) & (grid_idx <= t)], v)
        return out

    grids = []
    for p in pid.values():
        g_raw = series(p, "glucose_level", lo=40, hi=400)
        n_bad = int((pd.to_numeric(T[(p, "glucose_level")]["value"], errors="coerce").lt(40) |
                     pd.to_numeric(T[(p, "glucose_level")]["value"], errors="coerce").gt(400)).sum())
        idx = pd.date_range(g_raw.index.min(), g_raw.index.max(), freq="5min")
        gr = pd.DataFrame(index=idx)
        gr["glucose"] = g_raw.reindex(idx)
        gr["glucose_observed"] = gr["glucose"].notna().astype(int)
        gr["glucose"] = interp_short(gr["glucose"], 6)  # <= 30 min gaps
        gr["glucose_interpolated"] = ((gr["glucose"].notna()) & (gr["glucose_observed"] == 0)).astype(int)
        log(ds, f"{p}: {len(idx)} 5-min steps, {int(gr['glucose_observed'].sum())} observed, "
                f"{int(gr['glucose_interpolated'].sum())} interpolated, out-of-range glucose set NaN: {n_bad}")

        for name, col, agg, lo, hi, f2c in (("basis_heart_rate", "hr", "mean", 30, 220, False),
                                            ("basis_gsr", "gsr", "mean", 0, np.inf, False),
                                            ("basis_skin_temperature", "skin_temp_c", "mean", 15, 45, True),
                                            ("basis_air_temperature", "air_temp_c", "mean", -10, 50, True),
                                            ("basis_steps", "steps", "sum", 0, np.inf, False)):
            s = series(p, name, agg=agg, lo=lo if not f2c else lo, hi=hi, f2c=False) if not f2c else None
            if f2c:  # filter after F->C conversion
                d = T[(p, name)]
                v = (pd.to_numeric(d["value"], errors="coerce") - 32) * 5 / 9
                v = v.where((v >= lo) & (v <= hi))
                s = pd.DataFrame({"b": d["ts"].dt.floor("5min"), "v": v}).dropna().groupby("b")["v"].agg(agg)
            gr[col] = interp_short(s.reindex(idx), 3) if s is not None else np.nan
            gr[f"{col}_missing"] = gr[col].isna().astype(int)

        # insulin: basal step function, temp_basal overrides, bolus / meal as impulses
        b = T[(p, "basal")]
        bs = (pd.Series(pd.to_numeric(b["value"], errors="coerce").values, index=b["ts"]).dropna().sort_index()
              if not b.empty else pd.Series(0.0, index=pd.DatetimeIndex([idx[0]])))
        bs = bs[~bs.index.duplicated(keep="last")]
        gr["basal_rate"] = bs.reindex(bs.index.union(idx)).ffill().reindex(idx).bfill()
        tb = T[(p, "temp_basal")]
        gr["temp_basal_active"] = 0
        tb_iter = tb.dropna(subset=["ts_begin", "ts_end"]).iterrows() if "ts_begin" in tb and "ts_end" in tb else iter(())
        for _, r in tb_iter:
            msk = (idx >= r["ts_begin"]) & (idx <= r["ts_end"])
            gr.loc[msk, "basal_rate"] = pd.to_numeric(r["value"], errors="coerce"); gr.loc[msk, "temp_basal_active"] = 1
        bo = T[(p, "bolus")]
        gr["bolus_dose"] = 0.0; gr["bolus_carb_input"] = 0.0
        if not bo.empty:
            dd = pd.DataFrame({"b": bo["ts_begin"].dt.floor("5min"), "dose": pd.to_numeric(bo["dose"], errors="coerce"),
                               "cin": pd.to_numeric(bo.get("bwz_carb_input"), errors="coerce")}).dropna(subset=["b"])
            a = dd.groupby("b")[["dose", "cin"]].sum()
            gr["bolus_dose"] = a["dose"].reindex(idx).fillna(0.0); gr["bolus_carb_input"] = a["cin"].reindex(idx).fillna(0.0)
        me = T[(p, "meal")]
        gr["meal_carbs"] = 0.0
        if not me.empty:
            mm = pd.DataFrame({"b": me["ts"].dt.floor("5min"), "c": pd.to_numeric(me["carbs"], errors="coerce")}).dropna()
            gr["meal_carbs"] = mm.groupby("b")["c"].sum().reindex(idx).fillna(0.0)
        gr["is_sleep"] = interval(idx, T[(p, "sleep")]).astype(int)
        gr["work_intensity"] = interval(idx, T[(p, "work")], val="intensity")
        ex = T[(p, "exercise")].copy(); gr["exercise_intensity"] = 0.0
        if not ex.empty:
            ex["ts_begin"] = ex["ts"]; ex["ts_end"] = ex["ts"] + pd.to_timedelta(pd.to_numeric(ex["duration"], errors="coerce").fillna(0), unit="min")
            gr["exercise_intensity"] = interval(idx, ex, val="intensity")
        for ev in ("stressors", "hypo_event", "illness"):
            n = len(T[(p, ev)])
            if n: log(ds, f"{p}: {ev} has {n} record(s) (too sparse/empty to use)")

        # event features on the FULL grid (history is valid even across glucose gaps)
        pos = pd.Series(np.arange(len(idx)), index=idx)
        gr["carbs_last_2h"] = gr["meal_carbs"].rolling(24, min_periods=1).sum()
        gr["bolus_last_2h"] = gr["bolus_dose"].rolling(24, min_periods=1).sum()
        gr["bolus_last_4h"] = gr["bolus_dose"].rolling(48, min_periods=1).sum()
        for ev, col in ((gr["meal_carbs"] > 0, "mins_since_meal"), (gr["bolus_dose"] > 0, "mins_since_bolus")):
            gr[col] = ((pos - pos.where(ev).ffill()) * 5).fillna(360).clip(upper=360)
        gr["hr_mean_30m"] = gr["hr"].rolling(6, min_periods=1).mean()
        gr["steps_sum_30m"] = gr["steps"].rolling(6, min_periods=1).sum()
        a_i, a_c = 1 - 0.5 ** (5 / 90), 1 - 0.5 ** (5 / 60)
        gr["iob_proxy"] = gr["bolus_dose"].ewm(alpha=a_i, adjust=False).mean() / a_i
        gr["cob_proxy"] = gr["meal_carbs"].ewm(alpha=a_c, adjust=False).mean() / a_c
        valid = gr["glucose"].notna()
        gr["seg"] = np.where(valid, (valid != valid.shift()).cumsum(), -1)
        gr.insert(0, "patient", p)
        grids.append(gr.rename_axis("dt").reset_index())
    full = pd.concat(grids, ignore_index=True)
    full["seg_id"] = np.where(full["seg"] >= 0, full["patient"] + "_" + full["seg"].astype(str), "")
    clean = full.drop(columns=["seg"]).rename(columns={"seg_id": "segment_id"})

    # glucose-based features inside segments only
    v = full[full["seg"] >= 0].copy()
    g = v.groupby("seg_id")
    for L in (1, 2, 3, 6, 9, 12):
        v[f"glucose_lag_{5 * L}m"] = g["glucose"].shift(L)
    v["glucose_delta_15"] = v["glucose"] - v["glucose_lag_15m"]
    v["glucose_delta_30"] = v["glucose"] - v["glucose_lag_30m"]
    v["glucose_delta_60"] = v["glucose"] - v["glucose_lag_60m"]
    v["glucose_roll_mean_30m"] = g["glucose"].rolling(6, min_periods=6).mean().droplevel(0)
    v["glucose_roll_std_30m"] = g["glucose"].rolling(6, min_periods=6).std().droplevel(0)
    v["in_range"] = ((v["glucose"] >= 70) & (v["glucose"] <= 180)).astype(float)
    v["below_range"] = (v["glucose"] < 70).astype(float)
    for nm, col, w, fn in (("glucose_mean_6h", "glucose", 72, "mean"), ("glucose_std_6h", "glucose", 72, "std"),
                           ("glucose_mean_24h", "glucose", 288, "mean"), ("glucose_tir_24h", "in_range", 288, "mean"),
                           ("glucose_tbr_24h", "below_range", 288, "mean")):
        v[nm] = getattr(g[col].rolling(w, min_periods=12), fn)().droplevel(0)
    tod = v["dt"].dt.hour + v["dt"].dt.minute / 60
    v["hour_sin"] = np.sin(2 * np.pi * tod / 24); v["hour_cos"] = np.cos(2 * np.pi * tod / 24)
    v["target_glucose_t30"] = g["glucose"].shift(-6)
    v["target_glucose_t60"] = g["glucose"].shift(-12)
    fut = pd.concat([g["glucose"].shift(-i) for i in range(1, 13)], axis=1)
    v["target_hypo_60"] = np.where(fut.isna().any(axis=1), np.nan, (fut.min(axis=1) < 70).astype(float))

    gcols = ["glucose"] + [f"glucose_lag_{5 * L}m" for L in (1, 2, 3, 6, 9, 12)] + \
            ["glucose_delta_15", "glucose_delta_30", "glucose_delta_60", "glucose_roll_mean_30m", "glucose_roll_std_30m",
            "glucose_mean_6h", "glucose_std_6h", "glucose_mean_24h", "glucose_tir_24h", "glucose_tbr_24h"]
    targets = ["target_glucose_t30", "target_glucose_t60", "target_hypo_60"]
    m = v.dropna(subset=gcols + targets).copy()
    log(ds, f"model rows after dropping incomplete windows: {len(m)} of {len(full)} grid rows")

    # temporal split per patient with 1h purge gap at each boundary
    m = m.sort_values(["patient", "dt"]).reset_index(drop=True)
    rank = m.groupby("patient").cumcount(); n = m.groupby("patient")["dt"].transform("size")
    a, b = (n * 0.70).astype(int), (n * 0.85).astype(int)
    m["split"] = np.where(rank < a, "train", np.where(rank < b, "val", "test"))
    purge = ((rank >= a - 12) & (rank < a + 12)) | ((rank >= b - 12) & (rank < b + 12))
    m = m.loc[~purge].reset_index(drop=True)
    log(ds, "temporal split per patient 70/15/15 (first 70% of time = train); 12-step purge around boundaries")

    # impute wearable gaps with TRAIN medians (indicator columns stay)
    wear = ["hr", "gsr", "skin_temp_c", "air_temp_c", "steps", "hr_mean_30m", "steps_sum_30m"]
    trm = m["split"] == "train"
    med = {c: float(m.loc[trm, c].median()) for c in wear}
    for c in wear:
        m[c] = m[c].fillna(med[c] if np.isfinite(med[c]) else 0.0)
    cont = gcols + wear + ["basal_rate", "bolus_last_2h", "bolus_last_4h", "carbs_last_2h", "mins_since_meal",
                           "mins_since_bolus", "work_intensity", "exercise_intensity", "iob_proxy", "cob_proxy"]
    binary = ["glucose_interpolated", "temp_basal_active", "is_sleep", "hr_missing", "gsr_missing", "skin_temp_c_missing",
              "air_temp_c_missing", "steps_missing", "hour_sin", "hour_cos"]
    feats = cont + binary
    feats = [c for c in feats if m.loc[trm, c].nunique() > 1]
    cont = [c for c in cont if c in feats]; binary = [c for c in binary if c in feats]
    dropped_const = [c for c in (gcols + wear + binary) if c not in feats]
    if dropped_const: log(ds, f"constant-in-train features dropped: {dropped_const}")
    params = zscore(m, cont, trm)
    m = m.rename(columns={"seg_id": "segment_id"})
    out = m[["patient", "segment_id", "dt", "split"] + feats + targets]
    check_model_ready(ds, out, feats, targets)
    log(ds, f"hypo (<70 in next 60 min) rate: {out['target_hypo_60'].mean():.3f}")
    write(clean, ds, "clean"); write(out, ds, "model_ready")
    write_schema(ds, id_cols=["patient", "segment_id", "dt"], group_col="patient",
                 split_method="temporal 70/15/15 inside each patient, 1h purge gap", features=feats, targets=targets,
                 imputed_with_train_median=med, scaled_cols_mean_std=params, unscaled_cols=binary,
                 note="5-min grid. Temps converted F->C. Sleep intervals with end<begin were swapped. "
                      "Window never crosses a glucose gap > 30 min. Patients P1..P6 = source files in order.")


REQUIRED = {"glucotwin": ["GlucoTwin.csv"], "smart_wearable": ["smart_wearable.csv"],
            "cgm_t1dm_t2dm": ["t1dm.csv", "t2dm.csv"], "ohhim": ["ohhim.csv"]}
FUNCS = {"glucotwin": prep_glucotwin, "smart_wearable": prep_wearable, "cgm_t1dm_t2dm": prep_cgm, "ohhim": prep_ohio}


def run_all(raw_dir="data", out_dir="processed", only=None):
    """Clean + engineer + split + scale. `only` = list of dataset keys (default all)."""
    global IN, OUT
    IN, OUT = Path(raw_dir), Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (only or list(FUNCS)):
        missing = [f for f in REQUIRED[d] if not (IN / f).exists()]
        if missing:
            raise FileNotFoundError(f"[{d}] missing {missing} in {IN.resolve()}  (use --raw_dir)")
        FUNCS[d]()
    (OUT / "prep_log.json").write_text(json.dumps(LOG, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data"); ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--datasets", nargs="*", default=None, choices=list(FUNCS))
    a = ap.parse_args()
    run_all(a.raw_dir, a.out_dir, a.datasets)
    print("\nALL DONE ->", OUT)
