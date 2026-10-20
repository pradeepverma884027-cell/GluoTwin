# Data Preparation Report

Re-run everything: `python3 processed/prep_pipeline.py`. Each dataset gets `*_clean.csv` (readable), `*_model_ready.csv` (encoded, scaled with TRAIN-only stats, has `split` column, zero NaN in features) and `schema_*.json` (features, targets, scaling params).

| Dataset | Rows (model-ready) | Task / targets | Split |
|---|---|---|---|
| GlucoTwin | 6,538 meals, 240 people | `Target_*` (glucose t+30/60/90/120, peak, iAUC, spike>180) | by participant, stratified by diabetes status |
| smart_wearable | 1,000 | `health_risk` (0/1, 60/40) | stratified rows |
| t1dm + t2dm | 95,619 windows, 80 patients | `target_cgm_t30`, `target_cgm_t60`, `target_hypo_60` | by patient |
| ohhim | 14,935 windows, 6 patients | `target_glucose_t30`, `_t60`, `target_hypo_60` | time-based per patient, 1h purge gap |

## Steps done per dataset
**GlucoTwin** - no missing/duplicates. Dates confirmed mm/dd/yyyy. Merged "African American Black" into "African American". Time strings to minutes, hour sin/cos. Winsorised wearable columns (train 0.5/99.5 pct). **Removed leakage**: `Post_HR_Mean_60`, `Post_Mets_Mean_60`, `Next_Meal_Within_120min` (information from after the meal). Dropped redundant (|r|>0.95): `HR_Mean_60`, `Fingerstick_GLU_1`, `VLDL_Cal`, `LDL_Cal`. `Diabetes_Status` kept only as aux column (it is a direct function of A1c). One-hot meal type + ethnicity. 15 rows hit the 400 mg/dL sensor ceiling (flagged in clean file).

**smart_wearable** - 22 SpO2 values >100% capped at 100; 6 BMI <15 treated as errors, median-imputed (flag column added); `year` dropped; `pulse_pressure` added. Data looks synthetic-ish: only steps (r=-0.48), stress, BMI, workouts relate to the target.

**t1dm + t2dm** - merged with `diabetes_type`; dropped Chinese duplicate column and file-name columns; 1 row with no date/CGM removed. Free-text parsed to numbers: s.c./i.v. insulin units (IU), long-acting flag, meal grams, number of oral drugs (doses were mixed units, not summable), pump suspend flag. Basal rate carried forward. Series cut into 93 segments at gaps >45 min, short gaps (13 steps) interpolated, flagged. Lags, deltas, rolling stats, insulin-on-board proxy, time-since-meal/insulin built **inside segments only**. Sensor floor (39.6, 466 rows) and ceiling flagged, kept. CBG and ketone left out of features (97-100% missing).

**ohhim** - JSON text cells parsed into 18 event tables (type order inferred from standard OhioT1DM layout). 6 files = P1..P6. Dates dd-mm-yyyy. Temperatures F to C. Sleep intervals with end<begin swapped. 5-min grid; glucose gaps <=30 min interpolated, longer gaps split segments. Wearable gaps (~6%) filled with train median + missing flag. Basal step function + temp basal override, bolus/meal impulses, 2h/4h sums.

## Things you should know
- Scaling is fit on train only. Targets stay in original units.
- Use group-aware CV (`Participant_ID` / `patient_uid` / `patient`), never random row split on the time-series sets.
- Hypo class is rare (~5%): use class weights or PR-AUC.
- `ohhim` patient ids and event order are inferred; check if you know the real ids.
- Very few ohhim patients (6): think about per-patient models.
- Outputs are CSV; ~78 MB total.

## Update v2: extra feature engineering (already applied in the files)
- **GlucoTwin:** added `Dexcom_minus_Fasting`, `Carb_Fiber_Ratio`, `Carb_Cal_Share`, `TG_HDL_Ratio`, `Glucose_Range_60`, `Rise_From_Min_60`. 22 microbiome scores replaced by 5 PCA parts (fit on train, 47% variance). `HOMA_IR` was built but dropped as redundant with `Insulin`.
- **smart_wearable:** added `steps_x_stress`, `short_sleep` (<6h), `obese` (BMI>=30).
- **t1dm+t2dm / ohhim:** added 6h and 24h glucose mean/std, 24h time-in-range and time-below-range, decayed insulin and carb load (`iob_proxy`, `cob_proxy`; half-life 90 / 60 min). All use past values only.
