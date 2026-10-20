# Glucose prediction project — start here

## Important
- The original prepared datasets and schemas are in `DATASET/processed/` in the supplied ZIP's nested structure. This project helper searches for that location automatically.
- No dataset path or command-line argument is required. Change `DATASET_NAME` and `TARGET_COLUMN` in `training/common.py` only when you want a different dataset/target.
- The default is `cgm_t1dm_t2dm` with target `target_cgm_t30` (regression).
- The prepared data already contains `train`, `val`, and `test` splits. Keep them intact. Models use train to fit, validation for model/threshold selection, and test for final reporting.
- This is a research/educational project, not a clinical tool.

## Setup in VS Code (Windows)
1. Extract the complete dataset ZIP.
2. Open the extracted top-level `DATASET` folder in VS Code.
3. Keep the `training` folder beside the nested `DATASET` folder that contains `processed`.
4. Open Terminal > New Terminal and install core packages:
   `python -m pip install pandas numpy scikit-learn joblib`
5. Run `training/00_inspect_dataset.py` first using VS Code's Run Python File button.
6. Run each model separately using the Run Python File button:
   - `01_train_linear.py`
   - `02_train_random_forest.py`
   - `03_train_extra_trees.py`
7. For LightGBM, install `python -m pip install lightgbm`, then run `04_train_lightgbm.py`.
8. For XGBoost, install `python -m pip install xgboost`, then run `05_train_xgboost.py`.

## Output
Every model writes its own `.joblib` file to `training/results/models/` and a metrics CSV to `training/results/`. Models are compared on the same held-out test split. For regression, prioritize RMSE and MAE (lower is better) and R² (higher is generally better); do not call R² an accuracy percentage. For classification, use ROC-AUC, PR-AUC, precision, recall, and F1.

## Change dataset/target
Edit the two constants at the top of `training/common.py`:
- `DATASET_NAME = "cgm_t1dm_t2dm"`
- `TARGET_COLUMN = "target_cgm_t30"`

Use target names from the matching `schema_*.json`. The GRU/time-series model is not included in these first baseline scripts; establish tabular baselines before adding a sequence model.
