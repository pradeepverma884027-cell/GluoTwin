\# GlucoTwin — CGM Forecasting and Glucose-Risk Analysis



GlucoTwin is a research and educational project exploring continuous glucose monitoring (CGM) forecasting and retrospective low-glucose-risk classification using Python and machine-learning models.



> \*\*Disclaimer:\*\* This is a research project, not a medical device. Its predictions and evaluation results are not clinically validated and must not be used to make treatment decisions.



\## Project Goals



\- Forecast future CGM values at multiple time horizons.

\- Explore retrospective low-glucose risk over a 60-minute horizon.

\- Compare machine-learning models, including Random Forest, LightGBM, XGBoost and GRU.

\- Audit feature timing, preprocessing, calibration and episode-based evaluation.



\## Repository Structure



\- `training/` — training, evaluation, inference, dashboard and audit scripts.

\- `files/` — shared preprocessing and model-training helpers.

\- `glucose\_hpc/` — HPC and SLURM-oriented scripts.

\- `DATA\_PREP\_REPORT.md` — data preparation notes.

\- `README\_START\_HERE.md` — additional setup instructions.



Raw datasets, patient-level records and trained model artifacts are not included. You must have authorized access to the required dataset before running scripts that depend on it.



\## Setup (Windows)



Create and activate a virtual environment from the repository root:



```cmd

python -m venv .venv

.venv\\Scripts\\activate

python -m pip install --upgrade pip

python -m pip install numpy pandas scikit-learn joblib

```



Some scripts additionally require LightGBM, XGBoost, TensorFlow or other packages. Check `training/requirements.txt` and `glucose\_hpc/requirements.txt` for the workflow you want to run.



\## Getting Started



1\. Read `README\_START\_HERE.md` and `training/README.md`.

2\. Obtain the dataset from an authorized source.

3\. Place the prepared dataset in the location expected by the selected script.

4\. Inspect the data layout and schema.

5\. Check each script's input paths and dependencies before execution.

6\. Keep training, validation and test splits separate.



Example inspection command:



```cmd

python training/00\_inspect\_dataset.py

```



Not every script is standalone; some require prepared data, previously generated artifacts or additional dependencies.



\## Exploratory Model Results



One retrospective evaluation of the `target\_hypo\_60` classification task reported these row-weighted results:



| Model | PR-AUC | ROC-AUC | Brier Score | Precision at 0.20 | Recall at 0.20 |

|---|---:|---:|---:|---:|---:|

| LightGBM | 0.8483 | 0.9805 | 0.0479 | 0.5446 | 0.9478 |

| XGBoost | 0.9136 | 0.9837 | 0.0281 | 0.7010 | 0.9004 |



The 0.20 threshold is exploratory, not a recommended medical alert threshold. These retrospective metrics do not establish clinical safety or prospective performance.



\## Data Privacy and Responsible Use



CGM records may contain sensitive health information. Confirm redistribution rights before sharing datasets. Never publish patient identifiers, raw records, patient-level exports, credentials or private local paths.



\## Limitations



\- Results depend on dataset quality, preprocessing and split strategy.

\- Retrospective evaluation does not prove prospective performance or clinical benefit.

\- Some legacy scripts may require local path adjustments.

\- This project has not established clinical validation or regulatory approval.



\## Licenses



No license is currently specified. Reuse and redistribution permissions are therefore not automatically granted.

