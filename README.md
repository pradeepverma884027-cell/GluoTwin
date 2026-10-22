GlucoTwin — A CGM-Based Digital Twin Research Prototype
Project title: GlucoTwin: A CGM-Based Digital Twin Research Prototype
Repository: https://github.com/pradeepverma884027-cell/GluoTwin
Research-use disclaimer: GlucoTwin is a retrospective research and educational prototype, not a medical device. It is not clinically validated, does not process live CGM data, and must not be used for diagnosis, treatment, insulin dosing, or other medical decisions.

1. Team Details
- Team name: pradeepverma884027
- Team members and roles: 
.Pradeep Kumar  - Project Lead & ML Engineer
.Shivank Atri   - Data Analyst
.Rakesh Yadav   - Data Research
.Rohan Dwivedi  - Frontend Developer
- Team lead / public contact email: Pradeep Kumar & pradeepverma884027@gmail.com 

2. College / Incubator Information
- College / Institution: IMS ENGINEERING COLLEGE
- Department / Programme: Computer Science and Engineering — Artificial Intelligence and Machine Learning
- Incubator / Accelerator: Not applicable.
- 
3. Problem Statement
Continuous glucose monitoring (CGM) produces time-series glucose readings, but a current reading alone does not show how glucose may evolve over the next several minutes or hours. Forecasts should also communicate uncertainty, and retrospective performance must not be presented as clinical proof.

GlucoTwin explores a research workflow that replays historical CGM states, produces multi-horizon glucose forecasts, displays prediction intervals, and reports held-out evaluation metrics. A separate synthetic electronic health record (EHR) panel demonstrates a possible interface for future integration without claiming verified patient linkage.

4. Healthcare Use Case
Intended research use: retrospective exploration of glucose trends and forecast uncertainty for diabetes-related CGM research.

The dashboard supports:
- Historical CGM replay for a selected patient and timestamp.
- Forecast horizons of +30, +60, +90 and +120 minutes.
- 80% and 90% prediction intervals using saved conformal calibration radii.
- A held-out evaluation summary.
- A synthetic EHR demonstration showing demographic and record-count summaries.
Important boundaries
- No real-time CGM ingestion is implemented.
- Synthetic EHR data is for demonstration only and is not model input.
- The virtual EHR-to-CGM mapping is arbitrary; clinical linkage is not verified.
- The prototype is not clinically validated and must not guide treatment.
  
5. Technical Stack
- Language: Python
- Dashboard: Streamlit
- Data processing: Pandas, NumPy
- Machine learning: scikit-learn, LightGBM, XGBoost
- Sequence modelling explored: TensorFlow / Keras GRU
- Model persistence: joblib and saved model artifacts where applicable
- Synthetic EHR demonstration: Synthea-generated synthetic records
- Version control: Git and GitHub
Individual scripts may require additional packages. Review the relevant requirements files and script input paths before running them.

6. AI / ML Model and Framework Details
Models explored across the project include Linear Regression, Random Forest, Extra Trees, LightGBM, XGBoost and GRU. The dashboard uses saved artifacts and retrospective historical data; it does not retrain models when launched.
Retrospective held-out CGM forecast results
Recorded model-ready dataset summary: 95,619 rows, 80 unique CGM patient identities, and 33 prepared features. Recorded split sizes are 67,532 training rows, 10,713 validation rows and 17,374 test rows.
Forecast horizon	MAE	RMSE	R²
+30 minutes	9.068	12.862	0.957
+60 minutes	16.258	23.015	0.862
+90 minutes	21.226	29.595	0.772
+120 minutes	24.864	34.331	0.694


Prediction interval empirical coverage on the held-out test set:
- 80% intervals: 82.626%–83.290% across the four horizons.
- 90% intervals: 91.993%–93.006% across the four horizons.
These are retrospective research metrics. They do not demonstrate prospective accuracy, clinical utility, safety, or generalisation to other cohorts. Results depend on data preparation, split strategy and evaluation protocol.

7. Architecture
Historical CGM dataset
        |
        v
Historical patient / segment / timestamp selection
        |
        v
Saved forecasting models + inference pipeline
        |
        v
+30 / +60 / +90 / +120 minute forecasts
        |
        +----> 80% and 90% prediction intervals
        |
        +----> Held-out evaluation summary
        |
        v
Streamlit research dashboard

Synthetic Synthea EHR records
        |
        v
Separate EHR demo panel / arbitrary virtual mapping
        (not model input; clinical linkage unverified)
Architecture diagram (PDF/PPT): [UPLOAD THE ARCHITECTURE DIAGRAM TO THIS REPOSITORY AND ADD ITS LINK HERE]
8. Demo Video (15–20 Minutes)
Unlisted YouTube demo: https://youtu.be/nY0XYbmTmYQ

The recording should demonstrate the dashboard, historical replay, multi-horizon forecasts, uncertainty intervals, held-out evaluation and the separate synthetic EHR panel. Clearly state that the system is retrospective and not clinically validated.
9. Presentation
Project presentation (PDF/PPT): [UPLOAD THE FINAL PRESENTATION AND ADD ITS REPOSITORY LINK HERE]

The presentation should cover the project motivation, problem statement, architecture, dataset and split, forecasting workflow, uncertainty estimates, held-out results, synthetic EHR demonstration, limitations and next steps.
10. Open-Source License Details
Current status: No open-source license has been specified in the current README. Until a license is chosen and a corresponding LICENSE file is included, do not assume that the repository may be reused, modified or redistributed.

License: [SELECT AN APPROPRIATE LICENSE AND ADD THE LICENSE FILE AFTER CONFIRMING RIGHTS TO LICENSE ALL INCLUDED CONTENT]

Third-party datasets, model artifacts, libraries and synthetic-data tooling may have their own terms. Check those terms separately before redistribution.
11. Repository Structure
- training/ — training, evaluation, inference, dashboard and audit scripts.
- files/ — shared preprocessing and model-training helpers.
- glucose_hpc/ — HPC and SLURM-oriented scripts.
- DATA_PREP_REPORT.md — data preparation notes.
- README_START_HERE.md — additional setup instructions.
- add_ehr_panel.py, add_fusion_panel.py — EHR/fusion demo helper scripts.
- pect_event_records.py — event-record extraction / processing script.
Raw datasets, patient-level records and trained model artifacts are not included in every checkout. Obtain authorized access to required data and artifacts before running scripts that depend on them. Do not publish patient identifiers, private health records, credentials or private local paths.
12. Setup and Running the Dashboard (Windows)
From the repository root:
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib
Additional dependencies may be required by the selected model or script, including LightGBM, XGBoost or TensorFlow. Install versions appropriate to the workflow and available artifacts.

Run the digital-twin proof-of-concept dashboard from the directory containing the expected project data and artifacts:
python -m streamlit run training\glucotwin_digital_twin_poc.py
Some scripts use project-specific relative paths and require prepared datasets or previously generated artifacts. Review their source and input requirements before execution. The dashboard should not be assumed to work from an arbitrary directory without the expected data layout.
13. Submission Requirements Checklist
Before submitting, confirm that each required item is present in this README and that every linked resource opens without login or special permission:
- [ ] Team details completed.
- [ ] Official college / incubator information completed.
- [x] Project title included.
- [x] Problem statement included.
- [x] Healthcare use case included.
- [x] Technical stack included.
- [x] AI/ML model and framework details included.
- [ ] 15–20 minute unlisted YouTube demo uploaded and linked.
- [ ] Open-source license selected and LICENSE file added.
- [ ] Architecture diagram PDF/PPT uploaded and linked.
- [ ] Final project presentation PDF/PPT uploaded and linked.
- [ ] All required files and links confirmed publicly accessible without additional permissions.
14. Limitations and Future Work
Current limitations include retrospective-only evaluation, no live CGM connection, no clinical validation, no verified EHR-to-CGM linkage and no EHR features supplied to the forecasting model. Future work may include evaluation on independent cohorts, subgroup analysis, improved uncertainty calibration, verified and consented linkage, and carefully validated real-time integration.
15. Responsible Use
CGM and EHR data can be sensitive health information. Confirm data permissions before sharing datasets. Keep patient-level records, identifiers, credentials, access tokens and private local paths out of the public repository. Synthetic EHR examples are not real patient records and do not establish a clinical relationship to any CGM identity.
