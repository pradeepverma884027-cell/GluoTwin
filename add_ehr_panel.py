
from pathlib import Path

dashboard = Path("training/glucotwin_digital_twin_poc.py")
ehr_path = Path(
    r"C:\Users\pradeep\Desktop\gluotwinv2\synthea_setup"
    r"\glucotwin_ehr_demo\synthetic_ehr_patient_summary.csv"
)

source = dashboard.read_text(encoding="utf-8")

marker = '# Only held-out test rows are offered in this first prototype.'

if marker not in source:
    raise SystemExit("Insertion point not found; dashboard unchanged.")

if "Synthetic EHR demo profiles" in source:
    raise SystemExit("EHR panel already exists; dashboard unchanged.")

panel = '''
# Synthetic EHR demo profiles — separate from CGM identities.
EHR_PATH = Path(
    r"C:\\Users\\pradeep\\Desktop\\gluotwinv2\\synthea_setup"
    r"\\glucotwin_ehr_demo\\synthetic_ehr_patient_summary.csv"
)

st.subheader("Synthetic EHR Demo Profile")
st.caption(
    "These are synthetic Synthea records. They are NOT linked to the "
    "selected CGM patient and are shown for integration demonstration only."
)

if EHR_PATH.is_file():
    ehr = pd.read_csv(EHR_PATH)
    ehr_ids = ehr["synthetic_patient_id"].astype(str).tolist()
    selected_ehr_id = st.selectbox("Select synthetic EHR profile", ehr_ids)
    ehr_row = ehr[ehr["synthetic_patient_id"].astype(str) == selected_ehr_id].iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("Age (years)", str(ehr_row.get("age_years", "N/A")))
    b.metric("Condition records", int(ehr_row.get("condition_records", 0)))
    c.metric("Observation records", int(ehr_row.get("observation_records", 0)))
    d.metric("Medication records", int(ehr_row.get("medication_records", 0)))

    st.write({
        "Synthetic EHR ID": selected_ehr_id,
        "Gender": str(ehr_row.get("GENDER", "N/A")),
        "Race": str(ehr_row.get("RACE", "N/A")),
        "Ethnicity": str(ehr_row.get("ETHNICITY", "N/A")),
        "Marital status": str(ehr_row.get("MARITAL", "N/A")),
        "Source": "Synthea synthetic EHR",
        "Sensor linkage": "Not linked — demo only",
    })
else:
    st.warning(f"Synthetic EHR summary not found: {EHR_PATH}")

'''

dashboard.write_text(
    source.replace(marker, panel + "\n" + marker, 1),
    encoding="utf-8"
)

print("Synthetic EHR panel added.")
print("Original backup should remain available as .backup")
