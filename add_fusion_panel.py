
from pathlib import Path

dashboard = Path("training/glucotwin_digital_twin_poc.py")
marker = "# Only held-out test rows are offered in this first prototype."

source = dashboard.read_text(encoding="utf-8")

if marker not in source:
    raise SystemExit("Insertion point not found; dashboard unchanged.")

if "Synthetic virtual-patient fusion demo" in source:
    raise SystemExit("Fusion panel already exists; dashboard unchanged.")

panel = '''
# Synthetic virtual-patient fusion demo
FUSION_MAP_PATH = Path(
    r"C:\\Users\\pradeep\\Desktop\\gluotwinv2\\synthea_setup"
    r"\\glucotwin_ehr_demo\\synthetic_fusion_mapping.csv"
)
EHR_SUMMARY_PATH = Path(
    r"C:\\Users\\pradeep\\Desktop\\gluotwinv2\\synthea_setup"
    r"\\glucotwin_ehr_demo\\synthetic_ehr_patient_summary.csv"
)

st.subheader("Synthetic Virtual-Patient Fusion Demo")
st.warning(
    "DEMO ONLY: EHR and CGM identities are paired arbitrarily for UI "
    "demonstration. This is not verified clinical linkage. Forecasts use "
    "the existing CGM model only; synthetic EHR data is not model input."
)

if FUSION_MAP_PATH.is_file() and EHR_SUMMARY_PATH.is_file():
    fusion_map = pd.read_csv(FUSION_MAP_PATH)
    ehr_summary = pd.read_csv(EHR_SUMMARY_PATH)

    virtual_id = st.selectbox(
        "Select virtual demo patient",
        fusion_map["virtual_patient_id"].astype(str).tolist(),
        key="fusion_virtual_patient",
    )
    link = fusion_map[
        fusion_map["virtual_patient_id"].astype(str) == virtual_id
    ].iloc[0]
    ehr_match = ehr_summary[
        ehr_summary["synthetic_patient_id"].astype(str)
        == str(link["synthetic_ehr_id"])
    ]

    st.write({
        "Virtual demo ID": virtual_id,
        "Synthetic EHR ID": str(link["synthetic_ehr_id"]),
        "CGM demo identity": str(link["cgm_patient_uid"]),
        "Mapping type": str(link["mapping_type"]),
        "Clinical linkage verified": False,
    })

    if not ehr_match.empty:
        row = ehr_match.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Age", str(row.get("age_years", "Unknown")))
        b.metric("Condition records", int(row.get("condition_records", 0)))
        c.metric("Observation records", int(row.get("observation_records", 0)))
        d.metric("Medication records", int(row.get("medication_records", 0)))

    st.caption(
        "The virtual mapping is for demonstration only. It does not establish "
        "that the EHR and CGM records belong to the same person."
    )
else:
    st.warning("Fusion mapping or EHR summary file not found.")

'''

dashboard.write_text(
    source.replace(marker, panel + "\\n" + marker, 1),
    encoding="utf-8",
)

print("Synthetic virtual-patient fusion demo panel added.")
