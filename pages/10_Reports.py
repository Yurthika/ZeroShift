"""Reports and exports."""
import pandas as pd
import streamlit as st

from utils.data_loader import load_processed_data
from utils.db_manager import fetch_reports, init_db, insert_report_record
from utils.report_generator import generate_shift_pdf
from utils.ui import render_header


def render() -> None:
    init_db()
    render_header("📋 Reports")

    st.subheader("Report generator")
    rtype = st.radio("Report type", ["Shift Report", "Daily", "Weekly", "Monthly", "Anomaly Report"], horizontal=True)
    d1, d2 = st.columns(2)
    with d1:
        p0 = st.date_input("Period start", value=pd.Timestamp("2025-05-01").date())
    with d2:
        p1 = st.date_input("Period end", value=pd.Timestamp("2025-05-19").date())

    sec_exec = st.checkbox("Executive KPI Summary", value=True)
    sec_energy = st.checkbox("Energy Trend Analysis", value=True)
    sec_anom = st.checkbox("Anomaly Detection Log", value=True)
    sec_rec = st.checkbox("AI Recommendations", value=True)
    sec_carb = st.checkbox("Carbon Footprint Summary", value=True)
    sec_fc = st.checkbox("Forecast Accuracy", value=True)
    st.checkbox("Raw Data Export", value=False)

    sections = []
    if sec_exec:
        sections.append("Executive KPI Summary")
    if sec_energy:
        sections.append("Energy Trend Analysis")
    if sec_anom:
        sections.append("Anomaly Detection Log")
    if sec_rec:
        sections.append("AI Recommendations")
    if sec_carb:
        sections.append("Carbon Footprint Summary")
    if sec_fc:
        sections.append("Forecast Accuracy")

    df = load_processed_data()
    sub = df.loc[str(p0) : str(p1)]

    with st.expander("👁️ Preview"):
        st.write(sub.tail(12))

    if st.button("📥 Prepare PDF"):
        pdf_bytes = generate_shift_pdf(sub, rtype, f"{p0} → {p1}", sections)
        insert_report_record(
            name=f"{rtype} {p0}",
            report_type=rtype,
            period_start=str(p0),
            period_end=str(p1),
            file_size_kb=len(pdf_bytes) / 1024.0,
            sections=sections,
        )
        st.session_state["pdf_bytes"] = pdf_bytes
        st.session_state["pdf_name"] = f"zeroshift_{rtype.replace(' ', '_')}.pdf"

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            label="Download generated PDF",
            data=st.session_state["pdf_bytes"],
            file_name=st.session_state.get("pdf_name", "zeroshift.pdf"),
            mime="application/pdf",
        )

    csv_bytes = sub.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv_bytes, file_name="zeroshift_period.csv")

    st.subheader("Auto-scheduled reports")
    sched = pd.DataFrame(
        {
            "Report": ["Shift Report", "Daily Summary", "Weekly Analysis", "Monthly Audit"],
            "Cadence": ["Every 8 hours", "6:00 AM", "Monday 7AM", "1st of month"],
            "Last Sent": ["2025-05-19 06:00", "2025-05-19 06:00", "2025-05-13 07:00", "2025-05-01 08:00"],
            "Status": ["✅", "✅", "✅", "✅"],
        }
    )
    st.dataframe(sched, use_container_width=True)
    st.toggle("Shift schedule ON", value=True)
    st.toggle("Daily schedule ON", value=True)

    st.subheader("Recent reports")
    reps = fetch_reports(10)
    if reps:
        rdf = pd.DataFrame(reps)
        st.dataframe(rdf, use_container_width=True)
    else:
        st.caption("Generate a PDF to populate the reports table.")
