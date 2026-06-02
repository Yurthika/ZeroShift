"""
ZeroShift — main entry. Loads page modules dynamically so sidebar stays consistent.
Run from zeroshift/: streamlit run app.py
"""
import importlib.util
import os

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))

PAGES = {
    "🏠 Overview Dashboard": os.path.join("pages", "1_Overview.py"),
    "⚡ Energy Monitor": os.path.join("pages", "2_Energy_Monitor.py"),
    "🚨 Anomaly Detection": os.path.join("pages", "3_Anomaly_Detection.py"),
    "🔮 AI Forecasting": os.path.join("pages", "4_AI_Forecasting.py"),
    "🔧 Predictive Maintenance": os.path.join("pages", "5_Predictive_Maintenance.py"),
    "💡 Recommendations": os.path.join("pages", "6_Recommendations.py"),
    "♻️ Carbon Tracker": os.path.join("pages", "7_Carbon_Tracker.py"),
    "🤖 AI Agent": os.path.join("pages", "8_AI_Agent.py"),
    "📊 Data Analytics": os.path.join("pages", "9_Data_Analytics.py"),
    "📋 Reports": os.path.join("pages", "10_Reports.py"),
}


def _load_page_module(relpath: str):
    full = os.path.join(ROOT, relpath)
    mod_name = "zs_" + relpath.replace(os.sep, "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, full)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {full}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sidebar() -> str:
    st.sidebar.markdown(
        "<div style='font-size:2.1rem;font-weight:800;color:#FFB300;'>⚡ ZeroShift</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**UltraTech | Balaji Cement Works**")
    st.sidebar.divider()

    choice = st.sidebar.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        """
---
<span style='color:#66BB6A'>● SYSTEM ONLINE</span><br/>
""",
        unsafe_allow_html=True,
    )
    from datetime import datetime
    from zoneinfo import ZoneInfo

    h = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    if 6 <= h < 14:
        sh = "Morning"
    elif 14 <= h < 22:
        sh = "Afternoon"
    else:
        sh = "Night"
    st.sidebar.markdown(f"**Shift:** {sh}")
    st.sidebar.caption("DCS: Emerson Ovation ✓")
    st.sidebar.caption("Last Sync: just now")
    st.sidebar.markdown("---")
    try:
        from utils.db_manager import count_new_anomalies

        n = count_new_anomalies()
    except Exception:
        n = 0
    st.sidebar.markdown(
        f"Active Anomalies: <span style='color:#FF5252;font-weight:700'>{n}</span>",
        unsafe_allow_html=True,
    )
    return choice


def main() -> None:
    st.set_page_config(
        page_title="ZeroShift | CPP Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    from utils.db_manager import init_db

    init_db()
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    choice = _sidebar()
    mod = _load_page_module(PAGES[choice])
    if hasattr(mod, "render"):
        mod.render()
    else:
        st.error("Page module missing render()")


main()
