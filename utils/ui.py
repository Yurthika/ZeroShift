"""Shared UI helpers."""
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st


def render_header(page_title: str) -> None:
    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(f"## {page_title}")
    with col2:
        st.markdown(
            f"<div style='text-align:right;padding-top:8px;color:#B0BEC5'>{now} "
            f"<span style='background:#1B5E20;color:#fff;padding:4px 10px;border-radius:12px;'>OPERATIONAL</span></div>",
            unsafe_allow_html=True,
        )
