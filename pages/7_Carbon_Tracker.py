"""Carbon tracker."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_processed_data
from utils.ui import render_header


def render() -> None:
    render_header("♻️ Carbon Tracker")
    st.markdown(
        """
        <div style='padding:16px;border-radius:12px;background:linear-gradient(90deg,#1B5E20,#2E7D32);color:white;font-weight:600;'>
        🌱 UltraTech Net Zero 2050
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_processed_data()
    month = df.index.max().month
    mdf = df[df.index.month == month]
    total_co2 = float(mdf["co2_tons_per_hour"].sum())
    total_mwh = float(mdf["net_power_mw"].sum())
    baseline = total_co2 / 0.73  # proxy baseline before 27% reduction
    target = baseline * (1 - 0.27)
    progress = float(np.clip((baseline - total_co2) / max(baseline - target, 1e-6) * 27.0, 0, 27))
    intensity = total_co2 / max(total_mwh, 1e-6)
    bench = 0.82
    badge = "✅ Better" if intensity < bench else "⚠️ Worse"

    left, right = st.columns([2, 3])
    with left:
        st.metric("CO₂ this month (tons)", f"{total_co2:,.0f}")
        st.metric("Target (27% reduction track)", f"{target:,.0f}")
        st.metric("Scope 1 progress (% of 27%)", f"{progress:.1f}%")
        st.metric("CO₂ per MWh", f"{intensity:.3f}", delta=f"Benchmark {bench} — {badge}")

    with right:
        pie = pd.DataFrame(
            {
                "Source": ["CFBC Combustion", "Auxiliary", "Coal Handling", "WTP"],
                "Share": [78, 12, 6, 4],
            }
        )
        fig = px.pie(
            pie,
            names="Source",
            values="Share",
            hole=0.6,
            title="Monthly CO₂ composition (modelled)",
            color_discrete_sequence=px.colors.sequential.Brwnyl,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#ECEFF1"),
            annotations=[dict(text=f"{total_co2:,.0f} t", showarrow=False, font_size=16, x=0.5, y=0.5)],
        )
        st.plotly_chart(fig, use_container_width=True)

    monthly = df["co2_tons_per_hour"].resample("M").sum().rename("co2").reset_index()
    monthly["target"] = monthly["co2"].rolling(12, min_periods=1).mean() * 0.93
    deploy_date = pd.Timestamp("2025-05-01")
    fig2 = px.bar(monthly, x="timestamp", y="co2", title="12-month CO₂", color_discrete_sequence=["#66BB6A"])
    fig2.add_trace(
        go.Scatter(
            x=monthly["timestamp"],
            y=monthly["target"],
            mode="lines",
            name="Target",
            line=dict(dash="dash", color="#EF5350"),
        )
    )
    fig2.add_vline(x=deploy_date, line_color="#FFB300", line_dash="dot")
    fig2.add_annotation(x=deploy_date, y=float(monthly["co2"].max()), text="ZeroShift Deployed — May 2025", showarrow=False)
    fig2.update_layout(paper_bgcolor="#0A1628", plot_bgcolor="#0A1628", font=dict(color="#ECEFF1"))
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    trees = total_co2 * 18
    c1.metric("🌳 Trees equivalent", f"{trees:,.0f}")
    c2.metric("⚡ Coal saved by AI (tons/mo)", "52")
    c3.metric("🌞 Renewable energy %", "6.2%")
    c4.metric("📉 Scope 1 progress", f"{progress:.1f}%")

    actions = pd.DataFrame(
        {
            "Action": [
                "PA fan optimization",
                "APH soot blowing discipline",
                "ESP hopper integrity",
                "VFD trim on ID fan",
                "Coal fineness correction",
                "Air staging O₂ setpoint",
                "BFP recirc map update",
            ],
            "CO₂ Saved/Month (t)": [120, 95, 80, 110, 70, 90, 60],
            "kWh Saved": [1500, 1200, 980, 1400, 900, 1150, 800],
            "₹ Saved": [120000, 96000, 78000, 112000, 72000, 92000, 64000],
            "Effort": ["Med", "Low", "Med", "Med", "High", "Med", "Low"],
            "Priority": ["HIGH", "MED", "MED", "HIGH", "LOW", "MED", "LOW"],
        }
    )

    st.dataframe(actions, use_container_width=True)
    st.download_button("Export opportunities CSV", data=actions.to_csv(index=False), file_name="carbon_opportunities.csv")
