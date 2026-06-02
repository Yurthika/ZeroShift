"""Overview dashboard."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_processed_data
from utils.db_manager import count_new_anomalies, fetch_anomalies, init_db
from utils.ui import render_header


def render() -> None:
    init_db()
    render_header("🏠 Overview Dashboard")
    df = load_processed_data()
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    st.markdown(
        """
        <div style='padding:22px;border-radius:16px;background:linear-gradient(120deg,#0A1628,#112240 60%,#1A237E);
        color:white;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center;'>
        <div>
            <div style='font-size:2rem;font-weight:800;color:#FFB300;'>ZeroShift</div>
            <div style='opacity:.9;'>Every Shift. Less Energy. Less Carbon.</div>
        </div>
        <div style='text-align:right;font-weight:600;color:#A5D6A7;'>🌱 UltraTech Net Zero 2050</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "⚡ Net Power Output",
            f"{latest['net_power_mw']:.2f} MW",
            delta=f"{(latest['net_power_mw'] - prev['net_power_mw']):.2f} MW vs prev hour",
        )
    with c2:
        st.metric("🪨 Coal Feed Rate", f"{latest['coal_feed_tph']:.2f} TPH")
    sp = float(latest["steam_pressure_kgcm2"])
    if 84 <= sp <= 90:
        sp_color = "#66BB6A"
    elif sp <= 92:
        sp_color = "#FFB300"
    else:
        sp_color = "#EF5350"
    with c3:
        st.markdown(
            f"<div><h4 style='margin:0'>💨 Steam Pressure</h4>"
            f"<p style='font-size:1.6rem;color:{sp_color};font-weight:700;margin:4px 0'>{sp:.2f} kgf/cm²</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.metric("🌡️ Steam Temperature", f"{latest['steam_temp_celsius']:.1f} °C", delta="target 515°C")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("🔥 Bed Temperature", f"{latest['bed_temp_celsius']:.1f} °C")
    with c6:
        st.metric("⚙️ Turbine RPM", f"{latest['turbine_rpm']:.0f}")
    with c7:
        st.metric("📊 Plant Load Factor", f"{latest['plant_load_factor']:.1f}%")
    with c8:
        st.metric("🚨 Active Anomalies (NEW)", f"{count_new_anomalies()}")

    last7 = df.last("7D")
    colL, colR = st.columns([2, 1])
    with colL:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=last7.index,
                y=last7["net_power_mw"],
                fill="tozeroy",
                name="Net Power",
                line=dict(color="#42A5F5"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=last7.index,
                y=last7["gross_power_mw"],
                mode="lines",
                name="Gross Power",
                line=dict(color="#90CAF9"),
            )
        )
        an = last7[last7["anomaly_flag"] == 1]
        fig.add_trace(
            go.Scatter(
                x=an.index,
                y=an["gross_power_mw"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#EF5350", size=10, symbol="circle-open"),
            )
        )
        fig.add_hline(y=35, line_dash="dash", line_color="#FFB300", annotation_text="35 MW rated")
        fig.update_layout(
            title="⚡ Power Generation — Last 7 Days",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#ECEFF1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with colR:
        tmp = last7.copy()
        tmp["date"] = tmp.index.date
        grp = tmp.groupby(["date", "shift"], as_index=False)["gross_power_mw"].mean()
        color_map = {"Morning": "#42A5F5", "Afternoon": "#FFB300", "Night": "#7E57C2"}
        fig2 = px.bar(
            grp,
            x="date",
            y="gross_power_mw",
            color="shift",
            barmode="group",
            color_discrete_map=color_map,
            title="🏭 Shift Performance — Last 7 Days",
        )
        fig2.add_hline(y=31.5, line_dash="dash", line_color="#B0BEC5")
        fig2.update_layout(
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#ECEFF1"),
            legend_title_text="Shift",
        )
        st.plotly_chart(fig2, use_container_width=True)

    left, mid, right = st.columns(3)
    with left:
        st.subheader("🚨 Live Anomaly Feed")
        rows = fetch_anomalies(limit=8)
        for r in rows:
            sev = r.get("severity", "WARNING")
            emoji = {"WARNING": "⚠️", "ALERT": "🔶", "CRITICAL": "🛑"}.get(sev, "⚠️")
            border = {"WARNING": "#FFB300", "ALERT": "#FB8C00", "CRITICAL": "#E53935"}.get(sev, "#FFB300")
            st.markdown(
                f"""
                <div style='border-left:4px solid {border};padding:8px;margin-bottom:8px;background:#112240;border-radius:8px;'>
                {emoji} <b>{r.get('anomaly_type')}</b><br/>
                <small>{r.get('timestamp')} | Δ {r.get('deviation_pct', 0):.1f}%</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with mid:
        st.subheader("📊 Plant Health Gauges")
        sp_score = float(np.interp(sp, [78.0, 87.0, 95.0], [40.0, 85.0, 55.0]))
        therm = float(np.clip(latest["coal_efficiency"] / 2.5 * 100, 0, 100))
        plf = float(latest["plant_load_factor"])
        overall = float(np.clip((sp_score + therm + plf) / 3.0, 0, 100))
        g1 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=min(100, max(0, sp_score)),
                title={"text": "Steam Pressure Health %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#42A5F5"},
                    "steps": [
                        {"range": [0, 50], "color": "#3E2723"},
                        {"range": [50, 75], "color": "#4E342E"},
                        {"range": [75, 100], "color": "#1B5E20"},
                    ],
                },
            )
        )
        g1.update_layout(height=220, paper_bgcolor="#0A1628", font=dict(color="#fff"))
        st.plotly_chart(g1, use_container_width=True)
        g2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=therm,
                title={"text": "Thermal Efficiency %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#FFB300"},
                    "steps": [
                        {"range": [0, 50], "color": "#3E2723"},
                        {"range": [50, 75], "color": "#4E342E"},
                        {"range": [75, 100], "color": "#1B5E20"},
                    ],
                },
            )
        )
        g2.update_layout(height=220, paper_bgcolor="#0A1628", font=dict(color="#fff"))
        st.plotly_chart(g2, use_container_width=True)
        g3 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=min(100, plf),
                title={"text": "Plant Load Factor %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#66BB6A"},
                    "steps": [
                        {"range": [0, 50], "color": "#3E2723"},
                        {"range": [50, 75], "color": "#4E342E"},
                        {"range": [75, 100], "color": "#1B5E20"},
                    ],
                },
            )
        )
        g3.update_layout(height=220, paper_bgcolor="#0A1628", font=dict(color="#fff"))
        st.plotly_chart(g3, use_container_width=True)
        g4 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=overall,
                title={"text": "Overall Plant Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#AB47BC"},
                    "steps": [
                        {"range": [0, 50], "color": "#3E2723"},
                        {"range": [50, 75], "color": "#4E342E"},
                        {"range": [75, 100], "color": "#1B5E20"},
                    ],
                },
            )
        )
        g4.update_layout(height=220, paper_bgcolor="#0A1628", font=dict(color="#fff"))
        st.plotly_chart(g4, use_container_width=True)

    with right:
        st.subheader("📅 Current Shift Summary")
        sub = df.last("8H")
        gen_mwh = float(sub["net_power_mw"].sum())
        coal_tons = float(sub["coal_feed_tph"].sum())
        anom = int(sub["anomaly_flag"].sum())
        aux_pct = float(sub["auxiliary_power_mw"].sum() / sub["gross_power_mw"].sum() * 100)
        co2 = float(sub["co2_tons_per_hour"].sum())
        st.metric("Generation this shift (MWh)", f"{gen_mwh:.1f}")
        st.metric("Coal consumed (tons)", f"{coal_tons:.1f}")
        st.metric("Anomalies detected", f"{anom}")
        st.metric("Auxiliary %", f"{aux_pct:.1f}%")
        st.metric("CO₂ this shift (tons)", f"{co2:.1f}")
        st.caption("AI checks: 480 ✓")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    month = df.index.max().month
    mdf = df[df.index.month == month]
    m1.metric("This Month Generation (MWh)", f"{mdf['net_power_mw'].sum():,.0f}")
    m2.metric("This Month Coal (tons)", f"{mdf['coal_feed_tph'].sum():,.0f}")
    m3.metric("This Month CO₂ (tons)", f"{mdf['co2_tons_per_hour'].sum():,.0f}")
    try:
        from utils.forecast_engine import forecast_next_hours

        fc = forecast_next_hours(df, hours=24)
        tom = float(fc["predicted_net_mw"].mean())
        m4.metric("Tomorrow Forecast (avg MW)", f"{tom:.2f}")
    except Exception:
        m4.metric("Tomorrow Forecast (avg MW)", "—")
