"""Energy monitor with synchronized charts."""
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.data_loader import load_processed_data
from utils.ui import render_header


def _shade(fig: go.Figure, y0: float, y1: float, color: str, name: str) -> None:
    fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.18, line_width=0, annotation_text=name)


def render() -> None:
    render_header("⚡ Energy Monitor")
    df = load_processed_data()
    dmin, dmax = df.index.min().date(), df.index.max().date()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        dr = st.date_input("From", value=max(dmin, dmax - timedelta(days=30)), key="em_from")
    with c2:
        dr2 = st.date_input("To", value=dmax, key="em_to")
    with c3:
        shifts = st.multiselect("Shift", ["Morning", "Afternoon", "Night"], default=["Morning", "Afternoon", "Night"])
    with c4:
        st.write("")
        st.write("")
        csv_bytes = df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", data=csv_bytes, file_name="zeroshift_export.csv")

    sub = df.loc[str(dr) : str(dr2)]
    sub = sub[sub["shift"].isin(shifts)]

    def style(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=title,
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#ECEFF1"),
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1E3555"),
        )
        return fig

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=sub.index, y=sub["net_power_mw"], fill="tozeroy", name="Net MW", line=dict(color="#42A5F5"))
        )
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=sub["auxiliary_power_mw"],
                fill="tonexty",
                name="Aux MW",
                line=dict(color="#78909C"),
                opacity=0.35,
            )
        )
        fig.add_hline(y=35, line_dash="dash", line_color="#FFB300")
        an = sub[sub["anomaly_flag"] == 1]
        fig.add_trace(
            go.Scatter(
                x=an.index,
                y=an["net_power_mw"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#EF5350", size=8),
            )
        )
        st.plotly_chart(style(fig, "Net Power Output (MW)"), use_container_width=True)
    with r1c2:
        fig = go.Figure()
        _shade(fig, 12, 16, "#1B5E20", "Optimal 12–16 TPH")
        fig.add_trace(go.Scatter(x=sub.index, y=sub["coal_feed_tph"], name="Coal TPH", line=dict(color="#FFB300")))
        st.plotly_chart(style(fig, "Coal Feed Rate (TPH)"), use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = go.Figure()
        _shade(fig, 84, 90, "#1B5E20", "Safe 84–90")
        _shade(fig, 90, 95, "#EF6C00", "Warning >90")
        fig.add_trace(
            go.Scatter(x=sub.index, y=sub["steam_pressure_kgcm2"], name="Pressure", line=dict(color="#FB8C00"))
        )
        st.plotly_chart(style(fig, "Steam Pressure (kgf/cm²)"), use_container_width=True)
    with r2c2:
        fig = go.Figure()
        _shade(fig, 510, 520, "#1B5E20", "Safe 510–520°C")
        fig.add_trace(go.Scatter(x=sub.index, y=sub["steam_temp_celsius"], name="Steam °C", line=dict(color="#FF7043")))
        fig.add_hline(y=515, line_dash="dash", line_color="#FFB300")
        st.plotly_chart(style(fig, "Steam Temperature (°C)"), use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = go.Figure()
        _shade(fig, 800, 850, "#1B5E20", "Normal 800–850°C")
        fig.add_trace(go.Scatter(x=sub.index, y=sub["bed_temp_celsius"], name="Bed °C", line=dict(color="#FFEE58")))
        fig.add_hline(y=550, line_dash="dot", line_color="#B0BEC5", annotation_text="550°C firing threshold")
        st.plotly_chart(style(fig, "Bed Temperature (°C)"), use_container_width=True)
    with r3c2:
        fig = go.Figure()
        colors = np.where(sub["plant_load_factor"] >= 90, "#66BB6A", np.where(sub["plant_load_factor"] >= 80, "#FFB300", "#EF5350"))
        fig.add_trace(
            go.Bar(x=sub.index, y=sub["plant_load_factor"], name="PLF %", marker_color=colors)
        )
        fig.add_hline(y=90, line_dash="dash", line_color="#FFB300")
        st.plotly_chart(style(fig, "Plant Load Factor (%)"), use_container_width=True)

    st.subheader("Correlation Heatmap")
    cols = [
        "gross_power_mw",
        "coal_feed_tph",
        "steam_pressure_kgcm2",
        "steam_temp_celsius",
        "bed_temp_celsius",
        "flue_gas_temp_celsius",
        "drum_level_pct",
        "turbine_rpm",
        "lube_oil_pressure",
        "auxiliary_power_mw",
    ]
    corr = sub[cols].corr()
    fig_h = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Parameter Correlations",
    )
    fig_h.update_layout(paper_bgcolor="#0A1628", plot_bgcolor="#0A1628", font=dict(color="#ECEFF1"))
    st.plotly_chart(fig_h, use_container_width=True)
    # top pair
    c2d = corr.abs().unstack().sort_values(ascending=False)
    c2d = c2d[c2d < 1]
    top = c2d.index[0]
    st.info(f"Strongest linear relationship: **{top[0]}** vs **{top[1]}** (|r|={c2d.iloc[0]:.2f}).")

    st.subheader("Shift Performance Table")
    tmp = sub.copy()
    tmp["date"] = tmp.index.date
    tmp["aux_pct_row"] = tmp["auxiliary_power_mw"] / tmp["gross_power_mw"] * 100.0
    tbl = tmp.groupby(["date", "shift"], as_index=False).agg(
        avg_mw=("gross_power_mw", "mean"),
        peak=("gross_power_mw", "max"),
        minp=("gross_power_mw", "min"),
        coal=("coal_feed_tph", "mean"),
        plf=("plant_load_factor", "mean"),
        aux_pct=("aux_pct_row", "mean"),
        anom=("anomaly_flag", "sum"),
    )

    def grade(plf: float) -> str:
        if plf > 90:
            return "A"
        if plf >= 80:
            return "B"
        return "C"

    tbl["grade"] = tbl["plf"].map(grade)
    st.dataframe(tbl, use_container_width=True)
    buf = io.StringIO()
    tbl.to_csv(buf, index=False)
    st.download_button("Download table CSV", data=buf.getvalue(), file_name="shift_performance.csv")
