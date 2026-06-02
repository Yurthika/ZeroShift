"""PDF report generation using ReportLab; charts exported via Plotly Kaleido."""
import io
import json
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.paths import project_root


def _fig_to_png_bytes(fig: go.Figure) -> bytes:
    return fig.to_image(format="png", width=1100, height=420, scale=1)


def build_power_trend_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["net_power_mw"],
            fill="tozeroy",
            name="Net MW",
            line=dict(color="#42A5F5"),
        )
    )
    fig.add_hline(y=35, line_dash="dash", line_color="#FFB300")
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0A1628"),
        title="Net power trend",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def build_anomaly_timeline_fig(df: pd.DataFrame) -> go.Figure:
    sub = df.copy()
    sub["color"] = sub["anomaly_flag"].map({0: "#C8E6C9", 1: "#FFCDD2"})
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=sub.index, y=sub["gross_power_mw"], marker_color=sub["color"], name="Gross MW")
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title="Anomaly timeline (red bars = anomaly hour)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def generate_shift_pdf(
    df: pd.DataFrame,
    report_type: str,
    period_label: str,
    sections: List[str],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Oblique", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(inch, 0.45 * inch, "ZeroShift | UltraTech Cement")
        canvas.restoreState()

    # Page 1 cover
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("<b>ZeroShift</b>", styles["Title"]))
    story.append(Paragraph("Balaji CPP — Performance Report", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>Type:</b> {report_type}", styles["Normal"]))
    story.append(Paragraph(f"<b>Period:</b> {period_label}", styles["Normal"]))
    story.append(
        Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} | ZeroShift AI v1.0",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.4 * inch))

    latest = df.iloc[-1]
    if "Executive KPI Summary" in sections:
        story.append(Paragraph("<b>Executive Summary</b>", styles["Heading3"]))
        data = [
            ["Metric", "Value"],
            ["Net Power (MW)", f"{latest['net_power_mw']:.2f}"],
            ["Gross Power (MW)", f"{latest['gross_power_mw']:.2f}"],
            ["Coal Feed (TPH)", f"{latest['coal_feed_tph']:.2f}"],
            ["Steam Pressure (kgf/cm²)", f"{latest['steam_pressure_kgcm2']:.2f}"],
            ["Plant Load Factor (%)", f"{latest['plant_load_factor']:.2f}"],
            ["CO₂ (t/h)", f"{latest['co2_tons_per_hour']:.2f}"],
        ]
        t = Table(data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#112240")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.2 * inch))

    if "Energy Trend Analysis" in sections:
        story.append(Paragraph("<b>Energy Trend</b>", styles["Heading3"]))
        fig = build_power_trend_fig(df)
        img = Image(io.BytesIO(_fig_to_png_bytes(fig)), width=6.8 * inch, height=2.4 * inch)
        story.append(img)
        story.append(Spacer(1, 0.15 * inch))

    if "Anomaly Detection Log" in sections:
        story.append(Paragraph("<b>Anomaly Timeline</b>", styles["Heading3"]))
        fig2 = build_anomaly_timeline_fig(df)
        img2 = Image(io.BytesIO(_fig_to_png_bytes(fig2)), width=6.8 * inch, height=2.4 * inch)
        story.append(img2)

    if "Carbon Footprint Summary" in sections:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("<b>Carbon Footprint</b>", styles["Heading3"]))
        total_co2 = float(df["co2_tons_per_hour"].sum())
        total_mwh = float(df["net_power_mw"].sum())
        intensity = total_co2 / max(total_mwh, 1e-6)
        story.append(Paragraph(f"Total CO₂ over period: {total_co2:,.1f} t", styles["Normal"]))
        story.append(Paragraph(f"CO₂ per MWh: {intensity:.3f} t/MWh", styles["Normal"]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = buf.getvalue()
    buf.close()
    return pdf
