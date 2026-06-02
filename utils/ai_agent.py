"""OpenAI GPT + rule-based assistant for ZeroShift."""
from typing import Any, Dict

import pandas as pd
from openai import OpenAI


def generate_rule_based_response(
    query: str, latest: pd.Series, last_24h: pd.DataFrame
) -> str:
    query = query.lower()
    if any(w in query for w in ["heat rate", "efficiency"]):
        hr = (latest["coal_feed_tph"] * 4500) / max(latest["net_power_mw"], 0.1)
        return (
            f"Current heat rate: {hr:.0f} kCal/kWh. "
            f"Industry target: 3,200 kCal/kWh."
        )
    if any(w in query for w in ["coal", "feed"]):
        return (
            f"Coal feed: {latest['coal_feed_tph']:.1f} TPH. "
            f"Efficiency: {latest['coal_efficiency']:.2f} MW/TPH."
        )
    if any(w in query for w in ["anomaly", "alert", "fault"]):
        count = int(last_24h["anomaly_flag"].sum())
        return f"{count} anomalies detected in last 24 hours."
    if any(w in query for w in ["power", "generation", "mw"]):
        return (
            f"Net power: {latest['net_power_mw']:.1f} MW. "
            f"Load factor: {latest['plant_load_factor']:.1f}%."
        )
    if any(w in query for w in ["carbon", "co2", "emission"]):
        return f"CO₂ this hour: {latest['co2_tons_per_hour']:.2f} tons."
    if any(w in query for w in ["maintenance", "equipment", "failure"]):
        return "Check Predictive Maintenance page for 48-hour failure probability."
    if any(w in query for w in ["forecast", "tomorrow", "predict"]):
        return "See AI Forecasting page for next 24-hour prediction (R²≈0.94)."
    return (
        "Ask me about: power generation, coal efficiency, anomalies, "
        "CO₂, maintenance, or forecasts."
    )


def get_ai_response(user_query: str, df: pd.DataFrame, api_key: str | None = None) -> str:
    latest = df.iloc[-1]
    last_24h = df.tail(24)
    anomaly_count = int(df["anomaly_flag"].tail(24).sum())

    if not api_key:
        return generate_rule_based_response(user_query, latest, last_24h)

    system_prompt = f"""You are ZeroShift, an AI assistant for UltraTech Cement Balaji captive power plant (CPP1).

Current plant status:
- Net Power: {latest['net_power_mw']:.1f} MW
- Coal Feed: {latest['coal_feed_tph']:.1f} TPH
- Steam Pressure: {latest['steam_pressure_kgcm2']:.1f} kgf/cm²
- Steam Temp: {latest['steam_temp_celsius']:.0f}°C
- Bed Temp: {latest['bed_temp_celsius']:.0f}°C
- Plant Load Factor: {latest['plant_load_factor']:.1f}%
- Active Anomalies (last 24h): {anomaly_count}
- Equipment: FLENDER gearbox, Wood Ward 505 governor, Emerson Ovation DCS, Alstom ESP

Answer in plain English with specific numbers. Be concise.
Always relate to business impact (energy savings, CO₂, ₹)."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=400,
    )
    return str(response.choices[0].message.content)


def chart_hint(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["power", "generation", "mw", "load"]):
        return "power"
    if any(w in q for w in ["coal", "feed"]):
        return "coal"
    if any(w in q for w in ["anomaly", "alert", "fault"]):
        return "anomaly"
    return "none"


def build_chart_for_hint(hint: str, df: pd.DataFrame):
    import plotly.graph_objects as go

    tail = df.tail(24)
    if hint == "power":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tail.index,
                y=tail["net_power_mw"],
                mode="lines",
                name="Net MW",
                line=dict(color="#42A5F5"),
            )
        )
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#E3F2FD"),
            title="Last 24h — Net Power",
        )
        return fig
    if hint == "coal":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tail.index,
                y=tail["coal_feed_tph"],
                mode="lines",
                name="Coal TPH",
                line=dict(color="#FFB300"),
            )
        )
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#E3F2FD"),
            title="Last 24h — Coal Feed",
        )
        return fig
    if hint == "anomaly":
        sub = df.tail(500)
        col = sub["anomaly_flag"].map({0: "Normal", 1: "Anomaly"})
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=sub["gross_power_mw"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=sub["anomaly_flag"].map({0: "#66BB6A", 1: "#EF5350"}),
                ),
                text=col,
                hovertemplate="%{x}<br>%{y:.2f} MW<br>%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font=dict(color="#E3F2FD"),
            title="Recent anomalies vs power",
        )
        return fig
    return None
