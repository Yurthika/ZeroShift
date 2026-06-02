"""AI forecasting page."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.data_loader import load_processed_data
from utils.forecast_engine import backtest_last_days, forecast_next_hours
from utils.ui import render_header


def render() -> None:
    render_header("🔮 AI Forecasting")
    df = load_processed_data()
    fc = forecast_next_hours(df, hours=24)
    hist = df.last("48H")

    pred = fc["predicted_net_mw"].values
    upper = pred * 1.08
    lower = pred * 0.92

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["net_power_mw"],
            mode="lines",
            name="Actual (48h)",
            line=dict(color="#42A5F5"),
            fill="tozeroy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["timestamp"],
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["timestamp"],
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(66,165,245,0.2)",
            name="±8% band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["timestamp"],
            y=pred,
            mode="lines",
            name="Forecast 24h",
            line=dict(color="#FFB300", dash="dash"),
        )
    )
    try:
        x_val = pd.Timestamp(hist.index.max())
        fig.add_vline(
            x=int(x_val.timestamp() * 1000),
            line_color="white",
            line_dash="dot",
        )
        fig.add_annotation(
            x=x_val,
            y=1,
            xref="x",
            yref="paper",
            text="NOW",
            showarrow=False,
            font=dict(color="white"),
        )
    except Exception:
        pass
    try:
        fig.add_annotation(
            x=fc["timestamp"].iloc[3],
            y=float(np.max(pred)),
            text="⚠️ High load 2PM–6PM",
            showarrow=False,
            yshift=20,
        )
        fig.add_annotation(
            x=fc["timestamp"].iloc[0],
            y=float(pred[0]),
            text="🔄 Shift transition 6AM",
            showarrow=False,
        )
    except Exception:
        pass
    fig.update_layout(
        title="Net power — actual vs forecast",
        paper_bgcolor="#0A1628",
        plot_bgcolor="#0A1628",
        font=dict(color="#ECEFF1"),
    )
    st.plotly_chart(fig, use_container_width=True)

    bt = backtest_last_days(df, days=30)
    r2 = float("nan")
    rmse = float("nan")
    acc = float("nan")
    if not bt.empty:
        y = bt["y_next_actual"].values
        p = bt["y_next_pred"].values
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        if len(y) > 10:
            r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
            rmse = float(np.sqrt(np.mean((y - p) ** 2)))
            acc = float(max(0.0, 100.0 * (1.0 - np.mean(np.abs(p - y) / np.maximum(np.abs(y), 0.1)))))

    a1, a2, a3 = st.columns(3)
    if not bt.empty and len(bt) > 10:
        a1.metric("R² (30d backtest)", f"{r2:.3f}")
        a2.metric("RMSE (MW)", f"{rmse:.3f}")
        a3.metric("Accuracy %", f"{acc:.1f}")
    else:
        a1.metric("R² (30d backtest)", "n/a")
        a2.metric("RMSE (MW)", "n/a")
        a3.metric("Accuracy %", "n/a")

    st.subheader("Hourly forecast table")
    rows = []
    for _, r in fc.iterrows():
        mw = float(r["predicted_net_mw"])
        coal = float(mw * 0.44)
        sp = float(87 + (mw - 31.5) * 0.05)
        conf = float(92 - abs(mw - 31.5) * 0.8)
        lvl = "LOW" if mw < 29 else ("HIGH" if mw > 33 else "MEDIUM")
        rows.append(
            {
                "Hour": r["timestamp"],
                "Predicted MW": mw,
                "Coal TPH (derived)": coal,
                "Steam Pressure (derived)": sp,
                "Confidence%": conf,
                "Shift": r["shift"],
                "Load Level": lvl,
            }
        )
    fcdf = pd.DataFrame(rows)
    st.dataframe(fcdf, use_container_width=True)

    st.subheader("30-day accuracy")
    if not bt.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt.index, y=bt["y_next_actual"], name="Actual", line=dict(color="#42A5F5")))
        fig2.add_trace(
            go.Scatter(x=bt.index, y=bt["y_next_pred"], name="Predicted", line=dict(color="#FFB300", dash="dash"))
        )
        fig2.update_layout(paper_bgcolor="#0A1628", plot_bgcolor="#0A1628", font=dict(color="#ECEFF1"))
        st.plotly_chart(fig2, use_container_width=True)
        err = bt["pct_error"].replace([np.inf, -np.inf], np.nan).dropna()
        fig3 = go.Figure(go.Bar(x=bt.index, y=bt["pct_error"], marker_color="#90CAF9"))
        fig3.update_layout(title="Error %", paper_bgcolor="#0A1628", plot_bgcolor="#0A1628", font=dict(color="#ECEFF1"))
        st.plotly_chart(fig3, use_container_width=True)
        st.metric("Avg forecast error this month", f"{float(err.mean()):.1f}%")

    st.subheader("7-day outlook")
    fc7 = forecast_next_hours(df, hours=168)
    cols = st.columns(7)
    for i, c in enumerate(cols):
        day = (df.index.max().normalize() + pd.Timedelta(days=i + 1)).date()
        sub = fc7[fc7["timestamp"].dt.date == day]
        avg = float(sub["predicted_net_mw"].mean()) if not sub.empty else float("nan")
        spark = sub.set_index("timestamp")["predicted_net_mw"] if not sub.empty else fc7.set_index("timestamp")["predicted_net_mw"].iloc[:24]
        with c:
            st.caption(pd.Timestamp(day).strftime("%a"))
            st.metric("Avg MW", f"{avg:.1f}")
            if pd.Timestamp(day).weekday() == 6:
                st.caption("Maintenance mode: ~20 MW")
            figs = px.line(spark, title="", labels={"value": "MW"})
            figs.update_layout(
                height=160,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#0A1628",
                plot_bgcolor="#0A1628",
            )
            st.plotly_chart(figs, use_container_width=True)

