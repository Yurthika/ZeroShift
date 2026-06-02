"""Predictive maintenance dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_processed_data
from utils.maintenance_engine import equipment_dashboard
from utils.ui import render_header


def render() -> None:
    render_header("🔧 Predictive Maintenance")
    df = load_processed_data()
    dash = equipment_dashboard(df)

    st.subheader("Equipment risk overview")
    rows = [dash.iloc[i : i + 3] for i in range(0, len(dash), 3)]
    for chunk in rows:
        cols = st.columns(3)
        for j, (_, eq) in enumerate(chunk.iterrows()):
            with cols[j]:
                st.markdown(f"**{eq['name']}**  \n_{eq['spec']}_")
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(eq["failure_pct"]),
                        number={"suffix": "%"},
                        title={"text": "Failure in 48h"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#EF5350"},
                            "steps": [
                                {"range": [0, 30], "color": "#1B5E20"},
                                {"range": [30, 60], "color": "#4E342E"},
                                {"range": [60, 100], "color": "#B71C1C"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=240, paper_bgcolor="#0A1628", font=dict(color="#fff"))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Health %", f"{eq['health_pct']:.1f}")
                st.caption(f"Hours since maint: {eq['operating_hours']}")
                st.caption(f"Predicted days to next: {eq['days_to_maint']}")
                st.markdown(f"Risk: **{eq['risk']}**")

    st.subheader("Maintenance Gantt (next 90 days)")
    g = []
    for _, eq in dash.iterrows():
        start = df.index.max()
        end = start + pd.Timedelta(days=int(eq["days_to_maint"]))
        g.append(
            dict(
                Equipment=eq["name"],
                Start=start,
                Finish=end,
                Risk=eq["risk"],
            )
        )
    gdf = pd.DataFrame(g)
    fig2 = px.timeline(
        gdf,
        x_start="Start",
        x_end="Finish",
        y="Equipment",
        color="Risk",
        color_discrete_map={
            "LOW": "#66BB6A",
            "MEDIUM": "#FFB300",
            "HIGH": "#FB8C00",
            "CRITICAL": "#E53935",
        },
        title="Predicted maintenance windows",
    )
    fig2.update_layout(paper_bgcolor="#0A1628", plot_bgcolor="#0A1628", font=dict(color="#ECEFF1"))
    st.plotly_chart(fig2, use_container_width=True)

    worst = dash.sort_values("failure_pct", ascending=False).iloc[0]
    if worst["risk"] in ("HIGH", "CRITICAL"):
        st.error(
            f"Highest risk: **{worst['name']}** at {worst['failure_pct']:.1f}% — schedule inspection, "
            f"vibration snapshot, and lube oil sampling."
        )
    else:
        st.warning(
            f"Watchlist: **{worst['name']}** at {worst['failure_pct']:.1f}% — continue monitoring trends on DCS."
        )
