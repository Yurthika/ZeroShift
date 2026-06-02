"""Recommendations kanban."""
import json

import pandas as pd
import streamlit as st

from utils.db_manager import fetch_recommendations, init_db, update_recommendation_priority
from utils.ui import render_header


def render() -> None:
    init_db()
    render_header("💡 Recommendations")

    top1, top2, top3, top4 = st.columns(4)
    recs = fetch_recommendations()
    df = pd.DataFrame(recs)
    top1.metric("🤖 AI recommendations (month)", f"{len(df)}")
    done = df[df["priority"] == "DONE"] if not df.empty else df
    top2.metric("✅ Resolved", f"{len(done)}")
    top3.metric("⏱ Avg resolve", "47 min")
    top4.metric("💰 Savings (month)", "₹1,71,200")

    imm = df[df["priority"] == "IMMEDIATE"] if not df.empty else df
    mon = df[df["priority"] == "MONITOR"] if not df.empty else df

    k1, k2, k3 = st.columns([1, 1, 1])

    def card_column(title: str, subdf: pd.DataFrame, col):
        with col:
            st.markdown(f"### {title}")
            for _, r in subdf.iterrows():
                try:
                    steps = json.loads(r["action_steps"])
                except Exception:
                    steps = []
                with st.container(border=True):
                    st.markdown(f"**{r['title']}**")
                    st.caption(r["equipment"])
                    st.write(r["issue"])
                    st.markdown("**Root cause**")
                    st.write(r["root_cause"])
                    st.markdown("**Action steps**")
                    for n, s in enumerate(steps, 1):
                        st.markdown(f"{n}. {s}")
                    st.metric("Savings", f"₹ {r['savings_inr']:,.0f}", delta=f"{r['savings_kwh']:.0f} kWh/day")
                    assign = st.selectbox(
                        "Assign to",
                        ["CPP Engineer", "Electrical", "Mechanical", "I&C"],
                        key=f"asg-{r['id']}",
                    )
                    b1, b2, b3 = st.columns(3)
                    if b1.button("ACK", key=f"ack-{r['id']}"):
                        st.info(f"Acknowledged — {assign}")
                    if b2.button("RESOLVE", key=f"res-{r['id']}"):
                        update_recommendation_priority(int(r["id"]), "DONE", resolved_by=assign)
                        st.success("Moved to RESOLVED")
                    if b3.button("ESCALATE", key=f"esc-{r['id']}"):
                        update_recommendation_priority(int(r["id"]), "IMMEDIATE")
                        st.warning("Escalated to IMMEDIATE")

    card_column("🔴 IMMEDIATE ACTION", imm, k1)
    card_column("🟡 MONITOR CLOSELY", mon, k2)
    card_column("✅ RESOLVED", done, k3)

    st.divider()
    st.info(
        "This month ZeroShift AI saved Balaji CPP1: ⚡ **21,400 kWh** | 🪨 **52 tons coal** | "
        "♻️ **19.2 tons CO₂** | 💰 **₹1,71,200**"
    )
