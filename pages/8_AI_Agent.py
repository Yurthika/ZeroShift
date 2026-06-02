"""AI Agent chat."""
import streamlit as st

from utils.ai_agent import build_chart_for_hint, chart_hint, get_ai_response
from utils.data_loader import load_processed_data
from utils.ui import render_header


def render() -> None:
    render_header("🤖 ZeroShift AI Agent")
    st.caption("Ask anything about Balaji CPP performance in plain English")
    st.caption("Powered by GPT-3.5-turbo | Context: 8,760 hours of plant data")

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    st.info(
        "Add your OpenAI API key in the sidebar to enable live AI responses. "
        "Without a key, the agent uses fast on-device rule-based answers."
    )

    chips = [
        "📊 What is today's heat rate?",
        "🚨 Which equipment needs attention?",
        "📈 Compare this week vs last week performance",
        "🌱 How much CO₂ have we saved this month?",
        "⚡ When should we schedule next maintenance?",
        "🔮 What is tomorrow's predicted generation?",
        "🪨 Why did coal efficiency drop last night?",
        "🌡️ Explain the bed temperature trend",
    ]
    r1 = st.columns(4)
    r2 = st.columns(4)
    for i, q in enumerate(chips):
        col = r1[i] if i < 4 else r2[i - 4]
        with col:
            if st.button(q, key=f"chip-{i}"):
                st.session_state["pending"] = q

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask ZeroShift…")
    if st.session_state.get("pending"):
        prompt = st.session_state.pop("pending")

    df = load_processed_data()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = get_ai_response(prompt, df, api_key=api_key or None)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
            hint = chart_hint(prompt)
            fig = build_chart_for_hint(hint, df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
