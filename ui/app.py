"""
Task 5: User Interface and Demonstration

Streamlit app: upload a portfolio CSV (columns: ticker, weight), chat with
the agentic assistant, and see risk/allocation visualizations.

Run with:
    streamlit run ui/app.py
"""
import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.orchestrator import FinanceAgentOrchestrator

st.set_page_config(page_title="Portfolio Analysis Agent", layout="wide")
st.title("💬 Conversational Agent for Portfolio Analysis")
st.caption("Agentic AI (Retriever → Analyzer → Summarizer → Reporter) over a RAG knowledge base "
           "and ML models for return, direction, and risk clustering.")


@st.cache_resource
def get_orchestrator():
    return FinanceAgentOrchestrator()


orchestrator = get_orchestrator()

if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")):
    st.info("No ANTHROPIC_API_KEY / GROQ_API_KEY / OPENAI_API_KEY set -- running with the offline mock LLM. "
            "Set one of those env vars for full natural-language reasoning.", icon="ℹ️")

with st.sidebar:
    st.header("Your Portfolio")
    uploaded = st.file_uploader("Upload portfolio CSV (columns: ticker, weight)", type="csv")
    if uploaded:
        portfolio_df = pd.read_csv(uploaded)
    else:
        st.caption("No file uploaded -- using a sample portfolio.")
        portfolio_df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "TSLA", "JPM"],
            "weight": [0.35, 0.25, 0.20, 0.20],
        })
    st.dataframe(portfolio_df, use_container_width=True)
    portfolio = portfolio_df.to_dict(orient="records")

    st.divider()
    st.subheader("Sample prompts")
    for p in ["Summarize risk factors in my portfolio.",
              "What are the diversification gaps?",
              "Predict returns for the next quarter.",
              "Compare my portfolio to a tech-heavy allocation."]:
        st.code(p, language=None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your portfolio's risk, diversification, or predicted returns..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context, running models, and reasoning..."):
            report = orchestrator.run(prompt, portfolio=portfolio)
        st.markdown(report["answer"])

        if report.get("analysis") and "sector_allocation" in report["analysis"]:
            analysis = report["analysis"]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sector Allocation")
                st.bar_chart(pd.Series(analysis["sector_allocation"]))
            with col2:
                st.subheader("Risk Allocation")
                st.bar_chart(pd.Series(analysis["risk_allocation"]))

            st.subheader("Holdings")
            st.dataframe(pd.DataFrame(analysis["holdings"]), use_container_width=True)
            st.metric("Portfolio predicted 5-day return",
                      f"{analysis['portfolio_predicted_5d_return']*100:.2f}%")
            st.metric("Portfolio beta", f"{analysis['portfolio_beta']:.2f}")

        if report.get("sources"):
            with st.expander("Sources used for this answer"):
                st.dataframe(pd.DataFrame(report["sources"]), use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": report["answer"]})
