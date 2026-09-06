"""
Task 4: RAG-Based Agentic AI System

Chains Retriever -> Analyzer -> Summarizer -> Reporter with LangGraph,
with a router node that decides which sub-agents a given query needs,
plus a simple conversation-memory list for contextual multi-turn Q&A.

Falls back to a plain-Python sequential pipeline if the `langgraph`
package isn't installed, so the system still runs anywhere.
"""
import re
from typing import TypedDict, List, Optional

from .retriever_agent import RetrieverAgent
from .analyzer_agent import AnalyzerAgent
from .summarizer_agent import SummarizerAgent
from .reporter_agent import ReporterAgent

TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
KNOWN_TICKERS = {"AAPL", "MSFT", "GOOGL", "JPM", "XOM", "TSLA", "JNJ", "KO", "NVDA", "PG"}


class AgentState(TypedDict, total=False):
    query: str
    portfolio: Optional[list]
    history: List[dict]
    needs_retrieval: bool
    needs_analysis: bool
    retrieved_docs: list
    analysis: dict
    summary: str
    report: dict


def route_query(state: AgentState) -> AgentState:
    """Decide which sub-agents this query needs (Retriever / Analyzer / both)."""
    q = state["query"].lower()
    analysis_keywords = ["risk", "diversif", "predict", "return", "compare", "allocation", "beta"]
    state["needs_analysis"] = bool(state.get("portfolio")) or any(k in q for k in analysis_keywords)
    state["needs_retrieval"] = True  # retrieval almost always adds useful grounding context
    return state


def run_retriever(state: AgentState, agent: RetrieverAgent) -> AgentState:
    if not state.get("needs_retrieval"):
        state["retrieved_docs"] = []
        return state
    tickers_in_query = [t for t in TICKER_PATTERN.findall(state["query"]) if t in KNOWN_TICKERS]
    ticker_filter = tickers_in_query[0] if len(tickers_in_query) == 1 else None
    state["retrieved_docs"] = agent.retrieve(state["query"], ticker=ticker_filter, top_k=4)
    return state


def run_analyzer(state: AgentState, agent: AnalyzerAgent) -> AgentState:
    if not state.get("needs_analysis"):
        state["analysis"] = None
        return state
    portfolio = state.get("portfolio")
    if portfolio:
        state["analysis"] = agent.analyze_portfolio(portfolio)
    else:
        tickers_in_query = [t for t in TICKER_PATTERN.findall(state["query"]) if t in KNOWN_TICKERS]
        state["analysis"] = {t: agent.predict_ticker(t) for t in tickers_in_query} or None
    return state


def run_summarizer(state: AgentState, agent: SummarizerAgent) -> AgentState:
    state["summary"] = agent.summarize(state["query"], analysis=state.get("analysis"),
                                        retrieved_docs=state.get("retrieved_docs"))
    return state


def run_reporter(state: AgentState, agent: ReporterAgent) -> AgentState:
    state["report"] = agent.build_report(state["query"], state["summary"],
                                          analysis=state.get("analysis"),
                                          retrieved_docs=state.get("retrieved_docs"))
    return state


class FinanceAgentOrchestrator:
    """Public entry point used by the UI. Wraps a LangGraph StateGraph when
    available, otherwise runs the same nodes as a sequential pipeline."""

    def __init__(self):
        self.retriever = RetrieverAgent()
        self.analyzer = AnalyzerAgent()
        self.summarizer = SummarizerAgent()
        self.reporter = ReporterAgent()
        self.history: List[dict] = []  # conversational memory
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            return None  # sequential fallback used in .run()

        g = StateGraph(AgentState)
        g.add_node("route", route_query)
        g.add_node("retrieve", lambda s: run_retriever(s, self.retriever))
        g.add_node("analyze", lambda s: run_analyzer(s, self.analyzer))
        g.add_node("summarize", lambda s: run_summarizer(s, self.summarizer))
        g.add_node("report", lambda s: run_reporter(s, self.reporter))

        g.set_entry_point("route")
        g.add_edge("route", "retrieve")
        g.add_edge("retrieve", "analyze")
        g.add_edge("analyze", "summarize")
        g.add_edge("summarize", "report")
        g.add_edge("report", END)
        return g.compile()

    def run(self, query: str, portfolio: Optional[list] = None) -> dict:
        state: AgentState = {"query": query, "portfolio": portfolio, "history": self.history}

        if self.graph is not None:
            result = self.graph.invoke(state)
        else:
            result = route_query(state)
            result = run_retriever(result, self.retriever)
            result = run_analyzer(result, self.analyzer)
            result = run_summarizer(result, self.summarizer)
            result = run_reporter(result, self.reporter)

        self.history.append({"query": query, "answer": result["summary"]})
        return result["report"]
