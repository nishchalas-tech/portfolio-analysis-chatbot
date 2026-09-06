"""Summarizer Agent: turns retrieved docs + analysis into a natural-language answer."""
from .llm_client import call_llm

SYSTEM_PROMPT = """You are a financial analysis assistant embedded in a portfolio
advisory tool. You are given (a) quantitative analysis results from ML models and
(b) retrieved context snippets from filings/news/blogs. Write a clear, well-organized
answer to the user's question. Cite specific numbers from the analysis. Do not give
personalized investment advice or guarantee returns -- frame outputs as model
estimates. Keep it concise (under 250 words)."""


class SummarizerAgent:
    def summarize(self, user_query, analysis=None, retrieved_docs=None):
        context_parts = []
        if analysis:
            context_parts.append(f"QUANTITATIVE ANALYSIS:\n{analysis}")
        if retrieved_docs:
            snippets = "\n".join(f"- ({d['ticker']}, {d['doc_type']}): {d['text']}"
                                  for d in retrieved_docs)
            context_parts.append(f"RETRIEVED CONTEXT:\n{snippets}")

        user_prompt = f"User question: {user_query}\n\n" + "\n\n".join(context_parts)
        return call_llm(SYSTEM_PROMPT, user_prompt)
