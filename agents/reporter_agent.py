"""Reporter Agent: assembles the final structured response (text + data for charts)."""


class ReporterAgent:
    def build_report(self, user_query, summary_text, analysis=None, retrieved_docs=None):
        report = {
            "query": user_query,
            "answer": summary_text,
            "analysis": analysis,
            "sources": [
                {"ticker": d["ticker"], "doc_type": d["doc_type"], "score": round(d["score"], 3)}
                for d in (retrieved_docs or [])
            ],
        }
        return report
