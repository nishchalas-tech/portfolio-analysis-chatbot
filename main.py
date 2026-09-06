"""
End-to-end pipeline runner: generate data -> build vector DB -> train models
-> run a sample agentic query. Use this to sanity-check the whole system,
then launch `streamlit run ui/app.py` for the interactive demo.
"""
import subprocess
import sys
import os

STEPS = [
    ("Generating data", ["python3", "data_collection/generate_synthetic_data.py"]),
    ("Building vector DB", ["python3", "embeddings/build_vector_db.py"]),
    ("Training models", ["python3", "models/train_models.py"]),
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for label, cmd in STEPS:
        print(f"\n=== {label} ===")
        subprocess.run(cmd, cwd=root, check=True)

    print("\n=== Running a sample agentic query ===")
    sys.path.insert(0, root)
    from agents.orchestrator import FinanceAgentOrchestrator

    orchestrator = FinanceAgentOrchestrator()
    report = orchestrator.run(
        "Summarize risk factors in my portfolio.",
        portfolio=[
            {"ticker": "AAPL", "weight": 0.4},
            {"ticker": "MSFT", "weight": 0.3},
            {"ticker": "TSLA", "weight": 0.3},
        ],
    )
    print("\nAnswer:\n", report["answer"])
    print("\nRun `streamlit run ui/app.py` for the interactive chat UI.")


if __name__ == "__main__":
    main()
