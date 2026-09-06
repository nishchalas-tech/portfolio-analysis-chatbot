# Automating Portfolio Analysis with Agentic AI

A conversational assistant for financial portfolio analysis, built as specified in
`Conversational_Agent_Finance_Project_Spec.md`: RAG over financial documents +
a LangGraph-style multi-agent pipeline + ML models for return/risk prediction,
served through a Streamlit chat UI.

## What's implemented (mapped to the spec's tasks)

| Task | Where | Status |
|---|---|---|
| 1. Data Collection & Preprocessing | `data_collection/` | Real fetch scripts for Yahoo Finance, FMP, SEC EDGAR, news scraping **+** a synthetic data generator so the project runs with zero external network access |
| 2. Financial Knowledge Embedding | `embeddings/build_vector_db.py` | Chunking + embedding pipeline, pluggable backend (TF-IDF offline default / OpenAI / FinBERT), FAISS or numpy cosine index |
| 3. Predictive Financial Models | `models/train_models.py` | XGBoost return regressor, XGBoost direction classifier, KMeans risk clustering |
| 4. RAG-Based Agentic System | `agents/` | Retriever, Analyzer, Summarizer, Reporter sub-agents chained with LangGraph (`orchestrator.py`), with a sequential fallback if `langgraph` isn't installed, plus conversation memory |
| 5. UI & Demonstration | `ui/app.py` | Streamlit chat app: upload a portfolio CSV, ask questions, see risk/allocation charts |

## Why synthetic data by default

This environment's outbound network is restricted (only package registries
like PyPI/npm/GitHub are reachable — not Yahoo Finance, OpenAI, HuggingFace,
or SEC EDGAR). So the project ships with:

- **Real, correct client code** in `data_collection/fetch_structured_data.py`
  and `fetch_unstructured_data.py` — this is what you'd run in an environment
  with internet access and API keys.
- **`data_collection/generate_synthetic_data.py`** — produces data in the exact
  same schema, so every downstream step (embeddings, models, agents, UI) is
  fully runnable and testable right now, without any keys or network access.

Swap one for the other; nothing downstream needs to change.

## Quickstart

```bash
pip install -r requirements.txt

# 1) Generate data (swap for fetch_structured_data.py / fetch_unstructured_data.py
#    once you have API keys + network access)
python3 data_collection/generate_synthetic_data.py

# 2) Build the vector DB
python3 embeddings/build_vector_db.py

# 3) Train the predictive models
python3 models/train_models.py

# 4) Sanity-check the full agent pipeline
python3 main.py

# 5) Launch the interactive UI
streamlit run ui/app.py
```

## Plugging in real LLM reasoning

By default, `agents/llm_client.py` falls back to a deterministic offline mock
so the whole system runs with zero configuration. For real reasoning:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # preferred, uses Claude
# or
export OPENAI_API_KEY=sk-...            # uses GPT-4o
```

## Plugging in production-grade embeddings

`EMBEDDING_BACKEND` env var controls `embeddings/build_vector_db.py`:

```bash
export EMBEDDING_BACKEND=openai   # text-embedding-3-small, needs OPENAI_API_KEY
export EMBEDDING_BACKEND=finbert  # ProsusAI/finbert via sentence-transformers, needs HF access
# default: tfidf (fully offline)
```

If you switch backends, also update `RetrieverAgent._embed_query` in
`agents/retriever_agent.py` to embed the query the same way (only the TF-IDF
path is wired up out of the box).

## Architecture

```
User query + portfolio CSV
        │
        ▼
   [route_query]  -- decides which sub-agents are needed
        │
        ▼
  [RetrieverAgent] -- semantic search over vector_db/ (Task 2 output)
        │
        ▼
  [AnalyzerAgent]  -- runs models/saved/*.pkl (Task 3 output): return regression,
        │              direction classification, risk clustering; computes
        │              portfolio-level beta, sector allocation, diversification gaps
        ▼
 [SummarizerAgent] -- LLM (Claude/GPT-4/mock) turns analysis + retrieved context
        │              into a natural-language answer
        ▼
  [ReporterAgent]  -- assembles final structured report (answer + data + sources)
        │
        ▼
   Streamlit UI    -- chat thread + bar charts + holdings table + sources
```

## Directory structure

```
project-root/
├── data/
│   ├── raw/                    # stock_prices.csv, fundamentals.csv, macro_indicators.csv
│   └── processed/financial_docs/  # chunked-ready news/filing/blog .txt files
├── data_collection/
│   ├── fetch_structured_data.py     # real Yahoo Finance / FMP client
│   ├── fetch_unstructured_data.py   # real SEC EDGAR / article scraper
│   └── generate_synthetic_data.py   # offline stand-in, same schema
├── embeddings/
│   └── build_vector_db.py
├── models/
│   ├── train_models.py
│   └── saved/                  # trained .pkl models + metrics.json (generated)
├── agents/
│   ├── llm_client.py           # pluggable Claude / GPT-4 / offline mock
│   ├── retriever_agent.py
│   ├── analyzer_agent.py
│   ├── summarizer_agent.py
│   ├── reporter_agent.py
│   └── orchestrator.py         # LangGraph StateGraph (+ sequential fallback)
├── ui/
│   └── app.py                  # Streamlit chat app
├── vector_db/                  # embeddings.npy, docs.json, index.faiss (generated)
├── main.py                     # end-to-end pipeline runner / smoke test
└── requirements.txt
```

## Evaluation notes (per spec's Evaluation Criteria)

- **Data pipeline / embeddings**: chunking is character-based with overlap;
  swap in a token-aware splitter for production use.
- **ML models**: `models/saved/metrics.json` reports RMSE (return model) and
  accuracy/F1 (direction model) on a held-out test split — inspect this after
  running `train_models.py` on real historical data, since results on the
  synthetic data are illustrative only, not predictive of real performance.
- **Agentic architecture**: each sub-agent is independently testable/importable
  (see `agents/*.py`); `route_query` keeps the graph modular by only invoking
  the analyzer when the query needs it.
- **UI usability**: portfolio upload, chat, and visualization all live in one
  `ui/app.py` file for easy demoing/screencasting.
