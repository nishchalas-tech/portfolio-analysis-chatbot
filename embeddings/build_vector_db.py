"""
Task 2: Financial Knowledge Embedding

Chunks and embeds financial text documents, storing them in a local
vector index for semantic retrieval.

Embedding backend is pluggable:
  - "tfidf"  (default): scikit-learn TF-IDF vectors. Zero external calls,
             works fully offline -- used here so the project runs without
             internet access to OpenAI/HuggingFace.
  - "openai": text-embedding-3-small via the OpenAI API (needs OPENAI_API_KEY).
  - "finbert": sentence-transformers "ProsusAI/finbert" (needs internet
             access to HuggingFace to download weights on first run).

Swap EMBEDDING_BACKEND to "openai" or "finbert" for production-quality
embeddings once you have API access.
"""
import os
import glob
import json
import pickle
import numpy as np

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "tfidf")
CHUNK_SIZE = 400  # characters
CHUNK_OVERLAP = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def load_documents(docs_dir):
    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(path) as f:
            text = f.read()
        fname = os.path.basename(path)
        parts = fname.replace(".txt", "").split("_")
        ticker = parts[0] if parts else "UNKNOWN"
        doc_type = parts[1] if len(parts) > 1 else "unknown"
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "id": f"{fname}::chunk{i}",
                "source_file": fname,
                "ticker": ticker,
                "doc_type": doc_type,
                "text": chunk,
            })
    return docs


def embed_tfidf(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=2048, stop_words="english")
    matrix = vectorizer.fit_transform([d["text"] for d in docs])
    return np.asarray(matrix.todense(), dtype=np.float32), vectorizer


def embed_openai(docs):
    from openai import OpenAI
    client = OpenAI()
    vectors = []
    for d in docs:
        resp = client.embeddings.create(model="text-embedding-3-small", input=d["text"])
        vectors.append(resp.data[0].embedding)
    return np.array(vectors, dtype=np.float32), None


def embed_finbert(docs):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("ProsusAI/finbert")
    vectors = model.encode([d["text"] for d in docs], show_progress_bar=True)
    return np.array(vectors, dtype=np.float32), model


def build_index(embeddings):
    """Build a FAISS flat index (falls back to numpy cosine search if faiss unavailable)."""
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        index.add(norm)
        return {"backend": "faiss", "index": index}
    except ImportError:
        norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return {"backend": "numpy", "index": norm}


def main():
    base = os.path.dirname(__file__)
    docs_dir = os.path.join(base, "..", "data", "processed", "financial_docs")
    out_dir = os.path.join(base, "..", "vector_db")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading + chunking documents from {docs_dir} ...")
    docs = load_documents(docs_dir)
    print(f"{len(docs)} chunks loaded.")

    print(f"Embedding with backend='{EMBEDDING_BACKEND}' ...")
    if EMBEDDING_BACKEND == "openai":
        embeddings, vectorizer = embed_openai(docs)
    elif EMBEDDING_BACKEND == "finbert":
        embeddings, vectorizer = embed_finbert(docs)
    else:
        embeddings, vectorizer = embed_tfidf(docs)

    index_bundle = build_index(embeddings)

    with open(os.path.join(out_dir, "docs.json"), "w") as f:
        json.dump(docs, f)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"backend": EMBEDDING_BACKEND, "vectorizer": vectorizer,
                      "index_backend": index_bundle["backend"]}, f)
    if index_bundle["backend"] == "faiss":
        import faiss
        faiss.write_index(index_bundle["index"], os.path.join(out_dir, "index.faiss"))

    print(f"Vector DB built: {len(docs)} vectors, dim={embeddings.shape[1]}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
