"""Retriever Agent: semantic search over the vector DB built in Task 2."""
import os
import json
import pickle
import numpy as np

VDB_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_db")


class RetrieverAgent:
    def __init__(self, vdb_dir=VDB_DIR):
        with open(os.path.join(vdb_dir, "docs.json")) as f:
            self.docs = json.load(f)
        self.embeddings = np.load(os.path.join(vdb_dir, "embeddings.npy"))
        with open(os.path.join(vdb_dir, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.embeddings_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

    def _embed_query(self, query):
        if self.meta["backend"] == "tfidf":
            vec = self.meta["vectorizer"].transform([query])
            return np.asarray(vec.todense(), dtype=np.float32)[0]
        raise NotImplementedError("Wire up OpenAI/FinBERT query embedding to match build_vector_db.py")

    def retrieve(self, query, ticker=None, top_k=5):
        q = self._embed_query(query)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scores = self.embeddings_norm @ q_norm

        candidates = list(enumerate(scores))
        if ticker:
            candidates = [(i, s) for i, s in candidates if self.docs[i]["ticker"] == ticker]
        candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in candidates[:top_k]:
            d = dict(self.docs[i])
            d["score"] = float(score)
            results.append(d)
        return results
