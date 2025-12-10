# rag.py
import os
import sqlite3
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# --- Configuration ---
DB_PATH = "embeddings.db"
EMB_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3  # retrieval candidate count (we will pass only top_n to the LLM)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")  # override in .env if needed
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))  # seconds

# --- MiniRAG class ---
class MiniRAG:
    def __init__(self):
        # load embedding model
        self.model = SentenceTransformer(EMB_MODEL)

        # simple in-memory cache
        self._query_cache = {}

        # verify DB exists
        if not os.path.exists(DB_PATH):
            raise RuntimeError(f"{DB_PATH} not found. Run index_docs.py first to create embeddings.db")

        # connect and load index
        self.conn = sqlite3.connect(DB_PATH)
        self._load_index()

    def _load_index(self):
        rows = self.conn.execute("SELECT doc_name, chunk_index, text, emb FROM chunks").fetchall()
        if not rows:
            raise RuntimeError("No embeddings found in DB. Did you run index_docs.py?")

        self.texts = []
        vectors = []
        for doc, idx, text, emb_blob in rows:
            self.texts.append((doc, idx, text))
            vectors.append(np.frombuffer(emb_blob, dtype="float32"))
        self.vectors = np.vstack(vectors)

        # build nearest-neighbor index
        n_neighbors = min(TOP_K, len(self.vectors))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.nn.fit(self.vectors)

    def retrieve(self, query):
        """
        Compute embedding for query, return top-k retrieved chunks sorted by score (best first).
        Results are dicts with keys: doc, chunk, score, text
        """
        if query in self._query_cache:
            return self._query_cache[query]

        q_vec = self.model.encode(query)
        n_neighbors = min(TOP_K, len(self.vectors))
        dists, idxs = self.nn.kneighbors([q_vec], n_neighbors=n_neighbors)

        results = []
        for d, i in zip(dists[0], idxs[0]):
            doc, idx, text = self.texts[i]
            score = float(1 - d)
            results.append({"doc": doc, "chunk": idx, "score": score, "text": text})

        results = sorted(results, key=lambda r: r["score"], reverse=True)
        self._query_cache[query] = results
        return results

    def _build_prompt(self, query, retrieved, top_n=2):
        """
        Build a strict prompt using only top_n retrieved snippets.
        Prompt instructs the model to be concise and only use the provided snippets.
        """
        snippets = []
        for r in retrieved[:top_n]:
            header = f"{r['doc']}#chunk{r['chunk']} (score={r['score']:.2f})"
            snippets.append(f"{header}\n{r['text']}")
        context = "\n\n".join(snippets)

        prompt = (
            "You are a precise assistant. Use ONLY the provided document snippets below and nothing else.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer VERY CONCISELY in 1 or 2 short sentences, only using facts supported by the snippets. "
            "Do NOT add any information that is not present. "
            "If the answer is not present in the snippets, reply exactly: \"I couldn't find the answer in the documents.\" "
            "At the end, append a single 'Sources:' line listing the snippet headers you used (comma-separated), e.g. Sources: doc1.md#chunk0.\n"
        )
        return prompt

    def _call_ollama(self, prompt, max_tokens=256):
        """
        Call local Ollama HTTP API. Returns the textual response (string).
        Raises RuntimeError on failure.
        """
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            # preferred locations for the textual response
            if isinstance(data, dict):
                if "response" in data and isinstance(data["response"], str):
                    return data["response"].strip()
                if "results" in data and isinstance(data["results"], list):
                    parts = []
                    for r in data["results"]:
                        if isinstance(r, dict):
                            for k in ("response", "content", "text"):
                                if k in r and isinstance(r[k], str):
                                    parts.append(r[k].strip())
                    if parts:
                        return "\n\n".join(parts)
            # fallback: return stringified data
            return str(data)
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def ask(self, query):
        """
        Full RAG flow: retrieve -> build strict prompt (top 1-2) -> call Ollama -> post-process.
        If Ollama fails, fallback to returning the best snippet (short).
        """
        retrieved = self.retrieve(query)
        if not retrieved:
            return "I couldn't find the answer in the documents."

        # Build prompt using only top_n snippets (1 or 2 recommended)
        prompt = self._build_prompt(query, retrieved, top_n=2)

        try:
            raw = self._call_ollama(prompt)

            # Normalize and deduplicate lines
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            normalized = []
            prev = None
            for ln in lines:
                if ln == prev:
                    continue
                normalized.append(ln)
                prev = ln
            text = " ".join(normalized)

            # Ensure there's a Sources: line; if not, append a short one
            if "Sources:" not in text:
                srcs = [f"{r['doc']}#chunk{r['chunk']}" for r in retrieved[:2]]
                text = f"{text}\n\nSources: {', '.join(srcs)}"

            # Enforce a safe length cap
            if len(text) > 1000:
                text = text[:950].rsplit(" ", 1)[0] + "..."

            return text

        except Exception:
            # Ollama failed: return short snippet fallback
            top = retrieved[0]
            snippet = top["text"].strip()
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
            src = f"{top['doc']}#chunk{top['chunk']}"
            return f" Ollama unavailable — returning best snippet instead.\n\n{snippet}\n\nSources: {src}"
