# index_docs.py
import os
import sqlite3
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCS_FOLDER = "docs"
DB_PATH = "embeddings.db"
MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

model = SentenceTransformer(MODEL_NAME)

def ensure_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_name TEXT,
            chunk_index INTEGER,
            text TEXT,
            emb BLOB,
            chunk_hash TEXT UNIQUE
        )
    """)
    conn.commit()

def chunk_text(text):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+CHUNK_SIZE]
        chunks.append(chunk.strip())
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c]

def emb_to_blob(vec):
    return vec.astype("float32").tobytes()

def index_docs():
    print("📄 Starting document indexing...")
    
    if not os.path.exists(DOCS_FOLDER):
        print("❌ docs/ folder not found!")
        return
    
    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".md") or f.endswith(".txt")]
    print("Found docs:", files)

    if not files:
        print("❌ No markdown/text documents in docs/")
        return

    conn = sqlite3.connect(DB_PATH)
    ensure_db(conn)

    for fn in files:
        path = os.path.join(DOCS_FOLDER, fn)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        chunks = chunk_text(raw)
        print(f"Indexing {fn} ({len(chunks)} chunks)")

        for idx, chunk in enumerate(tqdm(chunks)):
            chash = hashlib.sha256((fn + str(idx) + chunk[:64]).encode()).hexdigest()

            cur = conn.execute("SELECT 1 FROM chunks WHERE chunk_hash=?", (chash,))
            if cur.fetchone():
                continue

            emb = model.encode(chunk)
            conn.execute(
                "INSERT INTO chunks (doc_name, chunk_index, text, emb, chunk_hash) VALUES (?, ?, ?, ?, ?)",
                (fn, idx, chunk, emb_to_blob(emb), chash)
            )

    conn.commit()
    conn.close()
    print("Indexing complete! embeddings.db created.")

if __name__ == "__main__":
    index_docs()
