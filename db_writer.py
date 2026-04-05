import struct
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"
EMBED_WORKERS = 4  # tune this — try 2, 4, 8


def init_db(db_path: str = "rag_chunks.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            zim_path TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_chunk
        ON chunks(zim_path, chunk_id)
    """)
    conn.commit()
    return conn


def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "input": text, "truncate": True},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]
    except Exception as e:
        print(f"  ⚠ Embedding failed: {e}")
        return None


def save_chunks_parallel(
    conn: sqlite3.Connection,
    chunks: list,
    embedding_model: str = EMBED_MODEL,
) -> int:
    def embed_chunk(chunk):
        embed_text = f"Title: {chunk['title']}\n\n{chunk['chunk_text']}"
        embed_text = embed_text[:1500]  # mxbai-embed-large: 512 token limit (~3 chars/token)
        embedding = get_embedding(embed_text, model=embedding_model)
        return chunk, embedding

    saved = 0
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = {executor.submit(embed_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk, embedding = future.result()
            if embedding is None:
                continue
            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (title, zim_path, chunk_id, chunk_text, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk["title"],
                chunk["zim_path"],
                chunk["chunk_id"],
                chunk["chunk_text"],
                struct.pack(f'{len(embedding)}f', *embedding),
            ))
            saved += 1

    return saved