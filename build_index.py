"""
Run this ONCE on the big computer after indexing your ZIM files.
Produces three files that search.py loads at startup on the Pi:
  - faiss.index      : pre-built FAISS vector index (IVF if large enough, flat otherwise)
  - chunks_cache.pkl : titles + texts arrays (avoids re-reading SQLite at startup)
  - fts_index.db     : SQLite FTS5 BM25 keyword index
"""

import struct
import sqlite3
import json
import pickle
import numpy as np
import faiss

import json as _json
with open("config.json") as _f:
    _cfg = _json.load(_f)

DB_FILE     = _cfg["db_file"]
FAISS_FILE  = _cfg["faiss_file"]
CACHE_FILE  = _cfg["cache_file"]
FTS_DB_FILE = _cfg["fts_db_file"]

MIN_CHUNK_LEN  = 100   # same filter as search.py
IVF_NPROBE     = 64    # how many IVF clusters to search at query time


def decode_embedding(emb):
    if isinstance(emb, bytes):
        return struct.unpack(f'{len(emb)//4}f', emb)
    return json.loads(emb)


def main():
    # ── 1. Load all chunks from SQLite ─────────────────────────────────────
    print("Loading chunks from SQLite...")
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT title, chunk_text, embedding FROM chunks").fetchall()
    conn.close()

    rows = [r for r in rows if len(r[1]) > MIN_CHUNK_LEN]
    n = len(rows)
    print(f"  {n} chunks after filtering")

    titles = [r[0] for r in rows]
    texts  = [r[1] for r in rows]

    # ── 2. Build normalized embedding matrix ───────────────────────────────
    print("Decoding embeddings...")
    matrix = np.array([decode_embedding(r[2]) for r in rows], dtype=np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-9, None)
    dim    = matrix.shape[1]
    print(f"  Matrix: {matrix.shape}")

    # ── 3. Build FAISS index ───────────────────────────────────────────────
    # IVFFlat partitions vectors into clusters so query only searches a subset
    # — much faster than flat search for large collections.
    # Needs at least 39 * nlist training points, otherwise fall back to flat.
    nlist     = max(1, min(int(np.sqrt(n)), 4096))
    min_train = nlist * 39

    if n >= min_train:
        print(f"Building IVF index (nlist={nlist}, nprobe={IVF_NPROBE})...")
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.add(matrix)
        index.nprobe = IVF_NPROBE
    else:
        print(f"Building flat index (n={n} too small for IVF)...")
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

    faiss.write_index(index, FAISS_FILE)
    print(f"  Saved {FAISS_FILE} ({index.ntotal} vectors)")

    # ── 4. Save titles + texts cache ───────────────────────────────────────
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"titles": titles, "texts": texts}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {CACHE_FILE}")

    # ── 5. Build FTS5 keyword index ────────────────────────────────────────
    # FTS5 is SQLite's built-in full-text search — inverted index in C.
    # We index "title + chunk_text" together so entity names (e.g. "Michael Jackson")
    # are searchable even though they live in the title column.
    # Porter stemmer maps die/died/dying → same root, fixing the query mismatch.
    print("Building FTS5 index...")
    fts = sqlite3.connect(FTS_DB_FILE)
    fts.execute("DROP TABLE IF EXISTS chunks_fts")

    # Try porter stemmer first (handles die→died, run→running etc.)
    try:
        fts.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                tokenize='porter unicode61'
            )
        """)
        print("  Tokenizer: porter unicode61")
    except Exception:
        fts.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                tokenize='unicode61'
            )
        """)
        print("  Tokenizer: unicode61 (porter not available)")

    # rowid is 1-based in SQLite; chunk index = rowid - 1
    # Concatenate title + text so BM25 rewards both entity name and content match
    batch = [
        (i + 1, f"{titles[i]}\n\n{texts[i]}")
        for i in range(n)
    ]
    fts.executemany("INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)", batch)
    fts.commit()
    fts.close()
    print(f"  Saved {FTS_DB_FILE} ({n} documents)")

    print("\nAll indexes built. Copy these to the Pi:")
    print(f"  {FAISS_FILE}")
    print(f"  {CACHE_FILE}")
    print(f"  {FTS_DB_FILE}")


if __name__ == "__main__":
    main()
