import re
import json
import time
import pickle
import sqlite3
import requests
import numpy as np
import faiss
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI()

# ── Config ───────────────────────────────────────────────────────────────────
with open("config.json") as f:
    _cfg = json.load(f)

OLLAMA_URL  = _cfg["ollama_url"]
EMBED_MODEL = _cfg["embed_model"]
LLM_MODEL   = _cfg["llm_model"]
TOP_K       = _cfg["top_k"]
FAISS_K     = _cfg["faiss_k"]
BM25_K      = _cfg["bm25_k"]
RRF_K       = _cfg["rrf_k"]
FAISS_FILE  = _cfg["faiss_file"]
CACHE_FILE  = _cfg["cache_file"]
FTS_DB_FILE = _cfg["fts_db_file"]

_STOPWORDS = {
    'the','a','an','is','was','were','did','do','does','when','where',
    'who','what','how','why','which','in','on','at','to','for','of',
    'and','or','but','it','he','she','they','we','i','you','his','her',
    'their','its','be','been','have','has','had','this','that','these',
    'those','will','would','could','should','about','from','with',
}

# ── Load pre-built indexes ────────────────────────────────────────────────────
print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_FILE)
if hasattr(faiss_index, 'nprobe'):
    faiss_index.nprobe = 64
print(f"  {faiss_index.ntotal} vectors")

print("Loading chunks cache...")
with open(CACHE_FILE, "rb") as f:
    _cache = pickle.load(f)
titles = _cache["titles"]
texts  = _cache["texts"]
print(f"  {len(titles)} chunks")
print("Ready.")


# ── Core functions ────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text, "truncate": True},
        timeout=30,
    )
    vec = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return vec / np.clip(np.linalg.norm(vec), 1e-9, None)


def _bm25_search(query: str, top_k: int):
    words = re.findall(r'\w+', query)
    if not words:
        return []
    match_expr = " ".join(f'"{w}"' for w in words if len(w) >= 2)
    if not match_expr:
        return []
    try:
        fts = sqlite3.connect(FTS_DB_FILE)
        rows = fts.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_expr, top_k),
        ).fetchall()
        fts.close()
        return [(int(row[0]) - 1, rank) for rank, row in enumerate(rows)]
    except Exception as e:
        print(f"BM25 error: {e}")
        return []


def _title_boost(query_words: set, title: str) -> float:
    if not query_words:
        return 0.0
    title_words = set(re.findall(r'\w+', title.lower()))
    return len(query_words & title_words) / len(query_words)


def retrieve(qvec: np.ndarray, query: str, top_k: int = TOP_K):
    # 1. Vector search
    vec_scores, vec_indices = faiss_index.search(qvec.reshape(1, -1), FAISS_K)
    vec_ranks = {int(idx): rank for rank, idx in enumerate(vec_indices[0]) if idx >= 0}

    # 2. BM25 keyword search
    bm25_ranks = {idx: rank for idx, rank in _bm25_search(query, BM25_K)}

    # 3. RRF fusion
    all_indices = set(vec_ranks) | set(bm25_ranks)
    rrf_scores = {}
    for idx in all_indices:
        score = 0.0
        if idx in vec_ranks:
            score += 1.0 / (RRF_K + vec_ranks[idx])
        if idx in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[idx])
        rrf_scores[idx] = score

    # 4. Title boost
    q_words = set(re.findall(r'\w+', query.lower())) - _STOPWORDS
    final = {
        idx: score + _title_boost(q_words, titles[idx]) * 0.15
        for idx, score in rrf_scores.items()
    }

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return [(titles[i], texts[i], rrf_scores[i]) for i, _ in ranked[:top_k]]


def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[{title}]\n{text}" for title, text, _ in chunks
    )
    return (
        "Answer ONLY using the context below. Be concise. "
        "Do not add information not in the context. "
        "If the context is not relevant, say \"I don't have information about that.\"\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )


# ── API ───────────────────────────────────────────────────────────────────────

class Query(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index_page():
    with open("index.html") as f:
        return f.read()


@app.post("/ask")
def ask(q: Query):
    t_start = time.time()

    # Step 1: embed query
    t0 = time.time()
    qvec = embed_query(q.question)
    embed_ms = round((time.time() - t0) * 1000)

    # Step 2: retrieve chunks
    t0 = time.time()
    chunks = retrieve(qvec, q.question)
    retrieve_ms = round((time.time() - t0) * 1000)

    # Step 3: build prompt
    prompt = build_prompt(q.question, chunks)

    def stream():
        t_gen = time.time()
        t_first_token_ms = None
        prompt_tokens = 0
        gen_tokens = 0

        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    if t_first_token_ms is None:
                        t_first_token_ms = round((time.time() - t_gen) * 1000)
                    yield token
                if data.get("done"):
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    gen_tokens    = data.get("eval_count", 0)

        gen_ms   = round((time.time() - t_gen) * 1000)
        total_ms = round((time.time() - t_start) * 1000)

        # Send stats + sources as a single JSON sentinel at the end of the stream
        stats = {
            "embed_ms":    embed_ms,
            "retrieve_ms": retrieve_ms,
            "prompt_tokens": prompt_tokens,
            "ttft_ms":     t_first_token_ms or 0,
            "gen_ms":      gen_ms,
            "gen_tokens":  gen_tokens,
            "total_ms":    total_ms,
            "sources": [
                {"title": t, "score": round(s, 4), "text": text}
                for t, text, s in chunks
            ],
        }
        yield f"\n\n[STATS]{json.dumps(stats)}"

    return StreamingResponse(stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
