# RAG_IIAB — Wikipedia Offline Search

A two-pipeline system that turns a Wikipedia ZIM file into a local semantic search engine. Run the **preprocessing pipeline** once on a powerful machine, then deploy the **search pipeline** on a Raspberry Pi (or any low-powered device).

---

## Install

```bash
pip install pyzim beautifulsoup4 numpy faiss-cpu requests fastapi uvicorn
```

> **Note:** If you want GPU-accelerated FAISS, install `faiss-gpu` instead of `faiss-cpu`.  
> Ollama must also be running locally with your chosen models pulled:
> ```bash
> ollama pull nomic-embed-text
> ollama pull qwen2.5:0.5b
> ```

---

## How to Run

### Pipeline 1 — Preprocessing (run once, on a big machine)

```bash
# Step 1: Chunk + embed all Wikipedia articles → saves to rag_chunks.db
python main.py

# Step 2: Build the search indexes from rag_chunks.db
python build_index.py
```

Then copy these 3 output files to your Pi:
```
faiss.index
chunks_cache.pkl
fts_index.db
```

---

### Pipeline 2 — Search Server (run on the Pi)

```bash
python search.py
```

Then open `http://localhost:8000` in your browser.

---

## File Overview

### Pipeline 1 — Preprocessing

| File | Role |
|---|---|
| `main.py` | Entry point. Loops through every article in the ZIM file, chunks it, embeds it, and saves to SQLite. |
| `zim_reader.py` | Opens the ZIM file, parses each article's HTML into clean plain text with `## Heading` markers. |
| `chunker.py` | Splits each article's text into small, overlapping chunks (~800 chars). Keeps sections separate and isolates the intro paragraph. |
| `db_writer.py` | Calls Ollama to get embeddings for each chunk, then saves the chunk + embedding (binary-packed) to `rag_chunks.db`. |
| `build_index.py` | Reads `rag_chunks.db` and builds 3 search artifacts: a FAISS vector index, a pickle cache of titles/texts, and an FTS5 keyword index. |

### Pipeline 2 — Search Server

| File | Role |
|---|---|
| `search.py` | FastAPI server. Embeds the query, runs vector + keyword search, fuses results, builds a prompt, and streams the LLM answer back. |
| `index.html` | Browser UI. Sends questions to `/ask` and displays the streaming answer, timing stats, and source chunks. |

### Config

| File | Role |
|---|---|
| `config.json` | Single config for both pipelines — models, file paths, chunk size, and search parameters. |

---

## How Pipeline 1 Works (Preprocessing)

```
wikipedia.zim
     │
     ▼
zim_reader.py       ← parses HTML → clean text with ## Section markers
     │
     ▼
chunker.py          ← splits text into ~800-char chunks per section
     │
     ▼
db_writer.py        ← embeds each chunk via Ollama → saves to rag_chunks.db
     │
     ▼
build_index.py      → faiss.index       (vector similarity search)
                    → chunks_cache.pkl  (titles + texts for fast lookup)
                    → fts_index.db      (BM25 keyword search via FTS5)
```

**What each step does:**

1. `zim_reader.py` opens the ZIM file entry by entry, strips HTML tags, removes junk sections (References, External Links, etc.), and emits clean text blocks with `## Heading` markers so the chunker knows section boundaries.

2. `chunker.py` walks those blocks and groups them into chunks. The very first paragraph of every article is always isolated (it's the most fact-dense part). After that, paragraphs are packed together up to ~800 chars within the same section, with 1 paragraph of overlap so context isn't lost at chunk boundaries. Each chunk is labeled `Section: <name>` so the embedding knows what part of the article it came from.

3. `db_writer.py` calls Ollama's embed endpoint for each chunk (in parallel, 4 workers by default), then binary-packs the float array and saves everything to SQLite. It uses `INSERT OR REPLACE` so the run is safely resumable if interrupted.

4. `build_index.py` reads all embeddings out of SQLite, normalizes them, and builds a FAISS IVF index for fast approximate nearest-neighbor search. It also saves a pickle of all titles/texts (so the server never has to hit SQLite at query time) and an FTS5 SQLite table for keyword search with a Porter stemmer.

---

## How Pipeline 2 Works (Search)

```
User question
     │
     ▼
embed_query()       ← Ollama: question → float vector
     │
     ├──► FAISS search    (top 60 by vector similarity)
     │
     ├──► BM25 search     (top 60 by keyword match, FTS5)
     │
     ▼
RRF fusion          ← combines both ranked lists into one score
     │
     ▼
Title boost         ← bumps chunks whose article title matches query words
     │
     ▼
Top 3 chunks selected
     │
     ▼
build_prompt()      ← wraps chunks + question into a prompt
     │
     ▼
Ollama LLM          ← streams the answer token by token
     │
     ▼
Browser             ← answer + timing stats + source chunks displayed
```

**What each step does:**

1. The query is embedded with the same model used during preprocessing so the vector space matches.

2. Two searches run independently — FAISS finds the most semantically similar chunks, and BM25 finds chunks with exact keyword matches. Neither alone is sufficient: vector search misses precise entity names; keyword search misses paraphrased concepts.

3. **RRF (Reciprocal Rank Fusion)** merges the two ranked lists by adding `1 / (60 + rank)` from each list. This rewards chunks that appear highly in both lists without requiring score normalization.

4. A small title boost (0.15 weight) nudges up chunks from articles whose title overlaps the query — useful for "who is X" style questions.

5. The top 3 chunks are formatted into a prompt that instructs the LLM to answer only from the provided context. The LLM streams its response back through FastAPI, and timing stats are appended at the end as a JSON sentinel so the browser can display them without a second request.

---

## Config Reference (`config.json`)

| Key | Default | What it controls |
|---|---|---|
| `embed_model` | `nomic-embed-text` | Ollama model used for embeddings |
| `llm_model` | `qwen2.5:0.5b` | Ollama model used for answer generation |
| `zim_file` | `wikipedia_en_100_2026-01.zim` | Input ZIM file |
| `db_file` | `rag_chunks.db` | SQLite DB storing chunks + embeddings |
| `faiss_file` | `faiss.index` | FAISS index output |
| `cache_file` | `chunks_cache.pkl` | Pickle cache of titles/texts |
| `fts_db_file` | `fts_index.db` | SQLite FTS5 keyword index |
| `top_k` | `3` | Chunks shown to LLM and user |
| `faiss_k` | `60` | Candidates fetched from FAISS before fusion |
| `bm25_k` | `60` | Candidates fetched from BM25 before fusion |
| `chunk_max_chars` | `800` | Max characters per chunk |
| `chunk_overlap_paragraphs` | `1` | Paragraphs of overlap between chunks |
