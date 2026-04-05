import gc
from zim_reader import iter_articles
from chunker import chunk_text
from db_writer import init_db, save_chunks_parallel

import json as _json
with open("config.json") as _f:
    _cfg = _json.load(_f)

ZIM_FILE           = _cfg["zim_file"]
DB_FILE            = _cfg["db_file"]
EMBED_MODEL        = _cfg["embed_model"]
MAX_CHARS          = _cfg["chunk_max_chars"]
OVERLAP_PARAGRAPHS = _cfg["chunk_overlap_paragraphs"]
MAX_ARTICLE_TEXT_CHARS = 200_000
COMMIT_EVERY = 10

# --- config ---
SAVE_TO_TEXT_FILE = True        # set False to skip text file output
TEXT_OUTPUT_FILE = "extracted_2.txt"
EMBED_CHUNKS = True             # set False to skip embedding (text only mode)
# --------------


def load_done_paths(conn):
    rows = conn.execute("SELECT DISTINCT zim_path FROM chunks").fetchall()
    return set(row[0] for row in rows)


def main():
    conn = init_db(DB_FILE)

    done_paths = load_done_paths(conn)
    print(f"Resuming — {len(done_paths)} articles already done, skipping them.")

    article_count = 0
    chunk_count = 0
    skipped_big = 0

    txt_file = open(TEXT_OUTPUT_FILE, "w", encoding="utf-8") if SAVE_TO_TEXT_FILE else None

    for article in iter_articles(ZIM_FILE, skip_paths=done_paths):
        article_count += 1

        if len(article["text"]) > MAX_ARTICLE_TEXT_CHARS:
            skipped_big += 1
            print(f"[{article_count}] SKIPPED (too large): {article['title']}")
            continue

        print(f"\n[{article_count}] Starting: {article['title']}")

        chunks = list(chunk_text(
            title=article["title"],
            zim_path=article["zim_path"],
            text=article["text"],
            max_chars=MAX_CHARS,
            overlap_paragraphs=OVERLAP_PARAGRAPHS,
        ))

        # write chunks to text file
        if txt_file:
            for chunk in chunks:
                txt_file.write("=" * 80 + "\n")
                txt_file.write(f"TITLE: {chunk['title']}\n")
                txt_file.write(f"ZIM_PATH: {chunk['zim_path']}\n")
                txt_file.write(f"CHUNK_ID: {chunk['chunk_id']}\n")
                txt_file.write("=" * 80 + "\n\n")
                txt_file.write(chunk["chunk_text"])
                txt_file.write("\n\n")
            txt_file.flush()

        # embed and save to DB
        if EMBED_CHUNKS:
            saved = save_chunks_parallel(conn, chunks, embedding_model=EMBED_MODEL)
            chunk_count += saved
        else:
            chunk_count += len(chunks)

        print(f"  ✓ Done — {len(chunks)} chunks | total so far: {chunk_count}")

        if article_count % COMMIT_EVERY == 0:
            conn.commit()
            gc.collect()

    if txt_file:
        txt_file.close()

    conn.commit()
    conn.close()

    print(f"\n--- Pipeline Summary ---")
    print(f"Articles processed:  {article_count}")
    print(f"Chunks saved:        {chunk_count}")
    print(f"Skipped (too large): {skipped_big}")
    if SAVE_TO_TEXT_FILE:
        print(f"Text saved to:       {TEXT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()