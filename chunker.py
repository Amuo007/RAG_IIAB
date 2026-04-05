import re
from typing import Generator, Dict


def _split_sentences(text: str, max_chars: int):
    """
    Split a paragraph that's too long into sentence-sized pieces,
    each under max_chars. Falls back to hard truncation if a single
    sentence is still too long.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = []
    current_len = 0
    for s in sentences:
        proposed = current_len + (1 if current else 0) + len(s)
        if proposed <= max_chars:
            current.append(s)
            current_len = proposed
        else:
            if current:
                yield " ".join(current)
            # Single sentence still too long — hard truncate
            if len(s) > max_chars:
                yield s[:max_chars]
            else:
                current = [s]
                current_len = len(s)
    if current:
        yield " ".join(current)


def chunk_text(
    title: str,
    zim_path: str,
    text: str,
    max_chars: int = 800,
    overlap_paragraphs: int = 1,
) -> Generator[Dict, None, None]:
    """
    Section-aware chunker.

    zim_reader now emits '## Heading' markers between paragraphs.
    This chunker:
      1. Parses those markers to track the current section.
      2. Always emits the very first paragraph alone (Wikipedia's lead/intro
         is the most fact-dense part of any article — isolating it gives the
         retrieval model a clean, focused chunk for "quick fact" queries).
      3. Chunks the rest within each section at ~800 chars with overlap so
         no two sections bleed into each other.
      4. Prepends 'Section: <name>' to every chunk_text so the embedding
         model (and BM25) know which part of the article the chunk is from.

    Stored chunk_text format:
        Section: Death

        On June 25, 2009, Jackson died from cardiac arrest...

    Combined with db_writer's title prefix the embedding input becomes:
        Title: Michael Jackson
        Section: Death

        On June 25, 2009, Jackson died from cardiac arrest...
    """

    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not raw_blocks:
        return

    # ── Parse into (section_name, [paragraphs]) groups ─────────────────────
    current_section = "Introduction"
    sections = []        # [(section_name, [para, para, ...]), ...]
    current_paras = []

    for block in raw_blocks:
        if block.startswith("## "):
            if current_paras:
                sections.append((current_section, current_paras))
                current_paras = []
            current_section = block[3:].strip()
        else:
            current_paras.append(block)

    if current_paras:
        sections.append((current_section, current_paras))

    if not sections:
        return

    chunk_id = 1

    def emit(section_name, paragraphs):
        nonlocal chunk_id
        body = "\n\n".join(paragraphs)
        yield {
            "title": title,
            "zim_path": zim_path,
            "chunk_id": chunk_id,
            "chunk_text": f"Section: {section_name}\n\n{body}",
        }
        chunk_id += 1

    # ── Always emit the very first paragraph alone (intro isolation) ────────
    first_section, first_paras = sections[0]
    if first_paras:
        yield from emit(first_section, [first_paras[0]])
        remaining_intro = first_paras[1:]
    else:
        remaining_intro = []

    # ── Chunk remaining paragraphs section by section ───────────────────────
    all_sections = [(first_section, remaining_intro)] + sections[1:]

    for section_name, paras in all_sections:
        if not paras:
            continue

        current = []
        current_len = 0

        for para in paras:
            # Oversized single paragraph — split by sentences then chunk normally
            if len(para) > max_chars:
                if current:
                    yield from emit(section_name, current)
                    current = []
                    current_len = 0
                for sentence_chunk in _split_sentences(para, max_chars):
                    yield from emit(section_name, [sentence_chunk])
                continue

            proposed = current_len + (2 if current else 0) + len(para)

            if proposed <= max_chars:
                current.append(para)
                current_len = proposed
            else:
                # Emit current chunk, carry overlap into next
                yield from emit(section_name, current)
                overlap = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
                current = overlap + [para]
                current_len = sum(len(x) for x in current) + 2 * max(0, len(current) - 1)

        if current:
            yield from emit(section_name, current)
