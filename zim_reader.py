import re
from typing import Generator, Dict

import pyzim
from bs4 import BeautifulSoup

# --- config ---
SKIP_NON_ASCII_TITLES = False  # set True to skip non-English titles
# --------------


# Section headings to skip — these contain no useful retrieval content
_SKIP_SECTIONS = {
    'references', 'notes', 'footnotes', 'bibliography',
    'external links', 'see also', 'further reading', 'citations',
}


def extract_wikipedia_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "sup", "ref"]):
        tag.decompose()

    for tag in soup.find_all("div", class_=["navbox", "reflist", "zim-footer", "mw-references-wrap"]):
        tag.decompose()
    for tag in soup.find_all("table", class_="navbox"):
        tag.decompose()

    main = soup.find("div", class_="mw-parser-output")
    if not main:
        main = soup.body
    if not main:
        soup.decompose()
        return ""

    blocks = []
    in_skipped_section = False

    # Walk h2/h3/p in document order so section headings appear before their paragraphs.
    # This lets the chunker know which section each paragraph belongs to.
    for el in main.find_all(["h2", "h3", "p"]):
        if el.name in ("h2", "h3"):
            raw = el.get_text(" ", strip=True)
            heading = re.sub(r"\[.*?\]", "", raw).strip()
            if not heading:
                continue
            # Stop indexing junk sections like References, External links etc.
            in_skipped_section = heading.lower() in _SKIP_SECTIONS
            if not in_skipped_section:
                blocks.append(f"## {heading}")

        elif el.name == "p":
            if in_skipped_section:
                continue
            text = el.get_text(" ", strip=True)
            if text:
                blocks.append(text)

    # Fall back to list items if nothing was extracted
    if not any(not b.startswith("##") for b in blocks):
        for li in main.find_all("li"):
            text = li.get_text(" ", strip=True)
            if text:
                blocks.append(text)

    soup.decompose()

    text = "\n\n".join(blocks)
    text = re.sub(r"\[\s*[a-zA-Z0-9]*\s*\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

    return text


def iter_articles(zim_file: str, skip_paths: set = None) -> Generator[Dict, None, None]:
    zim = pyzim.Zim.open(zim_file)
    skip_paths = skip_paths or set()

    skipped_non_ascii = 0
    skipped_empty = 0
    skipped_error = 0
    skipped_done = 0
    redirects = 0
    count = 0

    try:
        for entry in zim.iter_articles():
            title = entry.title

            if SKIP_NON_ASCII_TITLES and not title.isascii():
                skipped_non_ascii += 1
                continue

            is_redirect = entry.is_redirect
            if callable(is_redirect):
                is_redirect = is_redirect()

            if is_redirect:
                redirects += 1
                continue

            if entry.full_url in skip_paths:
                skipped_done += 1
                if skipped_done % 5000 == 0:
                    print(f"  ⏭ Skipped {skipped_done} already done...")
                continue

            try:
                raw_html = entry.read().decode("utf-8", errors="ignore")
                clean_text = extract_wikipedia_text(raw_html)

                if not clean_text:
                    skipped_empty += 1
                    continue

                count += 1
                yield {
                    "title": title,
                    "zim_path": entry.full_url,
                    "text": clean_text,
                }

            except Exception as e:
                skipped_error += 1
                print(f"Skipped [ERROR]: {title} | {e}")

    finally:
        zim.close()
        print(f"\n--- ZIM Reader Summary ---")
        print(f"Yielded:              {count}")
        print(f"Redirects:            {redirects}")
        print(f"Skipped already done: {skipped_done}")
        print(f"Skipped non-ASCII:    {skipped_non_ascii}")
        print(f"Skipped empty:        {skipped_empty}")
        print(f"Skipped errors:       {skipped_error}")
        print(f"Total skipped:        {skipped_non_ascii + skipped_empty + skipped_error}")