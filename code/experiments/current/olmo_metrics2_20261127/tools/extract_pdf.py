"""Extract a PDF's text with page markers, so a reader can check that every page was read.

Usage (run from outside the download directory, with -I so nothing in the download directory is imported):
    uv run --with pymupdf python -I <this script> <paper.pdf> <out.txt>
"""

import sys

import fitz  # pymupdf


def main() -> None:
    pdf, out = sys.argv[1], sys.argv[2]
    doc = fitz.open(pdf)
    n = doc.page_count
    chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text")
            chars += len(text)
            f.write(f"\n=== page {i}/{n} ===\n")
            f.write(text)
        f.write(f"\n=== end of document: {n} pages ===\n")
    print(f"{pdf}: {n} pages, {chars} characters -> {out}")


if __name__ == "__main__":
    main()
