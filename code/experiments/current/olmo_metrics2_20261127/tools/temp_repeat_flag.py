"""M1 repetition flag, FROZEN before any M1 data (sha256 in temp/FROZEN_repeat_flag.sha256; a test fails on any edit).

A response is flagged when, in its THINKING segment (the text before the first `</think>`; the whole text if there is
none), any sentence or paragraph unit of at least UNIT_MIN_CHARS characters occurs UNIT_REPEATS or more times, or any
word NGRAM-gram occurs NGRAM_REPEATS or more times. Thinking segment only, so the IFBench repeat-family prompts (which ask
the answer to repeat text) cannot trigger it. Modelled on the Olmo 3 report's repetition filter as quoted by the auditor
(10x+ sentences / paragraphs, 50x+ phrases); the exact open-instruct implementation was not available, so "phrase" is
the word 10-gram here.

CLI: temp_repeat_flag.py <samples.jsonl[.gz]> [...]   prints n, flagged, share per file (rows need a `text` field).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter

VERSION = "temp_repeat_flag v1"
UNIT_MIN_CHARS = 20
UNIT_REPEATS = 10
NGRAM = 10
NGRAM_REPEATS = 50

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+|\n")


def thinking_segment(text: str) -> str:
    return text.split("</think>", 1)[0]


def _units(seg: str) -> Counter:
    c: Counter = Counter()
    for splitter in (_PARA, _SENT):
        for u in splitter.split(seg):
            u = " ".join(u.split())
            if len(u) >= UNIT_MIN_CHARS:
                c[(splitter is _PARA, u)] += 1
    return c


def flag_text(text: str) -> bool:
    seg = thinking_segment(text)
    if any(n >= UNIT_REPEATS for n in _units(seg).values()):
        return True
    w = seg.split()
    if len(w) < NGRAM:
        return False
    grams = Counter(" ".join(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1))
    return max(grams.values()) >= NGRAM_REPEATS


def read_rows(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for ln in f:
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                return


def main(argv=None) -> int:
    for p in (sys.argv[1:] if argv is None else argv):
        n = k = 0
        for r in read_rows(p):
            n += 1
            k += flag_text(r.get("text", ""))
        print(f"{p}: n={n} flagged={k} share={(100 * k / n if n else float('nan')):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
