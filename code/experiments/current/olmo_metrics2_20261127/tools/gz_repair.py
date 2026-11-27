"""Rewrite a samples.jsonl.gz that a killed job left without its gzip end marker (orchestrator, 2026-11-27).

    gz_repair.py IN.jsonl.gz OUT.jsonl.gz

Decompresses what is there, keeps only complete JSON lines (a line cut mid-write is dropped and counted), writes a valid
gzip. The input is never modified. Prints `GZ_REPAIR rows=<n> dropped_partial=<k> complete_stream=<bool>`.
"""

import gzip
import json
import sys
import zlib


def repair(src: str, dst: str) -> dict:
    raw = open(src, "rb").read()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    txt = d.decompress(raw).decode("utf-8", "replace")
    complete = d.eof
    rows, dropped = [], 0
    for ln in txt.split("\n"):
        if not ln.strip():
            continue
        try:
            json.loads(ln)
            rows.append(ln)
        except ValueError:
            dropped += 1
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        for ln in rows:
            f.write(ln + "\n")
    return {"rows": len(rows), "dropped_partial": dropped, "complete_stream": bool(complete)}


if __name__ == "__main__":
    r = repair(sys.argv[1], sys.argv[2])
    print(
        f"GZ_REPAIR rows={r['rows']} dropped_partial={r['dropped_partial']} complete_stream={r['complete_stream']}"
    )
