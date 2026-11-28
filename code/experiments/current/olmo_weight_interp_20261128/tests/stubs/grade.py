"""Stands in for Dan's olmo_ifomega/grade.py <samples.gz> <problems> <graded.jsonl>.
strict = the row's stub_pass."""

import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as f, open(sys.argv[3], "w") as out:
    for ln in f:
        r = json.loads(ln)
        g = {k: r[k] for k in ("pid", "model", "rev", "job", "chunk", "i", "n_tok", "finish")}
        g.update(
            task="stub", strict=bool(r["stub_pass"]), per=[bool(r["stub_pass"])], think_end=True
        )
        out.write(json.dumps(g) + "\n")
