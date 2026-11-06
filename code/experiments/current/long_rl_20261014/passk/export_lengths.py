#!/usr/bin/env python3
"""results_10k/*.json -> report_lengths.html: body-line histograms of each checkpoint's distinct Lean-checked proofs.

  python3 passk/export_lengths.py pretrain v2_final v5_round025
"""

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    models = []
    for tag in sys.argv[1:]:
        res = json.load(open(f"{HERE}/results_10k/{tag}.json"))
        rows = res["sets"]["textbook72"]["rows"]
        pairs = [(w, p) for r in rows for w, p in zip(r["written"], r["pruned"], strict=True)]
        models.append({"tag": tag, "sha256": res["sha256"], "n": len(pairs),
                       "problems": sum(r["solved"] for r in rows),
                       "padded": sum(w > p for w, p in pairs),  # proofs with lines the conclusion never cites
                       "top_problem": max(len(r["written"]) for r in rows),
                       "written": collections.Counter(w for w, _ in pairs),
                       "pruned": collections.Counter(p for _, p in pairs)})
    page = open(f"{HERE}/report_lengths.template.html").read().replace("/*DATA*/", json.dumps(models))
    open(f"{HERE}/report_lengths.html", "w").write(page)
    for m in models:
        print(m["tag"], m["n"], "proofs over", m["problems"], "problems")


if __name__ == "__main__":
    main()
