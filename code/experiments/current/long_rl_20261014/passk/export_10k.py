#!/usr/bin/env python3
"""results_10k/*.json -> pass@k curves (mean + 95% theorem-bootstrap band) as JSON for the report page.

  uv run --with numpy passk/export_10k.py out.json pretrain v2_final v5_round025
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def passk(n, c, k):
    """Unbiased pass@k for one theorem: 1 - C(n - c, k) / C(n, k)."""
    return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def main():
    out, tags = sys.argv[1], sys.argv[2:]
    models = []
    for tag in tags:
        res = json.load(open(f"{HERE}/results_10k/{tag}.json"))
        rows = res["sets"]["textbook72"]["rows"]
        n = min(r["n_tried"] for r in rows)
        ks = sorted({int(round(x)) for x in np.geomspace(1, n, 60)} | {k for k in (1, 10, 100, 1000) if k <= n})
        per = np.array([[passk(r["n_tried"], r["n_ok"], k) for k in ks] for r in rows])
        boot = np.random.default_rng(0).integers(0, len(per), (2000, len(per)))
        lo, hi = np.percentile(per[boot].mean(1), [2.5, 97.5], axis=0)
        models.append({"tag": tag, "sha256": res["sha256"], "temp": res["temp"], "n": n, "ks": ks,
                       "mean": per.mean(0).round(4).tolist(), "lo": lo.round(4).tolist(), "hi": hi.round(4).tolist(),
                       "solved": sum(r["solved"] for r in rows),
                       "solved_dev": sum(r["solved"] for r in rows if r["split"] == "dev"),
                       "solved_train": sum(r["solved"] for r in rows if r["split"] == "train"),
                       "problems": {r["name"]: r["n_ok"] for r in rows}})
    json.dump(models, open(out, "w"))
    for m in models:
        print(m["tag"], m["n"], "solved", m["solved"], {k: v for k, v in zip(m["ks"], m["mean"]) if k in (1, 10, 100, 1000)},
              "pass@n", m["mean"][-1])


if __name__ == "__main__":
    main()
