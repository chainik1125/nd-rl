"""P3: K12 with every proof replaced by a minimal one (pool idea B1).

For every K12 row, run the fork's goal-directed minimal-proof search (`minlen.py`) with the bound set to
one line fewer than the row's proof. If it finds a shorter verified proof, the row keeps its prompt and
takes the new proof; otherwise the row is unchanged. The output has the K12 record fields plus
`orig_n_lines` and `minlen_status` in {shorter, same, timeout, none}.

    cd code && nohup uv run python pretrain_data/minimal_k12.py --workers 4 \
        --out ../data/pretrain_data/k12_minimal.jsonl.gz > ../data/pretrain_data/k12_minimal.log 2>&1 &

Analysis/data only; nothing is trained here. Every proof is re-checked by the pinned verifier.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../textbook_problem"), os.path.join(HERE, "../textbook_problem/fork_tools"), os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
from minlen import minlen  # noqa: E402
from patterns import parse  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

K12 = os.path.join("/private/tmp/claude-501/-Users-cyu-Desktop-github-nd-rl/0e8fdc82-572d-4fd1-ba25-ff48dbf29acf/scratchpad",
                   "dan_data", "train_k12.jsonl.gz")


def work(args):
    r, secs = args
    n0 = r["n_lines"]
    out = dict(r, orig_n_lines=n0)
    if n0 <= 2:
        out["minlen_status"] = "same"
        return out
    try:
        u = minlen(r["prompt"], bound=n0 - 1, time_limit=secs)
    except Exception as e:  # noqa: BLE001
        out["minlen_status"] = f"error:{type(e).__name__}"
        return out
    if u.get("timeout"):
        out["minlen_status"] = "timeout"
        return out
    if not u.get("proof") or u.get("min_lines_ub") is None:
        out["minlen_status"] = "none"
        return out
    ok, reason, n = safe_verify(r["prompt"], u["proof"])
    if not ok or n >= n0:
        out["minlen_status"] = "none" if ok else f"reject:{reason[:20]}"
        return out
    lines = parse(u["proof"])
    rules = collections.Counter(ln["rule"] for ln in lines if ln["rule"] not in ("PR", "AS"))
    out.update(proof=u["proof"], n_lines=n, rules=dict(sorted(rules.items())), minlen_status="shorter")
    if "text" in out:
        out["text"] = r["prompt"] + " " + u["proof"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = []
    with gzip.open(K12, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
            if a.limit and len(rows) >= a.limit:
                break
    print("rows", len(rows), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    stats = collections.Counter()
    saved = 0
    by_len = collections.defaultdict(collections.Counter)
    with gzip.open(a.out, "wt") as fo, ProcessPoolExecutor(a.workers) as ex:
        for i, out in enumerate(ex.map(work, ((r, a.secs) for r in rows), chunksize=64)):
            stats[out["minlen_status"].split(":")[0]] += 1
            by_len[out["orig_n_lines"]][out["minlen_status"].split(":")[0]] += 1
            saved += out["orig_n_lines"] - out["n_lines"]
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            if (i + 1) % 5000 == 0:
                print(f"{i + 1} rows {time.time() - t0:.0f}s {dict(stats)} lines saved {saved}", flush=True)
    summary = dict(rows=len(rows), secs=round(time.time() - t0), status=dict(stats), lines_saved=saved,
                   by_orig_len={k: dict(v) for k, v in sorted(by_len.items())})
    with open(a.out.replace(".jsonl.gz", ".stats.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
