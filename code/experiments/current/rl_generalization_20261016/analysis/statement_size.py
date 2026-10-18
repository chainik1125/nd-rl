"""Statement size: training pools vs test sets (O4), and pass@1 by length bin x statement-size tercile (O5).

    python3 statement_size.py [--reads DIR ...] > out/statement_size.txt

Part 1 (O4): per file, the prompt's whitespace-token count (THM ... SEQ ... PRF included) and, where the rows carry
it, `max_state` (the largest state of the reference proof, in tokens).

Part 2 (O5): per read directory (`runs/eval/v2/<set>_<arm>_s<seed>_plain_k<k>/rows_plain_B64.jsonl`), the problems
are split into statement-size terciles over the whole set and 2-line bins of `min_lines`; per cell, pass@1 (the
mean of the per-attempt `ok`) and the share of attempts that ended at `fail:S:binder mismatch`. One checkpoint, one
read: no intervals. Size is confounded with premises, connectives and padding (observational only).

Standard library only; CPU, one process.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

R = "/scratch/cowifornia/nd-rl/think/runs"
POOLS = f"{R}/pools"
EV = f"{R}/eval/v2"
SIZE_FILES = {
    "L2 pool": f"{POOLS}/curriculum/L2/pool.jsonl",
    "L3 pool": f"{POOLS}/curriculum/L3/pool.jsonl",
    "L3_3p rewrites": f"{POOLS}/curriculum_3p/L3_3p.jsonl",
    "hard712 pool": f"{POOLS}/hard712/pool.jsonl",
    "long_cands": f"{POOLS}/long_cands.jsonl",
    "test_13_20 (full sealed set)": f"{POOLS}/test_13_20/test_13_20.jsonl",
    "test_21_30 (full sealed set)": f"{POOLS}/test_21_30/test_21_30.jsonl",
    "Charles149 (charles_clean)": f"{POOLS}/charles_1002/charles_clean.jsonl",
    "K12 train (Stage-1 source)": "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl",
}
READS = [
    "t1320_grpo_think_vs_mb_kkl0_s0_plain_k16",
    "t1320_ei_think_v_s0_plain_k16",
    "t2130_grpo_think_vs_mb_kkl0_s0_plain_k8",
]


def quant(v: list[int]) -> str:
    v = sorted(v)
    n = len(v)
    return (
        f"mean {sum(v) / n:.0f} p10 {v[n // 10]} p50 {v[n // 2]} p90 {v[9 * n // 10]} max {v[-1]}"
    )


def part1() -> None:
    print("## O4: statement and state sizes (whitespace tokens of the prompt; max_state tokens)\n")
    for label, path in SIZE_FILES.items():
        if not os.path.exists(path):
            print(f"{label:30s} missing: {path}")
            continue
        pt: list[int] = []
        ms: list[int] = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = r.get("prompt") or ""
                if p:
                    pt.append(len(p.split()))
                if isinstance(r.get("max_state"), int):
                    ms.append(r["max_state"])
        over45 = 100 * sum(x > 45 for x in pt) / max(1, len(pt))
        print(f"{label:30s} n={len(pt):6d} prompt[{quant(pt)}] >45 tok {over45:4.1f}%")
        if ms:
            print(f"{'':30s}          max_state[{quant(ms)}]")
    print()


def part2(reads: list[str]) -> None:
    print("## O5: pass@1 and binder-mismatch share by min_lines bin x statement-size tercile\n")
    for d in reads:
        path = f"{EV}/{d}/rows_plain_B64.jsonl"
        if not os.path.exists(path):
            print(f"{d}: missing")
            continue
        rows = [json.loads(line) for line in open(path)]
        for r in rows:
            r["pt"] = len(r["prompt"].split())
        pts = sorted(r["pt"] for r in rows)
        t1, t2 = pts[len(pts) // 3], pts[2 * len(pts) // 3]
        tab: dict[tuple[int, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in rows:
            lb = r["min_lines"] // 2 * 2
            tb = "S" if r["pt"] <= t1 else ("M" if r["pt"] <= t2 else "L")
            c = tab[(lb, tb)]
            c[0] += sum(r["ok"])
            c[1] += r["k"]
            c[2] += sum(s == "fail:S:binder mismatch" for s in r["status"])
            c[3] += 1
        print(f"{d}: terciles S <= {t1}, M <= {t2}, L > {t2} tokens")
        for lb in sorted({k[0] for k in tab}):
            cells = []
            for tb in "SML":
                ok, n, bm, np_ = tab[(lb, tb)]
                cells.append(
                    f"{tb}: p1 {100 * ok / max(1, n):5.1f} binder {100 * bm / max(1, n):4.1f} (n={np_})"
                )
            print(f"   lines {lb}-{lb + 1}  " + "  ".join(cells))
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--reads", nargs="*", default=READS, help="read directories under runs/eval/v2")
    a = ap.parse_args()
    part1()
    part2(a.reads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
