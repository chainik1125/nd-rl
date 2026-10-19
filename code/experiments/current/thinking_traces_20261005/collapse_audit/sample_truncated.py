"""Collapse audit, probe 2: what a truncated action looks like (CPU, a handful of attempts).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.sample_truncated \
        --ckpt CKPT --budget 256 --items 6 --k 4

Samples a few 13-20 items with the eval's settings (plain arm, T 0.8, max_action 512, line forcing)
on CPU with traces kept, and prints every attempt's outcome and, for a failed one, the draw that
ended it: its think block and the head / tail of its action. Not an eval: a few dozen attempts to
read the text, nothing is written to the eval tree.
"""

from __future__ import annotations

import argparse
import collections
import json

import torch

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import M, valid_1320
from code.thinking.ckpt import load_ckpt
from code.thinking.sample import think_generate


def pick_items(n: int) -> list[dict]:
    """The first n valid items, ordered by the it49 B256 truncation count (most first)."""
    keep = valid_1320()
    with open(M / "diag/t1320_grpoT1v1e_it49/rows_plain_B256.jsonl") as f:
        rows = [json.loads(x) for x in f]
    rows = [r for r in rows if r["name"] in keep]
    rows.sort(key=lambda r: -sum(s == "fail:truncated" for s in r["status"]))
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", type=int, default=80)
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--quiet", action="store_true", help="only the outcome counts")
    a = ap.parse_args()
    torch.set_num_threads(8)
    model, tok, _ = load_ckpt(a.ckpt, "cpu")
    items = pick_items(a.items)
    prompts = [r["prompt"] for r in items for _ in range(a.k)]
    res = think_generate(
        model,
        tok,
        prompts,
        "plain",
        a.budget,
        0,
        True,
        0.8,
        False,
        512,
        96,
        10,
        256,
        a.seed,
        {},
        keep_trace=True,
        force_mode="line",
    )
    cnt = collections.Counter(r["status"] for r in res)
    print("outcomes:", dict(cnt.most_common()))
    if a.quiet:
        return 0
    for j, r in enumerate(res):
        print(
            f"\n=== {items[j // a.k]['name']} att {j % a.k} base {r['base']} status {r['status']} "
            f"steps {r['steps']}"
        )
        if r["status"] == "done" or not r["trace"]:
            continue
        last = r["trace"][-1]
        gen = list(last["gen"])
        txt = [tok.itos[i] for i in gen]
        if tok.end_think in gen and a.budget > 0:
            c = gen.index(tok.end_think)
            th, act = txt[:c], txt[c + 1 :]
        else:
            th, act = [], txt
        print(f"state: {' '.join(tok.itos[i] for i in last['state'])}")
        print(f"think ({len(th)}): {' '.join(th[:60])} ...")
        print(f"action ({len(act)}): {' '.join(act[: a.head])}")
        if len(act) > a.head:
            print(f"  ... tail: {' '.join(act[-a.tail :])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
