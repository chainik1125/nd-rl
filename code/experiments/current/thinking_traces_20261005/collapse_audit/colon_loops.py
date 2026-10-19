"""Collapse audit, probe 5: how often free-running samples fall into `:` loops (CPU, small).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.colon_loops \
        --ckpt CKPT --set h12pool --budget 256 --items 32 --k 4 [--train]

Samples `--items` x `--k` attempts with the eval's settings (T 0.8, max_action 512, max_steps 96)
or, with `--train`, GRPO's (T 1.0, max_action 256, max_steps 48), from the hard12 RL pool
(`h12pool`, items drawn at random), the 13-20 set (`t1320`, valid items) or the hard12 test set. Per
attempt: its status and whether any decoded segment (accepted or failing) holds a run of >= 8
consecutive `:` tokens. No Lean: `done` is the environment's verdict. Prints one summary line."""

from __future__ import annotations

import argparse
import collections
import json
import random

import torch

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import R, valid_1320
from code.thinking.ckpt import load_ckpt
from code.thinking.sample import think_generate

SETS = {
    "h12pool": R / "pools/rl_hard12.jsonl",
    "t1320": R / "pools/test_13_20/test_13_20.jsonl",
    "h12test": R / "restart2_hard12/hard12_test_1k.jsonl",
}


def max_run(ids: bytes, tid: int) -> int:
    best = cur = 0
    for x in ids:
        cur = cur + 1 if x == tid else 0
        best = max(best, cur)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default="h12pool", choices=list(SETS))
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true", help="GRPO's sampling settings")
    a = ap.parse_args()
    torch.set_num_threads(8)
    model, tok, _ = load_ckpt(a.ckpt, "cpu")
    with open(SETS[a.set]) as f:
        rows = [json.loads(x) for x in f]
    if a.set == "t1320":
        keep = valid_1320()
        rows = [r for r in rows if r["name"] in keep]
    random.Random(777).shuffle(rows)
    rows = rows[: a.items]
    prompts = [r["prompt"] for r in rows for _ in range(a.k)]
    T, ma, ms = (1.0, 256, 48) if a.train else (0.8, 512, 96)
    res = think_generate(
        model,
        tok,
        prompts,
        "plain",
        a.budget,
        0,
        True,
        T,
        False,
        ma,
        ms,
        10,
        512,
        a.seed,
        {},
        keep_trace=True,
        force_mode="line",
    )
    colon = tok.stoi[":"]
    close = tok.end_think
    st = collections.Counter()
    loops = done_loop = 0
    # ':' share of think-body tokens, split by the attempt's verdict (env 'done' vs failed)
    dens: dict[bool, list[float]] = {True: [], False: []}
    for r in res:
        st[r["status"]] += 1
        runs = [max_run(s["gen"], colon) for s in r["trace"] or []]
        if runs and max(runs) >= 8:
            loops += 1
            done_loop += r["status"] == "done"
        th = (
            [
                x
                for s in r["trace"] or []
                for x in s["gen"][
                    : (list(s["gen"]).index(close) if close in s["gen"] else len(s["gen"]))
                ]
            ]
            if a.budget
            else []
        )
        if th:
            dens[r["status"] == "done"].append(sum(x == colon for x in th) / len(th))
    n = len(res)
    top = ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in st.most_common(5))
    dd = " ".join(
        f"{'done' if k else 'fail'} {100 * sum(v) / len(v):.1f}% (n={len(v)})"
        for k, v in dens.items()
        if v
    )
    print(
        f"{a.ckpt.split('/')[-1]:28s} {a.set:8s} B{a.budget:<3d} {'train' if a.train else 'eval '} "
        f"n={n}: colon-run>=8 in {100 * loops / n:5.1f}% of attempts "
        f"(of which env-done {done_loop}); ':' share of think tokens: {dd}; {top}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
