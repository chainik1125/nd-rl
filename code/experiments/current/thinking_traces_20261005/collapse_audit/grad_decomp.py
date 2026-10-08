"""Collapse audit, probe 8: which part of the GRPO gradient raises `:` in long contexts?
(CPU, read-only: gradients are computed and projected, no optimizer step, nothing saved.)

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.grad_decomp \
        --ckpt s0_it49 --groups 32

A proxy GRPO batch: `--groups` (target, budget) pairs from the hard12 RL pool (targets at random,
budgets uniform over 0/16/64/256), 16 attempts each, sampled as `grpo.sample_groups` does (plain
arm, T 1.0, max_action 256, max_steps 48, line forcing). Reward is the environment's verdict
(`done`), not Lean (no Lean on this busy machine); `--overlong fail`. Items and advantages come from
`grpo.group_items`, and the loss gradient from `grpo.policy_loss` without old log-probs: the
on-policy first minibatch, where r = 1 and the clipped surrogate's gradient equals REINFORCE's.

The gradient is split by token into four parts: A > 0 / A < 0 x think block / action (each part
scaled to the whole batch's token mean, so the parts sum to the total). Each part is projected on
g_colon, the gradient of the colon score c = mean log P(':') at non-':' action targets of the
long-state 13-20 reference steps (`update_probe`). A descent step -lr x part changes c by about
-lr x (part . g_colon) ("raw"). Adam's step is preconditioned: with the per-coordinate scale taken
from the total gradient, -lr x part / |total| changes c by -lr x sum(part x g_colon / |total|)
("sign-like"). Positive numbers mean the part raises `:`.
"""

from __future__ import annotations

import argparse
import collections
import json
import random

import torch

from code.experiments.current.thinking_traces_20261005.collapse_audit.teacher_force import (
    CKPTS,
    build,
    cond_ids,
)
from code.experiments.current.thinking_traces_20261005.collapse_audit.update_probe import (
    colon_score,
)
from code.experiments.current.thinking_traces_20261005.matched_eval.figs import R
from code.thinking import grpo
from code.thinking.ckpt import load_ckpt
from code.thinking.sample import phase_bans, sample_format, think_generate

BUDGETS = (0, 16, 64, 256)


def grads(model) -> dict[str, torch.Tensor]:
    return {
        k: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for k, p in model.named_parameters()
    }


def masked_items(items, think_ids, part: str, sign: int):
    """Keep the items of advantage sign `sign`, with the loss mask restricted to their think
    tokens (`part` = 'think') or action tokens ('action')."""
    out = []
    for p, g, mk, a in items:
        if (a > 0) != (sign > 0) or a == 0:
            continue
        n_th = grpo.think_len(p, g, think_ids)
        m = [bool(x) and ((i < n_th) == (part == "think")) for i, x in enumerate(mk)]
        if any(m):
            out.append((p, g, m, a))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="s0_it49")
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(8)
    model, tok, _ = load_ckpt(CKPTS[a.ckpt], "cpu")
    think_ids = (tok.think, tok.end_think)
    bans = phase_bans(tok, model.cfg["vocab"], True, sample_format(model) == "root")
    with open(R / "pools/rl_hard12.jsonl") as f:
        pool = [json.loads(x) for x in f]
    rng = random.Random(a.seed)
    pairs = [(rng.randrange(len(pool)), rng.choice(BUDGETS)) for _ in range(a.groups)]
    groups: list[list[dict]] = [[] for _ in pairs]
    by = collections.defaultdict(list)
    for i, (_, b) in enumerate(pairs):
        by[b].append(i)
    for b, idx in sorted(by.items()):
        prompts = [pool[pairs[i][0]]["prompt"] for i in idx for _ in range(a.k)]
        res = think_generate(
            model,
            tok,
            prompts,
            "plain",
            [b],
            0,
            True,
            1.0,
            False,
            256,
            48,
            10,
            1024,
            a.seed * 10_007 + b,
            {},
            keep_trace=True,
            force_mode="line",
        )
        for r in res:
            r["ok"] = r["status"] == "done"  # env verdict as the reward (no Lean)
        for n, i in enumerate(idx):
            groups[i] = res[n * a.k : (n + 1) * a.k]
    items = []
    stat = collections.Counter()
    for att in groups:
        its, _ = grpo.group_items(att, tok, 0.0, "fail")
        stat["attempts"] += len(att)
        stat["ok"] += sum(x["ok"] for x in att)
        stat["truncated"] += sum(x["status"] == "fail:truncated" for x in att)
        if its is None:
            stat["skipped"] += 1
            continue
        items += its
    n_tok = sum(sum(m) for _, _, m, _ in items)
    print(f"{a.ckpt}: {dict(stat)}; {len(items)} items, {n_tok} scored tokens")

    # g_colon at this checkpoint
    colon = tok.stoi[":"]
    steps = [s for s in build("t1320", 200) if sum(1 for v in s["state"] if v == colon) >= 10]
    seqs = [cond_ids(tok, s, "B0") for s in steps]
    model.zero_grad(set_to_none=True)
    c, pc = colon_score(model, tok, seqs, grad=True)
    c.backward()
    gc = grads(model)
    print(
        f"colon score c = {float(c.detach()):.3f} (P(':') {100 * pc:.3f}%) "
        f"on {len(seqs)} long-state steps"
    )

    parts = {}
    for sign in (+1, -1):
        for part in ("think", "action"):
            sub = masked_items(items, think_ids, part, sign)
            n_sub = sum(sum(m) for _, _, m, _ in sub)
            model.zero_grad(set_to_none=True)
            if n_sub:
                grpo.policy_loss(
                    model, sub, 1.0, "cpu", 50_000, None, think_ids=think_ids, bans=bans
                )
            g = grads(model)
            scale = n_sub / n_tok  # policy_loss is a mean over the part's tokens
            parts[f"A{'>' if sign > 0 else '<'}0 {part}"] = (
                {k: v * scale for k, v in g.items()},
                n_sub,
            )
    tot = {k: sum(p[0][k] for p in parts.values()) for k in gc}
    eps = 1e-12
    print(f"{'part':14s} {'tokens':>8s} {'raw':>10s} {'sign-like':>10s}   (descent: + raises ':')")
    for name, (g, n_sub) in list(parts.items()) + [("total", (tot, n_tok))]:
        raw = -sum(float((g[k] * gc[k]).sum()) for k in gc)
        sl = -sum(float((g[k] / (tot[k].abs() + eps) * gc[k]).sum()) for k in gc)
        print(f"{name:14s} {n_sub:8d} {raw:+10.4f} {sl:+10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
