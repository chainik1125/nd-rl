"""Token sequences for crosscoder training: each model's own rollouts on the RL training pools,
and a sample of the Stage-1 pretraining pairs.

    python -m code.experiments.current.circuit_diffing_20261101.rollouts --gpu 0 \
        --models base:0,L3:0 [--n_problems 1200 --k 4]
    python -m code.experiments.current.circuit_diffing_20261101.rollouts --pretrain

A rollout is `sample.think_generate` (plain arm, per-attempt budgets drawn from 0 / 64 / 256 as in
GRPO, temperature 1.0, keep_trace). Every environment step is one sequence: the step's decoder
input (`state <eB> <think>`) followed by every sampled token of the draw (think, `</think>`,
action, `<eos>`); the failing draw of a failed attempt is kept too. The problems are a fixed
random subset of the L2 and L3 curriculum pools (the RL training distribution; Charles149,
13-20, 21-30 and textbook72 are not used), the same subset for every model.

Output `/scratch/cowifornia/nd-rl/circuits/seqs/<label>_s<seed>.npz`: `tokens` (uint8, all
sequences concatenated), `offsets` (n + 1), `gen_start` (index in each sequence where sampled
tokens begin), and per sequence `prob`, `attempt`, `step`, `budget`, `rejected`, `done` (attempt
status 'done'). The pretraining sample `pretrain.npz` has the same fields (prob = record index,
gen_start = state length, the rest 0 / -1).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time

import numpy as np

from .common import ART, RUNS, assert_gpu_free, pt_now, registry

SEQ = ART / "seqs"
POOLS = [RUNS / "pools/curriculum/L2/pool.jsonl", RUNS / "pools/curriculum/L3/pool.jsonl"]
PRETRAIN = RUNS / "data/S1g3.pkl"
MAX_SEQ = 1024


def problems(n: int, seed: int = 20261101) -> list[dict]:
    out = []
    rng = random.Random(seed)
    for p in POOLS:
        recs = [json.loads(line) for line in p.read_text().splitlines()]
        out += rng.sample(recs, n // len(POOLS))
    return out


def pack(seqs: list[tuple[list[int], int, dict]]) -> dict[str, np.ndarray]:
    lens = [min(len(s), MAX_SEQ) for s, _, _ in seqs]
    off = np.zeros(len(seqs) + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    tokens = np.zeros(off[-1], dtype=np.uint8)
    for i, (s, _, _) in enumerate(seqs):
        tokens[off[i] : off[i + 1]] = s[: lens[i]]
    out = {"tokens": tokens, "offsets": off, "gen_start": np.array([g for _, g, _ in seqs])}
    for key in ("prob", "attempt", "step", "budget", "rejected", "done"):
        out[key] = np.array([m[key] for _, _, m in seqs], dtype=np.int32)
    return out


def rollout(label: str, seed: int, n: int, k: int, device: str) -> None:
    import torch

    from code.thinking.ckpt import run_env
    from code.thinking.sample import think_generate

    from .common import load_model

    c = registry()[(label, seed)]
    out = SEQ / f"{label}_s{seed}.npz"
    run_env(ART / "runenv")
    model, tok, _ = load_model(c.path, device)
    recs = problems(n)
    prompts = [r["prompt"] for r in recs for _ in range(k)]
    t0 = time.time()
    res = think_generate(  # bf16 autocast inside, as in every read
        model,
        tok,
        prompts,
        arm="plain",
        budgets=[0, 64, 256],
        temperature=1.0,
        max_action=512,
        max_steps=96,
        batch=1024,
        seed=1000 + seed,
        keep_trace=True,
    )
    seqs = []
    for i, r in enumerate(res):
        for j, e in enumerate(r["trace"] or []):
            p, g = list(e["prompt"]), list(e["gen"])
            seqs.append(
                (
                    p + g,
                    len(p),
                    dict(
                        prob=i // k,
                        attempt=i,
                        step=j,
                        budget=int(r["budget"]),
                        rejected=int(bool(e.get("rejected", False))),
                        done=int(r["status"] == "done"),
                    ),
                )
            )
    d = pack(seqs)
    SEQ.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    done = np.mean([r["status"] == "done" for r in res])
    print(
        f"{pt_now()} {label} s{seed}: {len(res)} attempts, done {100 * done:.1f}%, "
        f"{len(seqs)} sequences, {d['offsets'][-1] / 1e6:.2f}M tokens, "
        f"{time.time() - t0:.0f}s -> {out}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()


def pretrain(n_pairs: int, seed: int = 20261101) -> None:
    with open(PRETRAIN, "rb") as f:
        data = pickle.load(f)
    rng = random.Random(seed)
    recs = data["records"]
    seqs = []
    while len(seqs) < n_pairs:
        ri = rng.randrange(len(recs))
        p, q, _ = recs[ri][rng.randrange(len(recs[ri]))]
        seqs.append(
            (
                list(p) + list(q),
                len(p),
                dict(prob=ri, attempt=-1, step=-1, budget=-1, rejected=0, done=0),
            )
        )
    d = pack(seqs)
    SEQ.mkdir(parents=True, exist_ok=True)
    np.savez(SEQ / "pretrain.npz", **d)
    print(f"pretrain: {len(seqs)} pairs, {d['offsets'][-1] / 1e6:.2f}M tokens", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="label:seed,label:seed,...")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--n_problems", type=int, default=1200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--n_pairs", type=int, default=60000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.pretrain:
        pretrain(a.n_pairs)
    if a.models:
        assert a.gpu is not None
        assert_gpu_free(a.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        for spec in a.models.split(","):
            label, seed = spec.split(":")
            if (SEQ / f"{label}_s{seed}.npz").exists() and not a.force:
                print(f"skip {label} s{seed}: exists", flush=True)
                continue
            rollout(label, int(seed), a.n_problems, a.k, "cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
