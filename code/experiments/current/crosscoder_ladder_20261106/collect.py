#!/usr/bin/env python3
"""Token sequences for the crosscoders, as Leon's rollouts.py: a checkpoint's own rollouts on
RL-pool targets, and
pretraining records. Lean is never run; the sequences are what the models write, accepted or not.

  python collect.py roll --run v4_untied-ei_s0 --stem rl_round_005 [--n_targets 2000 --k 4]
  python collect.py pretrain [--n 20000]

Writes CL_ART/seqs/<run>__<stem>.npz and CL_ART/seqs/pretrain.npz: `tokens` (all sequences
concatenated, int32),
`offsets` (n + 1), `gen_start` (per sequence, the index where sampled tokens begin). The targets are
one fixed
random draw from data/rl.jsonl (seed 0), the same for every checkpoint, so a_roll and every b_roll
share problems.
"""

import argparse
import random
import time

import common as C
import numpy as np

SEQ = C.ART / "seqs"


def save(path, seqs):
    tokens = np.concatenate([np.asarray(s, dtype=np.int32) for s, _ in seqs])
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s, _ in seqs])
    np.savez(
        path,
        tokens=tokens,
        offsets=offsets,
        gen_start=np.asarray([g for _, g in seqs], dtype=np.int64),
    )
    print(f"{path.name}: {len(seqs)} sequences, {offsets[-1] / 1e6:.2f}M tokens", flush=True)


def targets(n, seed=0):
    recs = C.read(C.DATA / "rl.jsonl")
    return random.Random(seed).sample(recs, min(n, len(recs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["roll", "pretrain"])
    ap.add_argument("--run")
    ap.add_argument("--stem")
    ap.add_argument("--n_targets", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=20000, help="pretraining records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=2304)
    a = ap.parse_args()
    SEQ.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.cmd == "pretrain":
        out = SEQ / "pretrain.npz"
        if out.exists():
            return print(f"exists: {out}")
        _, tok = C.build_model()
        recs = C.read(C.DATA / "pretrain.jsonl.gz")
        recs = random.Random(a.seed).sample(recs, min(a.n, len(recs)))
        seqs = []
        for r in recs:
            p = tok.encode_prompt(r["prompt"])
            seqs.append((p + tok.encode_proof(r["proof"]), len(p)))
        save(out, seqs)
    else:
        out = SEQ / f"{a.run}__{a.stem}.npz"
        if out.exists():
            return print(f"exists: {out}")
        model, tok = C.load_model(C.ckpt_path(a.run, a.stem))
        seqs = C.sample_seqs(model, tok, targets(a.n_targets), a.k, a.seed, a.max_new)
        save(out, seqs)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
