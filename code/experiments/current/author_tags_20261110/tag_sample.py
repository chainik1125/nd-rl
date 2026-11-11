"""Sample proofs from a tagged checkpoint under A, B and no tag, through Dan's sampler and the Lean gate.

    python tag_sample.py --ckpt CKPT [--tags A,B,none] [--n 8] [--k 4] [--max_lines 4] [--device cpu] [--out OUT.json]

Theorems: the first `n` records of the TRAINING corpus with at most `max_lines` lines (never an evaluation set). Per
tag: Dan's `state_sample.env_generate` (T 0.8, 256 tokens per action, 48 steps, his name-base draw) on n x k attempts,
then `lean_gate.gate` (Lean alone decides). The spy (`tag_core.spy_dan`) checks that every decoder input carried the
tag (or none) right before `<act>`, and counts them. Prints, and writes to OUT: per tag the attempts, Lean-accepted,
how attempts ended, and the states checked.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402


def theorems(n: int, max_lines: int) -> list[dict]:
    out = []
    with gzip.open(T.CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["n_lines"] <= max_lines:
                out.append({"key": r["key"], "prompt": r["prompt"], "tag": r[T.TAG_FIELD]})
                if len(out) >= n:
                    break
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_lines", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("ND_OFFLINE", "1")
    os.environ.setdefault("ND_REGISTRY", "0")
    os.environ.setdefault("LEAN_GATE_WORKERS", "8")
    if a.out:
        os.environ.setdefault("LEAN_GATE_LOG", str(Path(a.out).with_suffix(".lean_gate.jsonl")))
    T.install_dan()
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]
    from state_sample import env_generate, env_stats_json  # type: ignore[import-not-found]

    if a.device == "cpu":
        torch.set_num_threads(8)
    model, tok, _ = load_ckpt(a.ckpt, a.device)
    thms = theorems(a.n, a.max_lines)
    prompts = [t["prompt"] for t in thms for _ in range(a.k)]
    res = {"ckpt": a.ckpt, "mode": tok.mode, "n": len(thms), "k": a.k, "tags": {}}
    for tag in a.tags.split(","):
        T.set_process_tag(tag)
        T.spy_dan(tag)
        s0 = T.SPY["states"]
        st: dict = {}
        t0 = time.time()
        nds = env_generate(model, tok, prompts, greedy=False, temperature=0.8, max_action=256, max_steps=48,
                           batch=a.batch, seed=a.seed, stats=st, gate=True)  # fmt: skip
        ok = [not x.startswith(("LEANREJ", "LEANPARSE")) for x in nds]
        js = env_stats_json(st)
        res["tags"][tag] = {
            "attempts": len(nds), "lean_accepted": sum(ok),
            "theorems_solved": len({prompts[i] for i, o in enumerate(ok) if o}),
            "ends": js.get("env_end"), "states_checked": T.SPY["states"] - s0, "secs": round(time.time() - t0, 1),
        }  # fmt: skip
        print(f"tag {tag}: {res['tags'][tag]}", flush=True)
    res["spy_by_tag"] = dict(T.SPY["by_tag"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
