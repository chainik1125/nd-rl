"""Shared pieces: load Dan's fork, the textbook problems, and one Lean-judged evaluation in the environment."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK = Path(os.environ.get("ND_FORK", "/workspace/nd-takehome"))
RESULTS = Path(os.environ.get("ND_RESULTS", HERE / "results"))
CKPTS = Path(os.environ.get("ND_CKPTS", "/workspace/ckpts"))
SETS = ("v0", "v1", "batch3", "lean_v2")  # Charles's textbook inputs; hand14 is out (no model solves it), batch4 unavailable
sys.path.insert(0, str(FORK))
os.environ.setdefault("LEAN_GATE_LOG", "/tmp/lean_gate.jsonl")

import torch  # noqa: E402
from eval_set import judge  # noqa: E402
from model import load_ckpt  # noqa: E402
from state_sample import env_generate, env_stats_json  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def problems(limit=0):
    recs, seen = [], set()
    for s in SETS:
        for line in open(HERE / "data" / f"{s}.jsonl"):
            r = json.loads(line)
            if r["name"] not in seen:
                seen.add(r["name"])
                recs.append({**r, "set": s})
    return recs[:limit] if limit else recs


def ckpt_path(seed, rnd, pretrain=None):
    """`pretrain`: a Stage-1 step of rl-from-ckpt, whose ladders started from that step instead of the end of Stage 1."""
    if pretrain:
        name = f"stage1_best12_s{seed}_b1200_step{pretrain}" if rnd == 0 else f"la_T1_best12_s{seed}_p{pretrain}_r{rnd}"
    else:
        name = f"stage1_best12_s{seed}_b1200" if rnd == 0 else f"la_T1_best12_s{seed}_r{rnd}"
    return CKPTS / f"{name}.pt"


def load(seed, rnd, pretrain=None):
    return load_ckpt(str(ckpt_path(seed, rnd, pretrain)), DEVICE)


def evaluate(model, tok, recs, k, batch, seed=0):
    """Dan's state_eval protocol: T = 1.0, 48 steps, 256 tokens per action. Lean alone decides."""
    stats = {}
    prompts = [r["prompt"] for r in recs for _ in range(k)]
    flat = env_generate(model, tok, prompts, greedy=False, temperature=1.0, max_action=256, max_steps=48,
                        batch=batch, seed=seed, stats=stats)
    rows = judge(recs, [flat[i * k:(i + 1) * k] for i in range(len(recs))], "n_lines")
    env = env_stats_json(stats)
    return {"rows": [{"name": r["name"], "n_ok": r["n_ok"], "n_tried": r["n_tried"]} for r in rows],
            "env_end": env.get("env_end", {}), "actions": sum(int(s) * c for s, c in env.get("env_steps", {}).items())}


def done(path):
    return Path(path).exists()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)
