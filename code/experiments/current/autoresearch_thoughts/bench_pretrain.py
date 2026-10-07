#!/usr/bin/env python3
"""Throughput probe for the working-tree pretrain.py on one A40. Not an experiment: nothing is
registered, committed or scored, and the numbers never decide a keep.

  $PY bench_pretrain.py [--secs 60] [--samples 4096]

Rents one pod, runs pretrain.pretrain() for --secs of wall-clock on the real Stage-1 data
(schedule, compile and model construction included, as in a real run), then samples --samples
proofs at the harness's RL settings (32 per target, Lean gate included) and prints one JSON line:
steps, steps/s, n_params, peak GPU memory, final train/val loss, sampling samples/s. Budget
~4 min and ~$0.03 per call including pod boot. Inside a loop session the pod is recorded in a
bench-*.json manifest so cancellation terminates it.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pod

HERE = Path(__file__).resolve().parent

REMOTE_CODE = """
import json, random, sys, time, torch
import harness
from harness import FORK, MAX_NEW, SAMPLE_BATCH, TEMP, read, sample_and_judge
import pretrain
train = read(f"{FORK}/data/train.jsonl")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model, tok, stats = pretrain.pretrain(train, SECS, "cuda", 0, lambda m: print(m, file=sys.stderr))
torch.cuda.synchronize()
secs = time.time() - t0
model.eval()
pool = read(f"{FORK}/data/ladder/rl_targets.jsonl")
targets = random.Random(0).sample(pool, max(1, SAMPLES // 32))
rows, samp = sample_and_judge(model, tok, targets, 32, TEMP, 1, SAMPLE_BATCH, MAX_NEW)
curve = (stats or {}).get("curve") or [{}]
steps = (stats or {}).get("steps")
print("BENCH " + json.dumps({
    "pretrain_secs": round(secs, 1), "steps": steps,
    "steps_per_sec": round(steps / secs, 2) if steps else None,
    "n_params": sum(p.numel() for p in model.parameters()),
    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "train_loss": curve[-1].get("train_loss"), "val_loss": curve[-1].get("val_loss"),
    "samples": samp["samples"], "samples_per_sec": round(samp["samples_per_sec"]),
    "sample_acc": round(sum(r["n_ok"] for r in rows) / samp["samples"], 4),
    "torch": torch.__version__,
    "curve_steps": [[c.get("secs"), c.get("step")] for c in (stats or {}).get("curve") or []],
    "dynamo": {k: dict(v) for k, v in torch._dynamo.utils.counters.items()
               if k in ("stats", "recompiles")},
}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secs", type=int, default=60)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--pretrain", default="pretrain.py", help="file to run as pretrain.py")
    args = parser.parse_args()
    agent = os.environ.get("ND_AGENT", "manual")
    session = os.environ.get("ND_AUTORESEARCH_SESSION")
    name = f"ndrl-bench-{agent}-{os.getpid()}"
    if session:
        name += f"-{Path(session).name}"
        manifest = Path(session) / f"bench-{os.getpid()}.json"
        manifest.write_text(json.dumps({"pods": [name]}))  # before any paid API call
    run = [sys.executable, str(HERE / "pod.py")]
    try:
        up = [*run, "up", "--pod", name, "--pretrain", args.pretrain]
        if subprocess.run(up, stdout=sys.stderr).returncode:
            raise SystemExit("pod up failed")
        pod.STATE = os.path.join(pod.LOCAL_ART, f"pod_{name}.json")
        st = pod.load_state()
        code = f"SECS, SAMPLES = {args.secs}, {args.samples}\n" + REMOTE_CODE
        env = (
            "ND_FORK=/workspace/nd-takehome LEAN_GATE_WORKERS=$(nproc) "
            "LEAN_GATE_LOG=/tmp/bench_gate.jsonl"
        )
        out = subprocess.run(
            [
                "ssh",
                *pod.ssh_args(st),
                "bash",
                "-lc",
                shlex.quote(
                    f"cd {pod.REMOTE} && PY=$(for p in python python3; do $p -c 'import torch' "
                    f"2>/dev/null && echo $p && break; done) && {env} $PY -c {shlex.quote(code)}"
                ),
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,  # newer images' login profile would block reading stdin
        )
        lines = [line for line in out.stdout.splitlines() if line.startswith("BENCH ")]
        if not lines:
            sys.stderr.write(out.stderr[-3000:])
            raise SystemExit("benchmark failed on the pod (stderr tail above)")
        print(lines[-1][len("BENCH ") :])
    finally:
        t0 = time.time()
        subprocess.run([*run, "down", "--pod", name], stdout=sys.stderr, check=False)
        print(f"pod down in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
