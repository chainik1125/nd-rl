#!/usr/bin/env python3
"""Stands in for `sample_wrap.py ... -- <Dan's argv>` in the chain tests (no vLLM, no GPU).

STUB_MODE (or STUB_MODE_<job>): ok | hang | crash | badmeta. `hang` writes STUB_ROWS rows, leaves
the gzip open, starts a child `sleep` and sleeps, as a sampler stuck on long draws would. Each call
is logged to $M2_ROOT/stub_calls.log. A row passes when STUB_PASS_<job> (default 2) is above its
chunk index.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
i = argv.index("--")
own = argparse.ArgumentParser()
own.add_argument("--dan-root")
own.add_argument("--sidecar")
o = own.parse_args(argv[:i])
ap = argparse.ArgumentParser()
for k in ("model", "tokenizer", "name", "rev", "problems", "pids", "job", "out", "meta"):
    ap.add_argument(f"--{k}")
for k in ("n", "chunk", "seed"):
    ap.add_argument(f"--{k}", type=int)
a = ap.parse_args(argv[i + 1 :])
mode = os.environ.get(f"STUB_MODE_{a.job}", os.environ.get("STUB_MODE", "ok"))
root = Path(os.environ["M2_ROOT"])
with open(root / "stub_calls.log", "a") as f:
    f.write(f"{a.job} {mode} {a.model} seed={a.seed}\n")
if mode == "crash":
    print("stub: engine failed to load")
    sys.exit(1)
spec = {t.partition(":")[0]: int(t.partition(":")[2]) for t in Path(a.pids[1:]).read_text().split()}
keys = [(p, c) for p, n in spec.items() for c in range(n)]
npass = int(os.environ.get(f"STUB_PASS_{a.job}", "2"))
limit = int(os.environ.get("STUB_ROWS", "50")) if mode == "hang" else len(keys)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
g = gzip.open(a.out, "at")  # noqa: SIM115 (left open on purpose in `hang` mode)
for p, c in keys[:limit]:
    ok = c < npass
    row = {
        "pid": p,
        "model": a.name,
        "rev": a.rev,
        "job": a.job,
        "chunk": c,
        "i": 0,
        "n_tok": 300 + c,
        "finish": "stop",
        "text": "plan</think>" + ("An answer." if ok else "no"),
        "stub_pass": ok,
    }
    g.write(json.dumps(row) + "\n")
g.flush()
if mode == "hang":
    subprocess.Popen(["sleep", "611"])  # a helper process that must not outlive the cell
    time.sleep(600)
g.close()
meta = {
    "job": a.job,
    "name": a.name,
    "rev": a.rev,
    "model_dir": a.model,
    "seed": a.seed,
    "n_requests": len(keys),
    "samples": len(keys),
    "temperature": 0.6 if mode == "badmeta" else 1.0,
    "top_p": 1.0,
    "max_tokens": 32768,
    "chunk": a.chunk,
    "chunk0": 0,
    "stop_after": 0,
    "lora": None,
    "fill_context": None,
    "tp": 1,
    "vllm": "0.11.0",
    "tok_per_s": 1600.0,
    "t_gen_s": 1.0,
    "t_load_s": 1.0,
    "gpus": ["stub"],
    "length_cut": 0,
    "prompt_sha256_by_pid": json.loads(Path("inputs/prompt_sha256_reach.json").read_text()),
}
with open(a.meta, "a") as f:
    f.write(json.dumps(meta) + "\n")
Path(o.sidecar).write_text(
    json.dumps({"status": "done", "temperature": 1.0, "dan_argv": argv[i + 1 :]})
)
