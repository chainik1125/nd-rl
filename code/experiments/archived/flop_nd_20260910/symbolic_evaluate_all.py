"""Sequential frozen endpoint evaluations, four CPU threads maximum."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import common

ROOT = common.ROOT
jobs = []
for seed in (4101, 4102):
    for mode in ("full", "reset"):
        tag = f"{mode}_s{seed}"
        jobs.append((tag, ["--policy", "model", "--checkpoint",
                     str(ROOT / f"results/symbolic_{tag}/symbolic_checkpoint.pt")]))
    jobs.append((f"initial_s{seed}", ["--policy", "model", "--checkpoint",
                 str(ROOT / f"results/symbolic_full_s{seed}/symbolic_initial.pt")]))
jobs += [("random_minimal", ["--policy", "random", "--attempts", "16"]),
         ("random_loose", ["--policy", "random", "--attempts", "16", "--budget", "loose"]),
         ("heuristic", ["--policy", "heuristic"])]
ledger = ROOT / "results/symbolic_evaluation_ledger.jsonl"
for tag, extra in jobs:
    out = ROOT / f"results/symbolic_eval_{tag}"
    if (out / "symbolic_eval_summary.json").exists():
        print("Already complete:", tag, flush=True)
        continue
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "code/symbolic_nd.py"), "evaluate", "--population",
               str(ROOT / "data/symbolic_final.jsonl"), "--out", str(out), "--threads", "4",
               "--seed", "424242"] + extra
    started = time.time()
    with (out / "symbolic_eval.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        with ledger.open("a") as f:
            f.write(json.dumps({"event": "start", "tag": tag, "pid": proc.pid, "driver_pid": os.getpid(),
                                "command": command, "start_unix": started, "timeout_seconds": 600,
                                "code_sha256": common.sha(ROOT / "code/symbolic_nd.py"),
                                "population_sha256": common.sha(ROOT / "data/symbolic_final.jsonl")}) + "\n")
        code = proc.wait(timeout=600)
    elapsed = time.time() - started
    with ledger.open("a") as f:
        f.write(json.dumps({"event": "finish", "tag": tag, "exit_code": code, "elapsed_seconds": elapsed}) + "\n")
    print(tag, "exit", code, "seconds", round(elapsed, 2), flush=True)
    if code:
        raise SystemExit(code)
