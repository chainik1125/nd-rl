#!/usr/bin/env python3
# 120 columns, as in long_rl_20261014's files
# ruff: noqa: E501
"""One RunPod A40 per (arm, recipe, seed): rent, run knockout.py, pull the output, push checkpoints to HF, terminate.

  python launch.py go ARM RECIPE SEED     # ARM mp|cp|exfalso|dn|full, RECIPE ei|ei-focus|ei-hot|grpo
  python launch.py down ARM RECIPE SEED   # terminate a pod `go` left behind

Output lands in artifacts/rule_knockout_20261118/<arm>-<recipe>_s<seed>/; checkpoints go to the HF repo at
runs/rule-knockout-20261118_<arm>-<recipe>_s<seed>/. The harness, pretrain.py, tokenizer, Lean gate and data are
../long_rl_20261014's; pod plumbing is ../autoresearch/pod.py. Needs RUNPOD_API_KEY.
"""

import importlib.util
import json
import os
import shlex
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
LONG_RL = os.path.abspath(os.path.join(HERE, "..", "long_rl_20261014"))
# long_rl's launcher has the tested rent-wait-watchdog sequence (`up`); loaded by path because it is also launch.py
spec = importlib.util.spec_from_file_location("long_rl_launch", os.path.join(LONG_RL, "launch.py"))
assert spec and spec.loader
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)
pod = lr.pod

ARMS = {
    "mp": "IMPE",
    "cp": "IMPI",
    "exfalso": "BOTE",
    "dn": "DN",
    "full": "none",
}  # arm -> rule removed
# recipe -> (ND_RL, ND_POOL, sampling temperature, RL hours of GPU clock)
RECIPES = {
    "ei": ("ei", "full", 0.8, 3.0),
    "ei-focus": ("ei", "focus", 0.8, 2.0),
    "ei-hot": ("ei", "full", 1.0, 2.0),
    "grpo": ("grpo", "full", 0.8, 2.0),
}
EVAL_K = 16  # dev samples per theorem in the final eval (long_rl used 64); the RL census is the readout here
ART = os.path.join(lr.REPO, "artifacts", "rule_knockout_20261118")
REMOTE = "/workspace/long_rl"
LR_FILES = (
    "remote.sh harness.py pretrain.py overlay/lean_tok.py overlay/lean_gate.py data/pretrain.jsonl.gz "
    "data/rl.jsonl data/dev.jsonl data/holdout.jsonl"
)


def budget(recipe):
    """(RL seconds, the harness's total limit, the watchdog's limit). RL wall-clock ran 13-16% over the charged RL
    seconds in long_rl (Lean gate); pretraining is 600 s, setup and the eval about 900 s."""
    rl_secs = int(RECIPES[recipe][3] * 3600)
    total = 600 + int(1.25 * rl_secs) + 1500
    return rl_secs, total, total + 1500


def run(st, arm, recipe, name, seed):
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    lr.sh(f'{ssh} -n "mkdir -p {REMOTE}"')
    lr.sh(
        f"tar czf - -C {shlex.quote(LONG_RL)} {LR_FILES} -C {shlex.quote(HERE)} knockout.py remote_ko.sh "
        f'| {ssh} "tar xzf - -C {REMOTE}"'
    )
    algo, pool, temp, _ = RECIPES[recipe]
    rl_secs, total, _ = budget(recipe)
    job = (
        f"ND_ARCH=untied ND_RL={algo} ND_KNOCKOUT={ARMS[arm]} ND_POOL={pool} ND_TEMP={temp} ND_EVAL_K={EVAL_K} "
        f"ND_RL_SECS={rl_secs} ND_TOTAL_SECS={total} bash {REMOTE}/remote_ko.sh {REMOTE}/{name} --seed {seed}"
    )
    # 3 h before giving up on a dropped connection: the pod's watchdog bounds the bill, and this Mac may sleep
    rc = pod.run_detached(st, job, f"{REMOTE}/job", give_up_secs=3 * 3600)
    for _ in range(10):  # the pull can hit the same network drops as the stream
        pull = f'{ssh} -n "cd {REMOTE} && tar czf - {name} {name}.log"'
        if lr.sh(f"{pull} | tar xzf - -C {shlex.quote(ART)}", check=False) == 0:
            break
        time.sleep(30)
    return rc


def main():
    cmd, arm, recipe, seed = sys.argv[1:5]
    assert arm in ARMS and recipe in RECIPES and not (arm == "full" and recipe == "ei-focus")
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    os.makedirs(ART, exist_ok=True)
    name = f"{arm}-{recipe}_s{seed}"
    state = Path(ART, f"pod_{name}.json")
    if cmd == "go":
        if state.exists():
            raise SystemExit(f"{state} exists; `launch.py down {arm} {recipe} {seed}` first")
        lr.MAX_SECS = budget(recipe)[2]  # the watchdog `up` starts on the pod
        st = lr.up(key, f"ko-{name}", state)
        try:
            rc = run(st, arm, recipe, name, seed)
        finally:
            pod.terminate(key, st["id"])
            state.unlink()
            mins = (time.time() - st["t0"]) / 60
            print(
                f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})",
                flush=True,
            )
        print(f"output in {ART}/{name}; remote rc {rc}", flush=True)
        hf_name = f"rule-knockout-20261118_{name}"
        os.replace(os.path.join(ART, name), os.path.join(ART, hf_name))
        pod.LOCAL_ART = ART
        pod.push_checkpoints(hf_name)
    elif cmd == "down":
        pod.terminate(key, json.loads(state.read_text())["id"])
        state.unlink()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
