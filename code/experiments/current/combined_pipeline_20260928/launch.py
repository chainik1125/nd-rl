#!/usr/bin/env python3
"""One arm x seed on its own A40: rent, upload, run, pull (+ HF push), terminate.

  source .envrc.local && code/.venv/bin/python \
      code/experiments/current/combined_pipeline_20260928/launch.py ei|leon <seed> <ckpt-dir> \
      [--name combo-<variant>] [--cfg '{"full_pool": true}']

--name sets the run (default combo-<arm>, so runs are <name>_s<seed>); --cfg (leon only) is
written to the pod as leon_cfg.json, overriding leon_rl.CFG.

<ckpt-dir>/s<seed>/pretrain.pt is 186-claude's pretrained checkpoint for that seed. Reuses the
autoresearch pod.py (rental, watchdog, detached run, pull, HF push) and replaces only its upload:
the pod gets a staged directory instead of harness.py + remote.sh + pretrain.py.

  ei    the frozen harness.py verbatim: plain 4-round expert iteration (replays 186's RL stage)
  leon  harness_leon.py as harness.py (frozen harness, stage_rl = leon_rl.stage_rl)

Both arms get pretrain.py = ckpt_shim.py and run with ND_TOTAL_SECS=3600 (the harness's
benchmark override) instead of 2400, so a slower recipe is measured rather than killed.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "autoresearch")
sys.path.insert(0, AR)
import pod  # noqa: E402

TOTAL_SECS = 3600


def stage(arm, seed, ckpt_dir, name, cfg=None, pretrain_on_pod=False):
    d = os.path.join(pod.LOCAL_ART, "combined_stage", f"{name}_s{seed}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(AR, "remote.sh")) as f:
        remote = f.read()
    with open(os.path.join(d, "remote.sh"), "w") as f:
        f.write(f"export ND_TOTAL_SECS={TOTAL_SECS}\n" + remote)
    if arm == "ei":
        shutil.copy2(os.path.join(AR, "harness.py"), os.path.join(d, "harness.py"))
    else:
        shutil.copy2(os.path.join(AR, "harness.py"), os.path.join(d, "harness_frozen.py"))
        shutil.copy2(os.path.join(HERE, "harness_leon.py"), os.path.join(d, "harness.py"))
        shutil.copy2(os.path.join(HERE, "leon_rl.py"), d)
        for f in ("fast_sample.py", "guided.py", "line_checker.py"):
            shutil.copy2(os.path.join(HERE, f), d)
        if cfg:
            with open(os.path.join(d, "leon_cfg.json"), "w") as f:
                json.dump(json.loads(cfg), f)
    src = os.path.join(ckpt_dir, f"s{seed}")
    if pretrain_on_pod:  # 186's pretrain.py itself: the pod pretrains 300 s; nothing big to send
        shutil.copy2(os.path.join(src, "source", "pretrain.py"), os.path.join(d, "pretrain.py"))
        return d
    shutil.copy2(os.path.join(HERE, "ckpt_shim.py"), os.path.join(d, "pretrain.py"))
    shutil.copy2(os.path.join(src, "source", "pretrain.py"), os.path.join(d, "pretrain186.py"))
    shutil.copy2(os.path.join(src, "pretrain.pt"), os.path.join(d, "pretrain_ckpt.pt"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("ei", "leon"))
    ap.add_argument("seed", type=int)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--name")
    ap.add_argument("--cfg")
    ap.add_argument("--pretrain-on-pod", action="store_true")
    a = ap.parse_args()
    arm, seed, name = a.arm, a.seed, a.name or f"combo-{a.arm}"
    assert not (a.cfg and arm == "ei"), "--cfg configures leon_rl only"
    run_staged(stage(arm, seed, a.ckpt_dir, name, a.cfg, a.pretrain_on_pod), name, seed)


def run_staged(d, name, seed):
    """Rent a pod, upload staged dir `d` as /workspace/autoresearch, run remote.sh as
    <name>_s<seed>, pull the artifacts, terminate. Used by factorial_20260929/launch.py too."""
    # pod.py pushes checkpoints to HF before terminating the pod, so a slow uplink leaves the pod
    # idling; push later from this disk instead (`pod.py push <run>`).
    pod.push_checkpoints = lambda _name: None

    def upload(st, _pretrain_file):
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        subprocess.run(
            f"tar czf - -C {shlex.quote(d)} . "
            f'| {ssh} "mkdir -p {pod.REMOTE} && tar xzf - -C {pod.REMOTE}"',
            shell=True,
            check=True,
        )

    pod.upload = upload
    run, pod_name = f"{name}_s{seed}", f"{name}-s{seed}"
    try:
        for cmd in (["up"], ["run", run, "--seed", str(seed)]):
            sys.argv = ["pod.py", cmd[0], "--pod", pod_name, *cmd[1:]]
            pod.main()
    finally:
        if os.path.exists(os.path.join(pod.LOCAL_ART, f"pod_{pod_name}.json")):
            sys.argv = ["pod.py", "down", "--pod", pod_name]
            pod.main()


if __name__ == "__main__":
    main()
