#!/usr/bin/env python3
"""One arm x seed on its own A40: rent, upload, run, pull (+ HF push), terminate.

  source .envrc.local && code/.venv/bin/python \
      code/experiments/current/combined_pipeline_20260928/launch.py ei|leon <seed> <ckpt-dir>

<ckpt-dir>/s<seed>/pretrain.pt is 186-claude's pretrained checkpoint for that seed. Reuses the
autoresearch pod.py (rental, watchdog, detached run, pull, HF push) and replaces only its upload:
the pod gets a staged directory instead of harness.py + remote.sh + pretrain.py.

  ei    the frozen harness.py verbatim: plain 4-round expert iteration (replays 186's RL stage)
  leon  harness_leon.py as harness.py (frozen harness, stage_rl = leon_rl.stage_rl)

Both arms get pretrain.py = ckpt_shim.py and run with ND_TOTAL_SECS=3600 (the harness's
benchmark override) instead of 2400, so a slower recipe is measured rather than killed.
"""

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


def stage(arm, seed, ckpt_dir):
    d = os.path.join(pod.LOCAL_ART, "combined_stage", f"{arm}_s{seed}")
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
    shutil.copy2(os.path.join(HERE, "ckpt_shim.py"), os.path.join(d, "pretrain.py"))
    src = os.path.join(ckpt_dir, f"s{seed}")
    shutil.copy2(os.path.join(src, "source", "pretrain.py"), os.path.join(d, "pretrain186.py"))
    shutil.copy2(os.path.join(src, "pretrain.pt"), os.path.join(d, "pretrain_ckpt.pt"))
    return d


def main():
    arm, seed, ckpt_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert arm in ("ei", "leon")
    d = stage(arm, seed, ckpt_dir)

    def upload(st, _pretrain_file):
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        subprocess.run(
            f"tar czf - -C {shlex.quote(d)} . "
            f'| {ssh} "mkdir -p {pod.REMOTE} && tar xzf - -C {pod.REMOTE}"',
            shell=True,
            check=True,
        )

    pod.upload = upload
    name, pod_name = f"combo-{arm}_s{seed}", f"combo-{arm}-s{seed}"
    try:
        for cmd in (["up"], ["run", name, "--seed", str(seed)]):
            sys.argv = ["pod.py", cmd[0], "--pod", pod_name, *cmd[1:]]
            pod.main()
    finally:
        if os.path.exists(os.path.join(pod.LOCAL_ART, f"pod_{pod_name}.json")):
            sys.argv = ["pod.py", "down", "--pod", pod_name]
            pod.main()


if __name__ == "__main__":
    main()
