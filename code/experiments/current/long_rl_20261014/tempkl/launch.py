#!/usr/bin/env python3
"""Temperature-matched KL (tempkl.py) for the long_rl runs in RUNS, on one A40.

  python tempkl/launch.py            # rerun: tags already in tempkl/results/ are skipped on the pod

Checkpoints and tags come from passk/backfill.py's HF listing (<version>_<arm>_s<seed>_<stage>); each run's own
pretrain.pt is the reference. Results land in tempkl/results/<tag>.json; the pod's log in tempkl/pod.log. The pod is
terminated in a finally, and its watchdog self-destructs it after WATCHDOG_SECS regardless.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "passk")]
import backfill  # noqa: E402  (checkpoints(): the HF listing)
import launch  # noqa: E402  (up(), FILES, and ../autoresearch/pod.py as launch.pod)

REMOTE = "/workspace/long_rl"
RUNS = ("v4_untied-ei_s0", "v3_untied-grpo_s0")  # in priority order; the EI curve first
WATCHDOG_SECS = int(2.5 * 3600)
STAGE_ORDER = lambda tag: (0, 0) if tag.endswith("_pretrain") else (2, 0) if tag.endswith("_final") else (1, int(tag[-3:]))  # noqa: E731


def jobs():
    done = {p.stem for p in Path(HERE, "results").glob("*.json")}
    out = []
    for run in RUNS:
        cks = {tag: (url, arch) for url, tag, arch in backfill.checkpoints() if tag.startswith(run + "_")}
        pre_tag = f"{run}_pretrain"
        assert pre_tag in cks, f"{pre_tag} not on HF"
        for tag in sorted(cks, key=STAGE_ORDER):
            if tag not in done:
                out.append((cks[tag][0], tag, cks[tag][1], cks[pre_tag][0], pre_tag))
    return out


def main():
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    js = jobs()
    if not js:
        return print("nothing to score")
    jobs_file = Path(HERE, "jobs.txt")
    jobs_file.write_text("".join(" ".join(j) + "\n" for j in js))
    print(f"{len(js)} checkpoints: {' '.join(j[1] for j in js)}", flush=True)
    launch.MAX_SECS = WATCHDOG_SECS
    state = Path(HERE, "pod.json")
    st = launch.up(key, "longrl-tempkl", state)
    try:
        ssh = "ssh " + " ".join(launch.pod.ssh_args(st))
        sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731
        sh(f"tar czf - -C {os.path.join(HERE, '..')} {launch.FILES} tempkl/tempkl.py tempkl/analyze.py tempkl/remote.sh "
           f"| {ssh} 'mkdir -p {REMOTE} && tar xzf - -C {REMOTE}'")
        sh(f"{ssh} 'cat > {REMOTE}/tempkl/jobs.txt' < {jobs_file}")
        rc = launch.pod.run_detached(st, f"bash {REMOTE}/tempkl/remote.sh {REMOTE}/tempkl/jobs.txt", f"{REMOTE}/job")
        sh(f"{ssh} -n 'cd {REMOTE}/tempkl && tar czf - results lean_gate.jsonl' | tar xzf - -C {HERE}")
        print(f"remote rc {rc}", flush=True)
    finally:
        launch.pod.terminate(key, st["id"])
        state.unlink()
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})", flush=True)
    print(f"done: {len(list(Path(HERE, 'results').glob('*.json')))} result files", flush=True)


if __name__ == "__main__":
    main()
