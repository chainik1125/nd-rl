#!/usr/bin/env python3
"""textbook72 at k = 256 for every long_rl checkpoint on HF not yet in passk/results/, over several A40s.

  python passk/backfill.py [n_pods]     # default 4; rerun once more checkpoints upload: scored tags are skipped

Lists runs/long-rl-20261014*/*.pt in the HF repo and tags each <version>_<arm>_s<seed>_<stage> (stage pretrain, final
or roundNNN). Splits the unscored ones over the pods by expected cost (looped samples at 8 loops, about 3x untied);
each pod runs passk/remote.sh on its jobs file, pulling checkpoints from HF itself. Results land in
passk/results/<tag>.json; each pod's log in passk/backfill_<i>.log.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import launch  # noqa: E402  (up(), FILES, and ../autoresearch/pod.py as launch.pod)

HF = "https://huggingface.co/robbiethompson2018/nd-rl-checkpoints"
REMOTE = "/workspace/long_rl"
RUN_DIR = re.compile(r"runs/long-rl-20261014(?:-(v\d+))?_(\w+-\w+)_s(\d+)$")  # v1's directories have no version
# The first passk/launch.py job (17:58 PDT) scored these under short tags; report/export.py maps them back.
LEGACY = {"pretrain": "v2_untied-ei_s1_pretrain", "v2_final": "v2_untied-ei_s1_final",
          "v5_round020": "v5_untied-ei_s1_round020"}


def hf_tree(path):
    url, entries = f"{HF.replace('.co/', '.co/api/models/')}/tree/main/{path}", []
    while url:
        with urllib.request.urlopen(url) as r:
            entries += json.load(r)
            url = m[1] if (m := re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")) else None
    return entries


def checkpoints():
    """[(url, tag, arch)] for every long_rl checkpoint on HF."""
    jobs = []
    for run in hf_tree("runs"):
        if not (m := RUN_DIR.match(run["path"])):
            continue
        version, arm, seed = m[1] or "v1", m[2], m[3]
        for f in hf_tree(run["path"]):
            name = os.path.basename(f["path"])
            if name.endswith(".pt"):
                stage = name.removesuffix(".pt").replace("rl_round_", "round")
                jobs.append((f"{HF}/resolve/main/{f['path']}", f"{version}_{arm}_s{seed}_{stage}", arm.split("-")[0]))
    return jobs


def run_pod(i, jobs_file):
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    state = Path(HERE, f"backfill_pod_{i}.json")
    st = launch.up(key, f"longrl-backfill-{i}", state)
    try:
        ssh = "ssh " + " ".join(launch.pod.ssh_args(st))
        sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731
        sh(f'{ssh} -n "mkdir -p {REMOTE}/passk/textbook72"')
        sh(f"tar czf - -C {os.path.join(HERE, '..')} {launch.FILES} passk/passk.py passk/remote.sh | {ssh} 'tar xzf - -C {REMOTE}'")
        sh(f"tar czf - -C {os.path.join(HERE, '..', '..', 'autoresearch', 'textbook72')} . | {ssh} 'tar xzf - -C {REMOTE}/passk/textbook72'")
        sh(f"{ssh} 'cat > {REMOTE}/passk/jobs.txt' < {jobs_file}")
        rc = launch.pod.run_detached(st, f"bash {REMOTE}/passk/remote.sh {REMOTE}/passk/jobs.txt", f"{REMOTE}/job")
        sh(f"{ssh} -n 'cd {REMOTE} && tar czf - results' | tar xzf - -C {HERE}")
        print(f"remote rc {rc}", flush=True)
    finally:
        launch.pod.terminate(key, st["id"])
        state.unlink()
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})", flush=True)


def main():
    if sys.argv[1:2] == ["pod"]:
        return run_pod(int(sys.argv[2]), sys.argv[3])
    n_pods = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    done = {LEGACY.get(p.stem, p.stem) for p in Path(HERE, "results").glob("*.json")}
    if Path(HERE, "pod.json").exists():  # passk/launch.py is still scoring the legacy tags
        done |= set(LEGACY.values())
    jobs = [j for j in checkpoints() if j[1] not in done]
    if not jobs:
        return print("nothing to score")
    shares, loads = [[] for _ in range(n_pods)], [0] * n_pods
    for job in sorted(jobs, key=lambda j: j[2] != "looped"):  # looped first: largest jobs first balances best
        i = loads.index(min(loads))
        shares[i].append(job)
        loads[i] += 3 if job[2] == "looped" else 1
    procs = []
    for i, share in enumerate(s for s in shares if s):
        jobs_file = Path(HERE, f"backfill_jobs_{i}.txt")
        jobs_file.write_text("".join(" ".join(j) + "\n" for j in share))
        print(f"pod {i}: {len(share)} checkpoints: {' '.join(j[1] for j in share)}", flush=True)
        with open(Path(HERE, f"backfill_{i}.log"), "w") as log:
            procs.append(subprocess.Popen([sys.executable, __file__, "pod", str(i), str(jobs_file)],
                                          stdout=log, stderr=subprocess.STDOUT))
        time.sleep(2)  # stagger the RunPod API calls
    for p in procs:
        p.wait()
    print(f"done: {len(list(Path(HERE, 'results').glob('*.json')))} result files", flush=True)


if __name__ == "__main__":
    main()
