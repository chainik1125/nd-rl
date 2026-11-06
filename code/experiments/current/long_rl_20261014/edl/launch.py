#!/usr/bin/env python3
"""EDL (edl.py) for every long_rl checkpoint on HF not yet in edl/results/, over several A40s.

  python edl/launch.py [n_pods]     # default 4; rerun once more checkpoints upload: scored tags are skipped

Checkpoints and tags come from passk/backfill.py's HF listing (<version>_<arm>_s<seed>_<stage>), minus v1, whose RL
solved nothing. Each pod runs edl/remote.sh on its jobs file; results land in edl/results/<tag>.json, each pod's log
in edl/pod_<i>.log.
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


def run_pod(i, jobs_file):
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    state = Path(HERE, f"pod_{i}.json")
    st = launch.up(key, f"longrl-edl-{i}", state)
    try:
        ssh = "ssh " + " ".join(launch.pod.ssh_args(st))
        sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731
        sh(f"tar czf - -C {os.path.join(HERE, '..')} {launch.FILES} edl/edl.py edl/remote.sh | {ssh} 'mkdir -p {REMOTE} && tar xzf - -C {REMOTE}'")
        sh(f"{ssh} 'cat > {REMOTE}/edl/jobs.txt' < {jobs_file}")
        rc = launch.pod.run_detached(st, f"bash {REMOTE}/edl/remote.sh {REMOTE}/edl/jobs.txt", f"{REMOTE}/job")
        sh(f"{ssh} -n 'cd {REMOTE}/edl && tar czf - results' | tar xzf - -C {HERE}")
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
    done = {p.stem for p in Path(HERE, "results").glob("*.json")}
    jobs = [j for j in backfill.checkpoints() if j[1] not in done and not j[1].startswith("v1_")]
    if not jobs:
        return print("nothing to score")
    procs = []
    for i in range(min(n_pods, len(jobs))):
        share = jobs[i::n_pods]  # the listing is grouped by run, so striding spreads each run's stages over pods
        jobs_file = Path(HERE, f"jobs_{i}.txt")
        jobs_file.write_text("".join(" ".join(j) + "\n" for j in share))
        print(f"pod {i}: {len(share)} checkpoints: {' '.join(j[1] for j in share)}", flush=True)
        with open(Path(HERE, f"pod_{i}.log"), "w") as log:
            procs.append(subprocess.Popen([sys.executable, __file__, "pod", str(i), str(jobs_file)],
                                          stdout=log, stderr=subprocess.STDOUT))
        time.sleep(2)  # stagger the RunPod API calls
    for p in procs:
        p.wait()
    print(f"done: {len(list(Path(HERE, 'results').glob('*.json')))} result files", flush=True)


if __name__ == "__main__":
    main()
