#!/usr/bin/env python3
"""Score HF checkpoints on the transfer holdout, spread over many pods at once.

  python3 holdout.py score <pods> <procs-per-pod> <pool[,pool]> <hf-run-dir>...
      e.g. holdout.py score 11 3 holdout 153-claude_lean-width384_s0 ...
  python3 holdout.py record    # artifacts/autoresearch/holdout/*.json -> results/*/holdout.json

`score` rents up to <pods> pods (`pod.py up --pod holdout-<i>`; ND_POD_GPU widens the GPU types)
that pull <procs> checkpoints at a time from one shared queue, run score_holdout.sh, and pull
the per-checkpoint JSON into artifacts/autoresearch/holdout/. Pods are terminated in `finally`;
a pod still waiting for capacity when the queue drains is never rented. Checkpoints already
scored locally are skipped, so rerunning retries only failures.
`record` writes one holdout.json per experiment (per-seed ge7 plus the seed mean) for the
dashboard; the loop never reads it.
"""

import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pod

HERE = Path(__file__).parent
ART = Path(pod.LOCAL_ART).resolve()
OUT = ART / "holdout"


queue: list[tuple[str, str]] = []  # (hf run dir, pool) not yet claimed by a pod
lock = threading.Lock()


def claim(n):
    with lock:
        batch = queue[:n]
        del queue[:n]
        return batch


def run_batch(st, name, batch, procs, log, t0):
    """Score `batch` on the pod, pull the JSON; returns the jobs that produced no result."""
    tag = f"{name}.{int(time.time())}"
    (OUT / f"{tag}.jobs").write_text("".join(f"{run} {pool}\n" for run, pool in batch))
    scp = ["scp", "-q", "-P", str(st["port"]), *pod.ssh_args(st)[2:-1]]
    files = [HERE / "score_holdout.py", HERE / "score_holdout.sh", OUT / f"{tag}.jobs"]
    subprocess.run([*scp, *map(str, files), f"root@{st['ip']}:{pod.REMOTE}/"], check=True)
    remote_out = f"{pod.REMOTE}/holdout"
    # Detached so an ssh drop cannot kill it; poll for the exit-code file.
    # (pod.run_detached streams to sys.stdout, which the threads would share.)
    job = f"bash score_holdout.sh {tag}.jobs {procs} {remote_out} > {tag}.log 2>&1"
    pod.ssh_run(
        st,
        f"cd {pod.REMOTE} && nohup setsid bash -c "
        f"{shlex.quote(job + f'; echo $? > {tag}.rc')} >/dev/null 2>&1 </dev/null &",
    )
    ssh = "ssh " + " ".join(map(shlex.quote, pod.ssh_args(st)))
    while subprocess.run(
        f"{ssh} test -f {pod.REMOTE}/{tag}.rc", shell=True, capture_output=True
    ).returncode:
        if time.time() - t0 > 80 * 60:  # the pod watchdog kills it at 90 min anyway
            print(f"{name}: gave up waiting after 80 min", flush=True)
            break
        time.sleep(20)
    subprocess.run(f"{ssh} cat {pod.REMOTE}/{tag}.log", shell=True, stdout=log)
    subprocess.run(
        f"{ssh} 'tar czf - -C {remote_out} .' | tar xzf - -C {shlex.quote(str(OUT))}",
        shell=True, check=False,
    )  # fmt: skip
    return [(run, pool) for run, pool in batch if not (OUT / f"{run}.{pool}.json").exists()]


def score_on_pod(i, procs):
    """Rent a pod, then keep claiming `procs` jobs from the queue until it is empty."""
    name = f"holdout-{i}"
    with open(OUT / f"{name}.driver.log", "w") as log:

        def pod_cmd(*a):
            return subprocess.Popen(
                [sys.executable, str(HERE / "pod.py"), *a, "--pod", name],
                stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip

        t0, done = time.time(), 0
        try:
            up = pod_cmd("up")
            while up.poll() is None:  # still waiting for capacity or booting
                if not queue:  # other pods took everything: don't rent late
                    up.terminate()
                    up.wait()
                    return
                time.sleep(5)
            if up.returncode != 0:
                print(f"{name}: pod up FAILED", flush=True)
                return
            st = json.loads((ART / f"pod_{name}.json").read_text())
            while batch := claim(procs):
                failed = run_batch(st, name, batch, procs, log, t0)
                done += len(batch) - len(failed)
                if failed:
                    print(f"{name}: FAILED {failed}", flush=True)
            print(f"{name}: {done} scored in {(time.time() - t0) / 60:.0f} min", flush=True)
        finally:
            if (ART / f"pod_{name}.json").exists():
                pod_cmd("down").wait()


def score(n_pods, procs, pools, runs):
    OUT.mkdir(parents=True, exist_ok=True)
    queue.extend(
        (run, pool)
        for run in runs
        for pool in pools.split(",")
        if not (OUT / f"{run}.{pool}.json").exists()
    )
    print(f"{len(queue)} checkpoints to score on up to {n_pods} pods", flush=True)
    threads = [threading.Thread(target=score_on_pod, args=(i, procs)) for i in range(n_pods)]
    for t in threads:
        t.start()
        time.sleep(1)  # stagger the RunPod API calls
    for t in threads:
        t.join()


def record():
    by_exp = {}
    for f in sorted(OUT.glob("*.holdout.json")):
        r = json.loads(f.read_text())
        exp, _, seed = r["run"].rpartition("_s")
        by_exp.setdefault(exp, {})[seed] = r
    for exp, seeds in by_exp.items():
        res = HERE / "results" / exp
        if not res.is_dir():
            print(f"skip {exp}: no results/{exp}")
            continue
        ge7 = {s: r["solved_ge7"] for s, r in sorted(seeds.items())}
        n = next(iter(seeds.values()))["n"]
        out = {
            "pool": "data/ladder/transfer.jsonl (sha1 parity 1)",
            "n": n,
            "stage": "final",
            "holdout_metric": round(sum(ge7.values()) / len(ge7)),
            "seed_holdout": ge7,
            "seed_by_bin": {s: r["by_bin"] for s, r in sorted(seeds.items())},
            "scored_by": "holdout.py / score_holdout.py",
        }
        (res / "holdout.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"{exp}: holdout {out['holdout_metric']} / {n}  seeds {ge7}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        score(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif sys.argv[1:2] == ["record"]:
        record()
    else:
        raise SystemExit(__doc__)
