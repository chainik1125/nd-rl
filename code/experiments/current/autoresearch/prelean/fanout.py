#!/usr/bin/env python3
"""holdout.py's multi-pod queue, but for pre-Lean checkpoints (fork ab629c7, harness
5f0740b15fdb, abs tokenizer, nd_verify only). Run from the autoresearch dir:

  python3 prelean/fanout.py <pods> <procs> <jobs-file> [first-pod-index]
  python3 prelean/fanout.py drain <pod-name>...     # finish, pull and kill pods whose driver died

A jobs line is "<hf-run-dir> <pool>" (prelean/jobs.txt: every pre-Lean final.pt). Results land in
artifacts/autoresearch/holdout-prelean/<run>.<pool>.json; scored jobs are skipped on rerun.
"""

import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

P = Path(__file__).parent
sys.path.insert(0, str(P.parent))
import pod  # noqa: E402

ART = Path(pod.LOCAL_ART).resolve()
OUT = ART / "holdout-prelean"
REM = "/workspace/prelean"
FORK = "ab629c7d2cdf3dc50d471e302d14643ba0782749"
queue, lock = [], threading.Lock()


def claim(n):
    with lock:
        batch = queue[:n]
        del queue[:n]
        return batch


def setup(st):
    # Some L40S hosts boot with "CUDA unknown error": torch falls back to CPU, a job takes hours
    cuda = "import torch; assert torch.cuda.is_available()"
    pod.ssh_run(st, f"python3 -c '{cuda}' 2>/dev/null || python -c '{cuda}'")
    pod.ssh_run(
        st,
        f"git -C /workspace/nd-takehome worktree add -f /root/nd-prelean {FORK} > /dev/null && "
        f"gunzip -k /root/nd-prelean/data/train.jsonl.gz && mkdir -p {REM}",
    )
    ssh = "ssh " + " ".join(map(shlex.quote, pod.ssh_args(st)))
    subprocess.run(
        f"tar czf - -C {P} harness.py score_holdout.py score_holdout.sh src "
        f"-C {P.parent} textbook72 | {ssh} tar xzf - -C {REM}",
        shell=True, check=True,
    )  # fmt: skip


def run_batch(st, name, batch, procs, log):
    tag = f"{name}.{int(time.time())}"
    jobs = "".join(f"{run} {pool}\n" for run, pool in batch)
    ssh = "ssh " + " ".join(map(shlex.quote, pod.ssh_args(st)))
    subprocess.run(f"{ssh} 'cat > {REM}/{tag}.jobs'", shell=True, input=jobs.encode(), check=True)
    job = f"bash score_holdout.sh {tag}.jobs {procs} {REM}/out > {tag}.log 2>&1"
    pod.ssh_run(
        st,
        f"cd {REM} && nohup setsid bash -c "
        f"{shlex.quote(job + f'; echo $? > {tag}.rc')} >/dev/null 2>&1 </dev/null &",
    )
    t0 = time.time()
    has_rc = f"{ssh} test -f {REM}/{tag}.rc"
    while subprocess.run(has_rc, shell=True, capture_output=True).returncode:
        if time.time() - t0 > 60 * 60:
            print(f"{name}: batch gave up after 60 min", flush=True)
            break
        time.sleep(20)
    subprocess.run(f"{ssh} cat {REM}/{tag}.log", shell=True, stdout=log)
    subprocess.run(
        f"{ssh} 'tar czf - -C {REM}/out .' | tar xzf - -C {shlex.quote(str(OUT))}",
        shell=True, check=False,
    )  # fmt: skip
    return [(r, p) for r, p in batch if not (OUT / f"{r}.{p}.json").exists()]


def score_on_pod(i, procs):
    name = f"prelean-{i}"
    with open(OUT / f"{name}.driver.log", "a") as log:

        def pod_cmd(*a):
            return subprocess.Popen(
                [sys.executable, str(P.parent / "pod.py"), *a, "--pod", name],
                stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip

        t0, done = time.time(), 0
        try:
            up = pod_cmd("up")
            while up.poll() is None:
                if not queue:
                    up.terminate()
                    up.wait()
                    return
                time.sleep(5)
            if up.returncode != 0:
                print(f"{name}: pod up FAILED", flush=True)
                return
            st = json.loads((ART / f"pod_{name}.json").read_text())
            setup(st)
            print(f"{name}: ready after {(time.time() - t0) / 60:.0f} min", flush=True)
            while batch := claim(procs):
                failed = run_batch(st, name, batch, procs, log)
                done += len(batch) - len(failed)
                if failed:
                    print(f"{name}: FAILED {failed}", flush=True)
            print(f"{name}: {done} scored in {(time.time() - t0) / 60:.0f} min", flush=True)
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}", flush=True)
        finally:
            if (ART / f"pod_{name}.json").exists():
                pod_cmd("down").wait()


def drain(name):
    """A pod whose driver died: wait for its in-flight batches, pull the results, terminate it."""
    st = json.loads((ART / f"pod_{name}.json").read_text())
    ssh = "ssh " + " ".join(map(shlex.quote, pod.ssh_args(st)))
    busy = f"cd {REM} && for j in *.jobs; do [ -f ${{j%.jobs}}.rc ] || exit 1; done"
    t0 = time.time()
    while subprocess.run(f"{ssh} {shlex.quote(busy)}", shell=True, capture_output=True).returncode:
        if time.time() - t0 > 30 * 60:
            break
        time.sleep(20)
    subprocess.run(
        f"{ssh} 'tar czf - -C {REM}/out .' | tar xzf - -C {shlex.quote(str(OUT))}",
        shell=True, check=False,
    )  # fmt: skip
    with open(OUT / f"{name}.driver.log", "a") as log:
        down = [sys.executable, str(P.parent / "pod.py"), "down", "--pod", name]
        subprocess.run(down, stdout=log, stderr=log)
    print(f"{name}: drained", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "drain":
        ts = [threading.Thread(target=drain, args=(n,)) for n in sys.argv[2:]]
        [t.start() for t in ts]
        [t.join() for t in ts]
        sys.exit()
    n_pods, procs, jobs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # pod index offset, to avoid live names
    OUT.mkdir(parents=True, exist_ok=True)
    lines = Path(jobs).read_text().split("\n")[:-1]
    for line in lines:
        run, pool = line.split()
        if not (OUT / f"{run}.{pool}.json").exists():
            queue.append((run, pool))
    print(f"{len(queue)} jobs on up to {n_pods} pods", flush=True)
    threads = [
        threading.Thread(target=score_on_pod, args=(i, procs)) for i in range(first, first + n_pods)
    ]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    left = [line for line in lines if not (OUT / "{}.{}.json".format(*line.split())).exists()]
    print(f"done; {len(left)} jobs without a result", flush=True)
