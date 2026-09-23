#!/usr/bin/env python3
"""Run one pre-registered experiment on several seeds at once, one A40 pod per seed, and merge.

  python3 run_idea.py <NNN>_<slug> --seeds 3,4,5

Before renting pods, validates committed params.json + patch, source hashes, fork and seeds,
and refuses dirty harness.py / remote.sh / pod.py. Each seed runs as
artifacts/autoresearch/<name>_s<seed>/; the merged artifacts/autoresearch/<name>/summary.json
(what record.py reads) carries the mean over seeds as `metric` and `metric_rl`, plus
`seed_metrics`, `seed_metric_rl`, `seed_status`. Status is "ok" only if every seed is ok: a seed
that times out is the model's fault (slow sampling), not bad luck.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from params import check, fork_commit, git, sha

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "autoresearch")
FROZEN = ("harness.py", "remote.sh", "pod.py")


def preflight(name: str, seeds: list[int]) -> None:
    """Reject an unregistered launch before threads, API calls or pod creation."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be nonempty and unique")
    for filename in ("params.json", "pretrain.patch"):
        relative = f"results/{name}/{filename}"
        path = Path(HERE) / relative
        if not path.is_file():
            raise SystemExit(f"{relative} missing: pre-register and commit first")
        try:
            committed = git("show", f"HEAD:./{relative}")
        except subprocess.CalledProcessError:
            raise SystemExit(f"{relative} is not committed at HEAD") from None
        if path.read_text() != committed:
            raise SystemExit(f"{relative} differs from committed preregistration")
    p = check(name)
    if name != f"{p.id}_{p.slug}":
        raise SystemExit("experiment name does not match registered id/slug")
    dirty = git("status", "--porcelain", "--", *FROZEN)
    if dirty.strip():
        raise SystemExit(f"frozen files differ from HEAD; commit or revert first:\n{dirty}")
    for filename, expected in (("pretrain.py", p.pretrain_sha), ("harness.py", p.harness_sha)):
        actual = sha(Path(HERE) / filename)
        if actual != expected:
            raise SystemExit(f"{filename} sha {actual} differs from registered {expected}")
    if fork_commit() != p.protocol.fork_commit:
        raise SystemExit("remote.sh fork commit differs from registered protocol")
    for label, registered in (("config", p.config.seed), ("protocol", p.protocol.seed)):
        expected_seeds = [registered] if isinstance(registered, int) else registered
        if seeds != expected_seeds:
            raise SystemExit(f"requested seeds {seeds} differ from {label}.seed {expected_seeds}")


def one(name, seed, log_dir):
    run_name = f"{name}_s{seed}"
    with open(os.path.join(log_dir, f"{run_name}.driver.log"), "w") as log:

        def run(*a):
            return subprocess.run(
                [sys.executable, os.path.join(HERE, "pod.py"), *a, "--pod", run_name],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        t0 = time.time()
        try:
            if run("up").returncode != 0:
                print(f"{run_name}: pod up FAILED", flush=True)
                return
            run("run", run_name, "--seed", str(seed))
            print(f"{run_name}: done after {(time.time() - t0) / 60:.0f} min", flush=True)
        finally:
            run("down")


def merge(name, seeds):
    summaries = {}
    for s in seeds:
        fn = os.path.join(ART, f"{name}_s{s}", "summary.json")
        if os.path.exists(fn):
            with open(fn) as f:
                summaries[s] = json.load(f)
    if not summaries:
        raise SystemExit("no seed produced a summary.json")
    base = dict(summaries[min(summaries)])
    ok = [s for s in seeds if summaries.get(s, {}).get("status") == "ok"]
    base["seed_status"] = {str(s): summaries.get(s, {}).get("status", "missing") for s in seeds}
    base["seed_metrics"] = {str(s): summaries[s]["metric"] for s in ok}
    base["seed_metric_rl"] = {str(s): summaries[s]["metric_rl"] for s in ok}
    base["seeds"] = seeds
    if len(ok) == len(seeds):
        base["status"] = "ok"
        base["metric"] = round(sum(base["seed_metrics"].values()) / len(ok))
        base["metric_rl"] = round(sum(base["seed_metric_rl"].values()) / len(ok))
    else:
        bad = [s for s in seeds if s not in ok]
        base["status"] = summaries.get(bad[0], {}).get("status", "crash")
        base["metric"] = base["metric_rl"] = None
        base["reason"] = f"seeds {bad} not ok: {[base['seed_status'][str(s)] for s in bad]}"
    base["total_secs"] = max(v["total_secs"] for v in summaries.values())
    out = os.path.join(ART, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(base, f, indent=1)
    # keep the merged dir self-contained for anything that reads per-run files
    for s in ok:
        src = os.path.join(ART, f"{name}_s{s}", "summary.json")
        shutil.copy(src, os.path.join(out, f"summary_s{s}.json"))
    return base


def main():
    name = sys.argv[1]
    seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    preflight(name, seeds)
    os.makedirs(ART, exist_ok=True)
    threads = [threading.Thread(target=one, args=(name, s, ART)) for s in seeds]
    for t in threads:
        t.start()
        time.sleep(3)
    for t in threads:
        t.join()
    m = merge(name, seeds)
    keys = ("status", "metric", "metric_rl", "seed_metrics", "seed_status")
    print(json.dumps({k: m[k] for k in keys}))


if __name__ == "__main__":
    main()
