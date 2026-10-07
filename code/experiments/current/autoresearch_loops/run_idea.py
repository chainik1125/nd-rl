#!/usr/bin/env python3
"""Run one pre-registered experiment on several seeds at once, one A40 pod per seed, and merge.

  python3 run_idea.py <NNN>_<slug> --seeds 3,4,5

Before renting pods, validates source hashes, fork and seeds, commits/pushes params.json + patch,
and refuses dirty harness.py / remote.sh / pod.py. Each seed runs as
artifacts/autoresearch_loops/<name>_s<seed>/; the merged
artifacts/autoresearch_loops/<name>/summary.json (what record.py reads) carries the mean over
seeds as `metric`, `metric_diag` and `metric_rl`, plus `seed_metrics`, `seed_metric_rl`,
`seed_status`. Status is "ok" only if every seed is ok: a seed
that times out is the model's fault (slow sampling), not bad luck.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from params import check, commit_and_push, fork_commit, git, registration_files, sha

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "autoresearch_loops")
FROZEN = ("harness.py", "remote.sh", "pod.py")


def preflight(name: str, seeds: list[int], *, allow_new: bool = False) -> None:
    """Reject an unregistered launch before threads, API calls or pod creation."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be nonempty and unique")
    if (Path(HERE) / "results" / name / "results.json").exists():
        raise SystemExit(f"{name} already has a recorded result; do not rerun it")
    registration_files(name, allow_new=allow_new)
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


def register(name: str, seeds: list[int]) -> None:
    preflight(name, seeds, allow_new=True)
    commit_and_push(
        registration_files(name, allow_new=True),
        f"autoresearch {name.replace('_', ' ', 1)}: pre-registered",
    )
    preflight(name, seeds)  # Recheck committed bytes and candidate after hooks, before paid work.


def pod_prefix(name: str) -> str:
    """ndrl-loops-<agent>-<NNN>; pod.py adds `autoresearch-` and run_idea adds `-s<seed>`."""
    number, _, agent = check(name).id.partition("-")
    return f"ndrl-loops-{agent or os.environ.get('ND_AGENT', 'manual')}-{number}"


def ensure_idle(name: str) -> None:
    """Read-only guard against relaunching an interrupted attempt that is still alive."""
    for manifest in Path(ART).glob("loop/*/launch-*.json"):
        entry = json.loads(manifest.read_text())
        pid = entry.get("pid")
        if entry["name"] != name or not pid or pid == os.getpid():
            continue
        process = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # An old manifest alone is not evidence of a live process; PIDs can be reused.
        if "run_idea.py" in process.stdout and name in process.stdout.split():
            raise SystemExit(f"{name}: previous runner {pid} is still active")
    listed = subprocess.run(
        [sys.executable, str(Path(HERE) / "pod.py"), "list"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if listed.returncode:
        raise SystemExit("Cannot verify previous pods are gone: RunPod list failed")
    prefix = re.escape(f"autoresearch-{pod_prefix(name)}")
    for pod in json.loads(listed.stdout)["pods"]:
        if re.fullmatch(rf"{prefix}-s\d+(?:-.*)?", pod["name"]):
            raise SystemExit(
                f"{name}: previous pod {pod['name']} is still active; clean it up first"
            )


def archive_previous_attempt(name: str, seeds: list[int]) -> None:
    """Preserve cancelled-attempt evidence without mixing its scores into the next merge."""
    paths = [Path(ART) / suffix for suffix in [name, *(f"{name}_s{s}" for s in seeds)]]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    archive = Path(ART) / "previous_attempts"
    archive.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=archive))
    for path in existing:
        path.rename(directory / path.name)


def one(name, seed, log_dir, pod_name=None):
    run_name = f"{name}_s{seed}"
    with open(os.path.join(log_dir, f"{run_name}.driver.log"), "w") as log:

        def run(*a):
            return subprocess.run(
                [sys.executable, os.path.join(HERE, "pod.py"), *a, "--pod", pod_name or run_name],
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
            push_in_background(run_name)


def seed_pretrain(summary):
    pre = summary.get("pretrain", {})
    curve = pre.get("curve") or [{}]
    return {
        "val_loss": curve[-1].get("val_loss"),
        "heldout_greedy": pre.get("heldout_greedy"),
        "steps": pre.get("steps"),
    }


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
    base["seed_metric_diag"] = {str(s): summaries[s].get("metric_diag") for s in ok}
    base["seeds"] = seeds
    base["seed_pretrain"] = {str(s): seed_pretrain(summaries[s]) for s in ok}
    gates = [summaries[s]["lean_gate"] for s in ok if summaries[s].get("lean_gate")]
    base["lean_gate"] = {k: sum(g[k] for g in gates) for k in gates[0]} if gates else None
    if len(ok) == len(seeds):
        base["status"] = "ok"
        base["metric"] = round(sum(base["seed_metrics"].values()) / len(ok))
        base["metric_rl"] = round(sum(base["seed_metric_rl"].values()) / len(ok))
        base["metric_diag"] = round(sum(base["seed_metric_diag"].values()) / len(ok))
    else:
        bad = [s for s in seeds if s not in ok]
        base["status"] = summaries.get(bad[0], {}).get("status", "crash")
        base["metric"] = base["metric_rl"] = base["metric_diag"] = None
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


def push_in_background(*names):
    """Upload checkpoints to HF in a detached process: the upload from this Mac is slow and must
    not keep pods billing or the experiment waiting. push_checkpoints deletes a local .pt only
    after HF holds an identical copy, so overlapping pushes cannot lose one."""
    subprocess.Popen(
        [sys.executable, os.path.join(HERE, "pod.py"), "push", *names],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def push_leftover_checkpoints():
    """Checkpoints whose HF push was interrupted (e.g. the session was killed mid-push)."""
    push_in_background()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--check-only", action="store_true", help="Read-only restart validation")
    args = parser.parse_args()
    name, seeds = args.name, [int(x) for x in args.seeds.split(",")]
    if args.check_only:
        preflight(name, seeds)
        ensure_idle(name)
        print(f"Ready to restart {name}; no work launched")
        return
    register(name, seeds)
    ensure_idle(name)
    os.makedirs(ART, exist_ok=True)
    push_leftover_checkpoints()
    archive_previous_attempt(name, seeds)
    session = os.environ.get("ND_AUTORESEARCH_SESSION")
    log_dir, manifest = ART, None
    entry: dict[str, Any] = {}
    pods = [f"{pod_prefix(name)}-s{s}" for s in seeds]
    if session:
        # A separate group lets loop.py cancel this runner even if Codex detached its shell.
        if os.getpgrp() != os.getpid():
            os.setpgid(0, 0)
        directory = Path(session)
        log_dir = str(directory / f"run-{os.getpid()}")
        Path(log_dir).mkdir()
        pods = [f"{pod}-{directory.name}" for pod in pods]
        manifest = directory / f"launch-{os.getpid()}.json"
        entry = {"name": name, "seeds": seeds, "pid": os.getpid(), "pods": pods, "logs": log_dir}
        pending = manifest.with_suffix(".tmp")
        pending.write_text(json.dumps(entry))
        pending.replace(manifest)  # Publish ownership before any paid API calls.
    threads = [
        threading.Thread(target=one, args=(name, s, log_dir, pod))
        for s, pod in zip(seeds, pods, strict=True)
    ]
    try:
        for t in threads:
            t.start()
            time.sleep(3)
        for t in threads:
            t.join()
        m = merge(name, seeds)
        keys = ("status", "metric", "metric_diag", "metric_rl", "seed_metrics", "seed_status")
        print(json.dumps({k: m[k] for k in keys}))
    finally:
        if manifest and not any(t.is_alive() for t in threads):
            entry["pid"] = None
            pending = manifest.with_suffix(".tmp")
            pending.write_text(json.dumps(entry))
            pending.replace(manifest)


if __name__ == "__main__":
    main()
