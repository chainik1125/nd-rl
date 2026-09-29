"""Run frozen training stages sequentially under one inherited GPU lease."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    freeze = json.loads(Path(plan["source_freeze"]).read_text())
    out = Path(plan["out"]).resolve()
    if not out.is_relative_to(ROOT / "runs"):
        raise ValueError("Pipeline output must be inside this sprint")
    out.mkdir(parents=True, exist_ok=False)
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("Exactly one inherited GPU lease is required")
    def check_sources():
        if sha(args.plan) != plan_hash:
            raise ValueError("Loaded pipeline plan changed during execution")
        for path, expected in freeze["sha256"].items():
            if sha(path) != expected:
                raise ValueError(f"Frozen pipeline input changed: {path}")
    def event(kind, **values):
        row = dict(event=kind, utc=datetime.now(timezone.utc).isoformat(), **values)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    check_sources()
    (out / "plan.json").write_bytes(plan_bytes)
    deadline = datetime.fromisoformat(plan["not_after_utc"]).timestamp()
    completed = []
    for step in plan["steps"]:
        check_sources()
        if time.time() >= deadline:
            raise TimeoutError("Frozen training deadline reached")
        entry = ROOT / "code" / step["entrypoint"]
        if entry.name not in {"train.py", "capacity.py", "evaluate.py"} or entry.parent != ROOT / "code":
            raise ValueError("Unapproved pipeline entrypoint")
        command = [str(ROOT / ".venv/bin/python"), "-u", str(entry), *step["args"]]
        logfile = out / (step["name"] + ".log")
        if not logfile.resolve().is_relative_to(out):
            raise ValueError("Invalid stage name")
        event("stage_start", name=step["name"], command=command, logfile=str(logfile))
        started = time.time()
        with logfile.open("x") as stream:
            # Same process group as the parent; the owning launcher can stop it.
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            event("stage_pid", name=step["name"], pid=child.pid)
            status = child.wait()
        event("stage_exit", name=step["name"], exit_code=status, elapsed_seconds=time.time()-started)
        if status:
            raise RuntimeError(f"Stage {step['name']} failed with exit code {status}")
        for requirement in step.get("required_records", []):
            record = json.loads(Path(requirement["path"]).read_text())
            for key, value in requirement["equals"].items():
                if record.get(key) != value:
                    raise ValueError(f"Incomplete stage record: {requirement['path']} {key}")
        completed.append(step["name"])
    check_sources()
    (out / "completed.json").write_text(json.dumps({"status": "complete", "stages": completed,
        "plan_sha256": plan_hash, "source_freeze_sha256": sha(plan["source_freeze"])}, indent=2) + "\n")
    event("pipeline_completed", stages=completed)


if __name__ == "__main__":
    main()
