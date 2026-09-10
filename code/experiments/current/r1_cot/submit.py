"""Create and detach an immutable job on simplex1. Run this on simplex1."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def collect_input_hashes(command, cwd):
    """Freeze existing inputs, including those referenced by pipeline configs.

    Future stage outputs do not exist yet and are checked by their consumers.
    Config files themselves are always required and hashed before parsing.
    """
    inputs = {}
    visited_configs = set()
    flags = {
        "--checkpoint",
        "--data",
        "--guard-json",
        "--retention-data",
        "--fresh-config",
        "--config",
        "--final-release-json",
        "--development-release-json",
        "--evaluation-release-json",
    }

    def add(value, required=False):
        path = Path(value)
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        if path.is_file():
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif required:
            raise ValueError(f"Missing required pipeline configuration: {path}")
        return path

    def visit(argv):
        for index, argument in enumerate(argv):
            flag, separator, inline_value = argument.partition("=")
            if flag not in flags:
                continue
            if not separator and index + 1 >= len(argv):
                raise ValueError(f"Missing value for {flag}")
            value = inline_value if separator else argv[index + 1]
            path = add(value, required=flag == "--config")
            if flag != "--config" or path in visited_configs:
                continue
            visited_configs.add(path)
            config = json.loads(path.read_text())
            for stage in config.get("stages", []):
                if stage.get("kind") == "command":
                    visit(stage["command"])
                elif stage.get("kind") == "release_development":
                    for field in ("checkpoints", "data"):
                        for filename in stage[field]:
                            add(filename)
                    add(stage["completed_rl_checkpoint"])

    visit(command)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/users/dmitry/nd-rl/r1-cot-20260909")
    ap.add_argument("--name", required=True)
    ap.add_argument("--snapshot", default="v1")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--deadline", default="2026-09-09T13:25:00+00:00")
    ap.add_argument("--gpu", type=int, action="append")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    snapshot = root / "snapshots" / a.snapshot
    assert snapshot.is_dir()
    command = a.command
    if command and command[0] == "--":
        command = command[1:]
    assert command

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    inputs = {str(p): digest(p) for p in snapshot.rglob("*.py")}
    inputs.update(collect_input_hashes(command, root))
    assert datetime.now(timezone.utc) < datetime.fromisoformat(a.deadline)
    plan = dict(
        root=str(root),
        name=a.name,
        command=command,
        cwd=str(root),
        deadline_utc=a.deadline,
        timeout_seconds=a.timeout,
        env={
            "PYTHONPATH": str(snapshot / "vendor") + ":" + str(snapshot),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        input_sha256=inputs,
        candidate_gpus=a.gpu or [3, 4, 1, 5, 6, 7],
    )
    path = root / "plans" / (a.name + ".json")
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as f:
        json.dump(plan, f, indent=2)
    with (root / "plans" / (a.name + ".dispatch.log")).open("x") as log:
        process = subprocess.Popen(
            ["python3", str(root / "code/dispatch.py"), "--job", str(path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps(dict(name=a.name, dispatcher_pid=process.pid, plan=str(path))))


if __name__ == "__main__":
    main()
