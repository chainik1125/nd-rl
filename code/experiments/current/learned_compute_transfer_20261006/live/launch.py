"""Launch unchanged frozen live inference against the separately sealed transfer data."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OLD = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
PYTHON = "/data/users/dmitry/venvs/proof-state/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(registry_path, model_key, gpu):
    registry = json.loads(registry_path.read_text())
    if registry["status"] != "frozen" or registry["launcher_sha256"] != sha(__file__):
        raise ValueError("Unsealed launcher or protocol")
    if gpu not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("GPU outside root-owned transfer assignment")
    data = Path(registry["data_path"])
    if sha(data) != registry["test_data_sha256"]:
        raise ValueError("Transfer dataset changed")
    for path, expected in registry["implementation_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen implementation changed: {path}")
    for path, expected in registry["parent_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Parent model registration changed")
    entry = next(e for e in registry["registered_models"] if e["model_key"] == model_key)
    config_path = Path(entry["dev_evaluation_path"]) / "config.json"
    if sha(config_path) != entry["dev_config_sha256"]:
        raise ValueError("Original DEV inference configuration changed")
    config = json.loads(config_path.read_text())
    for name in ("base", "checkpoint"):
        if entry.get(name + "_path") and sha(entry[name + "_path"]) != entry[name + "_sha256"]:
            raise ValueError("Frozen model changed")
    out = ART / "live/results" / model_key
    if any(
        p.exists() or p.is_symlink()
        for p in (
            out,
            out.with_suffix(".launch.json"),
            out.with_suffix(".launch.log"),
            out.with_suffix(".launch.py"),
        )
    ):
        raise FileExistsError("Refuse existing transfer output")
    snapshot = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, used, _ = [v.strip() for v in snapshot.split(",")]
    occupants = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if int(used) > 100 or any(
        line.split(",")[0].strip() == uuid for line in occupants.splitlines()
    ):
        raise ValueError("Assigned GPU is occupied")
    command = [
        PYTHON,
        str(OLD / "evaluate_live_path.py"),
        "--source",
        str(ROOT / config["source"]),
        "--base",
        str(ROOT / entry["base_path"]),
        "--data",
        str(data),
        "--out",
        str(out),
        "--frozen-protocol",
        str(registry_path),
        "--seed",
        str(registry["test_seed"]),
        "--repeats",
        str(registry["test_repeats"]),
    ]
    if entry.get("checkpoint_path"):
        command += ["--checkpoint", str(entry["checkpoint_path"])]
    for name in (
        "context",
        "attempt_bound",
        "state_reserve",
        "max_attempts",
        "max_action",
        "max_steps",
        "temperature",
        "cost",
    ):
        command += ["--" + name.replace("_", "-"), str(config[name])]
    return out, command, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    registry_hash = sha(registry_path)
    for model_key in args.models:
        if sha(registry_path) != registry_hash:
            raise ValueError("Registry changed between children")
        out, command, occupancy = preflight(registry_path, model_key, args.gpu)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, out.with_suffix(".launch.py"))
        record = dict(
            model_key=model_key,
            gpu=args.gpu,
            command=command,
            occupancy=occupancy,
            registry_sha256=registry_hash,
            launcher_sha256=sha(__file__),
            started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            status="running",
            pid=os.getpid(),
        )
        ledger = out.with_suffix(".launch.json")
        with ledger.open("x") as stream:
            json.dump(record, stream, indent=2)
        started = time.monotonic()
        with out.with_suffix(".launch.log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "ND_OFFLINE": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record.update(
            status="complete" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            model_child_wall_seconds=time.monotonic() - started,
        )
        ledger.write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{model_key} failed; remaining children not launched")


if __name__ == "__main__":
    main()
