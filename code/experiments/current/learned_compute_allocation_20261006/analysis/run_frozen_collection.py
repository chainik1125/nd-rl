"""Collect frozen independent TEST pools while keeping outcome-bearing logs unread."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from final_protocol import ART, ROOT, read, sha, validate_parent


def ensure_idle(gpu):
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if any(line.split(",")[0].strip() == uuid for line in processes.splitlines()):
        raise ValueError(f"GPU {gpu} is occupied")


def run(args):
    parent = read(args.parent)
    validate_parent(parent)
    plan = parent["collection"][args.dataset]
    command = plan["argv"]
    input_path = command[command.index("--data") + 1]
    if sha(input_path) != plan["input_sha256"]:
        raise ValueError("Collection theorem source changed")
    checkpoint = ART / "checkpoints/la_T1_best12_s0_r16.pt"
    if sha(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Collection checkpoint changed")
    for name, stage in (
        ("collector", "argv"),
        ("renderer", "renderer_argv"),
        ("sealer", "sealer_argv"),
    ):
        if sha(plan[stage][1]) != plan[name + "_sha256"]:
            raise ValueError(f"Collection {name} changed")
    output = Path(plan["public_raw"]).parent
    private_log = Path(plan["private_stdout_stderr_redirect"])
    ledger = output.with_suffix(".collection_launch.json")
    public_log = output.with_suffix(".public_metadata.log")
    snapshot = output.with_suffix(".collection_sources")
    if any(path.exists() for path in (output, private_log, ledger, public_log, snapshot)):
        raise ValueError("Refusing to overwrite collection output or provenance")
    if args.preflight_only:
        print(json.dumps({"preflight": "pass", "dataset": args.dataset}))
        return
    ensure_idle(args.gpu)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir()
    for path in [
        Path(__file__),
        *[Path(plan[stage][1]) for stage in ("argv", "renderer_argv", "sealer_argv")],
    ]:
        shutil.copy2(path, snapshot / path.name)
    record = {
        "dataset": args.dataset,
        "gpu": args.gpu,
        "launch_clock_utc": args.utc,
        "parent_sha256": sha(args.parent),
        "launcher_sha256": sha(__file__),
        "collection_plan": plan,
        "checkpoint_sha256": sha(checkpoint),
        "status": "started",
        "private_outcomes_read": False,
        "processes": [],
    }
    ledger.write_text(json.dumps(record, indent=2) + "\n")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), ND_OFFLINE="1")
    for stage, path in (
        ("argv", private_log),
        ("renderer_argv", public_log),
        ("sealer_argv", public_log),
    ):
        start = time.monotonic()
        with path.open("a" if stage == "sealer_argv" else "x") as stream:
            result = subprocess.run(
                plan[stage], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
        record["processes"].append(
            {
                "stage": stage,
                "command": plan[stage],
                "monotonic_process_wall_seconds": time.monotonic() - start,
                "returncode": result.returncode,
            }
        )
        record["status"] = "in_progress" if result.returncode == 0 else "failed"
        ledger.write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"Collection stage {stage} failed; outcome log remains unopened")
    record["status"] = "public_seal_complete_pending_registry"
    record["public_seal_sha256"] = sha(output / "public_seal.json")
    ledger.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"dataset": args.dataset, "status": record["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--dataset", choices=["main", "historical"], required=True)
    parser.add_argument("--gpu", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--utc", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())
