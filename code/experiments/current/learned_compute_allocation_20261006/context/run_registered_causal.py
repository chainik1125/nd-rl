"""Run an already registered causal panel; never choose models or conditions."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import aggregate_causal
import analyze_causal
import causal_probe

from code.shared_code.evaluation_freeze import validate_frozen_evaluation

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"


def ensure_idle(gpu):
    device = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    occupied = [line for line in processes.splitlines() if line.split(",")[0].strip() == device]
    if occupied:
        raise ValueError(f"GPU {gpu} already occupied: {occupied}")


def prepare(args):
    registry = json.loads(Path(args.frozen_test_protocol).read_text())
    if registry.get("causal_launch_helper_sha256") != causal_probe.digest(__file__):
        raise ValueError("Causal launch helper is not registered")
    inventory = json.loads(Path(args.inventory).read_text())
    public = inventory["public"]
    validate_frozen_evaluation(registry, public, aggregate_causal.signature(args.inventory))
    entries = [run for run in inventory["runs"] if run["training_seed"] == args.training_seed]
    if {entry["condition"] for entry in entries} != set(aggregate_causal.CONDITIONS):
        raise ValueError("Require all four registered conditions for this checkpoint")
    models = [
        model
        for model in registry["registered_models"]
        if model["arm"] == "no_cost" and model["rl_seed"] == args.training_seed
    ]
    if len(models) != 1:
        raise ValueError("Require one registered zero-cost checkpoint")
    model = models[0]
    prepared = []
    for entry in entries:
        signature = entry["intervention_signature"]
        settings = dict(signature["protocol"])
        settings.update(
            source=str(ART / "source/nd-takehome-dan"),
            base=model["base_path"],
            checkpoint=model["checkpoint_path"],
            public=public,
        )
        if causal_probe.intervention_signature(SimpleNamespace(**settings)) != signature:
            raise ValueError("Current intervention differs from registered inventory")
        if (
            Path(entry["generation_registry"]).resolve()
            != Path(args.frozen_test_protocol).resolve()
        ):
            raise ValueError("Inventory names another generation registry")
        validate_frozen_evaluation(registry, public, signature)
        validate_frozen_evaluation(registry, public, analyze_causal.analysis_signature(signature))
        if args.condition and entry["condition"] != args.condition:
            continue
        out = Path(entry["run"])
        analysis_out = Path(entry["analysis"])
        ledger = out.with_suffix(".launch.json")
        log = out.with_suffix(".log")
        snapshot = out.with_suffix(".launcher.py")
        analysis_log = analysis_out.with_suffix(".log")
        if any(path.exists() for path in (out, analysis_out, ledger, log, snapshot, analysis_log)):
            raise ValueError(f"Refusing to overwrite causal output: {out}")
        command = [sys.executable, str(WORK / "context/causal_probe.py")]
        for key, value in settings.items():
            command.append("--" + key.replace("_", "-"))
            command.extend(map(str, value if isinstance(value, list) else [value]))
        command.extend(
            [
                "--out",
                str(out),
                "--split",
                "test",
                "--frozen-test-protocol",
                args.frozen_test_protocol,
            ]
        )
        prepared.append((entry, command, ledger, log, snapshot, analysis_log))
    if not prepared:
        raise ValueError("No registered interventions selected")
    return prepared


def run(args):
    registry_hash = causal_probe.digest(args.frozen_test_protocol)
    inventory_hash = causal_probe.digest(args.inventory)
    jobs = prepare(args)
    if args.preflight_only:
        print(json.dumps({"preflight": "pass", "jobs": len(jobs)}))
        return
    for entry, command, ledger, log, snapshot, analysis_log in jobs:
        if (
            causal_probe.digest(args.frozen_test_protocol) != registry_hash
            or causal_probe.digest(args.inventory) != inventory_hash
        ):
            raise ValueError("Registry or causal inventory changed after preflight")
        ensure_idle(args.gpu)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, snapshot)
        record = {
            "condition": entry["condition"],
            "training_seed": args.training_seed,
            "gpu": args.gpu,
            "command": command,
            "output": entry["run"],
            "registry_sha256": causal_probe.digest(args.frozen_test_protocol),
            "inventory_sha256": causal_probe.digest(args.inventory),
            "launcher_sha256": causal_probe.digest(__file__),
            "intervention_signature": entry["intervention_signature"],
            "status": "started",
        }
        ledger.write_text(json.dumps(record, indent=2) + "\n")
        start = time.monotonic()
        with log.open("x") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), ND_OFFLINE="1"),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record.update(
            process_wall_seconds=time.monotonic() - start,
            returncode=result.returncode,
            status="generation_complete" if result.returncode == 0 else "failed",
        )
        ledger.write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"Causal generation failed; see {log}")
        command = [
            sys.executable,
            str(WORK / "context/analyze_causal.py"),
            "--run",
            entry["run"],
            "--out",
            entry["analysis"],
            "--bootstrap",
            "5000",
            "--seed",
            "81006",
            "--frozen-test-protocol",
            args.frozen_test_protocol,
        ]
        with analysis_log.open("x") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record.update(status="complete", analysis_command=command)
        ledger.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"condition": entry["condition"], "status": "complete"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-test-protocol", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--training-seed", type=int, choices=[101, 102], required=True)
    parser.add_argument("--gpu", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--condition", choices=list(aggregate_causal.CONDITIONS))
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())
