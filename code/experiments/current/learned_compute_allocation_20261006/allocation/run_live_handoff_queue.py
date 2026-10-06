"""Coordinate remaining frozen live evaluations after explicit GPU owner handoffs."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch_live_test import ALLOC, ART, PYTHON, ROOT, sha
from run_final_eval_queue import all_main_complete, name

MODELS = {1: [("cost", 103)], 3: [("cost", 102), ("fixed1_control", 101)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(1, 3), required=True)
    parser.add_argument("--benchmark", choices=("main", "historical"), required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--main-registry", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    registry, main_registry = Path(args.registry), Path(args.main_registry)
    registry_hash, main_hash = sha(registry), sha(main_registry)
    freeze = json.loads(registry.read_text())
    expected_seed, expected_count = (62006, 400) if args.benchmark == "main" else (92006, 72)
    if (
        freeze.get("status") != "frozen"
        or freeze.get("selection_split") != "dev"
        or freeze.get("test_seed") != expected_seed
        or freeze.get("test_repeats") != 2
        or freeze.get("test_theorems") != expected_count
        or freeze.get("launch_helper_sha256") != sha(ALLOC / "launch_live_test.py")
        or (args.benchmark == "main" and registry_hash != main_hash)
    ):
        raise ValueError("Expected root-sealed registry with reviewed launcher")
    if args.benchmark == "historical" and not all_main_complete(
        ART / "allocation/final_main", main_hash
    ):
        raise ValueError("All nine MAIN jobs must complete before historical launch")
    ledger = Path(args.ledger).resolve()
    if not ledger.is_relative_to(ART.resolve()) or any(
        path.exists() for path in (ledger, ledger.with_suffix(".source.py"))
    ):
        raise ValueError("New artifact ledger and source paths required")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(".source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    record = dict(
        owner="/root/allocation",
        gpu=args.gpu,
        benchmark=args.benchmark,
        pid=os.getpid(),
        command=sys.argv,
        registry_sha256=registry_hash,
        main_registry_sha256=main_hash,
        queue_source_sha256=sha(__file__),
        started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        status="running",
        jobs=[],
        authorization="Root scheduling instruction plus explicit preceding-owner GPU release",
    )
    with ledger.open("x") as stream:
        json.dump(record, stream, indent=2)
    started = time.monotonic()
    try:
        for arm, seed in MODELS[args.gpu]:
            if sha(registry) != registry_hash or sha(main_registry) != main_hash:
                raise ValueError("Immutable registry changed")
            output = ART / f"allocation/final_{args.benchmark}" / name(arm, seed)
            data = ART / (
                "data/main_v1/test.jsonl"
                if args.benchmark == "main"
                else "data/textbook72_historical_public_v1/public.jsonl"
            )
            command = [
                PYTHON,
                str(ALLOC / "launch_live_test.py"),
                "--registry",
                str(registry),
                "--data",
                str(data),
                "--benchmark",
                args.benchmark,
                "--arm",
                arm,
                "--training-seed",
                str(seed),
                "--gpu",
                str(args.gpu),
                "--out",
                str(output),
            ]
            if args.benchmark == "historical":
                command.extend(["--main-registry", str(main_registry)])
            job_started = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, ND_OFFLINE="1"))
            record["jobs"].append(
                dict(
                    arm=arm,
                    seed=seed,
                    command=command,
                    output=str(output),
                    returncode=result.returncode,
                    monotonic_process_wall_seconds=time.monotonic() - job_started,
                )
            )
            ledger.write_text(json.dumps(record, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"Registered evaluation failed: {arm}/{seed}")
        record["status"] = "complete"
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        raise
    finally:
        record["whole_queue_wall_seconds"] = time.monotonic() - started
        ledger.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
