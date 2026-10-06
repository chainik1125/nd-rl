"""Resume remaining registered TEST queues through the approved shared-GPU launcher."""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

from launch_live_test import ALLOC, ART, PYTHON, ROOT, sha

QUEUES = {
    2: [("cost", 101), ("no_cost", 102), ("format_only", None)],
    4: [("no_cost", 101), ("no_cost", 103), ("fixed4_control", 101)],
}
ALL_MAIN = [(arm, seed) for arm in ("cost", "no_cost") for seed in (101, 102, 103)] + [
    ("fixed1_control", 101),
    ("fixed4_control", 101),
    ("format_only", None),
]


def name(arm, seed):
    return arm if seed is None else f"{arm}_s{seed}"


def completion(output, registry_hash):
    ledger = output.with_suffix(".launch.json")
    if not ledger.exists():
        return False
    record = json.loads(ledger.read_text())
    if record["registry_sha256"] != registry_hash:
        raise ValueError("Existing launch belongs to another registry")
    if record["status"] == "failed":
        raise RuntimeError(f"Registered evaluation failed: {output}")
    return record["status"] == "complete" and (output / "summary.json").exists()


def all_main_complete(directory, registry_hash):
    return all(completion(directory / name(arm, seed), registry_hash) for arm, seed in ALL_MAIN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(2, 4), required=True)
    parser.add_argument("--main-registry", required=True)
    parser.add_argument("--historical-registry", required=True)
    parser.add_argument("--main-data", default=str(ART / "data/main_v1/test.jsonl"))
    parser.add_argument("--main-family-data")
    parser.add_argument(
        "--historical-data", default=str(ART / "data/textbook72_historical_public_v1/public.jsonl")
    )
    parser.add_argument(
        "--ledger", required=True, help="New queue ledger; launch ledgers remain per model"
    )
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--operational-addendum", required=True)
    args = parser.parse_args()
    registries = {"main": Path(args.main_registry), "historical": Path(args.historical_registry)}
    hashes = {key: sha(path) for key, path in registries.items()}
    for benchmark, path in registries.items():
        freeze = json.loads(path.read_text())
        seed, count = (62006, 400) if benchmark == "main" else (92006, 72)
        if (
            freeze.get("status") != "frozen"
            or freeze.get("selection_split") != "dev"
            or freeze.get("test_seed") != seed
            or freeze.get("test_repeats") != 2
            or freeze.get("test_theorems") != count
            or freeze.get("launch_helper_sha256") != sha(ALLOC / "launch_live_test.py")
        ):
            raise ValueError("Both root registries must be sealed with the reviewed launch helper")
    ledger = Path(args.ledger).resolve()
    if (
        not ledger.is_relative_to(ART.resolve())
        or ledger.exists()
        or ledger.with_suffix(".source.py").exists()
    ):
        raise ValueError("Queue ledger/source must be new sprint-artifact paths")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(".source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    record = dict(
        gpu=args.gpu,
        owner="/root/allocation",
        pid=os.getpid(),
        command=sys.argv,
        registry_hashes=hashes,
        operational_addendum_sha256=sha(args.operational_addendum),
        queue_source_sha256=sha(__file__),
        started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        status="running",
        jobs=[],
    )
    with ledger.open("x") as stream:
        json.dump(record, stream, indent=2)
    import subprocess

    started = time.monotonic()
    try:
        for benchmark in ("main", "historical"):
            if benchmark == "historical":
                while not all_main_complete(ART / "allocation/final_main", hashes["main"]):
                    if any(sha(path) != hashes[key] for key, path in registries.items()):
                        raise ValueError("Registry changed during MAIN completion barrier")
                    time.sleep(10)
            for arm, seed in QUEUES[args.gpu]:
                if any(sha(path) != hashes[key] for key, path in registries.items()):
                    raise ValueError("Immutable root registry changed")
                output = ART / f"allocation/final_{benchmark}" / name(arm, seed)
                if args.skip_completed and completion(output, hashes[benchmark]):
                    record["jobs"].append(
                        dict(benchmark=benchmark, arm=arm, seed=seed, skipped_completed=True)
                    )
                    continue
                data = args.main_data if benchmark == "main" else args.historical_data
                command = [
                    PYTHON,
                    str(ALLOC / "launch_live_test_shared.py"),
                    "--operational-addendum",
                    args.operational_addendum,
                    "--registry",
                    str(registries[benchmark]),
                    "--benchmark",
                    benchmark,
                    "--data",
                    data,
                    "--arm",
                    arm,
                    "--gpu",
                    str(args.gpu),
                    "--out",
                    str(output),
                ]
                if seed is not None:
                    command.extend(["--training-seed", str(seed)])
                if benchmark == "main" and args.main_family_data:
                    command.extend(["--family-data", args.main_family_data])
                if benchmark == "historical":
                    command.extend(["--main-registry", str(registries["main"])])
                job_started = time.monotonic()
                result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, ND_OFFLINE="1"))
                record["jobs"].append(
                    dict(
                        benchmark=benchmark,
                        arm=arm,
                        seed=seed,
                        command=command,
                        monotonic_process_wall_seconds=time.monotonic() - job_started,
                        returncode=result.returncode,
                        output=str(output),
                    )
                )
                ledger.write_text(json.dumps(record, indent=2) + "\n")
                if result.returncode:
                    raise RuntimeError(f"Registered job failed: {arm}/{seed}/{benchmark}")
        record["status"] = "complete"
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        raise
    finally:
        record["whole_queue_wall_seconds_including_barrier"] = time.monotonic() - started
        ledger.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
